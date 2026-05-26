# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch._dynamo.utils import counters

QLINEAR_TARGETS = {
    torch.ops.onednn.qlinear_pointwise.default,
    torch.ops.onednn.qlinear_pointwise.tensor,
}


def _is_qlinear_target(target):
    if target in QLINEAR_TARGETS:
        return True
    return (
        callable(target)
        and getattr(target, "__module__", "")
        == "torch._inductor.fx_passes.quantization"
        and getattr(target, "__name__", "") == "qlinear"
    )


def _get_node_arg(node, kwarg_name, arg_index):
    if kwarg_name in node.kwargs:
        return node.kwargs[kwarg_name]
    return node.args[arg_index]


def _create_constant_tensor_node(graph, gm, base_name, value):
    node_name = base_name
    idx = 1
    while hasattr(gm, node_name):
        node_name = f"{base_name}_{idx}"
        idx += 1
    gm.register_buffer(node_name, value)
    setattr(gm, node_name, value)
    return graph.create_node("get_attr", node_name, (), {})


def _build_concat_arg(graph, gm, args, base_name):
    resolved_values = [_resolve_arg_value(arg, gm) for arg in args]
    if all(isinstance(value, torch.Tensor) for value in resolved_values):
        concat_value = _concat_tensors(resolved_values)
        return _create_constant_tensor_node(graph, gm, base_name, concat_value)

    input_nodes = []
    for idx, arg in enumerate(args):
        if isinstance(arg, torch.fx.Node):
            input_nodes.append(arg)
            continue
        resolved_value = _resolve_arg_value(arg, gm)
        assert isinstance(resolved_value, torch.Tensor)
        input_nodes.append(
            _create_constant_tensor_node(graph, gm, f"{base_name}_{idx}", resolved_value)
        )

    return graph.create_node(
        "call_function",
        torch.ops.aten.cat.default,
        (input_nodes, 0),
    )


def _is_get_attr_node(node):
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def _resolve_arg_value(arg, gm):
    if _is_get_attr_node(arg):
        return getattr(gm, arg.target)
    return arg


def _as_dense_tensor(value):
    if getattr(value, "is_mkldnn", False):
        return value.to_dense().t().contiguous()
    if (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float8_e4m3fn
        and value.dim() == 2
    ):
        return value.t().contiguous()
    return value


def _to_scalar(arg, gm, cast=float):
    value = _resolve_arg_value(arg, gm)
    if isinstance(value, torch.Tensor):
        return cast(value.item())
    if isinstance(value, torch.fx.Node):
        if value.target == torch.ops.aten.full.default:
            return cast(value.args[1])
        if value.target == torch.ops.aten.scalar_tensor.default:
            return cast(value.args[0])
        val = value.meta.get("val")
        assert isinstance(val, torch.Tensor)
        return cast(val.item())
    return cast(value)


def _numel(arg, gm):
    value = _resolve_arg_value(arg, gm)
    if isinstance(value, torch.Tensor):
        return value.numel()
    assert isinstance(value, torch.fx.Node)
    val = value.meta.get("val")
    assert isinstance(val, torch.Tensor)
    return val.numel()


def _concat_tensors(values):
    dense_values = [_as_dense_tensor(value) for value in values]
    if dense_values[0].dim() == 0:
        return torch.stack(dense_values)
    return torch.cat(dense_values, dim=0)


def _is_valid_concat_linear_quantized_fusion(qlinear_nodes, dtype):
    if len(qlinear_nodes) != 3:
        return False

    act = _get_node_arg(qlinear_nodes[0], "x", 0)
    act_scale = _get_node_arg(qlinear_nodes[0], "x_scale", 1)
    act_zp = _get_node_arg(qlinear_nodes[0], "x_zp", 2)
    output_dtype = _get_node_arg(qlinear_nodes[0], "output_dtype", 9)
    postop_name = _get_node_arg(qlinear_nodes[0], "postop_name", 10)
    postop_args = _get_node_arg(qlinear_nodes[0], "postop_args", 11)
    postop_algorithm = _get_node_arg(qlinear_nodes[0], "postop_algorithm", 12)
    with_bias = _get_node_arg(qlinear_nodes[0], "b", 6) is not None
    first_weight = _get_node_arg(qlinear_nodes[0], "packed_weight", 3)

    return all(
        (
            _is_qlinear_target(node.target)
            and _get_node_arg(node, "x", 0) == act
            and _get_node_arg(node, "x_scale", 1) == act_scale
            and (dtype != torch.uint8 or _get_node_arg(node, "x_zp", 2) == act_zp)
            and _is_get_attr_node(_get_node_arg(node, "packed_weight", 3))
            and (
                gemm_idx == 0 or _get_node_arg(node, "packed_weight", 3) != first_weight
            )
            and (
                (_get_node_arg(node, "b", 6) is not None)
                if with_bias
                else (_get_node_arg(node, "b", 6) is None)
            )
            and _get_node_arg(node, "output_dtype", 9) == output_dtype
            and _get_node_arg(node, "postop_name", 10) == postop_name
            and _get_node_arg(node, "postop_args", 11) == postop_args
            and _get_node_arg(node, "postop_algorithm", 12) == postop_algorithm
            and node.meta["val"].dtype == dtype
        )
        for gemm_idx, node in enumerate(qlinear_nodes)
    )


def _collect_first_dequant_on_single_user_chain(start_node, dtype):
    dequant_target = (
        torch.ops.quantized_decomposed.dequantize_per_tensor.default
        if dtype == torch.uint8
        else torch.ops.torchao.dequantize_affine_float8_non_decomposed.default
    )
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        nxt = users[0]
        if nxt.op == "call_function" and nxt.target == dequant_target:
            return nxt
        cur = nxt


def _collect_quant_dequant_pair_on_single_user_chain(start_node, dtype):
    quant_target = (
        torch.ops.quantized_decomposed.quantize_per_tensor.default
        if dtype == torch.uint8
        else torch.ops.torchao.quantize_affine_float8_non_decomposed.default
    )
    dequant_target = (
        torch.ops.quantized_decomposed.dequantize_per_tensor.default
        if dtype == torch.uint8
        else torch.ops.torchao.dequantize_affine_float8_non_decomposed.default
    )
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        nxt = users[0]
        if nxt.op == "call_function" and nxt.target == quant_target:
            q_users = list(nxt.users)
            if len(q_users) != 1:
                return None
            dq = q_users[0]
            if dq.op == "call_function" and dq.target == dequant_target:
                return (nxt, dq)
            return None
        cur = nxt


def _find_node_with_qparam_kwargs_on_single_user_chain(start_node, qparam_kwarg_keys):
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        cur = users[0]
        if all(key in cur.kwargs for key in qparam_kwarg_keys):
            return cur


def _concat_linear_quantized_cpu(graph: torch.fx.Graph):
    """
    Concat Linear optimization pass for quantized qlinear on CPU.
    This pass fuses the original pattern:
    def ...
        return (qlinear_cpu(x, ..., w1, ...), qlinear_cpu(x, ..., w2, ...), ...)
    into a single operation:
    def ...
        concat_res = qlinear_cpu(x, ..., concat_w, ...)
        return split(concat_res, split_size_list)
    """
    gm = graph.owning_module
    processed = set()

    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and _is_qlinear_target(node.target)
            and node not in processed
            and not node._erased
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].device.type == "cpu"
            and node.meta["val"].dtype in (torch.uint8, torch.float8_e4m3fn)
        ):
            dtype = node.meta["val"].dtype
            act = _get_node_arg(node, "x", 0)
            qlinear_users = [
                u
                for u in act.users
                if u.op == "call_function" and _is_qlinear_target(u.target)
            ][::-1]

            if _is_valid_concat_linear_quantized_fusion(qlinear_users, dtype):
                w_scale_args = [
                    _get_node_arg(user, "w_scale", 4) for user in qlinear_users
                ]
                w_scale_numels = [_numel(w_scale, gm) for w_scale in w_scale_args]
                assert all(numel == w_scale_numels[0] for numel in w_scale_numels)
                per_tensor_w_scale = w_scale_numels[0] == 1

                counters["inductor"]["quantized_concat_linear_fusion"] += 1
                if dtype == torch.uint8:
                    qmin = 0
                    qmax = 255
                    global_min = None
                    global_max = None
                    for qlinear_node in qlinear_users:
                        scale = _to_scalar(
                            _get_node_arg(qlinear_node, "output_scale", 7), gm, float
                        )
                        zp = _to_scalar(
                            _get_node_arg(qlinear_node, "output_zero_point", 8), gm, int
                        )
                        cur_min = (qmin - zp) * scale
                        cur_max = (qmax - zp) * scale
                        global_min = (
                            cur_min if global_min is None else min(global_min, cur_min)
                        )
                        global_max = (
                            cur_max if global_max is None else max(global_max, cur_max)
                        )
                    assert global_min is not None and global_max is not None
                    eps = 1e-12
                    fused_output_scale = max(
                        (global_max - global_min) / float(qmax - qmin), eps
                    )
                    fused_output_zero_point = int(
                        round(qmin - global_min / fused_output_scale)
                    )
                    fused_output_zero_point = max(
                        qmin, min(qmax, fused_output_zero_point)
                    )
                else:
                    fused_output_scale = float(
                        max(
                            _to_scalar(
                                _get_node_arg(qlinear_node, "output_scale", 7),
                                gm,
                                float,
                            )
                            for qlinear_node in qlinear_users
                        )
                    )
                    fused_output_zero_point = _get_node_arg(
                        qlinear_users[0], "output_zero_point", 8
                    )

                with graph.inserting_before(node):
                    computation_node_0 = qlinear_users[0]
                    packed_wgts = [
                        getattr(gm, _get_node_arg(user, "packed_weight", 3).target)
                        for user in qlinear_users
                    ]
                    act_shape = list(act.meta["val"].shape)
                    with_bias = _get_node_arg(qlinear_users[0], "b", 6) is not None
                    dense_wgts = [_as_dense_tensor(w) for w in packed_wgts]
                    out_feature_size_list = [w.size(0) for w in dense_wgts]

                    if per_tensor_w_scale and dtype == torch.float8_e4m3fn:
                        fp8_info = torch.finfo(torch.float8_e4m3fn)
                        real_wgts = [
                            dense_wgt.float() * _to_scalar(w_scale, gm, float)
                            for dense_wgt, w_scale in zip(dense_wgts, w_scale_args)
                        ]
                        real_concat_wgt = torch.cat(real_wgts, dim=0)
                        fused_w_scale = (
                            torch.max(torch.abs(real_concat_wgt)).item() / fp8_info.max
                        )
                        concat_wgt_input = torch.clamp(
                            real_concat_wgt / fused_w_scale,
                            fp8_info.min,
                            fp8_info.max,
                        ).to(torch.float8_e4m3fn)
                    elif per_tensor_w_scale:
                        qmin = 0
                        qmax = 255
                        w_zp_args = [
                            _get_node_arg(user, "w_zp", 5) for user in qlinear_users
                        ]
                        w_zp_numels = [_numel(w_zp, gm) for w_zp in w_zp_args]
                        assert all(numel == 1 for numel in w_zp_numels)
                        real_wgts = [
                            (dense_wgt.float() - _to_scalar(w_zp, gm, int))
                            * _to_scalar(w_scale, gm, float)
                            for dense_wgt, w_scale, w_zp in zip(
                                dense_wgts, w_scale_args, w_zp_args
                            )
                        ]
                        real_concat_wgt = torch.cat(real_wgts, dim=0)
                        eps = 1e-12
                        global_min = real_concat_wgt.min().item()
                        global_max = real_concat_wgt.max().item()
                        fused_w_scale = max(
                            (global_max - global_min) / float(qmax - qmin), eps
                        )
                        fused_w_zp = int(round(qmin - global_min / fused_w_scale))
                        fused_w_zp = max(qmin, min(qmax, fused_w_zp))
                        concat_wgt_input = torch.clamp(
                            torch.round(real_concat_wgt / fused_w_scale + fused_w_zp),
                            qmin,
                            qmax,
                        ).to(torch.uint8)
                    else:
                        concat_wgt_input = torch.cat(dense_wgts, dim=0)

                    concat_wgt = torch.ops.onednn.qlinear_prepack(
                        concat_wgt_input, act_shape
                    )

                    concat_w_node = _create_constant_tensor_node(
                        graph,
                        gm,
                        f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_concat",
                        concat_wgt,
                    )

                    with graph.inserting_after(concat_w_node):
                        if per_tensor_w_scale:
                            concat_wgt_scales_node = _create_constant_tensor_node(
                                graph,
                                gm,
                                f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_scales_concat",
                                torch.tensor([fused_w_scale]),
                            )
                        else:
                            concat_wgt_scales_node = _build_concat_arg(
                                graph,
                                gm,
                                w_scale_args,
                                f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_scales_concat",
                            )

                    if dtype == torch.uint8:
                        with graph.inserting_after(concat_wgt_scales_node):
                            if per_tensor_w_scale:
                                concat_wgt_qzeros_node = _create_constant_tensor_node(
                                    graph,
                                    gm,
                                    f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_qzeros_concat",
                                    torch.tensor([fused_w_zp], dtype=torch.int32),
                                )
                            else:
                                concat_wgt_qzeros_node = _build_concat_arg(
                                    graph,
                                    gm,
                                    [
                                        _get_node_arg(user, "w_zp", 5)
                                        for user in qlinear_users
                                    ],
                                    f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_qzeros_concat",
                                )
                        node_before_linear = concat_wgt_qzeros_node
                    else:
                        concat_wgt_qzeros_node = None
                        node_before_linear = concat_wgt_scales_node

                    if with_bias:
                        with graph.inserting_after(node_before_linear):
                            concat_bias_node = _build_concat_arg(
                                graph,
                                gm,
                                [_get_node_arg(user, "b", 6) for user in qlinear_users],
                                "concat_bias",
                            )
                        node_before_linear = concat_bias_node
                    else:
                        concat_bias_node = None

                    qlinear_kwargs = {
                        "x": act,
                        "x_scale": _get_node_arg(qlinear_users[0], "x_scale", 1),
                        "x_zp": _get_node_arg(qlinear_users[0], "x_zp", 2)
                        if dtype == torch.uint8
                        else None,
                        "packed_weight": concat_w_node,
                        "w_scale": concat_wgt_scales_node,
                        "w_zp": concat_wgt_qzeros_node,
                        "b": concat_bias_node,
                        "output_scale": fused_output_scale,
                        "output_zero_point": fused_output_zero_point,
                        "output_dtype": _get_node_arg(qlinear_users[0], "output_dtype", 9),
                        "postop_name": _get_node_arg(qlinear_users[0], "postop_name", 10),
                        "postop_args": _get_node_arg(qlinear_users[0], "postop_args", 11),
                        "postop_algorithm": _get_node_arg(
                            qlinear_users[0], "postop_algorithm", 12
                        ),
                    }
                    if node.target in QLINEAR_TARGETS:
                        qlinear_args = (
                            qlinear_kwargs["x"],
                            qlinear_kwargs["x_scale"],
                            qlinear_kwargs["x_zp"],
                            qlinear_kwargs["packed_weight"],
                            qlinear_kwargs["w_scale"],
                            qlinear_kwargs["w_zp"],
                            qlinear_kwargs["b"],
                            qlinear_kwargs["output_scale"],
                            qlinear_kwargs["output_zero_point"],
                            qlinear_kwargs["output_dtype"],
                            qlinear_kwargs["postop_name"],
                            qlinear_kwargs["postop_args"],
                            qlinear_kwargs["postop_algorithm"],
                        )
                        with graph.inserting_after(node_before_linear):
                            new_qlinear_node = graph.create_node(
                                "call_function", node.target, qlinear_args
                            )
                    else:
                        with graph.inserting_after(node_before_linear):
                            new_qlinear_node = graph.create_node(
                                "call_function", node.target, (), qlinear_kwargs
                            )

                    with graph.inserting_after(new_qlinear_node):
                        split_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.split_with_sizes.default,
                            (
                                new_qlinear_node,
                                out_feature_size_list,
                                -1,  # split along the out feature dimension
                            ),
                        )

                    delayed_erase_nodes = []
                    split_outputs = []
                    with graph.inserting_after(split_node):
                        for gemm_idx, user in enumerate(qlinear_users):
                            get_item = graph.create_node(
                                "call_function",
                                operator.getitem,
                                (
                                    split_node,
                                    gemm_idx,
                                ),
                            )
                            split_outputs.append(get_item)
                            user.replace_all_uses_with(get_item)

                            delayed_erase_nodes.append(user)

                    fused_dequant_scale = fused_output_scale
                    if dtype != torch.uint8:
                        with graph.inserting_after(split_outputs[-1]):
                            fused_dequant_scale = _create_constant_tensor_node(
                                graph,
                                gm,
                                f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_output_scale",
                                torch.tensor([fused_output_scale]),
                            )

                    for split_out in split_outputs:
                        dequant_node = _collect_first_dequant_on_single_user_chain(
                            split_out, dtype
                        )
                        if dequant_node is not None:
                            if dtype == torch.uint8:
                                dequant_node.args = (
                                    dequant_node.args[0],
                                    fused_output_scale,
                                    fused_output_zero_point,
                                    dequant_node.args[3],
                                    dequant_node.args[4],
                                    dequant_node.args[5],
                                )
                            else:
                                if "scale" in dequant_node.kwargs:
                                    new_kwargs = dict(dequant_node.kwargs)
                                    new_kwargs["scale"] = fused_dequant_scale
                                    dequant_node.kwargs = new_kwargs
                                else:
                                    assert len(dequant_node.args) >= 2
                                    dequant_node.args = (
                                        dequant_node.args[0],
                                        fused_dequant_scale,
                                        *dequant_node.args[2:],
                                    )

                        qdq_pair = _collect_quant_dequant_pair_on_single_user_chain(
                            split_out, dtype
                        )
                        if qdq_pair is not None:
                            quant_node, dequant_node = qdq_pair
                            dequant_node.replace_input_with(
                                quant_node, quant_node.args[0]
                            )
                            delayed_erase_nodes.append(quant_node)

                    qparam_kwarg_keys = (
                        ("q_scale", "q_zp", "k_scale", "k_zp", "v_scale", "v_zp")
                        if dtype == torch.uint8
                        else ("q_scale", "k_scale", "v_scale")
                    )
                    qparam_nodes = [
                        _find_node_with_qparam_kwargs_on_single_user_chain(
                            split_out,
                            qparam_kwarg_keys,
                        )
                        for split_out in split_outputs
                    ]
                    if (
                        qparam_nodes[0] is not None
                        and qparam_nodes[0] == qparam_nodes[1] == qparam_nodes[2]
                    ):
                        new_kwargs = dict(qparam_nodes[0].kwargs)
                        if dtype == torch.uint8:
                            for key in qparam_kwarg_keys:
                                new_kwargs[key] = (
                                    fused_output_scale
                                    if key.endswith("scale")
                                    else fused_output_zero_point
                                )
                        else:
                            for key in qparam_kwarg_keys:
                                new_kwargs[key] = fused_output_scale
                        qparam_nodes[0].kwargs = new_kwargs

                    for old_node in delayed_erase_nodes:
                        graph.erase_node(old_node)

                processed.update(qlinear_users)


def register_quantized_concat_linear_cpu_pass():
    from torch._inductor import config as inductor_config

    inductor_config.post_grad_custom_post_pass = _concat_linear_quantized_cpu
