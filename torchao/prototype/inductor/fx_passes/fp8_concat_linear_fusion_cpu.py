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
        and getattr(target, "__module__", "") == "torch._inductor.fx_passes.quantization"
        and getattr(target, "__name__", "") == "qlinear"
    )


def _get_node_arg(node, kwarg_name, arg_index):
    if kwarg_name in node.kwargs:
        return node.kwargs[kwarg_name]
    return node.args[arg_index]


def _build_qlinear_node(graph, target, **kwargs):
    if target in QLINEAR_TARGETS:
        ordered_args = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
            kwargs["b"],
            kwargs["output_scale"],
            kwargs["output_zero_point"],
            kwargs["output_dtype"],
            kwargs["postop_name"],
            kwargs["postop_args"],
            kwargs["postop_algorithm"],
        )
        return graph.create_node("call_function", target, ordered_args)
    return graph.create_node("call_function", target, (), kwargs)


def _create_constant_tensor_node(graph, gm, base_name, value):
    node_name = base_name
    idx = 1
    while hasattr(gm, node_name):
        node_name = f"{base_name}_{idx}"
        idx += 1
    gm.register_buffer(node_name, value)
    setattr(gm, node_name, value)
    return graph.create_node("get_attr", node_name, (), {})


def _is_get_attr_node(node):
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def _resolve_arg_value(arg, gm):
    if _is_get_attr_node(arg):
        return getattr(gm, arg.target)
    if isinstance(arg, torch.Tensor):
        return arg
    if isinstance(arg, (int, float)):
        return arg
    return None


def _unpack_weight(value):
    if getattr(value, "is_mkldnn", False):
        return value.to_dense().t().contiguous()
    return value.t().contiguous()


def _concat_weights(values):
    return torch.cat([_unpack_weight(value) for value in values], dim=0)


def _concat_tensors(values):
    if values[0].dim() == 0:
        return torch.stack(values)
    return torch.cat(values, dim=0)


def _build_concat_arg(graph, gm, args, dim, base_name):
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
        if not isinstance(resolved_value, torch.Tensor):
            return None
        constant_node = _create_constant_tensor_node(
            graph,
            gm,
            f"{base_name}_{idx}",
            resolved_value,
        )
        input_nodes.append(constant_node)

    return graph.create_node(
        "call_function",
        torch.ops.aten.cat.default,
        (input_nodes, dim),
    )


def _to_scalar(arg, gm, cast=float):
    value = _resolve_arg_value(arg, gm)
    if isinstance(value, (int, float)):
        return cast(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return cast(value.item())
    return None


def _get_fused_output_scale(qlinear_nodes, gm):
    scales = [
        _to_scalar(_get_node_arg(node, "output_scale", 7), gm, float)
        for node in qlinear_nodes
    ]
    if all(scale is not None for scale in scales):
        return max(scales)
    return _get_node_arg(qlinear_nodes[0], "output_scale", 7)


def _get_fused_output_zero_point(qlinear_nodes):
    return _get_node_arg(qlinear_nodes[0], "output_zero_point", 8)


def _is_fp8_packed_weight(arg, gm):
    if not _is_get_attr_node(arg):
        return False
    value = getattr(gm, arg.target)
    return _unpack_weight(value).dtype == torch.float8_e4m3fn


def _is_valid_concat_linear_fp8_fusion(qlinear_nodes, gm):
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
            and _get_node_arg(node, "x_zp", 2) == act_zp
            and _is_fp8_packed_weight(_get_node_arg(node, "packed_weight", 3), gm)
            and (
                gemm_idx == 0
                or _get_node_arg(node, "packed_weight", 3) != first_weight
            )
            and (((_get_node_arg(node, "b", 6) is not None)) if with_bias else ((_get_node_arg(node, "b", 6) is None)))
            and _get_node_arg(node, "output_dtype", 9) == output_dtype
            and _get_node_arg(node, "postop_name", 10) == postop_name
            and _get_node_arg(node, "postop_args", 11) == postop_args
            and _get_node_arg(node, "postop_algorithm", 12) == postop_algorithm
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].dtype == torch.float8_e4m3fn
        )
        for gemm_idx, node in enumerate(qlinear_nodes)
    )


def _collect_first_fp8_dequant_on_single_user_chain(start_node):
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        nxt = users[0]
        if (
            nxt.op == "call_function"
            and nxt.target
            == torch.ops.torchao.dequantize_affine_float8_non_decomposed.default
        ):
            return nxt
        cur = nxt


def _collect_fp8_quant_dequant_pair_on_single_user_chain(start_node):
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        nxt = users[0]
        if (
            nxt.op == "call_function"
            and nxt.target
            == torch.ops.torchao.quantize_affine_float8_non_decomposed.default
        ):
            q_users = list(nxt.users)
            if len(q_users) != 1:
                return None
            dq = q_users[0]
            if (
                dq.op == "call_function"
                and dq.target
                == torch.ops.torchao.dequantize_affine_float8_non_decomposed.default
            ):
                return (nxt, dq)
            return None
        cur = nxt


def _replace_fp8_dequant_scale(node, fused_output_scale):
    if "scale" in node.kwargs:
        new_kwargs = dict(node.kwargs)
        new_kwargs["scale"] = fused_output_scale
        node.kwargs = new_kwargs
        return
    node.args = (node.args[0], fused_output_scale, *node.args[2:])


_QSDPA_FP8_SCALE_KWARG_KEYS = ("q_scale", "k_scale", "v_scale")


def _find_node_with_qsdpa_scale_kwargs_on_single_user_chain(start_node):
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        cur = users[0]
        if all(key in cur.kwargs for key in _QSDPA_FP8_SCALE_KWARG_KEYS):
            return cur


def _replace_node_qsdpa_scale_kwargs(node, fused_output_scale):
    new_kwargs = dict(node.kwargs)
    for key in _QSDPA_FP8_SCALE_KWARG_KEYS:
        new_kwargs[key] = fused_output_scale
    node.kwargs = new_kwargs


def _concat_linear_fp8_cpu(graph: torch.fx.Graph):
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
        ):
            act = _get_node_arg(node, "x", 0)
            qlinear_users = [
                user
                for user in act.users
                if user.op == "call_function" and _is_qlinear_target(user.target)
            ][::-1]

            if _is_valid_concat_linear_fp8_fusion(qlinear_users, gm):
                counters["inductor"]["fp8_concat_linear_fusion"] += 1
                counters["inductor"]["fp8_concat_linear_nodes"] += len(qlinear_users)
                fused_output_scale = _get_fused_output_scale(qlinear_users, gm)
                fused_output_zero_point = _get_fused_output_zero_point(qlinear_users)

                with graph.inserting_before(node):
                    computation_node_0 = qlinear_users[0]
                    packed_wgts = [
                        getattr(gm, _get_node_arg(user, "packed_weight", 3).target)
                        for user in qlinear_users
                    ]
                    act_shape = None
                    if isinstance(act, torch.fx.Node):
                        act_val = act.meta.get("val")
                        if isinstance(act_val, torch.Tensor):
                            act_shape = list(act_val.shape)
                    with_bias = _get_node_arg(qlinear_users[0], "b", 6) is not None

                    concat_wgt = torch.ops.onednn.qlinear_prepack(
                        _concat_weights(packed_wgts), act_shape
                    )
                    out_feature_size_list = [
                        _unpack_weight(weight).size(0) for weight in packed_wgts
                    ]

                    concat_w_node = _create_constant_tensor_node(
                        graph,
                        gm,
                        f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_concat",
                        concat_wgt,
                    )

                    with graph.inserting_after(concat_w_node):
                        concat_wgt_scales_node = _build_concat_arg(
                            graph,
                            gm,
                            [_get_node_arg(user, "w_scale", 4) for user in qlinear_users],
                            0,
                            f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_scales_concat",
                        )

                    w_zp_args = [_get_node_arg(user, "w_zp", 5) for user in qlinear_users]
                    if all(arg is None for arg in w_zp_args):
                        concat_wgt_qzeros_node = None
                        node_before_linear = concat_wgt_scales_node
                    else:
                        with graph.inserting_after(concat_wgt_scales_node):
                            concat_wgt_qzeros_node = _build_concat_arg(
                                graph,
                                gm,
                                w_zp_args,
                                0,
                                f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_qzeros_concat",
                            )
                        node_before_linear = concat_wgt_qzeros_node

                    if with_bias:
                        with graph.inserting_after(node_before_linear):
                            concat_bias_node = _build_concat_arg(
                                graph,
                                gm,
                                [_get_node_arg(user, "b", 6) for user in qlinear_users],
                                0,
                                "concat_bias",
                            )
                        node_before_linear = concat_bias_node
                    else:
                        concat_bias_node = None

                    with graph.inserting_after(node_before_linear):
                        new_qlinear_node = _build_qlinear_node(
                            graph,
                            node.target,
                            x=act,
                            x_scale=_get_node_arg(qlinear_users[0], "x_scale", 1),
                            x_zp=_get_node_arg(qlinear_users[0], "x_zp", 2),
                            packed_weight=concat_w_node,
                            w_scale=concat_wgt_scales_node,
                            w_zp=concat_wgt_qzeros_node,
                            b=concat_bias_node,
                            output_scale=fused_output_scale,
                            output_zero_point=fused_output_zero_point,
                            output_dtype=_get_node_arg(qlinear_users[0], "output_dtype", 9),
                            postop_name=_get_node_arg(qlinear_users[0], "postop_name", 10),
                            postop_args=_get_node_arg(qlinear_users[0], "postop_args", 11),
                            postop_algorithm=_get_node_arg(qlinear_users[0], "postop_algorithm", 12),
                        )

                    with graph.inserting_after(new_qlinear_node):
                        split_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.split_with_sizes.default,
                            (
                                new_qlinear_node,
                                out_feature_size_list,
                                -1,
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

                    for split_out in split_outputs:
                        dequant_node = _collect_first_fp8_dequant_on_single_user_chain(split_out)
                        if dequant_node is not None:
                            _replace_fp8_dequant_scale(dequant_node, fused_output_scale)

                        qdq_pair = _collect_fp8_quant_dequant_pair_on_single_user_chain(split_out)
                        if qdq_pair is not None:
                            quant_node, dequant_node = qdq_pair
                            dequant_node.replace_input_with(quant_node, quant_node.args[0])
                            _replace_fp8_dequant_scale(dequant_node, fused_output_scale)
                            delayed_erase_nodes.append(quant_node)

                    qsdpa_nodes = [
                        _find_node_with_qsdpa_scale_kwargs_on_single_user_chain(split_out)
                        for split_out in split_outputs
                    ]
                    if (
                        len(qsdpa_nodes) == 3
                        and qsdpa_nodes[0] is not None
                        and qsdpa_nodes[0] == qsdpa_nodes[1] == qsdpa_nodes[2]
                    ):
                        _replace_node_qsdpa_scale_kwargs(
                            qsdpa_nodes[0], fused_output_scale
                        )

                    for old_node in delayed_erase_nodes:
                        graph.erase_node(old_node)

                processed.update(qlinear_users)


def register_fp8_concat_linear_cpu_pass():
    from torch._inductor import config as inductor_config

    inductor_config.post_grad_custom_post_pass = _concat_linear_fp8_cpu