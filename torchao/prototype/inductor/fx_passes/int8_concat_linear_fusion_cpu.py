# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch


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
    if not isinstance(value, torch.Tensor):
        return None
    node_name = _make_unique_buffer_name(gm, base_name)
    gm.register_buffer(node_name, value)
    return graph.create_node("get_attr", node_name, (), {})


def _build_concat_arg(graph, gm, args, dim, base_name):
    resolved_values = [_resolve_arg_value(arg, gm) for arg in args]
    if all(isinstance(value, torch.Tensor) for value in resolved_values):
        concat_value = _concat_tensors(resolved_values)
        if concat_value is None:
            return None
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
        if constant_node is None:
            return None
        input_nodes.append(constant_node)

    return graph.create_node(
        "call_function",
        torch.ops.aten.cat.default,
        (input_nodes, dim),
    )


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


def _as_dense_tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    if getattr(value, "is_mkldnn", False):
        return value.to_dense()
    return value


def _to_scalar(arg, gm, cast=float):
    value = _resolve_arg_value(arg, gm)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return cast(value.item())
    if isinstance(value, (int, float)):
        return cast(value)
    return None


def _concat_tensors(values):
    if len(values) == 0:
        return None
    dense_values = [_as_dense_tensor(value) for value in values]
    if not isinstance(dense_values[0], torch.Tensor):
        return None
    if dense_values[0].dim() == 0:
        return torch.stack(dense_values)
    return torch.cat(dense_values, dim=0)


def _concat_packed_weights(packed_weights):
    concat_dense_weight = _concat_tensors(packed_weights)
    if concat_dense_weight is None:
        return (None, None)
    concat_packed_weight = torch.ops.onednn.qlinear_prepack(concat_dense_weight, None)
    return (concat_packed_weight, concat_dense_weight)


def _make_unique_buffer_name(gm, base_name):
    name = base_name
    idx = 1
    while hasattr(gm, name):
        name = f"{base_name}_{idx}"
        idx += 1
    return name


def _compute_merged_qparams(qlinear_nodes, gm):
    qmin = 0
    qmax = 255
    global_min = None
    global_max = None

    for node in qlinear_nodes:
        scale = _to_scalar(_get_node_arg(node, "output_scale", 7), gm, float)
        zp = _to_scalar(_get_node_arg(node, "output_zero_point", 8), gm, int)
        if scale is None or zp is None:
            return None
        cur_min = (qmin - zp) * scale
        cur_max = (qmax - zp) * scale
        global_min = cur_min if global_min is None else min(global_min, cur_min)
        global_max = cur_max if global_max is None else max(global_max, cur_max)

    if global_min is None or global_max is None:
        return None

    eps = 1e-12
    scale = max((global_max - global_min) / float(qmax - qmin), eps)
    zp = int(round(qmin - global_min / scale))
    zp = max(qmin, min(qmax, zp))
    return (float(scale), int(zp))


def _get_fused_output_qparams(qlinear_nodes, gm):
    merged_qparams = _compute_merged_qparams(qlinear_nodes, gm)
    if merged_qparams is not None:
        return merged_qparams
    return (
        _get_node_arg(qlinear_nodes[0], "output_scale", 7),
        _get_node_arg(qlinear_nodes[0], "output_zero_point", 8),
    )


def _is_valid_concat_linear_int8_fusion(qlinear_nodes):
    if len(qlinear_nodes) != 3:
        return False

    computation_op = qlinear_nodes[0].target
    if not _is_qlinear_target(computation_op):
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
            and _is_get_attr_node(_get_node_arg(node, "packed_weight", 3))
            and (
                gemm_idx == 0
                or _get_node_arg(node, "packed_weight", 3) != first_weight
            )
            and (((_get_node_arg(node, "b", 6) is not None)) if with_bias else ((_get_node_arg(node, "b", 6) is None)))
            and _get_node_arg(node, "output_dtype", 9) == output_dtype
            and _get_node_arg(node, "postop_name", 10) == postop_name
            and _get_node_arg(node, "postop_args", 11) == postop_args
            and _get_node_arg(node, "postop_algorithm", 12) == postop_algorithm
        )
        for gemm_idx, node in enumerate(qlinear_nodes)
    )


def _collect_first_dequant_on_single_user_chain(start_node):
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        nxt = users[0]
        if (
            nxt.op == "call_function"
            and nxt.target
            == torch.ops.quantized_decomposed.dequantize_per_tensor.default
        ):
            return nxt
        cur = nxt


def _collect_quant_dequant_pair_on_single_user_chain(start_node):
    cur = start_node
    while True:
        users = list(cur.users)
        if len(users) != 1:
            return None
        nxt = users[0]
        if (
            nxt.op == "call_function"
            and nxt.target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        ):
            q_users = list(nxt.users)
            if len(q_users) != 1:
                return None
            dq = q_users[0]
            if (
                dq.op == "call_function"
                and dq.target
                == torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                return (nxt, dq)
            return None
        cur = nxt


def _concat_linear_int8_cpu(graph: torch.fx.Graph):
    """
    Concat Linear optimization pass for int8 on CPU
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
        ):
            act = _get_node_arg(node, "x", 0)
            qlinear_users = [
                u
                for u in act.users
                if u.op == "call_function" and _is_qlinear_target(u.target)
            ]
            if _is_valid_concat_linear_int8_fusion(qlinear_users):
                fused_output_scale, fused_output_zero_point = _get_fused_output_qparams(
                    qlinear_users, gm
                )

                with graph.inserting_before(node):
                    computation_node_0 = qlinear_users[0]
                    packed_wgts = [
                        getattr(gm, _get_node_arg(user, "packed_weight", 3).target)
                        for user in qlinear_users
                    ]
                    with_bias = _get_node_arg(qlinear_users[0], "b", 6) is not None

                    concat_wgt, concat_dense_wgt = _concat_packed_weights(packed_wgts)
                    if concat_wgt is None or concat_dense_wgt is None:
                        continue

                    out_feature_size_list = [
                        _as_dense_tensor(w).size(0) for w in packed_wgts
                    ]

                    concat_w_node = _create_constant_tensor_node(
                        graph,
                        gm,
                        f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_concat",
                        concat_wgt,
                    )
                    if concat_w_node is None:
                        continue

                    with graph.inserting_after(concat_w_node):
                        concat_wgt_scales_node = _build_concat_arg(
                            graph,
                            gm,
                            [_get_node_arg(user, "w_scale", 4) for user in qlinear_users],
                            0,
                            f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_scales_concat",
                        )
                    if concat_wgt_scales_node is None:
                        continue

                    with graph.inserting_after(concat_wgt_scales_node):
                        concat_wgt_qzeros_node = _build_concat_arg(
                            graph,
                            gm,
                            [_get_node_arg(user, "w_zp", 5) for user in qlinear_users],
                            0,
                            f"{_get_node_arg(computation_node_0, 'packed_weight', 3).target}_qzeros_concat",
                        )
                    if concat_wgt_qzeros_node is None:
                        continue

                    node_before_linear = concat_wgt_qzeros_node
                    if with_bias:
                        with graph.inserting_after(concat_wgt_qzeros_node):
                            concat_bias_node = _build_concat_arg(
                                graph,
                                gm,
                                [_get_node_arg(user, "b", 6) for user in qlinear_users],
                                0,
                                "concat_bias",
                            )
                        if concat_bias_node is None:
                            continue
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

                    for split_out in split_outputs:
                        dequant_node = _collect_first_dequant_on_single_user_chain(split_out)
                        if dequant_node is not None:
                            dequant_node.args = (
                                dequant_node.args[0],
                                fused_output_scale,
                                fused_output_zero_point,
                                dequant_node.args[3],
                                dequant_node.args[4],
                                dequant_node.args[5],
                            )

                        qdq_pair = _collect_quant_dequant_pair_on_single_user_chain(split_out)
                        if qdq_pair is not None:
                            quant_node, dequant_node = qdq_pair
                            dequant_node.replace_input_with(quant_node, quant_node.args[0])
                            delayed_erase_nodes.append(quant_node)

                    for old_node in delayed_erase_nodes:
                        if not old_node._erased:
                            graph.erase_node(old_node)

                processed.update(qlinear_users)


def register_int8_concat_linear_cpu_pass():
    from torch._inductor import config as inductor_config
    inductor_config.post_grad_custom_post_pass = _concat_linear_int8_cpu
