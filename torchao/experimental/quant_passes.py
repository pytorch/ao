# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from collections import defaultdict
from typing import Callable, Optional

import torch
from torch._export.passes.constant_folding import (
    ConstantFolder,
    replace_node_with_constant,
)
from torch.fx import subgraph_rewriter


def constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
    skip_constructors: bool = False,
):
    with torch.utils._python_dispatch._disable_current_modes():
        # The ConstantFolder has a bug where it throws if dequantize_affine is not defined
        # TODO: fix upstream
        try:
            getattr(torch.ops.pt2e_quant, "dequantize_affine")
        except AttributeError:
            setattr(torch.ops.pt2e_quant, "dequantize_affine", None)

        cf = ConstantFolder(gm, skip_constructors)
        cf.run()

        for node, constant in cf.node_replacements.items():
            if constraint_fn is not None and not constraint_fn(node):
                continue
            replace_node_with_constant(gm, node, constant)

        erased_params = []
        # Get all attr users by looking up the graph instead from node.users, because in this case
        # _tensor_constant0 and _tensor_constant0_1 are actually refereing to the same tensor.

        #     opcode         name                 target            args                         kwargs
        # -------------  -------------------  ----------------  ---------------------------  --------
        # placeholder    arg0_1               arg0              ()                           {}
        # get_attr       _tensor_constant0    state             ()                           {}
        # call_function  add                  aten.add.Tensor   (arg0_1, _tensor_constant0)  {}
        # get_attr       _tensor_constant0_1  state             ()                           {}
        # call_function  add_                 aten.add_.Tensor  (_tensor_constant0_1, 1)     {}
        # output         output               output            ([add],)                     {}

        get_attr_node_users = defaultdict(list)
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                get_attr_node_users[node.target].extend(node.users.keys())
        for node in gm.graph.find_nodes(op="get_attr"):
            if node.op == "get_attr" and len(get_attr_node_users[node.target]) == 0:
                if hasattr(gm, node.target):
                    delattr(gm, node.target)
                erased_params.append(node)
        for node in erased_params:
            gm.graph.erase_node(node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()


def _get_q_dq_linear_patterns_replacements_and_filters(
    weight_bit_width, has_weight_zeros, target
):
    glbs = globals()
    glbs["weight_bit_width"] = weight_bit_width
    glbs["target"] = target
    glbs["w_quant_min"] = -(1 << (weight_bit_width - 1))
    glbs["w_quant_max"] = (1 << (weight_bit_width - 1)) - 1
    glbs["a_quant_min"] = -128
    glbs["a_quant_max"] = 127
    glbs["a_mapping_type"] = "ASYMMETRIC"
    glbs["a_scale_dtype"] = torch.float32
    glbs["a_eps"] = None

    lcls = {}

    pattern_str = f"""
def pattern(
    a, a_block_size, a_target_dtype, a_zero_point_dtype,
    w_int_data, w_block_size, w_scale, w_zero_point, w_target_dtype,
    bias):
    a_scale, a_zero_point = torch.ops.quant.choose_qparams_affine.default(
        a,
        a_mapping_type,
        a_block_size,
        a_target_dtype,
        a_quant_min,
        a_quant_max,
        a_eps,
        a_scale_dtype,
        a_zero_point_dtype,
    )
    a_int_data = torch.ops.quant.quantize_affine.default(
        a, a_block_size, a_scale, a_zero_point, a_target_dtype, a_quant_min, a_quant_max, 
    )
    dq_a = torch.ops.quant.dequantize_affine.default(
        a_int_data, a_block_size, a_scale, a_zero_point, a_target_dtype, a_quant_min, a_quant_max
    )
    dq_w = torch.ops.quant.dequantize_affine.default(
        w_int_data,
        w_block_size,
        w_scale,
        w_zero_point,
        w_target_dtype,
        w_quant_min,
        w_quant_max,
        {"'INT'" if has_weight_zeros else "'NONE'"}
    )
    return torch.ops.aten.linear.default(dq_a, dq_w, bias)
"""
    exec(pattern_str, glbs, lcls)
    pattern = lcls["pattern"]

    replacement_str = f"""
def replacement(
    a, a_block_size, a_target_dtype, a_zero_point_dtype,
    w_int_data, w_block_size, w_scale, w_zero_point, w_target_dtype,
    bias,):
    n = w_int_data.size(0)
    k = a_block_size[-1]
    group_size = w_block_size[-1]
    out_shape = a.shape[:-1] + (n,)
    packed_weight = getattr(
        torch.ops.torchao,
        f"_pack_8bit_act_{weight_bit_width}bit_weight",
    )(
        w_int_data.to(torch.int8),
        w_scale.reshape(-1),
        {"w_zero_point.reshape(-1).to(torch.int8)" if has_weight_zeros else "None"},
        group_size,
        bias,
        target,
    )
    return getattr(
        torch.ops.torchao, f"_linear_8bit_act_{weight_bit_width}bit_weight"
    )(a.reshape(-1, k), packed_weight, group_size, n, k).reshape(out_shape)
"""

    exec(replacement_str, glbs, lcls)
    replacement = lcls["replacement"]

    def match_filter(match, x, y):
        def get_val(name):
            node = [n for n in match.nodes_map if n.name == name][0]
            return match.nodes_map[node]

        int_types = [torch.int8, torch.int16, torch.int32, torch.int64]

        a_target_dtype = get_val("a_target_dtype")
        if a_target_dtype not in int_types:
            return False

        a_zero_point_dtype = get_val("a_zero_point_dtype")
        if a_zero_point_dtype not in int_types:
            return False

        # We only want a_block_size with shape [1, ..., 1, k]
        a_block_size = get_val("a_block_size")
        for d in a_block_size[0:-1]:
            if d != 1:
                print("a_block_size not [1, ..., 1, k]")
                return False

        # We only want w_block_size with shape [1, group_size]
        w_block_size = get_val("w_block_size")
        if len(w_block_size) != 2 or w_block_size[0] != 1:
            return False

        return True

    return pattern, replacement, match_filter


def replace_q_dq_patterns_with_quantized_linear_ops_pass(
    ep: torch.export.ExportedProgram,
    target=None,
) -> torch.export.ExportedProgram:
    """
    This replaces Q/DQ patterns with torchao quantized linear ops.
    It is intended for converting Q/DQ nodes exported with QDQLayout to using
    the lowbit quantized linear ops.
    """
    # TODO: figure out how to do this with dynamic_shapes (not saved on EP for easy re-export)
    # See https://fb.workplace.com/groups/1028545332188949/permalink/1185289956514485/
    assert (
        len(ep.range_constraints) == 0
    ), "ExportedProgram with range constraints are not supported"

    # ep.module() unlifts the weight inputs, which we need for constant folding
    gm = ep.module()
    for weight_bit_width, has_weight_zeros in itertools.product(
        range(1, 9), [True, False]
    ):
        pattern, replacement, match_filter = (
            _get_q_dq_linear_patterns_replacements_and_filters(
                weight_bit_width, has_weight_zeros, target
            )
        )
        subgraph_rewriter.replace_pattern_with_filters(
            gm, pattern, replacement, match_filters=[match_filter]
        )

    # Constant fold evaluates and removes the packing ops
    constant_fold(gm)

    # Re-export
    return torch.export.export(gm, *ep.example_inputs)


def _get_q_dq_embedding_patterns_replacements_and_filters(
    weight_bit_width,
):
    w_quant_min = -(1 << (weight_bit_width - 1))
    w_quant_max = (1 << (weight_bit_width - 1)) - 1
    w_target_dtype = torch.int8

    def pattern(
        indices,
        w_int_data,
        w_block_size,
        w_scale,
        w_zero_point,
    ):
        dq_w = torch.ops.quant.dequantize_affine.default(
            w_int_data,
            w_block_size,
            w_scale,
            w_zero_point,
            w_target_dtype,
            w_quant_min,
            w_quant_max,
        )
        return torch.ops.aten.embedding.default(dq_w, indices)

    def replacement(
        indices,
        w_int_data,
        w_block_size,
        w_scale,
        w_zero_point,
    ):
        num_embeddings, embedding_dim = w_int_data.size()
        packed_weight_qvals = getattr(
            torch.ops.torchao, f"_pack_embedding_{weight_bit_width}bit"
        )(w_int_data)
        out_shape = indices.shape + (embedding_dim,)
        group_size = w_block_size[-1]
        n_groups = embedding_dim // group_size
        w_scale = w_scale.reshape(-1, n_groups)
        w_zero_point = w_zero_point.reshape(-1, n_groups)
        return getattr(torch.ops.torchao, f"_embedding_{weight_bit_width}bit")(
            packed_weight_qvals,
            num_embeddings,
            embedding_dim,
            w_scale,
            w_zero_point,
            indices.reshape(-1),
        ).reshape(out_shape)

    def match_filter(match, x, y):
        def get_val(name):
            node = [n for n in match.nodes_map if n.name == name][0]
            return match.nodes_map[node]

        # We only want w_block_size with shape [1, group_size]
        w_block_size = get_val("w_block_size")
        if len(w_block_size) != 2 or w_block_size[0] != 1:
            return False

        return True

    return pattern, replacement, match_filter


def replace_q_dq_patterns_with_quantized_embedding_ops_pass(
    ep: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
    """
    This replaces Q/DQ patterns with torchao quantized embedding ops.
    It is intended for converting Q/DQ nodes exported with QDQLayout to using
    the lowbit quantized embedding ops.
    """
    # TODO: figure out how to do this with dynamic_shapes (not saved on EP for easy re-export)
    # See https://fb.workplace.com/groups/1028545332188949/permalink/1185289956514485/
    assert (
        len(ep.range_constraints) == 0
    ), "ExportedProgram with range constraints are not supported"

    # ep.module() unlifts the weight inputs, which we need for constant folding
    gm = ep.module()
    for weight_bit_width in range(1, 9):
        pattern, replacement, match_filter = (
            _get_q_dq_embedding_patterns_replacements_and_filters(
                weight_bit_width,
            )
        )
        subgraph_rewriter.replace_pattern_with_filters(
            gm, pattern, replacement, match_filters=[match_filter]
        )

    # Constant fold evaluates and removes the packing ops
    constant_fold(gm)

    # Re-export
    return torch.export.export(gm, *ep.example_inputs)
