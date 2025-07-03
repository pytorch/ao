# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
)


# Inductor FX passes for concat linear for DA8W4
def _is_valid_concat_linear_da8w4_fusion(computation_nodes):
    if "CPU" not in torch._C._dispatch_dump("torchao::da8w4_linear_cpu"):
        # cpp kernels not built
        return False
    # OP schema:
    # da8w4_linear_cpu(Tensor input, Tensor input_scales, Tensor input_qzeros, Tensor weight, Tensor weight_scales, Tensor weight_qzeros, Tensor compensation, Tensor? bias, ScalarType output_dtype) -> Tensor
    computation_op = torch.ops.torchao.da8w4_linear_cpu.default
    act = computation_nodes[0].args[0]
    act_scales = computation_nodes[0].args[1]
    act_zp = computation_nodes[0].args[2]
    wgt = computation_nodes[0].args[3]
    in_feature_size = act.meta.get("val").size(1)  # type: ignore[union-attr]
    if len(wgt.meta.get("val").shape) != 4:
        return False
    block_k = wgt.meta.get("val").size(2)  # type: ignore[union-attr]
    with_bias = computation_nodes[0].args[7] is not None
    output_dtype = computation_nodes[0].args[-1]

    def check_in_feature_of_wgt(wgt):
        return (
            wgt.meta.get("val").size(1) * wgt.meta.get("val").size(2) == in_feature_size
        )  # type: ignore[union-attr]

    def check_block_k_of_wgt(wgt):
        return wgt.meta.get("val").size(2) == block_k

    def check_bias(b):
        return (b is not None) if with_bias else (b is None)

    return len(computation_nodes) >= 2 and all(
        (
            node.target == computation_op
            and node.args[0] == act  # share same activation
            and node.args[1] == act_scales  # same act scale
            and node.args[2] == act_zp  # same act zero point
            and check_in_feature_of_wgt(node.args[3])  # same in-feature size
            and (node.args[3] != wgt or gemm_idx == 0)
            and node.args[3].op == "get_attr"  # wgt are all constants
            and check_block_k_of_wgt(node.args[3])  # same block_k
            and check_bias(node.args[7])  # bias is either all None or all not None
            and node.args[-1] == output_dtype  # same output dtype
        )
        for gemm_idx, node in enumerate(computation_nodes)
    )


def _concat_linear_dq8w4_cpu(gm: torch.fx.GraphModule):
    """
    Concat Linear optimization pass for DA8W4 on CPU
    This pass fuses the original pattern:
    def ...
        return (da8w4_linear_cpu(x, ..., w1, ...), da8w4_linear_cpu(x, ..., w2, ...), ...)
    into a single operation:
    def ...
        concat_res = da8w4_linear_cpu(x, ..., concat_w, ...)
        return split(concat_res, split_size_list)
    """
    if "CPU" not in torch._C._dispatch_dump("torchao::da8w4_linear_cpu"):
        # cpp kernels not built
        return
    computation_op = torch.ops.torchao.da8w4_linear_cpu.default
    # OP schema:
    # da8w4_linear_cpu(Tensor input, Tensor input_scales, Tensor input_qzeros, Tensor weight, Tensor weight_scales, Tensor weight_qzeros, Tensor compensation, Tensor? bias, ScalarType output_dtype) -> Tensor
    graph = gm.graph
    for node in graph.find_nodes(op="call_function", target=computation_op):
        if (
            not node._erased
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].device.type == "cpu"
        ):
            act = node.args[0]
            act_scales = node.args[1]
            act_qzeros = node.args[2]
            users = list(act.users)
            if _is_valid_concat_linear_da8w4_fusion(users):
                with graph.inserting_before(node):
                    computation_node_0 = users[0]
                    packed_wgts = [getattr(gm, user.args[3].target) for user in users]
                    out_feature_size_list = [
                        (w.size(0) * w.size(-1) * 2) for w in packed_wgts
                    ]
                    wgt_scales = [getattr(gm, user.args[4].target) for user in users]
                    wgt_qzeros = [getattr(gm, user.args[5].target) for user in users]
                    compensations = [getattr(gm, user.args[6].target) for user in users]
                    bias = []
                    with_bias = users[0].args[7] is not None
                    if with_bias:
                        bias = [getattr(gm, user.args[7].target) for user in users]
                    output_dtype = node.args[-1]
                    # Shape of packed weight: [N/block_n, K/block_k, block_k, block_n/2]
                    # Shape of weight scales/qzeros: [N/block_n, G, block_n]
                    # Shape of compensation: [N/block_n, K/block_k, block_n]
                    # Concat them along N/block_n
                    concat_wgt = torch.cat(packed_wgts, dim=0)
                    concat_w_node_name = computation_node_0.args[3].target + "_concat"
                    concat_wgt_scales = torch.cat(wgt_scales, dim=0)
                    concat_ws_node_name = computation_node_0.args[4].target + "_concat"
                    concat_wgt_qzeros = torch.cat(wgt_qzeros, dim=0)
                    concat_wz_node_name = computation_node_0.args[5].target + "_concat"
                    concat_compensation = torch.cat(compensations, dim=0)
                    concat_comp_node_name = (
                        computation_node_0.args[6].target + "_concat"
                    )
                    concat_bias = torch.cat(bias, dim=0) if with_bias else None
                    concat_bias_node_name = (
                        computation_node_0.args[7].target + "_concat"
                        if with_bias
                        else None
                    )
                    gm.register_buffer(concat_w_node_name, concat_wgt)
                    setattr(gm, concat_w_node_name, concat_wgt)
                    gm.register_buffer(concat_ws_node_name, concat_wgt_scales)
                    setattr(gm, concat_ws_node_name, concat_wgt_scales)
                    gm.register_buffer(concat_wz_node_name, concat_wgt_qzeros)
                    setattr(gm, concat_wz_node_name, concat_wgt_qzeros)
                    gm.register_buffer(concat_comp_node_name, concat_compensation)
                    setattr(gm, concat_comp_node_name, concat_compensation)
                    if with_bias:
                        gm.register_buffer(concat_bias_node_name, concat_bias)
                        setattr(gm, concat_bias_node_name, concat_bias)

                    concat_w_node = graph.create_node(
                        "get_attr", concat_w_node_name, (), {}
                    )
                    with graph.inserting_after(concat_w_node):
                        concat_wgt_scales_node = graph.create_node(
                            "get_attr", concat_ws_node_name, (), {}
                        )
                    with graph.inserting_after(concat_wgt_scales_node):
                        concat_wgt_qzeros_node = graph.create_node(
                            "get_attr", concat_wz_node_name, (), {}
                        )
                    with graph.inserting_after(concat_wgt_qzeros_node):
                        concat_compensation_node = graph.create_node(
                            "get_attr", concat_comp_node_name, (), {}
                        )
                    node_before_linear = concat_compensation_node
                    if with_bias:
                        with graph.inserting_after(concat_compensation_node):
                            concat_bias_node = graph.create_node(
                                "get_attr", concat_bias_node_name, (), {}
                            )
                        node_before_linear = concat_bias_node
                    else:
                        concat_bias_node = None
                    with graph.inserting_after(node_before_linear):
                        new_linear_node = graph.create_node(
                            "call_function",
                            computation_op,
                            (
                                act,
                                act_scales,
                                act_qzeros,
                                concat_w_node,
                                concat_wgt_scales_node,
                                concat_wgt_qzeros_node,
                                concat_compensation_node,
                                concat_bias_node,
                                output_dtype,
                            ),
                        )
                    with graph.inserting_after(new_linear_node):
                        split_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.split_with_sizes.default,
                            (
                                new_linear_node,
                                out_feature_size_list,
                                -1,  # split along the out feature dimension
                            ),
                        )
                    with graph.inserting_after(split_node):
                        for gemm_idx, user in enumerate(users):
                            get_item = graph.create_node(
                                "call_function",
                                operator.getitem,
                                (
                                    split_node,
                                    gemm_idx,
                                ),
                            )
                            with graph.inserting_after(get_item):
                                clone_node = graph.create_node(
                                    "call_function",
                                    torch.ops.aten.clone.default,
                                    (get_item,),
                                    {"memory_format": torch.contiguous_format},
                                )
                                user.replace_all_uses_with(clone_node)
                                graph.erase_node(user)


# Define and register a custom pass for concat linear
# We always register the pass when calling this function
# but it only takes effect when config.cpp.enable_concat_linear is set to True
def register_da8w4_concat_linear_pass():
    from torch._inductor import config as inductor_config

    if TORCH_VERSION_AT_LEAST_2_8:
        from torch._inductor.codegen.common import (
            get_scheduling_for_device,
            get_wrapper_codegen_for_device,
            init_backend_registration,
            register_backend_for_device,
        )
        from torch._inductor.custom_graph_pass import (
            CustomGraphModulePass,
            get_hash_for_files,
        )

        class DA8W4ConcatLinearCpuPass(CustomGraphModulePass):
            def __init__(self):
                super().__init__()

            def __call__(self, gm: torch.fx.GraphModule) -> None:
                if inductor_config.cpp.enable_concat_linear:
                    _concat_linear_dq8w4_cpu(gm)

            def uuid(self) -> bytes:
                return get_hash_for_files((__file__,))

        da8w4_concat_linear_pass = DA8W4ConcatLinearCpuPass()
        device = "cpu"
        init_backend_registration()
        device_scheduling = get_scheduling_for_device(device)
        device_python_wrapper = get_wrapper_codegen_for_device(device, False)
        device_cpp_wrapper = get_wrapper_codegen_for_device(device, True)
        device_custom_pass = da8w4_concat_linear_pass
        register_backend_for_device(
            device,
            device_scheduling,
            device_python_wrapper,
            device_cpp_wrapper,
            device_custom_pass,
        )
