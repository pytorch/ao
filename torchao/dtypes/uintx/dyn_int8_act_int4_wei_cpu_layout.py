# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import operator
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import Layout, PlainLayout, is_device
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_7,
    TORCH_VERSION_AT_LEAST_2_8,
)

from .int4_cpu_layout import (
    Int4CPUAQTTensorImpl,
    _is_float,
)

aten = torch.ops.aten


@dataclass(frozen=True)
class Int8DynamicActInt4WeightCPULayout(Layout):
    """Layout class for da8w4 CPU layout for affine quantized tensor"""

    pass


@register_layout(Int8DynamicActInt4WeightCPULayout)
class DA8W4CPUAQTTensorImpl(Int4CPUAQTTensorImpl):
    """TensorImpl for da8w4 CPU layout for affine quantized tensor
    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 2-d tensor of
    dimension: [n][k / 2] (uint8 dtype)
    It is similar to Int4CPUAQTTensorImpl but with a different memory layout of weight data
    fields:
      packed_weight (torch.Tensor): the 2-d packed tensor in a Int4 CPU layout
      scales (torch.Tensor): the scales Tensor used to map between floating point tensor to quantized tensor
      qzeros (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        compensation: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["layout"] = (
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else packed_weight.layout
        )
        kwargs["dtype"] = packed_weight.dtype
        kwargs["requires_grad"] = False
        shape = packed_weight.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        compensation: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scales = scales
        self.qzeros = qzeros
        self.compensation = compensation
        self.transposed = transposed
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["packed_weight", "scales", "qzeros", "compensation"], [
            self.transposed,
            self._layout,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scales, qzeros, compensation = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scales"],
            tensor_data_dict["qzeros"],
            tensor_data_dict["compensation"],
        )
        (
            transposed,
            _layout,
        ) = tensor_attributes
        return cls(packed_weight, scales, qzeros, compensation, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):
        assert isinstance(_layout, Int8DynamicActInt4WeightCPULayout)
        assert int_data.dtype == torch.uint8, "DA8W4 CPU: expects uint8 weight"
        assert int_data.shape[1] % 2 == 0, "DA8W4 CPU: expects even number of columns"
        if scale.dim() == 1:
            scale.unsqueeze_(-1)
        scale = scale.to(torch.float)
        if zero_point.dim() == 1:
            zero_point.unsqueeze_(-1)

        weight_int4, scales, qzeros, compensation = (
            torch.ops.torchao.da8w4_linear_prepack_cpu(int_data, scale, zero_point)
        )
        return cls(weight_int4, scales, qzeros, compensation, False, _layout)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scales),
            fn(self.qzeros),
            fn(self.compensation),
            self.transposed,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            transposed = DA8W4CPUAQTTensorImpl(
                args[0].packed_weight,
                args[0].scales,
                args[0].qzeros,
                args[0].compensation,
                not args[0].transposed,
                args[0]._layout,
            )
            return return_and_correct_aliasing(func, args, kwargs, transposed)
        else:
            return super().__torch_dispatch__(func, types, args, kwargs)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @property
    def block_size(self):
        assert len(self.packed_weight.shape) == 2
        weight_shape = self.packed_weight.shape
        N = weight_shape[0]
        K = weight_shape[1] * 2
        groups = self.scales.numel() // N
        group_size = K // groups
        return (1, group_size)

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unpack weight by linear(eye(K), packed_weight).t()
        packed_w_shape = self.packed_weight.shape
        if len(packed_w_shape) == 4:
            K = packed_w_shape[1] * packed_w_shape[2]
        else:
            K = packed_w_shape[1]
        x = torch.eye(K).to(torch.uint8)
        x_scale = torch.ones(K).float()
        x_qzero = torch.zeros(K).to(torch.int32)
        w_scale = torch.ones_like(self.scales).float()
        w_qzero = torch.zeros_like(self.qzeros).to(torch.int8)
        plain_weight = torch.ops.torchao.da8w4_linear_cpu.default(
            x,
            x_scale,
            x_qzero,
            self.packed_weight,
            w_scale,
            w_qzero,
            self.compensation,
            None,  # bias
            torch.float,  # out_dtype
        )
        plain_weight = plain_weight.t().contiguous()
        plain_weight = plain_weight.to(torch.int8)

        if self.scales.dim() == 2:
            assert self.qzeros.dim() == 2
            plain_scales = self.scales
            plain_qzeros = self.qzeros
        else:
            assert self.scales.dim() == 3 and self.qzeros.dim() == 3
            packed_shape = self.scales.shape  # [Nc, G, block_n]
            plain_scales = (
                self.scales.permute([0, 2, 1]).contiguous().view([-1, packed_shape[1]])
            )
            plain_qzeros = (
                self.qzeros.permute([0, 2, 1]).contiguous().view([-1, packed_shape[1]])
            )

        return plain_weight, plain_scales, plain_qzeros


def _aqt_is_uint8(aqt):
    """Check if an AffineQuantizedTensor is uint8 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.uint8
        and aqt.quant_min == 0
        and aqt.quant_max == 255
    )


def _aqt_is_int8(aqt):
    """Check if an AffineQuantizedTensor is uint8 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.int8
        and aqt.quant_min == -127
        and aqt.quant_max == 127
    )


def _aqt_is_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.uint8
        and aqt.quant_min == 0
        and aqt.quant_max == 15
    )


def _linear_int8_act_int4_weight_cpu_check(input_tensor, weight_tensor, bias):
    return (
        TORCH_VERSION_AT_LEAST_2_7
        and is_device(input_tensor.device.type, "cpu")
        and is_device(weight_tensor.device.type, "cpu")
        and (bias is None or is_device(bias.device.type, "cpu"))
        and isinstance(input_tensor, AffineQuantizedTensor)
        and (_aqt_is_uint8(input_tensor) or _aqt_is_int8(input_tensor))
        and _is_float(input_tensor.dtype)
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_uint4(weight_tensor)
        and _is_float(weight_tensor.dtype)
        and isinstance(weight_tensor._layout, Int8DynamicActInt4WeightCPULayout)
    )


def _linear_int8_act_int4_weight_cpu_impl(input_tensor, weight_tensor, bias):
    assert TORCH_VERSION_AT_LEAST_2_7, (
        f"Requires PyTorch version at least 2.7, but got: {torch.__version__}"
    )
    if _aqt_is_int8(input_tensor):
        assert TORCH_VERSION_AT_LEAST_2_8, (
            f"Requires PyTorch version at least 2.8, but got: {torch.__version__}"
        )
    assert is_device(input_tensor.device.type, "cpu"), (
        f"For CPU device only but got: {input_tensor.device}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    act_mat = input_tensor
    act = act_mat.tensor_impl.int_data
    act_scales = act_mat.tensor_impl.scale
    act_qzeros = act_mat.tensor_impl.zero_point

    packed_weight = weight_tensor.tensor_impl.packed_weight
    wei_scales = weight_tensor.tensor_impl.scales
    wei_qzeros = weight_tensor.tensor_impl.qzeros
    compensation = weight_tensor.tensor_impl.compensation

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act = act.reshape(-1, act.shape[-1])

    y = torch.ops.torchao.da8w4_linear_cpu.default(
        act.contiguous(),
        act_scales,
        act_qzeros,
        packed_weight,
        wei_scales,
        wei_qzeros,
        compensation,
        bias.float() if bias is not None else bias,  # requires bias to be float
        orig_dtype,  # out_dtype
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    return y.to(orig_dtype)


# Inductor FX passes for concat linear
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
# We always register the pass when importing this file
# but it only takes effect when enable_concat_linear is set to True
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
