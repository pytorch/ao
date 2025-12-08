# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    _choose_qparams_and_quantize_affine_hqq,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.utils import TorchAOBaseTensor, torch_version_at_least

from .int4_choose_qparams_algorithm import Int4ChooseQParamsAlgorithm

__all__ = [
    "Int4PlainInt32Tensor",
]

aten = torch.ops.aten


class Int4PlainInt32Tensor(TorchAOBaseTensor):
    """
    int4 weight-only quantization on XPU with oneDNN as backend (groupwise quantization only)

    Tensor Attributes:
        qdata: (N, K/8), packed int4 weight, the data type is int32 here with 4*(int4*2), the original data type can be half and bfloat16
        scale: (K/group_size, N), dtype is the same as the original Tensor dtype
        zero_point: (K/group_size, N), dtype is int8

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity.
        shape: shape of the original Tensor

    Optional Tensor Data Attributes:
        act_pre_scale (Optional[Tensor]): Optional scale for activation Tensor, if present,
               we'll multiply activation Tensor with act_pre_scale before applying dynamic
               quantization to activation or running quantized mm op

    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale"]

    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.act_pre_scale = act_pre_scale

    def _quantization_type(self):
        s = f"shape={self.shape}, block_size={self.block_size}, device={self.device}"
        if self.act_pre_scale is not None:
            s += f", act_pre_scale.shape={self.act_pre_scale.shape}"
        return s

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
        int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = Int4ChooseQParamsAlgorithm.TINYGEMM,
    ):
        if w.device.type == "xpu":
            return _from_hp_xpu(cls, w, block_size, int4_choose_qparams_algorithm)
        elif w.device.type == "npu":
            return _from_hp_npu(cls, w, block_size, int4_choose_qparams_algorithm)
        else:
            raise NotImplementedError(
                f"Int4PlainInt32Tensor does not support device '{w.device.type}' yet."
            )


def _from_hp_xpu(
    cls,
    w: torch.Tensor,
    block_size: List[int],
    int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = Int4ChooseQParamsAlgorithm.TINYGEMM,
):
    assert w.ndim == 2 and w.device.type == "xpu", (
        f"Expecting 2D tensor on XPU, but got: {w.shape} on {w.device.type}"
    )
    assert len(block_size) == w.ndim
    assert w.dtype in [torch.float16, torch.bfloat16], (
        f"Expecting float16 or bfloat16 weight tensor, but got: {w.dtype}"
    )
    original_shape = w.shape
    target_dtype = torch.int32
    quant_min = 0
    quant_max = 15

    # 1. use HQQ (Half-Quadratic Quantization) algorithm to compute
    #    scale and zero_point, then convert to the format that's compatible with XPU kernels
    if int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.HQQ:
        import math

        nbits = int(math.log2(quant_max + 1))
        axis = 1
        group_size = block_size[-1]
        compute_dtype = w.dtype
        device = str(w.device)
        int_data, scale, zero_point, _ = _choose_qparams_and_quantize_affine_hqq(
            w,
            nbits=nbits,
            group_size=group_size,
            axis=axis,
            compute_dtype=compute_dtype,
            device=device,
            verbose=False,
            raw_output=False,
        )
        int_data = int_data.to(target_dtype)
    # 2. don't use HQQ, use default choose_qparams_affine algorithm to compute scale and zero_point
    else:
        assert int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.TINYGEMM, (
            f"Unsupported Int4ChooseQParamsAlgorithm: {int4_choose_qparams_algorithm}"
        )
        mapping_type = MappingType.ASYMMETRIC
        eps = 1e-6
        scale_dtype = None
        zero_point_dtype = torch.int32
        scale, zero_point = choose_qparams_affine(
            w,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
        )
        int_data = quantize_affine(
            w,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min,
            quant_max,
        )
    assert int_data.dtype == torch.int32, (
        "torch.ops.aten._convert_weight_to_int4pack expects `int32` dtype"
    )
    packed_weight = (int_data[::, 1::2] << 4 | int_data[::, ::2]).to(torch.uint8)
    packed_weight = torch.ops.aten._convert_weight_to_int4pack(
        packed_weight.contiguous(), 8
    )
    scale = scale.reshape(int_data.shape[0], -1)
    zero_point = zero_point.reshape(int_data.shape[0], -1)
    return Int4PlainInt32Tensor(
        packed_weight,
        scale.transpose(0, 1).contiguous(),
        zero_point.transpose(0, 1).contiguous().to(torch.int8),
        block_size,
        original_shape,
        act_pre_scale=None,
    )


def _from_hp_npu(
    cls,
    w: torch.Tensor,
    block_size: List[int],
    int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = Int4ChooseQParamsAlgorithm.TINYGEMM,
):
    assert (
        torch.accelerator.is_available()
        and torch.accelerator.current_accelerator().type == "npu"
        and torch_version_at_least("2.7.1")
    ), (
        f"PyTorch NPU 2.7.1+ needed for int4 packing and matmul ops, {torch.__version__} found"
    )

    assert w.ndim == 2 and w.device.type == "npu", (
        f"Expecting 2D tensor on NPU, but got: {w.shape} on {w.device.type}"
    )
    assert len(block_size) == w.ndim
    assert w.dtype in [torch.float16, torch.bfloat16], (
        f"Expecting float16 or bfloat16 weight tensor, but got: {w.dtype}"
    )

    group_size = block_size[1]
    k_dim = w.shape[-1]
    assert group_size >= 32 and group_size % 32 == 0 and group_size < k_dim, (
        f"Invalid group_size={group_size}: "
        f"expected to be a multiple of 32, "
        f"in range [32, {k_dim - 1}] for per-group quantization, "
        f"but got group_size={group_size} (k_dim={k_dim})."
    )

    original_shape = w.shape
    target_dtype = torch.int32
    quant_min = -8
    quant_max = 7

    # 1. use HQQ (Half-Quadratic Quantization) algorithm to compute
    #    scale and zero_point, then convert to the format that's compatible with XPU kernels
    if int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.HQQ:
        import math

        nbits = int(math.log2(quant_max - quant_min + 1))
        axis = 1
        compute_dtype = w.dtype
        device = str(w.device)
        int_data, scale, zero_point, _ = _choose_qparams_and_quantize_affine_hqq(
            w,
            nbits=nbits,
            group_size=group_size,
            axis=axis,
            compute_dtype=compute_dtype,
            device=device,
            verbose=False,
            raw_output=False,
        )
        int_data = int_data.to(target_dtype)
    else:
        assert int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.TINYGEMM, (
            f"Unsupported Int4ChooseQParamsAlgorithm: {int4_choose_qparams_algorithm}"
        )
        # 2. don't use HQQ, use default choose_qparams_affine algorithm to compute scale and zero_point
        mapping_type = MappingType.ASYMMETRIC
        eps = 1e-6
        scale_dtype = w.dtype
        zero_point_dtype = w.dtype

        scale, zero_point = choose_qparams_affine(
            w,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
        )

        int_data = quantize_affine(
            w,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min,
            quant_max,
        )

    assert int_data.dtype == torch.int32, (
        "torch.ops.npu.npu_convert_weight_to_int4pack expects `int32` dtype"
    )
    assert int_data.shape[-1] % 8 == 0, (
        f"torch.ops.npu.npu_convert_weight_to_int4pack expects last dim must be aligned to 8,but got {int_data.shape[-1]}"
    )

    packed_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
        int_data.contiguous(), 0
    )

    scale = scale.reshape(int_data.shape[0], -1)
    zero_point = zero_point.reshape(int_data.shape[0], -1)

    return Int4PlainInt32Tensor(
        packed_weight.contiguous(),
        scale.transpose(0, 1).contiguous(),
        zero_point.transpose(0, 1).contiguous(),
        block_size,
        original_shape,
        act_pre_scale=None,
    )


implements = Int4PlainInt32Tensor.implements
implements_torch_function = Int4PlainInt32Tensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    if input_tensor.device.type == "xpu":
        return _linear_xpu(input_tensor, weight_tensor, bias)
    elif input_tensor.device.type == "npu":
        return _linear_npu(input_tensor, weight_tensor, bias)
    else:
        raise NotImplementedError(
            f"Int4PlainInt32Tensor does not support device '{input_tensor.device.type}' yet."
        )


def _linear_xpu(
    input_tensor,
    weight_tensor,
    bias,
):
    assert input_tensor.device.type == "xpu", (
        f"For XPU device only but got: {input_tensor.device}"
    )
    assert isinstance(weight_tensor, Int4PlainInt32Tensor), (
        f"Expected weight_tensor to be Int4PlainInt32Tensor, got: {type(weight_tensor)}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"Shapes of input and weight do not match, input:{input_tensor.shape}, weight: {weight_tensor.shape}"
    )

    if weight_tensor.act_pre_scale is not None:
        input_tensor = input_tensor * weight_tensor.act_pre_scale

    act_mat = input_tensor
    packed_weight = weight_tensor.qdata
    scale = weight_tensor.scale
    zero_point = weight_tensor.zero_point

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
        act_mat, packed_weight, groupsize, scale, zero_point
    )

    # remove out_feature padding
    assert weight_tensor.ndim == 2
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


def _linear_npu(
    input_tensor,
    weight_tensor,
    bias,
):
    assert input_tensor.device.type == "npu", (
        f"For NPU device only but got: {input_tensor.device.type}"
    )
    assert isinstance(weight_tensor, Int4PlainInt32Tensor), (
        f"Expected weight_tensor to be Int4PlainInt32NPUTensor, got: {type(weight_tensor)}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"Shapes of input and weight do not match, input:{input_tensor.shape}, weight: {weight_tensor.shape}"
    )

    if weight_tensor.act_pre_scale is not None:
        input_tensor = input_tensor * weight_tensor.act_pre_scale

    act_mat = input_tensor
    packed_weight = weight_tensor.qdata
    scale = weight_tensor.scale
    zero_point = weight_tensor.zero_point

    orig_act_size = act_mat.shape
    orig_dtype = act_mat.dtype

    # dtype alignment
    if act_mat.dtype == torch.float16:
        scale = scale.to(torch.float16)
        zero_point = zero_point.to(torch.float16)
        if bias is not None:
            bias = bias.to(torch.float16)
    elif act_mat.dtype == torch.bfloat16:
        scale = scale.to(torch.bfloat16)
        zero_point = zero_point.to(torch.bfloat16)
        if bias is not None:
            bias = bias.to(torch.float32)

    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]

    y = torch.ops.npu.npu_weight_quant_batchmatmul(
        x=act_mat,
        weight=packed_weight.transpose(-1, -2),
        antiquant_scale=scale,
        antiquant_offset=zero_point,
        antiquant_group_size=groupsize,
        bias=bias,
    )

    # remove out_feature padding
    assert weight_tensor.ndim == 2
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    return y.to(orig_dtype)


Int4PlainInt32Tensor.__module__ = "torchao.quantization"

# Allow a model with Int4PlainInt32Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4PlainInt32Tensor])
