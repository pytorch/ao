# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "Int4PlainInt32Tensor",
]

aten = torch.ops.aten


class Int4PlainInt32Tensor(TorchAOBaseTensor):
    """
    int4 weight-only quantization on XPU with oneDNN as backend (groupwise quantization only)

    Tensor Attributes:
        qdata: (N, K/8)/(E, N, K/8), packed int4 weight, the data type is int32 here with 4*(int4*2), the original data type can be half and bfloat16
        scale: (K/group_size, N), dtype is the same as the original Tensor dtype, (E, K/group_size, N) for grouped weights
        zero_point: (K/group_size, N), dtype is int8, (E, K/group_size, N) for grouped weights

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

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Reconstruct the original high precision weight from the packed
        int4 (plain int32) representation.

        Currently only supported on XPU, where there is no native op to
        unpack `torch.ops.aten._convert_weight_to_int4pack` output.
        """
        if self.device.type != "xpu":
            raise NotImplementedError(
                f"dequantize is only supported on XPU for Int4PlainInt32Tensor, got: {self.device.type}"
            )
        if output_dtype is None:
            output_dtype = self.dtype

        is_transposed = self.qdata.stride(-2) < self.qdata.stride(-1)
        if is_transposed:
            qdata = self.qdata.transpose(-2, -1)
            scale = self.scale.transpose(-2, -1)
            zero_point = self.zero_point.transpose(-2, -1)
            n_dim, k_dim = self.shape[-1], self.shape[-2]
        else:
            qdata = self.qdata
            scale = self.scale
            zero_point = self.zero_point
            n_dim, k_dim = self.shape[-2], self.shape[-1]
        groupsize = self.block_size[-1]

        def _dequant_2d(qdata, scale, zero_point):
            identity = torch.eye(k_dim, dtype=self.dtype, device=qdata.device)
            w_t = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
                identity,
                qdata.contiguous(),
                groupsize,
                scale.contiguous(),
                zero_point.contiguous(),
            )
            return w_t[:, :n_dim].t().contiguous()

        if len(self.block_size) == 3:
            out = torch.stack(
                [
                    _dequant_2d(qdata[i], scale[i], zero_point[i])
                    for i in range(qdata.shape[0])
                ],
                dim=0,
            )
        else:
            out = _dequant_2d(qdata, scale, zero_point)

        if is_transposed:
            out = out.transpose(-2, -1)

        return out.to(output_dtype)

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
    ):
        if w.device.type == "xpu":
            return _from_hp_xpu(cls, w, block_size)
        elif w.device.type == "npu":
            return _from_hp_npu(cls, w, block_size)
        else:
            raise NotImplementedError(
                f"Int4PlainInt32Tensor does not support device '{w.device.type}' yet."
            )


def _quantize_and_pack_xpu_2d(w: torch.Tensor, block_size: List[int]):
    """Quantize and pack a single 2D (N, K) weight tensor into the plain
    int32 tinygemm packed format used on XPU. Returns (packed_weight, scale,
    zero_point) with scale/zero_point of shape (K/group_size, N).
    """
    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.int32
    quant_min = 0
    quant_max = 15
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
    return (
        packed_weight,
        scale.transpose(0, 1).contiguous(),
        zero_point.transpose(0, 1).contiguous().to(torch.int8),
    )


def _from_hp_xpu(
    cls,
    w: torch.Tensor,
    block_size: List[int],
):
    assert w.ndim in (2, 3) and w.device.type == "xpu", (
        f"Expecting 2D (N, K) or 3D (E, N, K) tensor on XPU, but got: {w.shape} on {w.device.type}"
    )
    assert len(block_size) == w.ndim
    assert w.dtype in [torch.float16, torch.bfloat16], (
        f"Expecting float16 or bfloat16 weight tensor, but got: {w.dtype}"
    )
    original_shape = w.shape

    if w.ndim == 3:  # for moe quant
        packed_weights, scales, zero_points = zip(
            *[
                _quantize_and_pack_xpu_2d(w[i], block_size[1:])
                for i in range(w.shape[0])
            ],
            strict=False,
        )
        packed_weight = torch.stack(packed_weights, dim=0)
        scale = torch.stack(scales, dim=0)
        zero_point = torch.stack(zero_points, dim=0)
    else:
        packed_weight, scale, zero_point = _quantize_and_pack_xpu_2d(w, block_size)

    return Int4PlainInt32Tensor(
        packed_weight,
        scale,
        zero_point,
        block_size,
        original_shape,
        act_pre_scale=None,
    )


def _from_hp_npu(
    cls,
    w: torch.Tensor,
    block_size: List[int],
):
    assert (
        torch.accelerator.is_available()
        and torch.accelerator.current_accelerator().type == "npu"
    ), "NPU device required for int4 packing and matmul ops"

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
    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.int32
    quant_min = -8
    quant_max = 7
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


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args
    assert self.ndim == 3, (
        f"transpose is only supported for 3D grouped/MoE Int4PlainInt32Tensor, got {self.ndim}D"
    )
    valid_dims = ((1, 2), (2, 1), (-1, -2), (-2, -1))
    assert (dim0, dim1) in valid_dims, f"transpose unsupported for {dim0=} {dim1=}"

    new_shape = list(self.shape)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

    new = Int4PlainInt32Tensor(
        func(self.qdata, dim0, dim1),
        func(self.scale, dim0, dim1),
        func(self.zero_point, dim0, dim1),
        self.block_size,
        new_shape,
        act_pre_scale=self.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements([aten._grouped_mm.default])
def _grouped_mm(func, types, args, kwargs):
    """Handles `torch._grouped_mm` when weight (mat_b) is an
    Int4PlainInt32Tensor.
    """
    mat_a, mat_b = args[0], args[1]
    offs = args[2] if len(args) > 2 else kwargs.get("offs", None)
    assert isinstance(mat_b, Int4PlainInt32Tensor), (
        f"Expected mat_b to be Int4PlainInt32Tensor, got: {type(mat_b)}"
    )
    assert offs is not None, "offs is required for _grouped_mm"
    assert mat_b.ndim == 3, (
        f"Int4PlainInt32Tensor grouped_mm expects a 3D weight tensor "
        f"(E, K, N), got shape: {mat_b.shape}"
    )
    is_transposed = mat_b.qdata.stride(-2) < mat_b.qdata.stride(-1)
    assert is_transposed, (
        "Int4PlainInt32Tensor grouped_mm expects mat_b to be transposed"
    )
    assert mat_a.shape[-1] == mat_b.shape[-2], (
        f"Shapes of input and weight do not match, input: {mat_a.shape}, weight: {mat_b.shape}"
    )

    weight_hp = mat_b.dequantize(mat_a.dtype)
    return torch._grouped_mm(mat_a, weight_hp, offs=offs)


Int4PlainInt32Tensor.__module__ = "torchao.quantization"

# Allow a model with Int4PlainInt32Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4PlainInt32Tensor])
