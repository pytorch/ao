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

    def _group_size(self) -> int:
        non_unit = [int(v) for v in self.block_size if int(v) != 1]
        assert len(non_unit) == 1 and non_unit[0] > 1, (
            f"Invalid block_size for Int4PlainInt32Tensor: {self.block_size}. "
            "Expected exactly one non-unit group dimension."
        )
        return non_unit[0]

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize the int4 packed weight back to high-precision dtype.

        Uses the existing _weight_int4pack_mm_with_scales_and_zeros kernel
        with an identity matrix to recover the original weight values.

        Handles logically transposed tensors: qdata/scale/zero_point are always
        stored in the original [N, K/8] / [K/gs, N] layout. If the logical shape
        has been transposed (detected via block_size), the dequantized result is
        transposed to match.
        """
        if self.device.type != "xpu":
            raise NotImplementedError(
                "Int4PlainInt32Tensor.dequantize currently supports only XPU. "
                f"Got device '{self.device.type}'."
            )

        if output_dtype is None:
            output_dtype = self.dtype

        if self.ndim >= 3:
            # Detect if logically transposed: original block_size is [1, 1, gs],
            # after transpose(-2,-1) it becomes [1, gs, 1].
            # Transposed means the group_size is NOT in the last position.
            is_transposed = self.block_size[-1] == 1 and any(
                b != 1 for b in self.block_size[:-1]
            )

            # Dequantize each 2D expert slice in its stored orientation
            E = self.qdata.shape[0]
            # Find the actual group_size (the non-1 value in block_size)
            group_size = self._group_size()

            slices = []
            for i in range(E):
                # qdata[i] is in original [N, K/8] packed format
                # scale[i] is in [K/gs, N] format
                # We need the original 2D shape [N, K] for the identity trick
                # N = self.shape[-2] if not transposed, self.shape[-1] if transposed
                if is_transposed:
                    orig_N = self.shape[-1]
                    orig_K = self.shape[-2]
                else:
                    orig_N = self.shape[-2]
                    orig_K = self.shape[-1]

                identity = torch.eye(orig_K, dtype=output_dtype, device=self.device)
                result = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
                    identity, self.qdata[i], group_size, self.scale[i], self.zero_point[i]
                )
                # result is [K, N_padded], trim to [K, N]
                result = result[:, :orig_N]
                # result is [K, N], transpose to get [N, K]
                slices.append(result.transpose(0, 1).contiguous().to(output_dtype))

            # Stack: [E, N, K] (original orientation)
            stacked = torch.stack(slices, dim=0)

            # If logically transposed, transpose last two dims to match self.shape
            if is_transposed:
                stacked = stacked.transpose(-2, -1).contiguous()

            return stacked

        # 2D case: use the matmul kernel with identity matrix
        # Find group_size from block_size
        group_size = self._group_size()
        K = self.shape[1]
        N = self.shape[0]

        identity = torch.eye(K, dtype=output_dtype, device=self.device)
        result = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
            identity, self.qdata, group_size, self.scale, self.zero_point
        )
        # Trim to original output features (may have been padded)
        result = result[:, :N]
        # result is [K, N], transpose to get [N, K]
        return result.transpose(0, 1).contiguous().to(output_dtype)

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


def _from_hp_xpu(
    cls,
    w: torch.Tensor,
    block_size: List[int],
):
    assert w.device.type == "xpu", (
        f"Expecting tensor on XPU, but got: {w.device.type}"
    )
    assert len(block_size) == w.ndim
    assert w.dtype in [torch.float16, torch.bfloat16], (
        f"Expecting float16 or bfloat16 weight tensor, but got: {w.dtype}"
    )

    if w.ndim >= 3:
        # Quantize each 2D slice independently and stack
        results = [_from_hp_xpu_2d(cls, w[i], block_size[1:]) for i in range(w.shape[0])]
        qdata = torch.stack([r.qdata for r in results], dim=0)
        scale = torch.stack([r.scale for r in results], dim=0)
        zero_point = torch.stack([r.zero_point for r in results], dim=0)
        return Int4PlainInt32Tensor(
            qdata,
            scale,
            zero_point,
            block_size,
            w.shape,
            act_pre_scale=None,
        )
    else:
        return _from_hp_xpu_2d(cls, w, block_size)


def _from_hp_xpu_2d(
    cls,
    w: torch.Tensor,
    block_size: List[int],
):
    """Quantize a single 2D weight tensor on XPU."""
    assert w.ndim == 2, (
        f"Expecting 2D tensor, but got: {w.shape}"
    )
    original_shape = w.shape
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
        f"Int4PlainInt32Tensor transpose only supports 3D tensors, got ndim={self.ndim}"
    )
    valid_dims = ((1, 2), (2, 1), (-1, -2), (-2, -1))
    assert (dim0, dim1) in valid_dims, (
        f"Only transpose of last two dims is supported, got dims {dim0}, {dim1}"
    )

    # For packed int4 tensors, we do NOT physically transpose qdata/scale/zero_point.
    # The packed format from _convert_weight_to_int4pack is not meaningfully transposable.
    # We only update shape and block_size to reflect the logical transpose.
    # dequantize() and grouped_mm know to handle the stored layout correctly.

    # Update block_size by swapping the dimensions
    block_size = self.block_size.copy()
    ndim = len(block_size)
    d0 = dim0 % ndim
    d1 = dim1 % ndim
    block_size[d0], block_size[d1] = block_size[d1], block_size[d0]

    # Update shape by swapping the dimensions
    new_shape = list(self.shape)
    new_shape[d0], new_shape[d1] = new_shape[d1], new_shape[d0]

    new = Int4PlainInt32Tensor(
        self.qdata,       # NOT transposed — packed format is layout-specific
        self.scale,       # NOT transposed — stays in [E, K/gs, N] layout
        self.zero_point,  # NOT transposed — stays in [E, K/gs, N] layout
        block_size,
        new_shape,
        act_pre_scale=self.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements([aten.index.Tensor])
def _(func, types, args, kwargs):
    """Handles tensor[indices] for Int4PlainInt32Tensor expert stacks.

    This is used by the MoE forward pass to select expert weights:
        selected_weights = self.gate_up_proj[expert_ids]
    where gate_up_proj is [E, N, K] and expert_ids is a 1D index tensor.

    Supported forms:
      - indices == [expert_ids]
      - indices == [expert_ids, None, None]
    where expert_ids is a 1D integer tensor. Any other indexing form is rejected.
    """
    self = args[0]
    indices = args[1]

    assert self.ndim == 3, (
        "Int4PlainInt32Tensor aten.index.Tensor currently supports only 3D expert stacks"
    )

    if not isinstance(indices, (list, tuple)):
        raise NotImplementedError(
            "Int4PlainInt32Tensor aten.index.Tensor expects list/tuple indices"
        )

    expert_ids = None
    if len(indices) == 1:
        expert_ids = indices[0]
    elif len(indices) == self.ndim and all(idx is None for idx in indices[1:]):
        expert_ids = indices[0]
    else:
        raise NotImplementedError(
            "Int4PlainInt32Tensor aten.index.Tensor supports only indexing dim0 "
            "with a single 1D integer tensor"
        )

    if not isinstance(expert_ids, torch.Tensor):
        raise NotImplementedError(
            "Int4PlainInt32Tensor aten.index.Tensor requires Tensor expert indices"
        )
    if expert_ids.ndim != 1:
        raise NotImplementedError(
            f"Int4PlainInt32Tensor aten.index.Tensor expects 1D indices, got ndim={expert_ids.ndim}"
        )
    if expert_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise NotImplementedError(
            "Int4PlainInt32Tensor aten.index.Tensor requires integer expert indices"
        )

    if expert_ids.device != self.device:
        expert_ids = expert_ids.to(self.device)

    # Expert stacks are aligned on dim0 for qdata/scale/zero_point.
    new_qdata = self.qdata.index_select(0, expert_ids)
    new_scale = self.scale.index_select(0, expert_ids)
    new_zero_point = self.zero_point.index_select(0, expert_ids)

    new_shape = [new_qdata.shape[0], self.shape[1], self.shape[2]]

    new = Int4PlainInt32Tensor(
        new_qdata,
        new_scale,
        new_zero_point,
        self.block_size.copy(),
        new_shape,
        act_pre_scale=self.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements([aten._grouped_mm.default])
def _(func, types, args, kwargs):
    """Handles torch._grouped_mm when weight (mat_b) is an Int4PlainInt32Tensor.

    Decomposes the grouped matmul into per-expert matmuls using the existing
    int4 kernel (_weight_int4pack_mm_with_scales_and_zeros on XPU).

    The calling convention is:
        torch._grouped_mm(mat_a, weight.transpose(-2, -1), offs=offs)
    where weight is [E, N, K] and after transpose is [E, K, N].
    mat_a is [total_M, K], offs is [E] with cumulative row counts.
    """
    mat_a, mat_b = args[0], args[1]
    offs = args[2] if len(args) > 2 else kwargs.get("offs", None)
    assert isinstance(mat_b, Int4PlainInt32Tensor)
    assert offs is not None, "offs is required for _grouped_mm"
    if mat_b.device.type != "xpu":
        raise NotImplementedError(
            "Int4PlainInt32Tensor grouped_mm fallback currently supports only XPU. "
            f"Got device '{mat_b.device.type}'."
        )

    # Dequantize and call the native grouped_mm
    return torch._grouped_mm(mat_a, mat_b.dequantize(), offs=offs)


@implements([aten.bmm.default])
def _(func, types, args, kwargs):
    """Handles torch.bmm when one operand is an Int4PlainInt32Tensor.

    Used by the MoE batched_linear path:
        torch.bmm(weight, input.unsqueeze(-1))  — weight is [S, N, K] Int4
    or:
        torch.bmm(input.unsqueeze(1), weight)   — weight is [S, K, N] Int4

    Falls back to dequantizing the Int4 tensor and calling native bmm.
    """
    a, b = args[0], args[1]
    if isinstance(a, Int4PlainInt32Tensor):
        if a.device.type != "xpu":
            raise NotImplementedError(
                "Int4PlainInt32Tensor bmm fallback currently supports only XPU. "
                f"Got device '{a.device.type}'."
            )
        return aten.bmm.default(a.dequantize(), b)
    assert isinstance(b, Int4PlainInt32Tensor)
    if b.device.type != "xpu":
        raise NotImplementedError(
            "Int4PlainInt32Tensor bmm fallback currently supports only XPU. "
            f"Got device '{b.device.type}'."
        )
    return aten.bmm.default(a, b.dequantize())


Int4PlainInt32Tensor.__module__ = "torchao.quantization"

# Allow a model with Int4PlainInt32Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4PlainInt32Tensor])
