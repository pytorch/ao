# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025, NVIDIA CORPORATION.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the tensor subclasses to represent the MX format spec from
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Exponent E8M0 encoding details (OCP spec section 5.4.1):
  * bias: 127
  * supported exponent range: -127 to 127
  * infinities: N/A
  * NaN: 11111111
  * Zeros: N/A
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)
from torch.utils._pytree import tree_map

from torchao.utils import torch_version_at_least

# ScalingType and SwizzleType are only available in PyTorch 2.10+
if torch_version_at_least("2.10.0"):
    from torch.nn.functional import ScalingType, SwizzleType

from torchao.prototype.mx_formats.config import (
    MXFP8Dim0CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.constants import (
    BLOCK_SIZE_DEFAULT,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    E8M0_EXPONENT_BIAS,
    E8M0_EXPONENT_NAN_VAL,
    F4_E2M1_MAX,
    F4_E2M1_MAX_POW2,
    F6_E2M3_MAX,
    F6_E2M3_MAX_POW2,
    F6_E3M2_MAX,
    F6_E3M2_MAX_POW2,
    F8E4M3_MAX,
    F8E4M3_MAX_POW2,
    F8E5M2_MAX,
    F8E5M2_MAX_POW2,
    F32_EXP_BIAS,
    F32_MIN_NORMAL,
    SUPPORTED_ELEM_DTYPES,
)
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f6_e2m3_unpacked_to_f32,
    f6_e3m2_unpacked_to_f32,
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    pack_uint4,
    triton_to_mxfp8_dim0,
    unpack_uint4,
)
from torchao.prototype.mx_formats.utils import (
    _swizzle_aware_slice,
    from_blocked,
    hp_data_dims_to_swizzled_scale_dims_mx,
    to_blocked,
)
from torchao.quantization.quantize_.common import (
    QuantizeTensorKwargs,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.utils import TorchAOBaseTensor, fill_defaults

aten = torch.ops.aten

# TODO(later): read from somewhere else?
SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2


@dataclass
class QuantizeTensorToMXKwargs(QuantizeTensorKwargs):
    elem_dtype: Union[torch.dtype, str] = torch.float8_e4m3fn
    block_size: int = 32
    # TODO(future PR): flip the scaling_mode default to RCEIL
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR
    kernel_preference: KernelPreference = KernelPreference.EMULATED
    is_swizzled_scales: bool = False


def _to_mx_rceil(
    data_hp: torch.Tensor,
    max_abs: torch.Tensor,
    max_pos: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A prototype implementation of MXFP scale factor derivation method described in
    https://docs.nvidia.com/cuda/cublas/#d-block-quantization

    For Nvidia GPU with Blackwell+ architecture, the scale factor derivation method
    could be accelerated by the `cvt.rp.satfinite.ue8m0x2.f32` instruction.

    Args:
        data_hp: High precision data.
        max_abs: Maximum absolute value for data_hp along specified dimension/block_size.
        max_pos: The maximum value of the low precision data type.

    Returns:
        exponent: The biased exponent with dtype E8M0 in uint8 container.
        data_lp: The targeted low precision data, in high precision container
            (requires cast to low precision data type).
    """
    descale = max_abs / max_pos
    # TODO: nan/inf needs to be set for any value
    # of nan/inf in input not just amax.
    exponent = torch.where(
        torch.isnan(descale),
        0xFF,  # Handle biased exponent for nan
        # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
        (
            torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-E8M0_EXPONENT_BIAS,
                max=E8M0_EXPONENT_BIAS,
            )
            + E8M0_EXPONENT_BIAS
        ).to(torch.uint8),
    )

    descale_fp = torch.where(
        exponent == 0, 1.0, torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32))
    )

    # scale and saturated cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
    return exponent, data_lp


def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    is_swizzled_scales: bool = False,
):
    """
    Takes a high precision tensor and converts to MX scale and raw data, in
    naive layout (scale and raw data are separate tensors).
    """
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    # TODO(future PR): consider supporting padding
    assert data_hp.shape[-1] % block_size == 0, (
        f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}"
    )
    assert data_hp.is_contiguous(), "unsupported"
    assert elem_dtype in SUPPORTED_ELEM_DTYPES, "unsupported"

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    )

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    # We cast to float32 here because
    # in the `max_abs_int32 = max_abs.view(hp_int_dtype)` line below,
    # if tensor parallel is enabled then the resulting shape is 2x larger
    # than it should be under some conditions, likely because of a bug in
    # the `view` op with DTensor and target dtype int16.  I reproduce in
    # torchtitan but not in a unit test, so not enough info to file a good
    # issue in pytorch/pytorch. For now, work around. In the future we should
    # debug and fix this properly.
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type, and get the mbits at the same time
    if elem_dtype == torch.float8_e4m3fn:
        target_max_pow2 = F8E4M3_MAX_POW2
        mbits = MBITS_F8_E4M3
        max_pos = F8E4M3_MAX
    elif elem_dtype == torch.float8_e5m2:
        target_max_pow2 = F8E5M2_MAX_POW2
        mbits = MBITS_F8_E5M2
        max_pos = F8E5M2_MAX
    elif elem_dtype == DTYPE_FP6_E2M3:
        target_max_pow2 = F6_E2M3_MAX_POW2
        mbits = MBITS_F6_E2M3
        max_pos = F6_E2M3_MAX
    elif elem_dtype == DTYPE_FP6_E3M2:
        target_max_pow2 = F6_E3M2_MAX_POW2
        mbits = MBITS_F6_E3M2
        max_pos = F6_E3M2_MAX
    elif elem_dtype == torch.float4_e2m1fn_x2:
        target_max_pow2 = F4_E2M1_MAX_POW2
        mbits = MBITS_F4_E2M1
        max_pos = F4_E2M1_MAX
    else:
        raise AssertionError("unsupported element dtype")

    if scaling_mode == ScaleCalculationMode.RCEIL:
        scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)
    else:
        assert data_hp.dtype is torch.float32
        hp_int_dtype = torch.int32
        hp_mbits = MBITS_F32
        hp_ebits = EBITS_F32
        hp_exp_bias = F32_EXP_BIAS

        # rounding before calculating the largest power of 2
        # X = 2^(floor(log2(rounding(max_abs(v)))-max_exp))
        if scaling_mode == ScaleCalculationMode.EVEN:
            nan_mask = torch.isnan(max_abs)
            max_abs = max_abs.view(hp_int_dtype)
            val_to_add = 1 << (hp_mbits - mbits - 1)
            mask = ((1 << (hp_ebits + SBITS)) - 1) << hp_mbits
            max_abs = (max_abs + val_to_add) & mask
            max_abs = max_abs.view(data_hp.dtype)
            max_abs[nan_mask] = torch.tensor(
                float("nan"), device=max_abs.device, dtype=max_abs.dtype
            )

        # Calculate the scale for different modes
        max_abs_int32 = max_abs.view(hp_int_dtype)
        # For now, use `torch.bitwise_right_shift` instead of `>>` to support DTensor
        # See https://github.com/pytorch/pytorch/issues/156533.
        extracted_pow2 = (
            (torch.bitwise_right_shift(max_abs_int32, hp_mbits)) & 0b11111111
        ) - hp_exp_bias

        if scaling_mode in (ScaleCalculationMode.FLOOR, ScaleCalculationMode.EVEN):
            scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
        elif scaling_mode == ScaleCalculationMode.CEIL:
            # round up: add one to scale if the mantissa is larger than 0
            # 0x7FFFFF is equal to 23 ones
            mantissa_gt_one = (max_abs_int32 & 0x7FFFFF) > 0
            extracted_pow2 += mantissa_gt_one
            scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
        else:
            raise AssertionError("unsupported scaling calculation mode")

        # Clamp to exponents that can be represented in e8m0
        # add one to positive range to capture NaNs
        scale_e8m0_unbiased = torch.clamp(
            scale_e8m0_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS + 1
        )

        # Create the biased e8m0 representation and cast it to 8 bits
        scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS
        scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)

        # Conversion to torch.uint8 sets NaN values to 0, fix this by
        # explicitly setting known NaN values to 255
        scale_e8m0_biased = torch.where(
            torch.isnan(max_abs),
            E8M0_EXPONENT_NAN_VAL,
            scale_e8m0_biased,
        )

        # For now, calculate the scale in floating point.
        # For now, use `torch.bitwise_left_shift` instead of `<<` to support DTensor
        # See https://github.com/pytorch/pytorch/issues/156533.
        scale_fp32 = (
            torch.bitwise_left_shift(scale_e8m0_biased.to(torch.int32), MBITS_F32)
        ).view(torch.float32)

        # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
        # float32 denormal range. For now, manually adjust the fp scale. This is
        # relevant if all of the incoming block values are zeroes.
        # See https://github.com/pytorch/pytorch/issues/125557 for details.
        # Note: it would be more correct to set the minimum to 2**-127, but this
        # does not work in triton either as it looks like subnormal value handling
        # has some gaps.  So, for now just set to the minimum normal value.
        scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

        # scale and saturated cast the data elements to max of target dtype
        data_lp = data_hp / scale_fp32

        if (
            elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not torch._dynamo.is_compiling()
        ):
            # As of 20250317, the Pytorch eager mode cast to `torch.float8_e4m3fn`
            # is unsaturated. This cast is saturated in triton. If we are compute bound,
            # we see a speedup if we remove this redundant clamp if we are compiling
            # to triton.
            # TODO(#1912): make the saturated cast work in eager mode and remove this
            # workaround.
            data_lp = torch.clamp(data_lp, min=-1 * max_pos, max=max_pos)

    # cast to target dtype
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_lp = data_lp.to(elem_dtype)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == DTYPE_FP6_E2M3:
        data_lp = f32_to_f6_e2m3_unpacked(data_lp)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == DTYPE_FP6_E3M2:
        data_lp = f32_to_f6_e3m2_unpacked(data_lp)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == torch.float4_e2m1fn_x2:
        # can't reshape at the end without handling it in the packing code,
        # punt until later since we'll need to rethink the torch.compile
        # approach for fp4x2 in any case
        data_lp = data_lp.reshape(orig_shape)
        data_lp = f32_to_f4_unpacked(data_lp)
        data_lp = pack_uint4(data_lp)
    else:
        raise AssertionError("unsupported")

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)

    # if user requested scale swizzling, do it here
    if is_swizzled_scales:
        leading_dims, M, K = orig_shape[:-2], orig_shape[-2], orig_shape[-1]
        scale_shape = (math.prod(leading_dims) * M, K // block_size)
        scale = to_blocked(scale_e8m0_biased.view(scale_shape)).flatten()
        scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_mx(M, K)
        scale_e8m0_biased = scale.view(*leading_dims, scale_M, scale_K)

    return scale_e8m0_biased, data_lp


def get_fp_scale(scale_e8m0):
    scale_e8m0 = scale_e8m0.view(torch.uint8)
    s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
    # TODO(later): it would be nice if there was a way to do the 2^x operation
    # in PyTorch without creating a tensor of twos
    two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
    # pow(two, s_offset) can be out of range of floating point formats.
    # TODO(later): handle this for float16 if we decide to support float16
    # scales.
    s_fp = torch.pow(two, s_offset)

    # If a block exponent was 255, set values of that block to NaN
    s_fp = torch.where(scale_e8m0 != E8M0_EXPONENT_NAN_VAL, s_fp, float("nan"))

    return s_fp


def to_dtype(
    data_lp,
    scale_e8m0,
    elem_dtype,
    block_size,
    target_dtype,
):
    orig_shape = data_lp.shape
    is_transposed = not data_lp.is_contiguous()
    # if the underlying data is transposed, convert to row major before
    # unpacking and unscaling
    if is_transposed:
        data_lp = data_lp.t()
        scale_e8m0 = scale_e8m0.t()
        assert data_lp.is_contiguous()
        orig_shape = (orig_shape[1], orig_shape[0])

    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_hp = data_lp.to(target_dtype)
    elif elem_dtype == DTYPE_FP6_E2M3:
        data_hp = f6_e2m3_unpacked_to_f32(data_lp)
        data_hp = data_hp.to(target_dtype).reshape(orig_shape)
    elif elem_dtype == DTYPE_FP6_E3M2:
        data_hp = f6_e3m2_unpacked_to_f32(data_lp)
        data_hp = data_hp.to(target_dtype).reshape(orig_shape)
    elif elem_dtype == torch.float4_e2m1fn_x2:
        # fp4
        f4_unpacked = unpack_uint4(data_lp)
        # for now we only have a cast to f32
        # TODO(future PR): add cast directly to bf16
        f32 = f4_unpacked_to_f32(f4_unpacked)
        data_hp = f32.to(target_dtype)
        # manually adjust shape to account for the unpacking
        # TODO(future PR): clean up the shape code and remove the hack
        # below
        orig_shape = (*orig_shape[:-1], orig_shape[-1] * 2)
    else:
        raise AssertionError("unsupported")

    data_hp = data_hp.reshape(-1, block_size)
    s_fp = get_fp_scale(scale_e8m0).reshape(-1, 1).to(target_dtype)
    data_hp = data_hp * s_fp
    data_hp = data_hp.reshape(orig_shape)

    # if we converted to row-major before unscaling convert back
    if is_transposed:
        data_hp = data_hp.t()

    return data_hp


def tensor_size_hp_to_fp4x2(orig_size, is_contiguous):
    new_size = orig_size
    if is_contiguous:
        new_size = [*list(new_size[:-1]), new_size[-1] // 2]
    else:
        if len(orig_size) == 2:
            new_size = [new_size[0] // 2, *list(new_size[1:])]
        else:
            assert len(orig_size) == 3, "unsupported"
            # only supporting dim0, dim1, dim2 and dim0, dim2, dim1 orders
            new_size = [new_size[0], new_size[2] // 2, new_size[1]]
    return new_size


def tensor_size_fp4x2_to_hp(orig_size, is_contiguous):
    new_size = orig_size
    if is_contiguous:
        new_size = [*list(new_size[:-1]), new_size[-1] * 2]
    else:
        if len(orig_size) == 2:
            new_size = [new_size[0] * 2, *list(new_size[1:])]
        else:
            assert len(orig_size) == 3, "unsupported"
            # only supporting dim0, dim1, dim2 and dim0, dim2, dim1 orders
            new_size = [new_size[0], new_size[2] * 2, new_size[1]]
    return new_size


class MXTensor(TorchAOBaseTensor):
    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = [
        "_elem_dtype",
        "block_size",
        "_orig_dtype",
        "kernel_preference",
        "act_quant_kwargs",
        "_is_swizzled_scales",
    ]

    def __new__(
        cls,
        qdata,
        scale_e8m0_bits,
        elem_dtype,
        block_size,
        orig_dtype,
        kernel_preference,
        act_quant_kwargs,
        is_swizzled_scales,
    ):
        new_size = qdata.size()
        if elem_dtype == torch.float4_e2m1fn_x2:
            # set the tensor size to what it would be without 2x4 packing
            # Note: `is_contiguous` is going to return True for a tensor of size
            # (M, 1) regardless or the order of dims, so this logic is currently
            # broken for tensors of size (M, 1) or (1, M). Leaving broken until
            # a time when fixing this becomes important.
            new_size = tensor_size_fp4x2_to_hp(
                new_size,
                qdata.is_contiguous(),
            )
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            new_size,
            strides=qdata.stride(),
            storage_offset=qdata.storage_offset(),
            layout=qdata.layout,
            dtype=orig_dtype,
            device=qdata.device,
        )
        assert scale_e8m0_bits.dtype == torch.float8_e8m0fnu, (
            f"scale_e8m0_bits.dtype must be `torch.float8_e8m0fnu`, got {scale_e8m0_bits.dtype}"
        )
        assert qdata.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.uint8,
        ), "unsupported"
        self.qdata = qdata
        self.scale = scale_e8m0_bits
        self._elem_dtype = elem_dtype
        self.block_size = block_size
        self._orig_dtype = orig_dtype
        self.kernel_preference = kernel_preference
        self.act_quant_kwargs = act_quant_kwargs
        self._is_swizzled_scales = is_swizzled_scales
        return self

    def __repr__(self):
        # TODO better elem dtype print for fp4
        return f"MXTensor: elem_dtype: {self._elem_dtype}, s_e8m0: {self.scale}, d: {self.qdata}, act_quant_kwargs: {self.act_quant_kwargs}, _is_swizzled_scales={self._is_swizzled_scales}"  # noqa: E501

    def _quantization_type(self):
        return f"{self._elem_dtype=}, {self.block_size=}, {self._orig_dtype=}, {self.kernel_preference=}, {self.act_quant_kwargs=}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        scale = self.scale
        if self._is_swizzled_scales:
            is_transposed = self.qdata.stride(-2) < self.qdata.stride(-1)
            if is_transposed:
                leading_dims, M, K = self.shape[:-2], self.shape[-1], self.shape[-2]
                scale = scale.transpose(-2, -1)
            else:
                leading_dims, M, K = self.shape[:-2], self.shape[-2], self.shape[-1]
            scale = from_blocked(
                scale, math.prod(leading_dims) * M, K // self.block_size
            )
            scale = scale.view(*leading_dims, M, K // self.block_size)
            if is_transposed:
                scale = scale.transpose(-2, -1)

        return to_dtype(
            self.qdata,
            scale,
            self._elem_dtype,
            self.block_size,
            output_dtype,
        )

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_mx(
        data_hp: torch.Tensor,
        elem_dtype: Union[torch.dtype, str],
        block_size: int = BLOCK_SIZE_DEFAULT,
        # TODO(future PR): flip the scaling_mode default to RCEIL
        scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
        # TODO(future PR): switch default gemm to cublas
        kernel_preference: KernelPreference = KernelPreference.EMULATED,
        act_quant_kwargs: Optional[QuantizeTensorToMXKwargs] = None,
        is_swizzled_scales: bool = False,
        mxfp8_dim0_cast_kernel_choice: MXFP8Dim0CastKernelChoice = MXFP8Dim0CastKernelChoice.TORCH,
    ):
        assert mxfp8_dim0_cast_kernel_choice in (
            MXFP8Dim0CastKernelChoice.TRITON,
            MXFP8Dim0CastKernelChoice.TORCH,
        ), (
            f"unsupported kernel choice for mxfp8_dim0_cast_kernel_choice: {mxfp8_dim0_cast_kernel_choice}"
        )

        triton_kernel_supported = (
            elem_dtype == torch.float8_e4m3fn and not is_swizzled_scales
        )
        if mxfp8_dim0_cast_kernel_choice == MXFP8Dim0CastKernelChoice.TRITON:
            assert triton_kernel_supported, (
                f"triton kernel unsupported for {data_hp.dtype=}, {elem_dtype=}, {scaling_mode=}, {is_swizzled_scales=}"
            )
            data_lp, scale_e8m0_biased = triton_to_mxfp8_dim0(
                data_hp,
                inner_block_size=block_size,
                scaling_mode=scaling_mode.value,
            )
        else:
            scale_e8m0_biased, data_lp = to_mx(
                data_hp, elem_dtype, block_size, scaling_mode, is_swizzled_scales
            )
        if isinstance(scale_e8m0_biased, DTensor):
            assert isinstance(data_lp, DTensor), "unsupported"
            local_scale_e8m0_biased = scale_e8m0_biased.to_local()
            local_data_lp = data_lp.to_local()
            inner_mx_tensor = MXTensor(
                local_data_lp,
                local_scale_e8m0_biased,
                elem_dtype,
                block_size,
                data_hp.dtype,
                kernel_preference,
                act_quant_kwargs,
                is_swizzled_scales,
            )
            return DTensor.from_local(
                inner_mx_tensor,
                data_lp.device_mesh,
                data_lp.placements,
                run_check=False,
                shape=data_lp.size(),
                stride=data_lp.stride(),
            )
        return MXTensor(
            data_lp,
            scale_e8m0_biased,
            elem_dtype,
            block_size,
            data_hp.dtype,
            kernel_preference,
            act_quant_kwargs,
            is_swizzled_scales,
        )

    # Do not force the MXTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl


implements = MXTensor.implements


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(func)
    )


def _get_gemm_choice(
    choice_a: Optional[KernelPreference], choice_b: Optional[KernelPreference]
) -> KernelPreference:
    if choice_a is not None and choice_b is not None:
        assert choice_a == choice_b, (
            "Both MXTensor inputs must have the same gemm config if specified"
        )
        return choice_a

    # Assert that at least one is set and return that one
    assert choice_a is not None or choice_b is not None, (
        "At least one gemm choice must be specified"
    )
    return choice_a if choice_a is not None else choice_b


def _addmm_mx_dispatch(
    a: torch.Tensor, b: MXTensor, aten_op, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Core implementation shared between mx_mm and mx_addmm.
    The only difference is whether bias is None or not.
    """

    if not isinstance(a, MXTensor):
        assert b.act_quant_kwargs is not None, "weight-only quant not yet supported"
        k = b.act_quant_kwargs
        a = MXTensor.to_mx(
            a,
            k.elem_dtype,
            k.block_size,
            k.scaling_mode,
            k.kernel_preference,
            k.is_swizzled_scales,
        )

    gemm_choice = _get_gemm_choice(a.kernel_preference, b.kernel_preference)

    if gemm_choice == KernelPreference.AUTO:
        # real MX gemm backed by torchao's CUTLASS kernels
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        assert a.qdata.is_contiguous()
        assert b.qdata.t().is_contiguous()
        assert a.block_size == 32, f"Invalid block size {a.block_size}"
        assert b.block_size == 32, f"Invalid block size {b.block_size}"

        if a._is_swizzled_scales:
            a_scale_block = a.scale
        else:
            a_scale = a.scale.view(M, K // a.block_size)
            a_scale_block = to_blocked(a_scale)

        if b._is_swizzled_scales:
            b_scale_block = b.scale.t()
        else:
            b_scale = b.scale.t().view(N, K // b.block_size)
            b_scale_block = to_blocked(b_scale)

        if a._elem_dtype == torch.float8_e4m3fn:
            assert b._elem_dtype == torch.float8_e4m3fn
            res = torch._scaled_mm(
                a.qdata,
                b.qdata,
                a_scale_block.view(torch.float8_e8m0fnu),
                b_scale_block.view(torch.float8_e8m0fnu),
                bias=bias,
                out_dtype=torch.bfloat16,
            )
        else:
            assert a._elem_dtype == torch.float4_e2m1fn_x2
            assert b._elem_dtype == torch.float4_e2m1fn_x2
            if not torch_version_at_least("2.10.0"):
                raise RuntimeError(
                    "MXFP4 matmul requires PyTorch 2.10.0 or later for F.scaled_mm support"
                )
            # FP4 operations using F.scaled_mm
            res = F.scaled_mm(
                a.qdata.view(torch.float4_e2m1fn_x2),
                b.qdata.view(torch.float4_e2m1fn_x2),
                scale_a=a_scale_block,
                scale_recipe_a=ScalingType.BlockWise1x32,
                scale_b=b_scale_block,
                scale_recipe_b=ScalingType.BlockWise1x32,
                swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                bias=bias,
                output_dtype=torch.bfloat16,
            )

    else:
        assert gemm_choice == KernelPreference.EMULATED, "unimplemented"
        # emulated MX gemm
        a_hp = a.dequantize(a._orig_dtype)
        b_hp = b.dequantize(b._orig_dtype)
        # assert memory layout we expect to be required in hardware
        assert a_hp.is_contiguous()
        assert b_hp.t().is_contiguous()

        # Call appropriate aten_op based on whether bias is provided
        if bias is not None:
            res = aten_op(bias, a_hp, b_hp)  # addmm
        else:
            res = aten_op(a_hp, b_hp)  # mm

    return res


@implements([aten.mm.default, aten.matmul.default])
def mx_mm(func, types, args, kwargs):
    a = args[0]
    b = args[1]
    assert isinstance(b, MXTensor)

    return _addmm_mx_dispatch(a, b, func)


@implements([aten.addmm.default])
def mx_addmm(func, types, args, kwargs):
    assert isinstance(args[0], torch.Tensor) and isinstance(args[2], MXTensor)
    bias = args[0]
    a = args[1]
    b = args[2]
    return _addmm_mx_dispatch(a, b, func, bias=bias)


@implements([aten.linear.default])
def mx_linear(func, types, args, kwargs):
    assert isinstance(args[0], torch.Tensor) and isinstance(args[1], MXTensor)
    a = args[0]

    # make a 2d
    orig_a_shape = a.shape
    a_2d = a.view(-1, orig_a_shape[-1])

    b = args[1].t()
    if len(args) > 2:
        bias = args[2]
        res = _addmm_mx_dispatch(a_2d, b, aten.addmm.default, bias)
    else:
        res = _addmm_mx_dispatch(a_2d, b, aten.mm.default)

    # reshape back to original shape
    res = res.view(*orig_a_shape[:-1], res.shape[-1])
    return res


@implements([aten.t.default])
def mx_t(func, types, args, kwargs):
    # For now, only transpose(input, 0, 1) is supported.
    old = args[0]
    new = MXTensor(
        old.qdata.t(),
        old.scale.t(),
        old._elem_dtype,
        old.block_size,
        old._orig_dtype,
        old.kernel_preference,
        old.act_quant_kwargs,
        old._is_swizzled_scales,
    )
    return new


@implements([aten.sum.dim_IntList])
def mx_cast_up_op(func, types, args, kwargs):
    """Be careful with this function, this is a "fallback" op that
    casts the output of the op to the original precision. And performs the op.

    We currently need this to support the backward for admmm bias.
    "addmm" -> out
    "hp_gradBias" <-"sum" <- "identity" <- gradOut <- "hp_gradOut"
    """

    def unwrap(x):
        if isinstance(x, MXTensor):
            return x.dequantize(x._orig_dtype)
        return x

    new_args = tree_map(unwrap, args)
    new_kwargs = tree_map(unwrap, kwargs)
    return func(*new_args, **new_kwargs)


@implements([aten.view.default])
def mx_view_op(func, types, args, kwargs):
    data = args[0].qdata
    new_size = args[1]
    if args[0]._elem_dtype == torch.float4_e2m1fn_x2:
        # special case fp4 as we pack two elements per byte
        new_size = tensor_size_hp_to_fp4x2(new_size, data.is_contiguous())
    new_data = func(data, new_size, *args[2:], **kwargs)
    return MXTensor(
        new_data,
        args[0].scale,
        args[0]._elem_dtype,
        args[0].block_size,
        args[0]._orig_dtype,
        args[0].kernel_preference,
        args[0].act_quant_kwargs,
        args[0]._is_swizzled_scales,
    )


@implements([aten.slice.Tensor])
def mx_slice(func, types, args, kwargs):
    x, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    sliced_data, sliced_scale = _swizzle_aware_slice(x, dim, start, end, step)

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        MXTensor(
            sliced_data,
            sliced_scale,
            x._elem_dtype,
            x.block_size,
            x._orig_dtype,
            x.kernel_preference,
            x.act_quant_kwargs,
            x._is_swizzled_scales,
        ),
    )


@implements([aten.clone.default])
def mx_clone(func, types, args, kwargs):
    self = args[0]
    memory_format = kwargs.get("memory_format", None)

    if memory_format is not None:
        clone_fn = lambda x: x.clone(memory_format=memory_format)
    else:
        clone_fn = lambda x: x.clone()

    return self._apply_fn_to_data(clone_fn)


@implements([aten.select.int])
def mx_select(func, types, args, kwargs):
    old_mx_tensor, dim, index = args
    assert dim == 0, f"MXTensor aten.select.int with {dim=} is not yet supported"
    assert len(old_mx_tensor.qdata.shape) == len(old_mx_tensor.scale.shape), (
        "unsupported"
    )
    assert not old_mx_tensor._is_swizzled_scales, "unsupported"
    new_mx_tensor = old_mx_tensor.__class__(
        old_mx_tensor.qdata[index],
        old_mx_tensor.scale[index],
        old_mx_tensor._elem_dtype,
        old_mx_tensor.block_size,
        old_mx_tensor._orig_dtype,
        old_mx_tensor.kernel_preference,
        old_mx_tensor.act_quant_kwargs,
        old_mx_tensor._is_swizzled_scales,
    )
    return return_and_correct_aliasing(func, args, kwargs, new_mx_tensor)


@implements([torch.ops._c10d_functional.all_gather_into_tensor.default])
def mx_all_gather(func, types, args, kwargs):
    """
    All-gather for MXTensor

    Args:
        func: The operation (all_gather_into_tensor)
        types: Tensor types involved
        args: (mx_tensor, group_tag, ...)
        kwargs: Additional arguments
    """
    mx_tensor = args[0]
    group_tag = args[1] if len(args) > 1 else "default"

    # TODO: Add support for concat CC as a future optimization

    # Gather both data and scale
    gathered_qdata = torch.ops._c10d_functional.all_gather_into_tensor.default(
        mx_tensor.qdata,  # The quantized data
        group_tag,
        *args[2:],
        **kwargs,
    )

    gathered_scale = torch.ops._c10d_functional.all_gather_into_tensor.default(
        mx_tensor.scale.view(
            torch.uint8
        ),  # The scale factors, Need to cast to uint8 as float8_e8m0fnu is not support for all gather.
        group_tag,
        *args[2:],
        **kwargs,
    )

    gathered_scale = gathered_scale.view(torch.float8_e8m0fnu)

    # Return new MXTensor with gathered data
    return MXTensor(
        gathered_qdata,
        gathered_scale,
        mx_tensor._elem_dtype,
        mx_tensor.block_size,
        mx_tensor._orig_dtype,
        mx_tensor.kernel_preference,
        mx_tensor.act_quant_kwargs,
        mx_tensor._is_swizzled_scales,
    )


@implements([torch.ops._c10d_functional.wait_tensor.default])
def mx_wait_tensor(func, types, args, kwargs):
    """
    Wait for async collective to complete on MXTensor

    This is called after collectives like all_gather to ensure
    the operation has completed before using the tensor.
    """
    mx_tensor = args[0]

    # Wait on both components
    waited_qdata = torch.ops._c10d_functional.wait_tensor.default(
        mx_tensor.qdata, *args[1:], **kwargs
    )

    waited_scale = torch.ops._c10d_functional.wait_tensor.default(
        mx_tensor.scale, *args[1:], **kwargs
    )

    return MXTensor(
        waited_qdata,
        waited_scale,
        mx_tensor._elem_dtype,
        mx_tensor.block_size,
        mx_tensor._orig_dtype,
        mx_tensor.kernel_preference,
        mx_tensor.act_quant_kwargs,
        mx_tensor._is_swizzled_scales,
    )
