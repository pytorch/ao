# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from enum import Enum, auto
from typing import Dict, Union

import torch

from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.constants import (
    BLOCK_SIZE_DEFAULT,
    DTYPE_FP4,
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
    F32_MIN_NORMAL,
    SUPPORTED_ELEM_DTYPES,
)
from torchao.prototype.mx_formats.custom_cast import (
    f4_unpacked_to_f32,
    f6_e2m3_unpacked_to_f32,
    f6_e3m2_unpacked_to_f32,
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    pack_uint4,
    triton_f4_to_scaled_bf16,
    unpack_uint4,
)

# TODO(later): read from somewhere else?
SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2


class ScaleCalculationMode(Enum):
    """
    Enum representing the different methods for calculating MX block scaling.
    There are three methods available:
    FLOOR: This method is recommended by the OCP MX Spec 1.0 and uses X = 2^floor(log2(max_abs(v))-max_exp).
           It result in overflow issues for large values and bad for gradient quantization.
    CEIL: This method avoids overflow issues, but small values may shift to 0 due to a large scaling factor.
           It uses X = 2^ceil(log2(max_abs(v))-max_exp).
    EVEN: This method is a trade-off between Option 1 and Option 2. It uses X = 2^(floor(log2(rounding(max_abs(v)))-max_exp)).
           It provides better accuracy for MX4 training compared to FLOOR and CEIL.
    By default, we use the EVEN method for better accuracy.
    """

    FLOOR = auto()
    CEIL = auto()
    EVEN = auto()


def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
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
    assert data_hp.numel() % block_size == 0, "unsupported"
    assert data_hp.is_contiguous(), "unsupported"
    assert elem_dtype in SUPPORTED_ELEM_DTYPES, "unsupported"

    # calculate the scale in e8m0 format

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(data_hp), 1)

    # Add an epsilon to prevent the log2 function call for returning -inf
    # where the values are zero.
    eps = F32_MIN_NORMAL * (max_abs == 0).type(max_abs.dtype)

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type, and get the mbits at the same time
    if elem_dtype == torch.float8_e4m3fn:
        target_max_pow2 = F8E4M3_MAX_POW2
        mbits = MBITS_F8_E4M3
    elif elem_dtype == torch.float8_e5m2:
        target_max_pow2 = F8E5M2_MAX_POW2
        mbits = MBITS_F8_E5M2
    elif elem_dtype == DTYPE_FP6_E2M3:
        target_max_pow2 = F6_E2M3_MAX_POW2
        mbits = MBITS_F6_E2M3
    elif elem_dtype == DTYPE_FP6_E3M2:
        target_max_pow2 = F6_E3M2_MAX_POW2
        mbits = MBITS_F6_E3M2
    elif elem_dtype == DTYPE_FP4:
        target_max_pow2 = F4_E2M1_MAX_POW2
        mbits = MBITS_F4_E2M1
    else:
        raise AssertionError("unsupported element dtype")

    # rounding before calculating the largest power of 2
    # X = 2^(floor(log2(rounding(max_abs(v)))-max_exp))
    if scaling_mode == ScaleCalculationMode.EVEN:
        nan_mask = torch.isnan(max_abs)
        max_abs = max_abs.to(torch.float32).view(torch.int32)
        val_to_add = 1 << (MBITS_F32 - mbits - 1)
        mask = ((1 << (EBITS_F32 + SBITS)) - 1) << MBITS_F32
        max_abs = (max_abs + val_to_add) & mask
        max_abs = max_abs.view(torch.float32)
        max_abs[nan_mask] = torch.tensor(float("nan"), device=max_abs.device)

    # Calculate the scale for different modes
    if scaling_mode in (ScaleCalculationMode.FLOOR, ScaleCalculationMode.EVEN):
        scale_e8m0_unbiased = torch.floor(torch.log2(max_abs + eps)) - target_max_pow2
    elif scaling_mode == ScaleCalculationMode.CEIL:
        scale_e8m0_unbiased = torch.ceil(torch.log2(max_abs + eps)) - target_max_pow2
    else:
        raise AssertionError("unsupported scaling calculation mode")

    # Clamp to exponents that can be represented in e8m0
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS
    )

    # Create the biased e8m0 representation and cast it to 8 bits
    scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS
    scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)

    # Conversion to torch.uint8 sets NaN values to 0, fix this by
    # explicitly setting known NaN values to 255
    scale_e8m0_biased = torch.where(
        torch.isnan(scale_e8m0_unbiased),
        E8M0_EXPONENT_NAN_VAL,
        scale_e8m0_biased,
    )

    # For now, calculate the scale in floating point.
    # TODO(future) audit if there is a need to bit shift exponents instead.
    scale_fp = torch.pow(
        torch.full(max_abs.size(), 2.0, device=scale_e8m0_biased.device),
        scale_e8m0_unbiased,
    )

    # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
    # float32 denormal range. For now, manually adjust the fp scale. This is
    # relevant if all of the incoming block values are zeroes.
    # See https://github.com/pytorch/pytorch/issues/125557 for details.
    # Note: it would be more correct to set the minimum to 2**-127, but this
    # does not work in triton either as it looks like subnormal value handling
    # has some gaps.  So, for now just set to the minimum normal value.
    scale_fp = torch.clamp(scale_fp, min=F32_MIN_NORMAL)

    # scale and saturated cast the data elements to max of target dtype
    if elem_dtype == torch.float8_e4m3fn:
        max_pos = F8E4M3_MAX
    elif elem_dtype == torch.float8_e5m2:
        max_pos = F8E5M2_MAX
    elif elem_dtype == DTYPE_FP6_E2M3:
        max_pos = F6_E2M3_MAX
    elif elem_dtype == DTYPE_FP6_E3M2:
        max_pos = F6_E3M2_MAX
    elif elem_dtype == DTYPE_FP4:
        max_pos = F4_E2M1_MAX
    else:
        raise AssertionError("unsupported")
    data_lp = torch.clamp(
        data_hp / scale_fp.unsqueeze(1), min=-1 * max_pos, max=max_pos
    )

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
    elif elem_dtype == DTYPE_FP4:
        # can't reshape at the end without handling it in the packing code,
        # punt until later since we'll need to rethink the torch.compile
        # approach for fp4x2 in any case
        data_lp = data_lp.reshape(orig_shape)
        data_lp = f32_to_f4_unpacked(data_lp)
        data_lp = pack_uint4(data_lp)
    else:
        raise AssertionError("unsupported")

    return scale_e8m0_biased, data_lp


def get_fp_scale(scale_e8m0):
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
    use_fp4_custom_triton_dequant_kernel,
):
    orig_shape = data_lp.shape
    is_transposed = not data_lp.is_contiguous()
    # if the underlying data is transposed, convert to row major before
    # unpacking and unscaling
    if is_transposed:
        data_lp = data_lp.t()
        assert data_lp.is_contiguous()
        orig_shape = (orig_shape[1], orig_shape[0])

    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_hp = data_lp.to(target_dtype)
    elif elem_dtype == DTYPE_FP6_E2M3:
        data_hp = f6_e2m3_unpacked_to_f32(data_lp)
        data_hp = data_hp.to(target_dtype)
    elif elem_dtype == DTYPE_FP6_E3M2:
        data_hp = f6_e3m2_unpacked_to_f32(data_lp)
        data_hp = data_hp.to(target_dtype)
    elif elem_dtype == DTYPE_FP4:
        if use_fp4_custom_triton_dequant_kernel:
            data_hp_rescaled = triton_f4_to_scaled_bf16(
                data_lp,
                scale_e8m0,
                block_size,
            )
            if is_transposed:
                data_hp_rescaled = data_hp_rescaled.t()
            return data_hp_rescaled.to(target_dtype)
        else:
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
        new_size = [new_size[0] // 2, *list(new_size[1:])]
    return new_size


def tensor_size_fp4x2_to_hp(orig_size, is_contiguous):
    new_size = orig_size
    if is_contiguous:
        new_size = [*list(new_size[:-1]), new_size[-1] * 2]
    else:
        new_size = [new_size[0] * 2, *list(new_size[1:])]
    return new_size


@torch._dynamo.allow_in_graph
class ToMXConstrFunc(torch.autograd.Function):
    """
    Differentiable cast to MX, no-op in backward
    """

    @staticmethod
    def forward(
        ctx,
        data_hp,
        elem_dtype,
        block_size,
        scaling_mode,
        use_fp4_custom_triton_dequant_kernel,
        gemm_kernel_choice,
    ):
        scale_e8m0_biased, data_lp = to_mx(
            data_hp, elem_dtype, block_size, scaling_mode
        )
        return MXTensor(
            scale_e8m0_biased,
            data_lp,
            elem_dtype,
            block_size,
            data_hp.dtype,
            use_fp4_custom_triton_dequant_kernel,
            gemm_kernel_choice,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None, None


@torch._dynamo.allow_in_graph
class FromMXConstrFunc(torch.autograd.Function):
    """
    Differentiable cast from MX, no-op in backward
    """

    @staticmethod
    def forward(ctx, tensor_lp, target_dtype):
        return to_dtype(
            tensor_lp._data,
            tensor_lp._scale_e8m0,
            tensor_lp._elem_dtype,
            tensor_lp._block_size,
            target_dtype,
            tensor_lp._use_fp4_custom_triton_dequant_kernel,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class MXTensor(torch.Tensor):
    def __new__(
        cls,
        scale_e8m0_bits,
        data_bits,
        elem_dtype,
        block_size,
        orig_dtype,
        use_fp4_custom_triton_dequant_kernel,
        gemm_kernel_choice,
    ):
        new_size = data_bits.size()
        if elem_dtype == DTYPE_FP4:
            # set the tensor size to what it would be without 2x4 packing
            # Note: `is_contiguous` is going to return True for a tensor of size
            # (M, 1) regardless or the order of dims, so this logic is currently
            # broken for tensors of size (M, 1) or (1, M). Leaving broken until
            # a time when fixing this becomes important.
            new_size = tensor_size_fp4x2_to_hp(
                new_size,
                data_bits.is_contiguous(),
            )
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            new_size,
            strides=data_bits.stride(),
            storage_offset=data_bits.storage_offset(),
            layout=data_bits.layout,
            dtype=orig_dtype,
            device=data_bits.device,
        )
        assert scale_e8m0_bits.dtype == torch.uint8, "unsupported"
        assert len(scale_e8m0_bits.shape) == 1, "unsupported"
        assert data_bits.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.uint8,
        ), "unsupported"
        if elem_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            DTYPE_FP6_E2M3,
            DTYPE_FP6_E3M2,
        ):
            target_numel = scale_e8m0_bits.numel() * block_size
        elif elem_dtype == DTYPE_FP4:
            assert data_bits.dtype is torch.uint8  # fp4
            target_numel = scale_e8m0_bits.numel() * block_size / 2
        else:
            raise AssertionError("unsupported")
        if not issubclass(
            torch._subclasses.fake_tensor.FakeTensor,
            type(data_bits),
        ):
            # this check is sometimes broken for FakeTensor
            # TODO investigate
            assert (
                target_numel == data_bits.numel()
            ), f"{target_numel} != {data_bits.numel()}"

        # `_scale_e8m0` has rank 1 and applies to a row-major memory layout of
        # `_data`
        self._scale_e8m0 = scale_e8m0_bits
        self._data = data_bits
        self._elem_dtype = elem_dtype
        self._block_size = block_size
        self._orig_dtype = orig_dtype
        self._use_fp4_custom_triton_dequant_kernel = (
            use_fp4_custom_triton_dequant_kernel
        )
        self._gemm_kernel_choice = gemm_kernel_choice
        return self

    def __repr__(self):
        # TODO better elem dtype print for fp4
        return f"MXTensor: elem_dtype: {self._elem_dtype}, s_e8m0: {self._scale_e8m0}, d: {self._data}, d_hp: {self.to_dtype(self._orig_dtype)}"  # noqa: E501

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # avoid circular dependency
        from torchao.prototype.mx_formats.mx_ops import MX_OPS_TABLE

        if func in MX_OPS_TABLE:
            return MX_OPS_TABLE[func](func, args, kwargs)

        raise NotImplementedError(f"{func} not implemented")

    def to_dtype(self, target_dtype):
        return FromMXConstrFunc.apply(self, target_dtype)

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_mx(
        data_hp: torch.Tensor,
        elem_dtype: Union[torch.dtype, str],
        block_size: int = BLOCK_SIZE_DEFAULT,
        scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
        use_fp4_custom_triton_dequant_kernel: bool = False,
        gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.EMULATED,
    ):
        return ToMXConstrFunc.apply(
            data_hp,
            elem_dtype,
            block_size,
            scaling_mode,
            use_fp4_custom_triton_dequant_kernel,
            gemm_kernel_choice,
        )

    def __tensor_flatten__(self):
        ctx = {
            "_elem_dtype": self._elem_dtype,
            "_block_size": self._block_size,
            "_orig_dtype": self._orig_dtype,
            "_use_fp4_custom_triton_dequant_kernel": self._use_fp4_custom_triton_dequant_kernel,
            "_gemm_kernel_choice": self._gemm_kernel_choice,
        }
        return ["_scale_e8m0", "_data"], ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict,
        metadata,
        outer_size,
        outer_stride,
    ):
        return MXTensor(
            inner_tensors["_scale_e8m0"],
            inner_tensors["_data"],
            metadata["_elem_dtype"],
            metadata["_block_size"],
            metadata["_orig_dtype"],
            metadata["_use_fp4_custom_triton_dequant_kernel"],
            metadata["_gemm_kernel_choice"],
        )

    # Do not force the MXTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
