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

from typing import Dict, Union

import torch

import torchao.prototype.mx_formats.config as config
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


def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
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

    # Find largest power of 2 less than or equal to max_abs.
    largest_p2_lt_max_abs = torch.floor(torch.log2(max_abs + eps))

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type
    if elem_dtype == torch.float8_e4m3fn:
        target_max_pow2 = F8E4M3_MAX_POW2
    elif elem_dtype == torch.float8_e5m2:
        target_max_pow2 = F8E5M2_MAX_POW2
    elif elem_dtype == DTYPE_FP6_E2M3:
        target_max_pow2 = F6_E2M3_MAX_POW2
    elif elem_dtype == DTYPE_FP6_E3M2:
        target_max_pow2 = F6_E3M2_MAX_POW2
    elif elem_dtype == DTYPE_FP4:
        target_max_pow2 = F4_E2M1_MAX_POW2
    else:
        raise AssertionError("unsupported")
    scale_e8m0_unbiased = largest_p2_lt_max_abs - target_max_pow2

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
    data_lp = data_lp.reshape(orig_shape)

    # cast to target dtype
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_lp = data_lp.to(elem_dtype)
    elif elem_dtype == DTYPE_FP6_E2M3:
        data_lp = f32_to_f6_e2m3_unpacked(data_lp)
    elif elem_dtype == DTYPE_FP6_E3M2:
        data_lp = f32_to_f6_e3m2_unpacked(data_lp)
    elif elem_dtype == DTYPE_FP4:
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


def to_dtype(data_lp, scale_e8m0, elem_dtype, block_size, target_dtype):
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
        if config.use_fp4_custom_triton_dequant_kernel:
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
    def forward(ctx, data_hp, elem_dtype, block_size):
        scale_e8m0_biased, data_lp = to_mx(data_hp, elem_dtype, block_size)
        return MXTensor(
            scale_e8m0_biased, data_lp, elem_dtype, block_size, data_hp.dtype
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None


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
    ):
        new_size = data_bits.size()
        if elem_dtype == DTYPE_FP4:
            # set the tensor size to what it would be without 2x4 packing
            new_size = tensor_size_fp4x2_to_hp(
                new_size,
                data_bits.is_contiguous(),
            )
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            new_size,
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
    ):
        return ToMXConstrFunc.apply(data_hp, elem_dtype, block_size)

    def __tensor_flatten__(self):
        ctx = {
            "_elem_dtype": self._elem_dtype,
            "_block_size": self._block_size,
            "_orig_dtype": self._orig_dtype,
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
        )

    # Do not force the MXTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
