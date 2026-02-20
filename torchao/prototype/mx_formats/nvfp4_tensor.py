# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.prototype.mx_formats.config import QuantizeToNVFP4KernelChoice
from torchao.prototype.mx_formats.constants import F4_E2M1_MAX, F8E4M3_MAX
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
    pack_uint4,
    triton_quantize_nvfp4,
    unpack_uint4,
)
from torchao.prototype.mx_formats.mx_tensor import (
    tensor_size_fp4x2_to_hp,
    tensor_size_hp_to_fp4x2,
)
from torchao.prototype.mx_formats.utils import (
    _swizzle_aware_slice,
    from_blocked,
    hp_data_dims_to_swizzled_scale_dims_nvfp4,
    to_blocked,
)
from torchao.quantization.quantize_.common import (
    QuantizeTensorKwargs,
)
from torchao.utils import TorchAOBaseTensor, fill_defaults

E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny

aten = torch.ops.aten


def _handle_use_triton_kernel(
    use_triton_kernel: bool,
    quantize_to_nvfp4_kernel_choice: QuantizeToNVFP4KernelChoice,
) -> QuantizeToNVFP4KernelChoice:
    """Handle deprecated use_triton_kernel parameter.

    Raises an exception if use_triton_kernel does not match
    quantize_to_nvfp4_kernel_choice.
    """
    expected = (
        QuantizeToNVFP4KernelChoice.TRITON
        if use_triton_kernel
        else QuantizeToNVFP4KernelChoice.TORCH
    )
    if expected != quantize_to_nvfp4_kernel_choice:
        raise ValueError(
            f"`use_triton_kernel={use_triton_kernel}` does not match "
            f"`quantize_to_nvfp4_kernel_choice={quantize_to_nvfp4_kernel_choice}`. "
            "`use_triton_kernel` is deprecated and will be removed after 0.17. "
            "Please use `quantize_to_nvfp4_kernel_choice` instead. "
            "`use_triton_kernel=True` is equivalent to "
            "`quantize_to_nvfp4_kernel_choice=QuantizeToNVFP4KernelChoice.TRITON`, "
            "`use_triton_kernel=False` is equivalent to "
            "`quantize_to_nvfp4_kernel_choice=QuantizeToNVFP4KernelChoice.TORCH`."
        )
    return quantize_to_nvfp4_kernel_choice


@dataclass
class QuantizeTensorToNVFP4Kwargs(QuantizeTensorKwargs):
    block_size: int = 16
    is_swizzled_scales: bool = False
    quantize_to_nvfp4_kernel_choice: QuantizeToNVFP4KernelChoice = (
        QuantizeToNVFP4KernelChoice.TORCH
    )
    use_dynamic_per_tensor_scale: bool = False
    use_triton_kernel: bool = False

    def __post_init__(self):
        self.quantize_to_nvfp4_kernel_choice = _handle_use_triton_kernel(
            self.use_triton_kernel, self.quantize_to_nvfp4_kernel_choice
        )


class NVFP4Tensor(TorchAOBaseTensor):
    """NVIDIA FP4 (NVFP4) Tensor subclass.

    This implements the NVIDIA variant of MX FP4 format, which uses a specific
    quantization algorithm for FP4 data with UE4M3 scales.

    Attributes:
        qdata: Packed FP4 data (2 values per byte)
        scale: Blockwise scales in float8_e4m3fn format (may be swizzled)
        per_tensor_scale: Optional global per-tensor scale in float32 format
        act_per_tensor_scale: Optional global per-tensor scale in float32 format, for activation
        block_size (int): Block size for quantization (fixed at 16)
        orig_dtype (torch.dtype): Original tensor dtype before quantization
        is_swizzled_scales (bool): Whether scales are stored in swizzled (blocked) format
        quantize_to_nvfp4_kernel_choice (QuantizeToNVFP4KernelChoice): Kernel preference for quantization
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = [
        "block_size",
        "orig_dtype",
    ]
    optional_tensor_data_names = ["per_tensor_scale", "act_per_tensor_scale"]
    optional_tensor_attribute_names = [
        "is_swizzled_scales",
        "quantize_to_nvfp4_kernel_choice",
        "act_quant_kwargs",
    ]

    def __new__(
        cls,
        qdata,
        scale,
        block_size,
        orig_dtype,
        per_tensor_scale=None,
        act_per_tensor_scale=None,
        is_swizzled_scales=False,
        quantize_to_nvfp4_kernel_choice=QuantizeToNVFP4KernelChoice.TORCH,
        act_quant_kwargs=None,
    ):
        # FP4 tensor size handling two paths, contiguous or not
        new_size = qdata.size()

        new_size = tensor_size_fp4x2_to_hp(
            new_size,
            qdata.stride(-2) > qdata.stride(-1),
        )

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            new_size,
            dtype=orig_dtype,
            device=qdata.device,
            requires_grad=False,
        )

        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        self.orig_dtype = orig_dtype
        self.per_tensor_scale = per_tensor_scale
        self.act_per_tensor_scale = act_per_tensor_scale
        self.is_swizzled_scales = is_swizzled_scales
        self.quantize_to_nvfp4_kernel_choice = quantize_to_nvfp4_kernel_choice
        self.act_quant_kwargs = act_quant_kwargs
        return self

    def __repr__(self):
        return f"NVFP4Tensor: scale: {self.scale}, per_tensor_scale: {self.per_tensor_scale}, d: {self.qdata}, d_hp: {self.dequantize(self.orig_dtype)}"

    def _quantization_type(self):
        return f"{self.is_swizzled_scales=}, {self.quantize_to_nvfp4_kernel_choice=}, {self.act_quant_kwargs=}"

    @staticmethod
    def to_nvfp4(
        data_hp: torch.Tensor,
        block_size: int = 16,
        per_tensor_scale: Optional[torch.Tensor] = None,
        act_per_tensor_scale: Optional[torch.Tensor] = None,
        is_swizzled_scales: bool = False,
        quantize_to_nvfp4_kernel_choice: QuantizeToNVFP4KernelChoice = QuantizeToNVFP4KernelChoice.TORCH,
        act_quant_kwargs: Optional[QuantizeTensorToNVFP4Kwargs] = None,
        use_triton_kernel: bool = False,
    ):
        """Convert high precision tensor to NVFP4 format.

        Args:
            data_hp: High precision input tensor (bfloat16 or float32)
            block_size: Block size for quantization (must be 16)
            per_tensor_scale: Optional pre-computed absolute maximum for calibration.
                If provided, uses per-tensor scaling. If None, uses block-wise scaling only.
            act_per_tensor_scale: Optional pre-computed absolute maximum for calibration for activation
                If provided, uses per-tensor scaling. If None, uses block-wise scaling only.
            is_swizzled_scales: If True, store scales in swizzled format for faster matrix multiplication
            quantize_to_nvfp4_kernel_choice: Kernel preference for quantization
            act_quant_kwargs: If specified, config for quantizing the activation

        Returns:
            NVFP4Tensor: Quantized tensor in NVFP4 format
        """
        assert len(data_hp.shape) in (2, 3), "unsupported"
        leading_dims, M, K = data_hp.shape[:-2], data_hp.shape[-2], data_hp.shape[-1]

        quantize_to_nvfp4_kernel_choice = _handle_use_triton_kernel(
            use_triton_kernel, quantize_to_nvfp4_kernel_choice
        )

        if quantize_to_nvfp4_kernel_choice == QuantizeToNVFP4KernelChoice.TRITON:
            assert is_swizzled_scales, "Triton kernel only supports swizzled scales"
            assert K % 16 == 0, (
                f"Triton kernel requires K (dim -1) to be divisible by 16, got {K}"
            )
            blockwise_scales, data_lp = triton_quantize_nvfp4(data_hp, per_tensor_scale)
        elif quantize_to_nvfp4_kernel_choice == QuantizeToNVFP4KernelChoice.TORCH:
            blockwise_scales, data_lp = nvfp4_quantize(
                data_hp, block_size, per_tensor_scale
            )
            if is_swizzled_scales:
                scale_shape = (math.prod(leading_dims) * M, K // block_size)
                blockwise_scales = to_blocked(
                    blockwise_scales.view(scale_shape)
                ).flatten()
        else:
            raise ValueError(
                f"Unsupported quantize_to_nvfp4_kernel_choice: {quantize_to_nvfp4_kernel_choice}"
            )

        if is_swizzled_scales:
            scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
        else:
            # a 1x16 unpacked or 1x8 packed qdata tile corresponds to 1
            # scale element
            scale_M, scale_K = M, K // block_size
        blockwise_scales = blockwise_scales.view(*leading_dims, scale_M, scale_K)

        return NVFP4Tensor(
            data_lp,
            blockwise_scales,
            block_size,
            data_hp.dtype,
            per_tensor_scale,
            act_per_tensor_scale,
            is_swizzled_scales,
            quantize_to_nvfp4_kernel_choice,
            act_quant_kwargs,
        )

    # Do not force the NVFP4Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert NVFP4Tensor back to high precision dtype.

        Args:
            target_dtype: Target dtype for dequantization (e.g., torch.float32, torch.bfloat16)

        Returns:
            torch.Tensor: Dequantized tensor in the target dtype
        """
        if output_dtype is None:
            output_dtype = self.dtype
        is_transposed = self.qdata.stride(-2) < self.qdata.stride(-1)
        if is_transposed:
            leading_dims, M, K = self.shape[:-2], self.shape[-1], self.shape[-2]
        else:
            leading_dims, M, K = self.shape[:-2], self.shape[-2], self.shape[-1]
        data = self.qdata.transpose(-2, -1) if is_transposed else self.qdata
        data_unpacked = unpack_uint4(data.contiguous().view(torch.uint8))
        data_f32 = f4_unpacked_to_f32(data_unpacked)

        data_f32 = data_f32.view(
            *leading_dims, M, K // self.block_size, self.block_size
        )
        scale_e4m3_reshaped = self.get_hp_scales().view(
            *leading_dims, M, K // self.block_size, 1
        )
        data_scaled = data_f32 * scale_e4m3_reshaped.to(torch.float32)
        result = data_scaled.view(*leading_dims, M, K).to(output_dtype)

        if is_transposed:
            result = result.transpose(-2, -1)

        return result

    def get_hp_scales(self) -> torch.Tensor:
        """Get the scales of the NVFP4Tensor in original dtype.

        Returns:
            torch.Tensor: Scales of the NVFP4Tensor
        """
        is_transposed = self.qdata.stride(-2) < self.qdata.stride(-1)
        if is_transposed:
            leading_dims, M, K = self.shape[:-2], self.shape[-1], self.shape[-2]
            scale_e4m3 = self.scale.transpose(-2, -1)
        else:
            leading_dims, M, K = self.shape[:-2], self.shape[-2], self.shape[-1]
            scale_e4m3 = self.scale

        if self.is_swizzled_scales:
            scale_e4m3 = from_blocked(
                scale_e4m3, math.prod(leading_dims) * M, K // self.block_size
            )

        return (
            scale_e4m3.to(self.orig_dtype)
            if self.per_tensor_scale is None
            else self.per_tensor_scale * scale_e4m3.to(self.orig_dtype)
        )

    @classmethod
    def _same_metadata(cls, self: "NVFP4Tensor", src: "NVFP4Tensor") -> bool:
        """Check if two NVFP4Tensors have the same metadata.

        Args:
            self: First NVFP4Tensor to compare
            src: Second NVFP4Tensor to compare

        Returns:
            bool: True if both tensors have identical metadata, False otherwise
        """
        per_tensor_scale_equal = (
            self.per_tensor_scale is None and src.per_tensor_scale is None
        ) or (self.per_tensor_scale.shape == src.per_tensor_scale.shape)
        act_per_tensor_scale_equal = (
            self.act_per_tensor_scale is None and src.act_per_tensor_scale is None
        ) or (self.act_per_tensor_scale.shape == src.act_per_tensor_scale.shape)

        return (
            isinstance(self, NVFP4Tensor)
            and isinstance(src, NVFP4Tensor)
            and self.block_size == src.block_size
            and self.orig_dtype == src.orig_dtype
            and self.is_swizzled_scales == src.is_swizzled_scales
            and self.scale.shape == src.scale.shape
            and per_tensor_scale_equal
            and act_per_tensor_scale_equal
            and self.qdata.shape == src.qdata.shape
            and self.act_quant_kwargs == src.act_quant_kwargs
        )


implements = NVFP4Tensor.implements


# TODO(future PR): move this to AOBaseTensor (will require debugging/fixing CI)
@implements([aten._to_copy.default])
def nvfp4_to_copy(func, types, args, kwargs):
    """Autocast + device movement"""
    assert isinstance(args[0], NVFP4Tensor)

    # Handle dtype parameter
    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        assert dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "Only support floating point conversion for autocast w/ NVFP4Tensor"

    # Handle device parameter
    device = kwargs.pop("device", None)
    if device is not None:
        tensor = args[0]._apply_fn_to_data(lambda x: func(x, device=device))
        tensor = return_and_correct_aliasing(func, args, {}, tensor)
    else:
        tensor = args[0]

    if dtype is not None:
        res = NVFP4Tensor(
            tensor.qdata,
            tensor.scale,
            tensor.block_size,
            dtype,
            tensor.per_tensor_scale,
            tensor.act_per_tensor_scale,
            tensor.is_swizzled_scales,
            tensor.quantize_to_nvfp4_kernel_choice,
            tensor.act_quant_kwargs,
        )
        return res

    return tensor


@implements([aten.slice.Tensor])
def nvfp4_slice(func, types, args, kwargs):
    x, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    assert x.qdata.is_contiguous(), "Only support contiguous data for now"
    assert len(x.shape) == 2, (
        f"only rank 2 is supported for slice, got rank {len(x.shape)}"
    )

    sliced_data, sliced_scale = _swizzle_aware_slice(x, dim, start, end, step)

    # Create result tensor
    result = NVFP4Tensor(
        sliced_data,
        sliced_scale,
        x.block_size,
        x.orig_dtype,
        x.per_tensor_scale,
        x.act_per_tensor_scale,
        x.is_swizzled_scales,
        x.quantize_to_nvfp4_kernel_choice,
        x.act_quant_kwargs,
    )

    return return_and_correct_aliasing(func, args, kwargs, result)


@implements([aten.t.default])
def nvfp4_t(func, types, args, kwargs):
    # For now, only transpose(input, 0, 1) is supported.
    old = args[0]
    new = NVFP4Tensor(
        old.qdata.t(),
        old.scale.t(),
        old.block_size,
        old.orig_dtype,
        old.per_tensor_scale,
        old.act_per_tensor_scale,
        old.is_swizzled_scales,
        old.quantize_to_nvfp4_kernel_choice,
        old.act_quant_kwargs,
    )
    return new


@implements([aten.transpose.int])
def nvfp4_transpose(func, types, args, kwargs):
    old, dim0, dim1 = args
    assert len(old.shape) == 3, f"unsupported rank {len(old.shape)}"
    valid_3d_dims = ((1, 2), (2, 1), (-1, -2), (-2, -1))
    assert (dim0, dim1) in valid_3d_dims, f"transpose unsupported for {dim0=} {dim1=}"
    new_qdata = func(old.qdata, dim0, dim1, **kwargs)
    new_scale = func(old.scale, dim0, dim1, **kwargs)
    new = NVFP4Tensor(
        new_qdata,
        new_scale,
        old.block_size,
        old.orig_dtype,
        old.per_tensor_scale,
        old.act_per_tensor_scale,
        old.is_swizzled_scales,
        old.quantize_to_nvfp4_kernel_choice,
        old.act_quant_kwargs,
    )
    return new


@implements([aten.view.default])
def nvfp4_view_op(func, types, args, kwargs):
    data = args[0].qdata
    new_size = args[1]
    new_size = tensor_size_hp_to_fp4x2(new_size, data.is_contiguous())
    new_data = func(data, new_size, *args[2:], **kwargs)
    return NVFP4Tensor(
        new_data,
        args[0].scale,
        args[0].block_size,
        args[0].orig_dtype,
        args[0].per_tensor_scale,
        args[0].act_per_tensor_scale,
        args[0].is_swizzled_scales,
        args[0].quantize_to_nvfp4_kernel_choice,
        args[0].act_quant_kwargs,
    )


@implements([aten.select.int])
def nvfp4_select(func, types, args, kwargs):
    old, dim, index = args
    assert dim == 0, f"NVFP4Tensor aten.select.int with {dim=} is not yet supported"
    assert len(old.qdata.shape) == len(old.scale.shape), "unsupported"
    new = old.__class__(
        old.qdata[index],
        old.scale[index],
        old.block_size,
        old.orig_dtype,
        old.per_tensor_scale,
        old.act_per_tensor_scale,
        old.is_swizzled_scales,
        old.quantize_to_nvfp4_kernel_choice,
        old.act_quant_kwargs,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


def _addmm_nvfp4_dispatch(
    a: NVFP4Tensor, b: NVFP4Tensor, aten_op, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Core implementation shared between nvfp4_mm, nvfp4_addmm, and nvfp4_linear.
    The only difference is whether bias is None or not.
    """
    assert a.qdata.is_contiguous()
    assert a.scale.is_contiguous()
    assert b.qdata.t().is_contiguous()
    assert b.scale.t().is_contiguous()
    assert a.block_size == 16, f"NVFP4 requires block_size=16, got {a.block_size}"
    assert b.block_size == 16, f"NVFP4 requires block_size=16, got {b.block_size}"
    assert len(a.shape) == 2 and len(b.shape) == 2

    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]

    # Swizzle Dizzle
    if a.is_swizzled_scales:
        a_scale_blocked = a.scale  # Already swizzled
    else:
        a_scale = a.scale.view(M, K // a.block_size)
        a_scale_blocked = to_blocked(a_scale)

    if b.is_swizzled_scales:
        b_scale_blocked = b.scale.t()  # Already swizzled
    else:
        b_scale = b.scale.t().view(N, K // b.block_size)
        b_scale_blocked = to_blocked(b_scale)

    # Merge double quant scales into 1 scale for Scale_In^D
    if a.per_tensor_scale is not None:
        assert b.per_tensor_scale is not None
        scale_result = a.per_tensor_scale * b.per_tensor_scale
    else:
        assert b.per_tensor_scale is None and a.per_tensor_scale is None
        scale_result = None

    # THIS IS A WORKAROUND FOR TWO ERRORS:
    #
    # (1) RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
    # When we have per-tensor scaling, we need to apply it before bias
    # since bias is not quantized
    #
    # (2) RuntimeError: Bias is not supported when out_dtype is set to Float32
    # This is not supported by _scaled_mm
    should_add_bias_separately = (
        scale_result is not None or a.orig_dtype == torch.float32
    ) and (bias is not None)
    # should_add_bias_separately = bias is not None

    # For gemm(A, B) with original high precision inputs A and B:
    #
    # 1. A and B are always cast to fp32 before being quantized and packed
    #    into uint8 (2 fp4 values per byte)
    # 2. _scaled_mm (cublas) always accumulates in fp32 since use_fast_accum=False
    # 3. Outputs are cast to A.dtype before returning
    # 4. Bias is added outside _scaled_mm if per_tensor_scale exists
    #    or output dtype is fp32
    #
    # -----------------------------------------------------------------------------
    # | A.dtype | B.dtype | Accum dtype | Out dtype | Bias added in _scaled_mm?   |
    # -----------------------------------------------------------------------------
    # | fp32    | fp32    | fp32        | fp32      | No                          |
    # | fp32    | bf16    | fp32        | fp32      | No                          |
    # | bf16    | fp32    | fp32        | bf16      | Only if no per_tensor_scale |
    # | bf16    | bf16    | fp32        | bf16      | Only if no per_tensor_scale |
    # -----------------------------------------------------------------------------
    result = torch._scaled_mm(
        a.qdata.view(torch.float4_e2m1fn_x2),
        b.qdata.view(torch.float4_e2m1fn_x2),
        a_scale_blocked.view(torch.float8_e4m3fn),
        b_scale_blocked.view(torch.float8_e4m3fn),
        bias=None if should_add_bias_separately else bias,
        out_dtype=a.orig_dtype,
        # scale_result=scale_result,  # Not supported yet
    )

    if scale_result is not None:
        result = result * scale_result.to(a.orig_dtype)

    # Add bias after scaling if needed
    if should_add_bias_separately:
        result = result + bias.to(a.orig_dtype)

    return result


@implements([torch.nn.functional.linear, aten.linear.default])
def nvfp4_linear(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    if not isinstance(weight_tensor, NVFP4Tensor):
        raise NotImplementedError("NVFP4Tensor: weight must be NVFP4Tensor")

    if weight_tensor.act_quant_kwargs is None:
        # weight_only quant
        weight_dequant = weight_tensor.dequantize(weight_tensor.orig_dtype)
        return torch.nn.functional.linear(input_tensor, weight_dequant, bias)
    else:
        # dynamic quant
        k = weight_tensor.act_quant_kwargs
        if k.use_dynamic_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(input_tensor))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = weight_tensor.act_per_tensor_scale
        orig_shape = input_tensor.shape
        input_tensor = input_tensor.view(-1, orig_shape[-1])
        input_tensor = NVFP4Tensor.to_nvfp4(
            input_tensor,
            block_size=k.block_size,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=k.is_swizzled_scales,
            quantize_to_nvfp4_kernel_choice=k.quantize_to_nvfp4_kernel_choice,
        )
        res = _addmm_nvfp4_dispatch(input_tensor, weight_tensor.t(), func, bias=bias)
        res = res.reshape(*orig_shape[:-1], res.shape[-1])
        return res


@implements([aten.mm.default, aten.matmul.default])
def nvfp4_mm(func, types, args, kwargs):
    input_tensor, weight_tensor = args[0], args[1]

    if not isinstance(weight_tensor, NVFP4Tensor):
        raise NotImplementedError("NVFP4Tensor: weight must be NVFP4Tensor")

    if weight_tensor.act_quant_kwargs is None:
        weight_dequant = weight_tensor.dequantize(weight_tensor.orig_dtype)
        if isinstance(input_tensor, NVFP4Tensor):
            input_dequant = input_tensor.dequantize(input_tensor.orig_dtype)
            return func(input_dequant, weight_dequant)
        else:
            return func(input_tensor, weight_dequant)
    else:
        if not isinstance(input_tensor, NVFP4Tensor):
            k = weight_tensor.act_quant_kwargs
            if k.use_dynamic_per_tensor_scale:
                tensor_amax = torch.max(torch.abs(input_tensor))
                per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
            else:
                per_tensor_scale = weight_tensor.act_per_tensor_scale
            input_tensor = NVFP4Tensor.to_nvfp4(
                input_tensor,
                block_size=k.block_size,
                per_tensor_scale=per_tensor_scale,
                is_swizzled_scales=k.is_swizzled_scales,
                quantize_to_nvfp4_kernel_choice=k.quantize_to_nvfp4_kernel_choice,
            )
        return _addmm_nvfp4_dispatch(input_tensor, weight_tensor, func)


@implements([aten.addmm.default])
def nvfp4_addmm(func, types, args, kwargs):
    bias, input_tensor, weight_tensor = args[0], args[1], args[2]

    if not isinstance(weight_tensor, NVFP4Tensor):
        raise NotImplementedError("NVFP4Tensor: weight must be NVFP4Tensor")

    if weight_tensor.act_quant_kwargs is None:
        weight_dequant = weight_tensor.dequantize(weight_tensor.orig_dtype)
        if isinstance(input_tensor, NVFP4Tensor):
            input_dequant = input_tensor.dequantize(input_tensor.orig_dtype)
            return torch.addmm(bias, input_dequant, weight_dequant)
        else:
            return torch.addmm(bias, input_tensor, weight_dequant)
    else:
        # TODO: refactor duplicate code
        if not isinstance(input_tensor, NVFP4Tensor):
            k = weight_tensor.act_quant_kwargs
            if k.use_dynamic_per_tensor_scale:
                tensor_amax = torch.max(torch.abs(input_tensor))
                per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
            else:
                per_tensor_scale = weight_tensor.act_per_tensor_scale
            input_tensor = NVFP4Tensor.to_nvfp4(
                input_tensor,
                block_size=k.block_size,
                per_tensor_scale=per_tensor_scale,
                is_swizzled_scales=k.is_swizzled_scales,
                quantize_to_nvfp4_kernel_choice=k.quantize_to_nvfp4_kernel_choice,
            )
        return _addmm_nvfp4_dispatch(input_tensor, weight_tensor, func, bias=bias)


def per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert per-tensor amax to per-tensor scale for NVFP4 quantization.

    Divides by both F8E4M3_MAX and F4_E2M1_MAX to ensure block scales can utilize
    the full FP8 E4M3 range (up to 448) when block_max equals tensor_max.
    Without F4_E2M1_MAX, the maximum scale would only reach FP8_MAX / FP4_MAX.

    Args:
        amax: Per-tensor absolute maximum value from calibration

    Returns:
        torch.Tensor: Per-tensor scale for two-level NVFP4 scaling
    """
    return amax.to(torch.float32) / (F8E4M3_MAX * F4_E2M1_MAX)


def nvfp4_quantize(
    data_hp: torch.Tensor,
    block_size: int = 16,
    per_tensor_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVIDIA FP4 quantization with UE4M3 scales.

    Implements the NVIDIA algorithm for quantizing tensors to FP4 format
    with unsigned E4M3 (UE4M3) scales.

    Args:
        data_hp: High precision input tensor (bfloat16 or float32)
        block_size: Block size for quantization (must be 16)
        per_tensor_amax: Optional pre-computed absolute maximum for calibration.
            If provided, uses per-tensor scaling. If None, uses block-wise scaling only.

    Returns:
        tuple: A tuple containing:
            - total_scale_fp8: Blockwise scales in float8_e4m3fn format
            - per_tensor_scale: Global per-tensor scale if per_tensor_amax provided, else None
            - data_lp: Packed FP4 data (2 values per byte)

    Raises:
        AssertionError: If input dtype is not supported, tensor size is not
            divisible by block_size, tensor is not contiguous, or block_size != 16
    """
    assert data_hp.dtype in (torch.bfloat16, torch.float), (
        f"{data_hp.dtype} not supported"
    )
    assert data_hp.size(-1) % block_size == 0, "K dim must be divisible by block_size"
    assert data_hp.is_contiguous(), "Only support contiguous data for now"
    assert block_size == 16, "NVFP4 requires block_size=16"

    orig_shape = data_hp.shape
    # Convert to float32 early for consistent precision with Triton implementation
    data_hp = data_hp.float().reshape(orig_shape[0], -1, block_size)

    max_abs = torch.amax(torch.abs(data_hp), dim=-1)
    # These scales are currently in fp32, we are going to `quantize` them to e4m3
    block_scale = max_abs / F4_E2M1_MAX

    out_scales = None
    if per_tensor_scale is None:
        # We are doing single level scaling
        block_scale_fp8 = torch.clamp(block_scale, min=E4M3_EPS, max=F8E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        block_scale_fp32 = block_scale_fp8.to(torch.float32)
        data_scaled = data_hp / block_scale_fp32.unsqueeze(-1)
        out_scales = block_scale_fp8
    else:
        # We are doing two level scaling,
        # This will likely be calibrated but
        # we want the per_tensor_scale ~= amax of the block_scale_fp32
        block_scale_fp32 = block_scale.to(torch.float32)
        # Quantize the blockwise scales w/ the per_tensor_scale
        scaled_block_scales = block_scale_fp32 / per_tensor_scale
        scaled_block_scales_fp8 = torch.clamp(
            scaled_block_scales, min=E4M3_EPS, max=F8E4M3_MAX
        ).to(torch.float8_e4m3fn)
        scaled_block_scales_fp32 = scaled_block_scales_fp8.to(torch.float32)
        # We "temporarily" dequant the scaled_block_scales_fp32 to get the per_tensor_scale
        # To apply to data
        total_scale = per_tensor_scale * scaled_block_scales_fp32
        data_scaled = data_hp / total_scale.unsqueeze(-1)
        out_scales = scaled_block_scales_fp8

    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)
    data_lp = f32_to_f4_unpacked(data_scaled)
    # TODO: NotImplementedError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'
    # data_lp = pack_uint4(data_lp).view(torch.float4_e2m1fn_x2)
    data_lp = pack_uint4(data_lp)
    return out_scales, data_lp
