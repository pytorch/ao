# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import sys
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

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
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked
from torchao.utils import ceil_div, fill_defaults

E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny

aten = torch.ops.aten

NVFP4_OPS_TABLE: Dict[Any, Any] = {}


class NVFP4MMConfig(Enum):
    DYNAMIC = "dynamic"
    WEIGHT_ONLY = "weight_only"


def implements(aten_ops):
    """Register aten ops to the NVFP4 op table"""

    def decorator(func):
        for op in aten_ops:
            NVFP4_OPS_TABLE[op] = func
        return func

    return decorator


class NVFP4Tensor(torch.Tensor):
    """NVIDIA FP4 (NVFP4) Tensor subclass.

    This implements the NVIDIA variant of MX FP4 format, which uses a specific
    quantization algorithm for FP4 data with UE4M3 scales.

    Attributes:
        _scale_e4m3: Blockwise scales in float8_e4m3fn format (may be swizzled)
        _per_tensor_scale: Optional global per-tensor scale in float32 format
        _data: Packed FP4 data (2 values per byte)
        _block_size: Block size for quantization (fixed at 16)
        _orig_dtype: Original tensor dtype before quantization
        _is_swizzled_scales: Whether scales are stored in swizzled (blocked) format
        mm_config: Matrix multiplication configuration
    """

    _scale_e4m3: torch.Tensor
    _per_tensor_scale: Optional[torch.Tensor]
    _data: torch.Tensor
    _block_size: int
    _orig_dtype: torch.dtype
    _is_swizzled_scales: bool
    mm_config: NVFP4MMConfig
    use_triton_kernel: bool

    def __new__(
        cls,
        blockwise_scales,
        per_tensor_scale,
        data_bits,
        block_size,
        orig_dtype,
        mm_config=NVFP4MMConfig.DYNAMIC,
        is_swizzled_scales=False,
        use_triton_kernel=False,
    ):
        # FP4 tensor size handling two paths, contiguous or not
        new_size = data_bits.size()

        new_size = tensor_size_fp4x2_to_hp(
            new_size,
            data_bits.stride(0) > data_bits.stride(1),
        )

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            new_size,
            dtype=orig_dtype,
            device=data_bits.device,
            requires_grad=False,
        )

        self._scale_e4m3 = blockwise_scales
        self._is_swizzled_scales = is_swizzled_scales
        self._per_tensor_scale = per_tensor_scale
        self._data = data_bits
        self._block_size = block_size
        self._orig_dtype = orig_dtype
        self.mm_config = mm_config
        self.use_triton_kernel = use_triton_kernel
        return self

    def __repr__(self):
        return f"NVFP4Tensor: blockwise_scales: {self._scale_e4m3}, per_tensor_scale: {self._per_tensor_scale}, d: {self._data}, d_hp: {self.to_dtype(self._orig_dtype)}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Use NVFP4-specific ops table
        if func in NVFP4_OPS_TABLE:
            return NVFP4_OPS_TABLE[func](func, types, args, kwargs)

        raise NotImplementedError(f"{func} not implemented for NVFP4Tensor")

    @staticmethod
    def to_nvfp4(
        data_hp: torch.Tensor,
        block_size: int = 16,
        per_tensor_scale: Optional[torch.Tensor] = None,
        mm_config: NVFP4MMConfig = NVFP4MMConfig.DYNAMIC,
        is_swizzled_scales: bool = False,
        use_triton_kernel: bool = False,
    ):
        """Convert high precision tensor to NVFP4 format.

        Args:
            data_hp: High precision input tensor (bfloat16 or float32)
            block_size: Block size for quantization (must be 16)
            per_tensor_scale: Optional pre-computed absolute maximum for calibration.
                If provided, uses per-tensor scaling. If None, uses block-wise scaling only.
            mm_config: Matrix multiplication configuration
            is_swizzled_scales: If True, store scales in swizzled format for faster matrix multiplication
            use_triton_kernel: If True, use Triton kernel for quantization

        Returns:
            NVFP4Tensor: Quantized tensor in NVFP4 format
        """
        if use_triton_kernel:
            assert is_swizzled_scales, "Triton kernel only supports swizzled scales"
            assert data_hp.shape[1] % 16 == 0, (
                f"Triton kernel requires K (dim 1) to be divisible by 16, got {data_hp.shape[1]}"
            )
            blockwise_scales, data_lp = triton_quantize_nvfp4(data_hp, per_tensor_scale)
        else:
            blockwise_scales, data_lp = nvfp4_quantize(
                data_hp, block_size, per_tensor_scale
            )
            if is_swizzled_scales:
                M, K = data_hp.shape[0], data_hp.shape[1]
                scale_shape = (M, K // block_size)
                blockwise_scales = to_blocked(
                    blockwise_scales.view(scale_shape)
                ).flatten()

        return NVFP4Tensor(
            blockwise_scales,
            per_tensor_scale,
            data_lp,
            block_size,
            data_hp.dtype,
            mm_config,
            is_swizzled_scales,
            use_triton_kernel,
        )

    def __tensor_flatten__(self):
        ctx = {
            "_block_size": self._block_size,
            "_orig_dtype": self._orig_dtype,
            "_is_swizzled_scales": self._is_swizzled_scales,
            "mm_config": self.mm_config,
            "use_triton_kernel": self.use_triton_kernel,
        }
        tensor_list = ["_scale_e4m3", "_data"]
        if self._per_tensor_scale is not None:
            tensor_list.append("_per_tensor_scale")
        return tensor_list, ctx

    def _apply_fn_to_data(self, fn: Callable):
        """Applies a fn to all tensor components stored on this class"""
        tensor_names, ctx = self.__tensor_flatten__()
        new_tensors = {}
        for name in tensor_names:
            new_tensors[name] = fn(getattr(self, name))
        if "_per_tensor_scale" not in tensor_names:
            new_tensors["_per_tensor_scale"] = None
        return self.__class__.__tensor_unflatten__(
            new_tensors,
            ctx,
            None,
            None,
        )

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors,
        metadata,
        outer_size,
        outer_stride,
    ):
        return NVFP4Tensor(
            inner_tensors["_scale_e4m3"],
            inner_tensors.get("_per_tensor_scale", None),
            inner_tensors["_data"],
            metadata["_block_size"],
            metadata["_orig_dtype"],
            metadata["mm_config"],
            metadata.get("_is_swizzled_scales", False),
            metadata.get("use_triton_kernel", False),
        )

    # Do not force the NVFP4Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

    def to_dtype(self, target_dtype: torch.dtype) -> torch.Tensor:
        """Convert NVFP4Tensor back to high precision dtype.

        Args:
            target_dtype: Target dtype for dequantization (e.g., torch.float32, torch.bfloat16)

        Returns:
            torch.Tensor: Dequantized tensor in the target dtype
        """
        is_transposed = self._data.stride(0) < self._data.stride(1)
        if is_transposed:
            M, K = self.shape[1], self.shape[0]
        else:
            M, K = self.shape[0], self.shape[1]
        data = self._data.t() if is_transposed else self._data
        data_unpacked = unpack_uint4(data.contiguous().view(torch.uint8))
        data_f32 = f4_unpacked_to_f32(data_unpacked)

        data_f32 = data_f32.view(M, K // self._block_size, self._block_size)
        scale_e4m3_reshaped = self.get_hp_scales().view(M, K // self._block_size, 1)
        data_scaled = data_f32 * scale_e4m3_reshaped.to(torch.float32)
        result = data_scaled.view(M, K).to(target_dtype)

        if is_transposed:
            result = result.t()

        return result

    def get_hp_scales(self) -> torch.Tensor:
        """Get the scales of the NVFP4Tensor in original dtype.

        Returns:
            torch.Tensor: Scales of the NVFP4Tensor
        """
        is_transposed = self._data.stride(0) < self._data.stride(1)
        if is_transposed:
            M, K = self.shape[1], self.shape[0]
        else:
            M, K = self.shape[0], self.shape[1]

        if self._is_swizzled_scales:
            scale_e4m3 = from_blocked(self._scale_e4m3, M, K // self._block_size)
        else:
            scale_e4m3 = self._scale_e4m3

        return (
            scale_e4m3.to(self._orig_dtype)
            if not self._per_tensor_scale
            else self._per_tensor_scale * scale_e4m3.to(self._orig_dtype)
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
            self._per_tensor_scale is None and src._per_tensor_scale is None
        ) or (self._per_tensor_scale.shape == src._per_tensor_scale.shape)

        return (
            isinstance(self, NVFP4Tensor)
            and isinstance(src, NVFP4Tensor)
            and self._block_size == src._block_size
            and self._orig_dtype == src._orig_dtype
            and self._is_swizzled_scales == src._is_swizzled_scales
            and self._scale_e4m3.shape == src._scale_e4m3.shape
            and per_tensor_scale_equal
            and self._data.shape == src._data.shape
        )


@implements([aten.detach.default, aten.alias.default])
def nvfp4_detach_alias(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(func)
    )


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
            tensor._scale_e4m3,
            tensor._per_tensor_scale,
            tensor._data,
            tensor._block_size,
            dtype,
            tensor.mm_config,
            tensor._is_swizzled_scales,
            tensor.use_triton_kernel,
        )
        return res

    return tensor


@implements([aten.copy_.default])
def nvfp4_copy_(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if NVFP4Tensor._same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return self
    raise ValueError(
        f"Not supported args for copy_ due to metadata mismatch: {self}, {src}"
    )


@implements([aten.clone.default])
def nvfp4_clone(func, types, args, kwargs):
    self = args[0]
    memory_format = kwargs.get("memory_format", None)

    if memory_format is not None:
        clone_fn = lambda x: x.clone(memory_format=memory_format)
    else:
        clone_fn = lambda x: x.clone()

    return self._apply_fn_to_data(clone_fn)


@implements([aten.slice.Tensor])
def nvfp4_slice(func, types, args, kwargs):
    x, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    assert x._data.is_contiguous(), "Only support contiguous data for now"

    M, K = x.shape[0], x.shape[1]

    if x._is_swizzled_scales:
        scale_rows = M
        scale_cols = K // x._block_size
        n_row_blocks = ceil_div(scale_rows, 128)
        n_col_blocks = ceil_div(scale_cols, 4)
        elements_per_block = 32 * 16  # 512 elements

        if dim == 0:
            # Row slicing
            # Handle sys.maxsize (default slice end)
            if end == sys.maxsize:
                end = M

            # Check if start/end align with 128-row boundaries
            if start is not None and start % 128 != 0:
                raise RuntimeError(
                    f"Row slicing of NVFP4Tensor with swizzled scales requires "
                    f"start index to be a multiple of 128, got {start}"
                )
            if end is not None and end != M and end % 128 != 0:
                raise RuntimeError(
                    f"Row slicing of NVFP4Tensor with swizzled scales requires "
                    f"end index to be a multiple of 128 or equal to tensor size {M}, got {end}"
                )

            # Calculate which row blocks to keep
            start_block = 0 if start is None else start // 128
            end_block = n_row_blocks if end is None or end >= M else end // 128

            # The swizzled tensor has shape (n_row_blocks * n_col_blocks * 32 * 16,)
            blocks_per_row = n_col_blocks
            start_idx = start_block * blocks_per_row * elements_per_block
            end_idx = (
                end_block * blocks_per_row * elements_per_block
                if end_block < n_row_blocks
                else None
            )

            sliced_scale = aten.slice.Tensor(x._scale_e4m3, 0, start_idx, end_idx, 1)
            sliced_data = aten.slice.Tensor(x._data, 0, start, end, step)

        elif dim == 1:
            # Column slicing
            # Handle sys.maxsize (default slice end)
            if end == sys.maxsize:
                end = K

            # Check if start/end align with 64-column boundaries (4 scale columns * 16 block_size)
            if start is not None and start % 64 != 0:
                raise RuntimeError(
                    f"Column slicing of NVFP4Tensor with swizzled scales requires "
                    f"start index to be a multiple of 64, got {start}"
                )
            if end is not None and end != K and end % 64 != 0:
                raise RuntimeError(
                    f"Column slicing of NVFP4Tensor with swizzled scales requires "
                    f"end index to be a multiple of 64 or equal to tensor size {K}, got {end}"
                )

            # Also check FP4 packing alignment
            if start is not None and start % 2 != 0:
                raise RuntimeError(f"Start index {start} must be even for FP4 packing")
            if end is not None and end != K and end % 2 != 0:
                raise RuntimeError(f"End index {end} must be even for FP4 packing")

            # Calculate which column blocks to keep
            start_scale_col = 0 if start is None else start // 16
            end_scale_col = scale_cols if end is None or end >= K else end // 16

            start_col_block = start_scale_col // 4
            end_col_block = end_scale_col // 4

            # Verify the end aligns with block boundary
            if end_scale_col % 4 != 0:
                raise RuntimeError(
                    f"Column slicing end index {end} does not align with scale block boundaries. "
                    f"End must result in a multiple of 4 scale columns (64 data columns)."
                )

            if start_col_block == 0 and end_col_block == n_col_blocks:
                # Full width - no slicing needed
                sliced_scale = x._scale_e4m3
            else:
                # Extract specific column blocks from each row block
                # Each row block in swizzled format contains n_col_blocks chunks of (32, 16)
                elements_per_row_block = n_col_blocks * elements_per_block

                # Build list of slices to extract
                slices_to_extract = []
                for row_block in range(n_row_blocks):
                    row_start = row_block * elements_per_row_block
                    col_start = row_start + start_col_block * elements_per_block
                    col_end = row_start + end_col_block * elements_per_block
                    slices_to_extract.append(x._scale_e4m3[col_start:col_end])

                # Concatenate all the slices
                sliced_scale = torch.cat(slices_to_extract, dim=0)

            # Slice the data tensor
            packed_start = None if start is None else start // 2
            packed_end = None if end is None else end // 2
            sliced_data = aten.slice.Tensor(
                x._data, dim, packed_start, packed_end, step
            )

        else:
            raise ValueError(
                f"NVFP4Tensor only supports slicing along dimensions 0 and 1, got dim={dim}"
            )

    else:
        scale_shaped = x._scale_e4m3.view(M, K // x._block_size)

        if dim == 0:
            sliced_scale = aten.slice.Tensor(scale_shaped, dim, start, end, step)
            sliced_data = aten.slice.Tensor(x._data, dim, start, end, step)

        elif dim == 1:
            if start is not None:
                assert start % x._block_size == 0, (
                    f"Start index {start} must be a multiple of block_size {x._block_size}"
                )
                assert start % 2 == 0, (
                    f"Start index {start} must be even for FP4 packing"
                )

            if end is not None and end != sys.maxsize:
                assert end % x._block_size == 0, (
                    f"End index {end} must be a multiple of block_size {x._block_size}"
                )
                assert end % 2 == 0, f"End index {end} must be even for FP4 packing"

            packed_start = None if start is None else start // 2
            packed_end = None if end is None else end // 2
            sliced_data = aten.slice.Tensor(
                x._data, dim, packed_start, packed_end, step
            )

            start_block = 0 if start is None else start // x._block_size
            end_block = None if end is None else end // x._block_size
            sliced_scale = aten.slice.Tensor(
                scale_shaped, 1, start_block, end_block, step
            )

        sliced_scale = sliced_scale.flatten()

    # Create result tensor
    result = NVFP4Tensor(
        sliced_scale,
        x._per_tensor_scale,
        sliced_data,
        x._block_size,
        x._orig_dtype,
        x.mm_config,
        x._is_swizzled_scales,
        x.use_triton_kernel,
    )

    return return_and_correct_aliasing(func, args, kwargs, result)


@implements([aten.t.default])
def nvfp4_t(func, types, args, kwargs):
    # For now, only transpose(input, 0, 1) is supported.
    old = args[0]
    new = NVFP4Tensor(
        old._scale_e4m3,
        old._per_tensor_scale,
        old._data.t(),
        old._block_size,
        old._orig_dtype,
        old.mm_config,
        old._is_swizzled_scales,
        old.use_triton_kernel,
    )
    return new


@implements([aten.view.default])
def nvfp4_view_op(func, types, args, kwargs):
    data = args[0]._data
    new_size = args[1]
    new_size = tensor_size_hp_to_fp4x2(new_size, data.is_contiguous())
    new_data = func(data, new_size, *args[2:], **kwargs)
    return NVFP4Tensor(
        args[0]._scale_e4m3,
        args[0]._per_tensor_scale,
        new_data,
        args[0]._block_size,
        args[0]._orig_dtype,
        args[0].mm_config,
        args[0]._is_swizzled_scales,
        args[0].use_triton_kernel,
    )


def _addmm_nvfp4_dispatch(
    a: NVFP4Tensor, b: NVFP4Tensor, aten_op, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Core implementation shared between nvfp4_mm, nvfp4_addmm, and nvfp4_linear.
    The only difference is whether bias is None or not.
    """
    assert a._data.is_contiguous()
    assert b._data.t().is_contiguous()
    assert a._block_size == 16, f"NVFP4 requires block_size=16, got {a._block_size}"
    assert b._block_size == 16, f"NVFP4 requires block_size=16, got {b._block_size}"

    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]

    # Swizzle Dizzle
    if a._is_swizzled_scales:
        a_scale_blocked = a._scale_e4m3  # Already swizzled
    else:
        a_scale = a._scale_e4m3.view(M, K // a._block_size)
        a_scale_blocked = to_blocked(a_scale)

    if b._is_swizzled_scales:
        b_scale_blocked = b._scale_e4m3  # Already swizzled
    else:
        b_scale = b._scale_e4m3.view(N, K // b._block_size)
        b_scale_blocked = to_blocked(b_scale)

    # Merge double quant scales into 1 scale for Scale_In^D
    if a._per_tensor_scale is not None:
        assert b._per_tensor_scale is not None
        scale_result = a._per_tensor_scale * b._per_tensor_scale
    else:
        assert b._per_tensor_scale is None and a._per_tensor_scale is None
        scale_result = None

    # THIS IS A WORKAROUND:
    # RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
    # When we have per-tensor scaling, we need to apply it before bias
    # since bias is not quantized
    should_add_bias_separately = (scale_result is not None) and (bias is not None)
    # should_add_bias_separately = bias is not None

    result = torch._scaled_mm(
        a._data.view(torch.float4_e2m1fn_x2),
        b._data.view(torch.float4_e2m1fn_x2),
        a_scale_blocked.view(torch.float8_e4m3fn),
        b_scale_blocked.view(torch.float8_e4m3fn),
        bias=None if should_add_bias_separately else bias,
        out_dtype=a._orig_dtype,
        # scale_result=scale_result,  # Not supported yet
    )

    if scale_result is not None:
        result = result * scale_result.to(a._orig_dtype)

    # Add bias after scaling if needed
    if should_add_bias_separately:
        result = result + bias

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

    config = weight_tensor.mm_config

    if config == NVFP4MMConfig.WEIGHT_ONLY:
        weight_dequant = weight_tensor.to_dtype(weight_tensor._orig_dtype)
        return torch.nn.functional.linear(input_tensor, weight_dequant, bias)
    else:
        input_tensor = NVFP4Tensor.to_nvfp4(
            input_tensor,
            mm_config=config,
            is_swizzled_scales=True,
            use_triton_kernel=weight_tensor.use_triton_kernel,
        )
        return _addmm_nvfp4_dispatch(input_tensor, weight_tensor.t(), func, bias=bias)


@implements([aten.mm.default, aten.matmul.default])
def nvfp4_mm(func, types, args, kwargs):
    input_tensor, weight_tensor = args[0], args[1]

    if not isinstance(weight_tensor, NVFP4Tensor):
        raise NotImplementedError("NVFP4Tensor: weight must be NVFP4Tensor")

    config = weight_tensor.mm_config

    if config == NVFP4MMConfig.WEIGHT_ONLY:
        weight_dequant = weight_tensor.to_dtype(weight_tensor._orig_dtype)
        if isinstance(input_tensor, NVFP4Tensor):
            input_dequant = input_tensor.to_dtype(input_tensor._orig_dtype)
            return func(input_dequant, weight_dequant)
        else:
            return func(input_tensor, weight_dequant)
    else:
        if not isinstance(input_tensor, NVFP4Tensor):
            input_tensor = NVFP4Tensor.to_nvfp4(
                input_tensor,
                mm_config=config,
                is_swizzled_scales=True,
                use_triton_kernel=weight_tensor.use_triton_kernel,
            )
        return _addmm_nvfp4_dispatch(input_tensor, weight_tensor, func)


@implements([aten.addmm.default])
def nvfp4_addmm(func, types, args, kwargs):
    bias, input_tensor, weight_tensor = args[0], args[1], args[2]

    if not isinstance(weight_tensor, NVFP4Tensor):
        raise NotImplementedError("NVFP4Tensor: weight must be NVFP4Tensor")

    config = weight_tensor.mm_config

    if config == NVFP4MMConfig.WEIGHT_ONLY:
        weight_dequant = weight_tensor.to_dtype(weight_tensor._orig_dtype)
        if isinstance(input_tensor, NVFP4Tensor):
            input_dequant = input_tensor.to_dtype(input_tensor._orig_dtype)
            return torch.addmm(bias, input_dequant, weight_dequant)
        else:
            return torch.addmm(bias, input_tensor, weight_dequant)
    else:
        if not isinstance(input_tensor, NVFP4Tensor):
            input_tensor = NVFP4Tensor.to_nvfp4(
                input_tensor,
                mm_config=config,
                is_swizzled_scales=True,
                use_triton_kernel=weight_tensor.use_triton_kernel,
            )
        return _addmm_nvfp4_dispatch(input_tensor, weight_tensor, func, bias=bias)


def per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert per-tensor amax to per-tensor scale.
    Used to scale fp32 scales down to fp8 scales

    Args:
        amax: Per-tensor amax tensor

    Returns:
        torch.Tensor: Per-tensor scale tensor
    """
    return torch.clamp(amax / F8E4M3_MAX, min=E4M3_EPS, max=F8E4M3_MAX).to(
        torch.float32
    )


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
