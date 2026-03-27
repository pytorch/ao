# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for scaling high precision tensors to float8 using "jagged"
rowwise scales (i.e., separate scales for each group/subtensor as determined by
the offsets).
"""

from typing import Tuple

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl

    EPS = 1e-12

    FP8_DTYPE_MAP = {
        torch.int8: tl.int8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.float8_e4m3fn: tl.float8e4nv,
        torch.float8_e4m3fnuz: tl.float8e4b8,
        torch.float8_e5m2: tl.float8e5,
        torch.float8_e5m2fnuz: tl.float8e5b16,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }

    # Two-pass kernel configs: iterate over rows in chunks of BLOCK_SIZE_ITER,
    # reading data twice (once for amax, once for scale+write).
    kernel_configs_2D = [
        triton.Config(
            {"BLOCK_SIZE": block_size, "BLOCK_SIZE_ITER": block_size_iter},
            num_warps=warps,
            num_stages=stages,
        )
        for block_size in [32, 64, 128]
        for block_size_iter in [64, 128, 256]
        for warps in [4, 8]
        for stages in [2, 3]
    ]
    kernel_configs_2D_dual = kernel_configs_2D

    # Fused single-pass kernel configs: MAX_GROUP_SIZE (passed as constexpr)
    # determines the tile height so all rows are loaded at once.
    # Only BLOCK_SIZE and num_warps are autotuned.
    kernel_configs_fused = [
        triton.Config(
            {"BLOCK_SIZE": block_size, "MAX_GROUP_SIZE": max_gs},
            num_warps=warps,
            num_stages=1,
        )
        for block_size in [32, 64]
        for max_gs in [256, 512, 1024, 2048]
        for warps in [8]
    ]

    @torch.library.custom_op(
        "torchao::triton_fp8_per_group_rowwise_scales", mutates_args={}
    )
    def triton_fp8_per_group_rowwise_scales(
        hp_tensor: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a high precision tensor to a float8 tensor in row-major memory layout,
        using 'jagged' rowwise scales (i.e., separate scales for each group/subtensor as
        determined by the offsets).

        Args:
            - hp_tensor: 2D high precision tensor to be converted
            - offsets: end index for each group/subtensor along dim 0
            - output_dtype: desired float8 dtype for the output tensor
            - round_scales_to_power_of_2: boolean indicating if scales should be rounded
                down to the nearest power of 2.
        Returns:
            - float8 tensor
            - jagged rowwise scales (i.e., rowwise scales for each group)
        """
        assert hp_tensor.ndim == 2, "input tensor must be 2D"

        tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
        tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

        fp8_dtype_min = torch.finfo(output_dtype).min
        fp8_dtype_max = torch.finfo(output_dtype).max

        m, k = hp_tensor.shape
        n_groups = offsets.numel()

        # allocate on-device buffers for output and scales
        output_buffer = torch.empty((m, k), dtype=output_dtype, device=hp_tensor.device)
        scales_buffer = torch.empty(
            (m * n_groups), dtype=torch.float32, device=hp_tensor.device
        )

        # parallelize across rows and groups (offsets)
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_SIZE"]),
            offsets.numel(),
        )
        _triton_fp8_per_group_rowwise_scales_kernel[grid](
            hp_tensor,
            offsets,
            output_buffer,
            scales_buffer,
            m,
            k,
            n_groups,
            hp_tensor.stride(0),
            output_buffer.stride(1),
            fp8_dtype_min,
            fp8_dtype_max,
            tl_input_dtype,
            tl_output_dtype,
            round_scales_to_power_of_2,
            EPS=EPS,
            STRIDE_OUTPUT_ROW=1,
            STRIDE_INPUT_COL=hp_tensor.stride(1),
        )
        return output_buffer, scales_buffer

    @triton_fp8_per_group_rowwise_scales.register_fake
    def _fake_triton_fp8_per_group_rowwise_scales_kernel(
        hp_tensor: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hp_tensor.ndim == 2, "input tensor must be 2D"
        m, k = hp_tensor.shape
        n_groups = offsets.numel()
        output = torch.empty_like(hp_tensor, dtype=output_dtype).as_strided(
            (m, k),  # shape
            (k, 1),  # stride
        )
        scales = torch.empty(
            (m * n_groups), dtype=torch.float32, device=hp_tensor.device
        )
        return output, scales

    # This kernel is used on grad_output.t() which has shape (K, M),
    # before the calculation `grad_B = grad_output_t @ input`.
    # However, in this code, we use the conventional dim names (M, K)
    # so the kernel is easily interpretable in a standalone fasion.
    # The tokens per expert will vary per iteration, so don't want
    # to recompile on `token` dim (K, in this case) changes.
    @triton.autotune(configs=kernel_configs_2D, key=["M", "N_GROUPS"])
    @triton.jit
    def _triton_fp8_per_group_rowwise_scales_kernel(
        input_ptr,
        offsets_ptr,
        out_ptr,
        scales_ptr,
        M: tl.int64,
        K: tl.int64,
        N_GROUPS: tl.int64,
        stride_input_row: tl.int64,
        stride_output_col: tl.int64,
        fp8_dtype_min: tl.constexpr,
        fp8_dtype_max: tl.constexpr,
        input_dtype: tl.constexpr,
        output_dtype: tl.constexpr,
        round_scales_to_power_of_2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_ITER: tl.constexpr,
        EPS: tl.constexpr,
        STRIDE_OUTPUT_ROW: tl.constexpr,
        STRIDE_INPUT_COL: tl.constexpr,
    ):
        # parallel across rows and groups (offsets)
        block_row_id = tl.program_id(axis=0)
        offset_idx = tl.program_id(axis=1)

        # determine start and end column idx for this group
        group_col_start_idx = tl.load(
            offsets_ptr + offset_idx - 1, mask=offset_idx > 0, other=0
        )
        group_col_end_idx = tl.load(offsets_ptr + offset_idx)
        block_row_offs = (block_row_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(
            tl.int64
        )

        # compute rowwise amaxes for this group
        amax_buffer = tl.zeros((BLOCK_SIZE,), dtype=input_dtype)
        for col_start_idx in range(
            group_col_start_idx, group_col_end_idx, BLOCK_SIZE_ITER
        ):
            block_col_offs = (col_start_idx + tl.arange(0, BLOCK_SIZE_ITER)).to(
                tl.int64
            )
            block_offs = (
                block_row_offs[:, None] * stride_input_row
                + block_col_offs[None, :] * STRIDE_INPUT_COL
            )
            block_mask = (block_row_offs[:, None] < M) & (
                block_col_offs[None, :] < group_col_end_idx
            )
            data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
                input_dtype
            )
            # we need to cast back to input dtype since triton promotes bf16 to fp32:
            # https://github.com/triton-lang/triton/blob/981e987eed9053b952f81153bc0779c99d8c642e/python/triton/language/standard.py#L173
            amax_buffer = tl.maximum(amax_buffer, tl.max(tl.abs(data), axis=1)).to(
                input_dtype
            )

        # compute rowwise scales for this group. round scales to nearest power of 2.
        amax_buffer = amax_buffer.to(tl.float64)
        scales = (fp8_dtype_max / tl.clamp(amax_buffer, min=EPS, max=float("inf"))).to(
            tl.float32
        )
        if round_scales_to_power_of_2:
            scales = tl.exp2(tl.floor(tl.log2(scales)))

        # store rowwise scales for each group in contiguous memory:
        # [group0_row0, group_0_row1, ..., group2_row0, group2_row1]
        scales_offs = block_row_offs + (M * offset_idx)
        scales_mask = tl.arange(0, BLOCK_SIZE) < M
        tl.store(scales_ptr + scales_offs, scales, mask=scales_mask)

        # perform float8 conversion for this group
        for col_start_idx in range(
            group_col_start_idx, group_col_end_idx, BLOCK_SIZE_ITER
        ):
            block_col_offs = (col_start_idx + tl.arange(0, BLOCK_SIZE_ITER)).to(
                tl.int64
            )
            block_offs = (
                block_row_offs[:, None] * stride_input_row
                + block_col_offs[None, :] * STRIDE_INPUT_COL
            )
            block_mask = (block_row_offs[:, None] < M) & (
                block_col_offs[None, :] < group_col_end_idx
            )
            data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
                input_dtype
            )
            scaled_data = data * scales[:, None]
            fp8_data = tl.clamp(scaled_data, min=fp8_dtype_min, max=fp8_dtype_max).to(
                output_dtype
            )
            out_offs = (
                block_row_offs[:, None] * STRIDE_OUTPUT_ROW
                + block_col_offs[None, :] * stride_output_col
            )
            tl.store(out_ptr + out_offs, fp8_data, mask=block_mask)

    @torch.library.custom_op(
        "torchao::triton_fp8_per_group_colwise_scales", mutates_args={}
    )
    def triton_fp8_per_group_colwise_scales(
        hp_tensor: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a high precision tensor to a float8 tensor in row-major memory layout,
        using 'jagged' column-wise scales (i.e., separate scales for each group/subtensor as
        determined by the offsets).

        Args:
            - hp_tensor: 2D high precision tensor to be converted
            - offsets: end index for each group/subtensor along dim 0
            - output_dtype: desired float8 dtype for the output tensor
            - round_scales_to_power_of_2: boolean indicating if scales should be rounded
                down to the nearest power of 2.
        Returns:
            - float8 tensor
            - jagged column-wise scales (i.e., column-wise scales for each group)
        """
        assert hp_tensor.ndim == 2, "input tensor must be 2D"

        num_elements = hp_tensor.numel()
        tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
        tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

        fp8_dtype_min = torch.finfo(output_dtype).min
        fp8_dtype_max = torch.finfo(output_dtype).max

        k, n = hp_tensor.shape
        n_groups = offsets.numel()

        # Output buffer in column major
        output_buffer = torch.empty_like(
            hp_tensor, dtype=output_dtype, device=hp_tensor.device
        ).as_strided(hp_tensor.size(), (1, k))

        scales_buffer = torch.empty(
            (n * n_groups), dtype=torch.float32, device=hp_tensor.device
        )

        # parallelize across columns and groups (offsets)
        grid = lambda meta: (
            triton.cdiv(n, meta["BLOCK_SIZE"]),
            offsets.numel(),
        )

        # Compute max group size to decide which kernel to use.
        # For uniform groups: max_group_size = k / n_groups.
        # For jagged groups: need the max of consecutive offset diffs.
        max_group_size = (k + n_groups - 1) // n_groups

        # Use fused single-pass kernel when the max group size fits in
        # the largest BLOCK_SIZE_ITER (2048). The fused kernel loads all
        # rows for a group in one shot, keeping data in registers for
        # both amax and scaling, reducing HBM traffic from 5 to 3 B/elem.
        # Round up to next power of 2 for the constexpr tile size.
        # Fused kernel config has MAX_GROUP_SIZE in [256, 512, 1024, 2048].
        # Use fused only when rounded-up group size matches a config.
        _FUSED_GROUP_SIZES = {c.kwargs["MAX_GROUP_SIZE"] for c in kernel_configs_fused}
        bsi = 1
        while bsi < max_group_size:
            bsi *= 2
        if bsi in _FUSED_GROUP_SIZES:
            # Filter configs to only the matching MAX_GROUP_SIZE
            valid_configs = [
                c for c in kernel_configs_fused if c.kwargs["MAX_GROUP_SIZE"] == bsi
            ]
            _triton_fp8_per_group_colwise_scales_fused_kernel.configs = valid_configs
            _triton_fp8_per_group_colwise_scales_fused_kernel[grid](
                hp_tensor,
                offsets,
                output_buffer,
                scales_buffer,
                k,
                n,
                hp_tensor.stride(0),
                output_buffer.stride(1),
                num_elements,
                fp8_dtype_min,
                fp8_dtype_max,
                tl_input_dtype,
                tl_output_dtype,
                round_scales_to_power_of_2,
                EPS=EPS,
                STRIDE_OUTPUT_ROW=1,
                STRIDE_INPUT_COL=hp_tensor.stride(1),
            )
        else:
            _triton_fp8_per_group_colwise_scales_kernel[grid](
                hp_tensor,
                offsets,
                output_buffer,
                scales_buffer,
                k,
                n,
                n_groups,
                hp_tensor.stride(0),
                output_buffer.stride(1),
                fp8_dtype_min,
                fp8_dtype_max,
                tl_input_dtype,
                tl_output_dtype,
                round_scales_to_power_of_2,
                EPS=EPS,
                STRIDE_OUTPUT_ROW=1,
                STRIDE_INPUT_COL=hp_tensor.stride(1),
            )
        return output_buffer, scales_buffer

    @triton_fp8_per_group_colwise_scales.register_fake
    def _fake_triton_fp8_per_group_colwise_scales(
        hp_tensor: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hp_tensor.ndim == 2, "input tensor must be 2D"
        k, n = hp_tensor.shape
        n_groups = offsets.numel()
        output_buffer = torch.empty_like(
            hp_tensor, dtype=output_dtype, device=hp_tensor.device
        ).as_strided(hp_tensor.size(), (1, k))

        scales_buffer = torch.empty(
            (n * n_groups), dtype=torch.float32, device=hp_tensor.device
        )
        return output_buffer, scales_buffer

    # This kernel is used on `input` which has shape (M, K),
    # before the calculation `grad_B = grad_output_t @ input`.
    # The tokens per expert will vary per iteration, so don't want
    # to recompile on `token` dim (M) changes.
    @triton.autotune(configs=kernel_configs_2D, key=["K", "N_GROUPS"])
    @triton.jit
    def _triton_fp8_per_group_colwise_scales_kernel(
        input_ptr,
        offsets_ptr,
        out_ptr,
        scales_ptr,
        K: tl.int64,
        N: tl.int64,
        N_GROUPS: tl.int64,
        stride_input_row: tl.int64,
        stride_output_col: tl.int64,
        fp8_dtype_min: tl.constexpr,
        fp8_dtype_max: tl.constexpr,
        input_dtype: tl.constexpr,
        output_dtype: tl.constexpr,
        round_scales_to_power_of_2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_ITER: tl.constexpr,
        EPS: tl.constexpr,
        STRIDE_OUTPUT_ROW: tl.constexpr,
        STRIDE_INPUT_COL: tl.constexpr,
    ):
        # parallel across columns and groups (offsets)
        block_col_id = tl.program_id(axis=0)
        offset_idx = tl.program_id(axis=1)

        # determine start and end row idx for this group
        group_row_start_idx = tl.load(
            offsets_ptr + offset_idx - 1, mask=offset_idx > 0, other=0
        )
        group_row_end_idx = tl.load(offsets_ptr + offset_idx)
        block_col_offs = (block_col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(
            tl.int64
        )

        # compute colwise amaxes for this group
        amax_buffer = tl.zeros((BLOCK_SIZE,), dtype=input_dtype)
        for row_start_idx in range(
            group_row_start_idx, group_row_end_idx, BLOCK_SIZE_ITER
        ):
            block_row_offs = (row_start_idx + tl.arange(0, BLOCK_SIZE_ITER)).to(
                tl.int64
            )
            block_offs = (
                block_row_offs[:, None] * stride_input_row
                + block_col_offs[None, :] * STRIDE_INPUT_COL
            )
            block_mask = (block_row_offs[:, None] < group_row_end_idx) & (
                block_col_offs[None, :] < N
            )
            data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
                input_dtype
            )
            # we need to cast back to input dtype since triton promotes bf16 to fp32:
            # https://github.com/triton-lang/triton/blob/981e987eed9053b952f81153bc0779c99d8c642e/python/triton/language/standard.py#L173
            amax_buffer = tl.maximum(amax_buffer, tl.max(tl.abs(data), axis=0)).to(
                input_dtype
            )

        # compute colwise scales for this group.
        amax_buffer = amax_buffer.to(tl.float64)
        scales = (fp8_dtype_max / tl.clamp(amax_buffer, min=EPS, max=float("inf"))).to(
            tl.float32
        )
        if round_scales_to_power_of_2:
            scales = tl.exp2(tl.floor(tl.log2(scales)))

        # store colwise scales for each group in contiguous memory:
        # [group0_col0, group_0_col1, ..., group2_col0, group2_col1]
        # note: input tensor is in col-major memory layout.
        scales_offs = block_col_offs + (N * offset_idx)
        scales_mask = tl.arange(0, BLOCK_SIZE) < N
        tl.store(scales_ptr + scales_offs, scales, mask=scales_mask)

        # perform float8 conversion for this group
        # transpose tile before writing so consecutive SIMD lanes write
        # consecutive rows (stride 1 in column-major output) for coalescing
        for row_start_idx in range(
            group_row_start_idx, group_row_end_idx, BLOCK_SIZE_ITER
        ):
            block_row_offs = (row_start_idx + tl.arange(0, BLOCK_SIZE_ITER)).to(
                tl.int64
            )
            block_offs = (
                block_row_offs[:, None] * stride_input_row
                + block_col_offs[None, :] * STRIDE_INPUT_COL
            )
            block_mask = (block_row_offs[:, None] < group_row_end_idx) & (
                block_col_offs[None, :] < N
            )
            data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
                input_dtype
            )
            scaled_data = data * scales[None, :]
            fp8_data = tl.clamp(scaled_data, min=fp8_dtype_min, max=fp8_dtype_max).to(
                output_dtype
            )
            out_offs = (
                block_row_offs[:, None] * STRIDE_OUTPUT_ROW
                + block_col_offs[None, :] * stride_output_col
            )
            tl.store(out_ptr + out_offs, fp8_data, mask=block_mask)

    # Fused single-pass kernel: loads all rows for a group in one shot,
    # keeping data in registers for both amax computation and fp8 scaling.
    # Reduces HBM traffic from 5 to 3 bytes/elem (eliminates second read).
    # Requires BLOCK_SIZE_ITER >= max group size (enforced by wrapper).
    @triton.autotune(configs=kernel_configs_fused, key=["K"])
    @triton.jit
    def _triton_fp8_per_group_colwise_scales_fused_kernel(
        input_ptr,
        offsets_ptr,
        out_ptr,
        scales_ptr,
        K: tl.int64,
        N: tl.int64,
        stride_input_row: tl.int64,
        stride_output_col: tl.int64,
        num_elements: tl.int64,
        fp8_dtype_min: tl.constexpr,
        fp8_dtype_max: tl.constexpr,
        input_dtype: tl.constexpr,
        output_dtype: tl.constexpr,
        round_scales_to_power_of_2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        EPS: tl.constexpr,
        MAX_GROUP_SIZE: tl.constexpr,
        STRIDE_OUTPUT_ROW: tl.constexpr,
        STRIDE_INPUT_COL: tl.constexpr,
    ):
        # parallel across columns and groups (offsets)
        block_col_id = tl.program_id(axis=0)
        offset_idx = tl.program_id(axis=1)

        # determine start and end row idx for this group
        group_row_start_idx = tl.load(
            offsets_ptr + offset_idx - 1, mask=offset_idx > 0, other=0
        )
        group_row_end_idx = tl.load(offsets_ptr + offset_idx)
        block_col_offs = block_col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Single pass: load ALL rows for this group at once.
        # MAX_GROUP_SIZE is a power-of-2 >= actual max group size,
        # so the mask handles any excess rows.
        # Keep offsets in int32 to reduce register pressure; the
        # multiplication with int64 strides handles promotion automatically.
        block_row_offs = group_row_start_idx + tl.arange(0, MAX_GROUP_SIZE)
        block_offs = (
            block_row_offs[:, None] * stride_input_row
            + block_col_offs[None, :] * STRIDE_INPUT_COL
        )
        block_mask = (block_row_offs[:, None] < group_row_end_idx) & (
            block_col_offs[None, :] < N
        )

        # Load once from HBM -- data stays in registers
        data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
            input_dtype
        )

        # Compute colwise amax from registers
        amax_buffer = tl.max(tl.abs(data), axis=0).to(input_dtype)

        # Compute scales
        amax_buffer = amax_buffer.to(tl.float64)
        scales = (fp8_dtype_max / tl.clamp(amax_buffer, min=EPS, max=float("inf"))).to(
            tl.float32
        )
        if round_scales_to_power_of_2:
            scales = tl.exp2(tl.floor(tl.log2(scales)))

        # Store scales
        scales_offs = block_col_offs + (N * offset_idx)
        scales_mask = tl.arange(0, BLOCK_SIZE) < N
        tl.store(scales_ptr + scales_offs, scales, mask=scales_mask)

        # Scale from registers (no second HBM read) and write fp8
        scaled_data = data * scales[None, :]
        fp8_data = tl.clamp(scaled_data, min=fp8_dtype_min, max=fp8_dtype_max).to(
            output_dtype
        )
        out_offs = (
            block_row_offs[:, None] * STRIDE_OUTPUT_ROW
            + block_col_offs[None, :] * stride_output_col
        )
        tl.store(out_ptr + out_offs, fp8_data, mask=block_mask)

    @triton.autotune(configs=kernel_configs_2D_dual, key=["K", "N_GROUPS"])
    @triton.jit
    def _triton_fp8_per_group_colwise_scales_dual_kernel(
        # Tensor 1 (e.g. padded_grad_output): shape (K, N1)
        input_ptr_1,
        out_ptr_1,
        scales_ptr_1,
        N1: tl.int64,
        stride_input_row_1: tl.int64,
        stride_output_col_1: tl.int64,
        # Tensor 2 (e.g. padded_A): shape (K, N2)
        input_ptr_2,
        out_ptr_2,
        scales_ptr_2,
        N2: tl.int64,
        stride_input_row_2: tl.int64,
        stride_output_col_2: tl.int64,
        # Shared: group offsets and dimensions
        offsets_ptr,
        K: tl.int64,
        N_GROUPS: tl.int64,
        fp8_dtype_min: tl.constexpr,
        fp8_dtype_max: tl.constexpr,
        input_dtype_1: tl.constexpr,
        input_dtype_2: tl.constexpr,
        output_dtype: tl.constexpr,
        round_scales_to_power_of_2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_ITER: tl.constexpr,
        EPS: tl.constexpr,
        STRIDE_INPUT_COL_1: tl.constexpr,
        STRIDE_OUTPUT_ROW_1: tl.constexpr,
        STRIDE_INPUT_COL_2: tl.constexpr,
        STRIDE_OUTPUT_ROW_2: tl.constexpr,
    ):
        block_col_id = tl.program_id(axis=0)
        offset_idx = tl.program_id(axis=1)

        # Load group row boundaries (shared between both tensors)
        group_row_start_idx = tl.load(
            offsets_ptr + offset_idx - 1, mask=offset_idx > 0, other=0
        )
        group_row_end_idx = tl.load(offsets_ptr + offset_idx)
        block_col_offs = (block_col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(
            tl.int64
        )

        # Each block may be responsible for tensor_1, tensor_2, or both,
        # depending on their respective column counts.
        n1_blocks = tl.cdiv(N1, BLOCK_SIZE)
        n2_blocks = tl.cdiv(N2, BLOCK_SIZE)
        process_1 = block_col_id < n1_blocks
        process_2 = block_col_id < n2_blocks

        # Pass 1: compute colwise amaxes — row loops for both tensors are merged
        # so each row is visited only once per tensor pair per block.
        amax_buffer_1 = tl.zeros((BLOCK_SIZE,), dtype=input_dtype_1)
        amax_buffer_2 = tl.zeros((BLOCK_SIZE,), dtype=input_dtype_2)
        for row_start_idx in range(
            group_row_start_idx, group_row_end_idx, BLOCK_SIZE_ITER
        ):
            block_row_offs = (row_start_idx + tl.arange(0, BLOCK_SIZE_ITER)).to(
                tl.int64
            )
            row_mask = block_row_offs < group_row_end_idx

            if process_1:
                block_offs_1 = (
                    block_row_offs[:, None] * stride_input_row_1
                    + block_col_offs[None, :] * STRIDE_INPUT_COL_1
                )
                mask_1 = row_mask[:, None] & (block_col_offs[None, :] < N1)
                data_1 = tl.load(input_ptr_1 + block_offs_1, mask=mask_1, other=0.0).to(
                    input_dtype_1
                )
                amax_buffer_1 = tl.maximum(
                    amax_buffer_1, tl.max(tl.abs(data_1), axis=0)
                ).to(input_dtype_1)

            if process_2:
                block_offs_2 = (
                    block_row_offs[:, None] * stride_input_row_2
                    + block_col_offs[None, :] * STRIDE_INPUT_COL_2
                )
                mask_2 = row_mask[:, None] & (block_col_offs[None, :] < N2)
                data_2 = tl.load(input_ptr_2 + block_offs_2, mask=mask_2, other=0.0).to(
                    input_dtype_2
                )
                amax_buffer_2 = tl.maximum(
                    amax_buffer_2, tl.max(tl.abs(data_2), axis=0)
                ).to(input_dtype_2)

        # Compute scales and store
        scales_1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        scales_2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        if process_1:
            amax_f64_1 = amax_buffer_1.to(tl.float64)
            scales_1 = (
                fp8_dtype_max / tl.clamp(amax_f64_1, min=EPS, max=float("inf"))
            ).to(tl.float32)
            if round_scales_to_power_of_2:
                scales_1 = tl.exp2(tl.floor(tl.log2(scales_1)))
            scales_offs_1 = block_col_offs + (N1 * offset_idx)
            tl.store(
                scales_ptr_1 + scales_offs_1,
                scales_1,
                mask=block_col_offs < N1,
            )

        if process_2:
            amax_f64_2 = amax_buffer_2.to(tl.float64)
            scales_2 = (
                fp8_dtype_max / tl.clamp(amax_f64_2, min=EPS, max=float("inf"))
            ).to(tl.float32)
            if round_scales_to_power_of_2:
                scales_2 = tl.exp2(tl.floor(tl.log2(scales_2)))
            scales_offs_2 = block_col_offs + (N2 * offset_idx)
            tl.store(
                scales_ptr_2 + scales_offs_2,
                scales_2,
                mask=block_col_offs < N2,
            )

        # Pass 2: FP8 cast — row loops for both tensors are merged
        for row_start_idx in range(
            group_row_start_idx, group_row_end_idx, BLOCK_SIZE_ITER
        ):
            block_row_offs = (row_start_idx + tl.arange(0, BLOCK_SIZE_ITER)).to(
                tl.int64
            )
            row_mask = block_row_offs < group_row_end_idx

            if process_1:
                block_offs_1 = (
                    block_row_offs[:, None] * stride_input_row_1
                    + block_col_offs[None, :] * STRIDE_INPUT_COL_1
                )
                mask_1 = row_mask[:, None] & (block_col_offs[None, :] < N1)
                data_1 = tl.load(input_ptr_1 + block_offs_1, mask=mask_1, other=0.0).to(
                    input_dtype_1
                )
                fp8_1 = tl.clamp(
                    data_1 * scales_1[None, :], min=fp8_dtype_min, max=fp8_dtype_max
                ).to(output_dtype)
                out_offs_1 = (
                    block_row_offs[:, None] * STRIDE_OUTPUT_ROW_1
                    + block_col_offs[None, :] * stride_output_col_1
                )
                tl.store(out_ptr_1 + out_offs_1, fp8_1, mask=mask_1)

            if process_2:
                block_offs_2 = (
                    block_row_offs[:, None] * stride_input_row_2
                    + block_col_offs[None, :] * STRIDE_INPUT_COL_2
                )
                mask_2 = row_mask[:, None] & (block_col_offs[None, :] < N2)
                data_2 = tl.load(input_ptr_2 + block_offs_2, mask=mask_2, other=0.0).to(
                    input_dtype_2
                )
                fp8_2 = tl.clamp(
                    data_2 * scales_2[None, :], min=fp8_dtype_min, max=fp8_dtype_max
                ).to(output_dtype)
                out_offs_2 = (
                    block_row_offs[:, None] * STRIDE_OUTPUT_ROW_2
                    + block_col_offs[None, :] * stride_output_col_2
                )
                tl.store(out_ptr_2 + out_offs_2, fp8_2, mask=mask_2)

    @torch.library.custom_op(
        "torchao::triton_fp8_per_group_colwise_scales_dual", mutates_args={}
    )
    def triton_fp8_per_group_colwise_scales_dual(
        hp_tensor_1: torch.Tensor,
        hp_tensor_2: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused version of triton_fp8_per_group_colwise_scales that quantizes two tensors
        in a single kernel launch. Both tensors must share the same row count (K) and
        group offsets. Row iteration loops are merged so each row is visited once per
        pass instead of twice, halving kernel launches and reducing per-row overhead
        when N1 == N2.

        Used in the MoE backward pass to quantize padded_grad_output and padded_A
        simultaneously for the wgrad GEMM: grad_B = grad_output_t @ A.

        Args:
            - hp_tensor_1: 2D high precision tensor, shape (K, N1)
            - hp_tensor_2: 2D high precision tensor, shape (K, N2); must share K with hp_tensor_1
            - offsets: end index for each group/subtensor along dim 0
            - output_dtype: desired float8 dtype
            - round_scales_to_power_of_2: round scales down to nearest power of 2
        Returns:
            - fp8 tensor for hp_tensor_1 (column-major)
            - colwise scales for hp_tensor_1
            - fp8 tensor for hp_tensor_2 (column-major)
            - colwise scales for hp_tensor_2
        """
        assert hp_tensor_1.ndim == 2, "hp_tensor_1 must be 2D"
        assert hp_tensor_2.ndim == 2, "hp_tensor_2 must be 2D"
        assert hp_tensor_1.shape[0] == hp_tensor_2.shape[0], (
            "hp_tensor_1 and hp_tensor_2 must have the same row count"
        )

        tl_input_dtype_1 = FP8_DTYPE_MAP[hp_tensor_1.dtype]
        tl_input_dtype_2 = FP8_DTYPE_MAP[hp_tensor_2.dtype]
        tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

        fp8_dtype_min = torch.finfo(output_dtype).min
        fp8_dtype_max = torch.finfo(output_dtype).max

        k, n1 = hp_tensor_1.shape
        _, n2 = hp_tensor_2.shape
        n_groups = offsets.numel()

        # Column-major output buffers
        output_buffer_1 = torch.empty_like(
            hp_tensor_1, dtype=output_dtype, device=hp_tensor_1.device
        ).as_strided(hp_tensor_1.size(), (1, k))
        output_buffer_2 = torch.empty_like(
            hp_tensor_2, dtype=output_dtype, device=hp_tensor_2.device
        ).as_strided(hp_tensor_2.size(), (1, k))

        scales_buffer_1 = torch.empty(
            (n1 * n_groups), dtype=torch.float32, device=hp_tensor_1.device
        )
        scales_buffer_2 = torch.empty(
            (n2 * n_groups), dtype=torch.float32, device=hp_tensor_2.device
        )

        # Grid covers the larger of the two tensors' column blocks.
        # Blocks beyond one tensor's range only process the other tensor.
        grid = lambda meta: (
            max(
                triton.cdiv(n1, meta["BLOCK_SIZE"]), triton.cdiv(n2, meta["BLOCK_SIZE"])
            ),
            n_groups,
        )
        _triton_fp8_per_group_colwise_scales_dual_kernel[grid](
            hp_tensor_1,
            output_buffer_1,
            scales_buffer_1,
            n1,
            hp_tensor_1.stride(0),
            output_buffer_1.stride(1),
            hp_tensor_2,
            output_buffer_2,
            scales_buffer_2,
            n2,
            hp_tensor_2.stride(0),
            output_buffer_2.stride(1),
            offsets,
            k,
            n_groups,
            fp8_dtype_min,
            fp8_dtype_max,
            tl_input_dtype_1,
            tl_input_dtype_2,
            tl_output_dtype,
            round_scales_to_power_of_2,
            EPS=EPS,
            STRIDE_INPUT_COL_1=hp_tensor_1.stride(1),
            STRIDE_OUTPUT_ROW_1=1,
            STRIDE_INPUT_COL_2=hp_tensor_2.stride(1),
            STRIDE_OUTPUT_ROW_2=1,
        )
        return output_buffer_1, scales_buffer_1, output_buffer_2, scales_buffer_2

    @triton_fp8_per_group_colwise_scales_dual.register_fake
    def _fake_triton_fp8_per_group_colwise_scales_dual(
        hp_tensor_1: torch.Tensor,
        hp_tensor_2: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert hp_tensor_1.ndim == 2 and hp_tensor_2.ndim == 2
        k, n1 = hp_tensor_1.shape
        _, n2 = hp_tensor_2.shape
        n_groups = offsets.numel()
        output_buffer_1 = torch.empty_like(hp_tensor_1, dtype=output_dtype).as_strided(
            hp_tensor_1.size(), (1, k)
        )
        output_buffer_2 = torch.empty_like(hp_tensor_2, dtype=output_dtype).as_strided(
            hp_tensor_2.size(), (1, k)
        )
        scales_buffer_1 = torch.empty(
            (n1 * n_groups), dtype=torch.float32, device=hp_tensor_1.device
        )
        scales_buffer_2 = torch.empty(
            (n2 * n_groups), dtype=torch.float32, device=hp_tensor_2.device
        )
        return output_buffer_1, scales_buffer_1, output_buffer_2, scales_buffer_2

else:

    def triton_fp8_per_group_rowwise_scales(
        hp_tensor: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_fp8_per_group_rowwise_scales requires torch 2.7.0+ and triton installed"
        )

    def triton_fp8_per_group_colwise_scales(
        hp_tensor: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_fp8_per_group_colwise_scales requires torch 2.7.0+ and triton installed"
        )

    def triton_fp8_per_group_colwise_scales_dual(
        hp_tensor_1: torch.Tensor,
        hp_tensor_2: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_fp8_per_group_colwise_scales_dual requires torch 2.7.0+ and triton installed"
        )
