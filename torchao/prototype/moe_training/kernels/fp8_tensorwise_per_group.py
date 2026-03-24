# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused Triton kernels for per-group tensorwise FP8 quantization of 2D tensors.

Two-pass approach with flat 1D grid (same design as fp8_tensorwise_2d.py):
  - Pass 1 (_fp8_per_group_amax_kernel): flat grid over the entire M×K tensor.
    Each block processes BLOCK_SIZE contiguous elements, fusing nan_to_num +
    abs + local max, then atomic_max into the owning group's amax slot.
  - Pass 2 (_fp8_per_group_quantize_kernel): same flat grid.  Each block
    loads its group's precomputed scale and writes FP8 output with fused
    nan_to_num + scale + clamp + cast.

The group for each block is determined by a 16-iteration in-kernel scan of the
(small) offsets array — no Python-side precomputation needed.
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
        torch.float8_e4m3fn: tl.float8e4nv,
        torch.float8_e4m3fnuz: tl.float8e4b8,
        torch.float8_e5m2: tl.float8e5,
        torch.float8_e5m2fnuz: tl.float8e5b16,
    }

    # Flat 1D configs matching the fp8_tensorwise_2d.py design.
    if torch.version.hip is not None:
        _configs = [
            triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
        ]
    else:
        _configs = [
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        ]

    # =========================================================================
    # Pass 1: flat per-group amax with fused nan_to_num
    # =========================================================================
    @triton.autotune(configs=_configs, key=["numel"])
    @triton.jit
    def _fp8_per_group_amax_kernel(
        input_ptr,
        group_amax_ptr,  # (N_GROUPS,) float32, pre-zeroed
        offsets_ptr,     # (N_GROUPS,) int32 group end offsets
        numel: int,
        K: tl.int64,
        N_GROUPS: tl.int64,
        INPUT_DTYPE_MAX: tl.constexpr,
        MAX_GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Each block reduces BLOCK_SIZE contiguous elements to a local max,
        then atomic_max into the owning group's amax slot.
        Group determined by scanning the small offsets array (≤16 entries).
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Fused nan_to_num
        x = tl.where(x != x, 0.0, x)
        x = tl.where(x > INPUT_DTYPE_MAX, INPUT_DTYPE_MAX, x)
        x = tl.where(x < -INPUT_DTYPE_MAX, -INPUT_DTYPE_MAX, x)

        local_max = tl.max(tl.abs(x))

        # Determine group from first element's row index.
        # Scan offsets — at most MAX_GROUPS (16) scalar loads from L1 cache.
        first_row = (pid.to(tl.int64) * BLOCK_SIZE) // K
        group_id = tl.zeros([], dtype=tl.int32)
        for g in tl.static_range(MAX_GROUPS):
            if g < N_GROUPS:
                off_g = tl.load(offsets_ptr + g).to(tl.int64)
                if first_row >= off_g:
                    group_id = tl.full([], g + 1, dtype=tl.int32)
        group_id = tl.minimum(
            group_id, tl.full([], N_GROUPS - 1, dtype=tl.int32)
        )

        if tl.constexpr(torch.version.hip is not None):
            tl.atomic_max(group_amax_ptr + group_id, local_max, sem="relaxed")
        else:
            tl.atomic_max(group_amax_ptr + group_id, local_max)

    # =========================================================================
    # Pass 2: flat per-group quantize with fused nan_to_num
    # =========================================================================
    @triton.autotune(configs=_configs, key=["numel"])
    @triton.jit
    def _fp8_per_group_quantize_kernel(
        input_ptr,
        out_ptr,
        scales_ptr,    # (N_GROUPS,) float32 scales
        offsets_ptr,   # (N_GROUPS,) int32 group end offsets
        numel: int,
        K: tl.int64,
        N_GROUPS: tl.int64,
        fp8_max: tl.constexpr,
        fp8_min: tl.constexpr,
        INPUT_DTYPE_MAX: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr,
        MAX_GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Each block loads its group's scale and applies fused nan_to_num +
        scale + clamp + FP8 cast to BLOCK_SIZE contiguous elements.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Fused nan_to_num (must match pass 1).
        x = tl.where(x != x, 0.0, x)
        x = tl.where(x > INPUT_DTYPE_MAX, INPUT_DTYPE_MAX, x)
        x = tl.where(x < -INPUT_DTYPE_MAX, -INPUT_DTYPE_MAX, x)

        # Group lookup — same scan as pass 1.
        first_row = (pid.to(tl.int64) * BLOCK_SIZE) // K
        group_id = tl.zeros([], dtype=tl.int32)
        for g in tl.static_range(MAX_GROUPS):
            if g < N_GROUPS:
                off_g = tl.load(offsets_ptr + g).to(tl.int64)
                if first_row >= off_g:
                    group_id = tl.full([], g + 1, dtype=tl.int32)
        group_id = tl.minimum(
            group_id, tl.full([], N_GROUPS - 1, dtype=tl.int32)
        )

        scale = tl.load(scales_ptr + group_id)

        scaled = x * scale
        # Use minimum/maximum instead of tl.clamp — tl.clamp produces
        # incorrect FP8 values on AMD ROCm.
        clamped = tl.minimum(tl.maximum(scaled, fp8_min), fp8_max)
        tl.store(out_ptr + offs, clamped.to(OUTPUT_DTYPE), mask=mask)

    # =========================================================================
    # Python wrapper
    # =========================================================================
    def triton_fp8_tensorwise_per_group_quantize(
        tensor: torch.Tensor,
        offs: torch.Tensor,
        output_dtype: torch.dtype,
        output_scale_dim: int,
        round_scales_to_power_of_2: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused two-pass per-group tensorwise FP8 quantization.

        GPU launches: amax kernel + quantize kernel + small PyTorch ops
        on (n_groups,) vectors = ~5 total (vs ~17 in the PyTorch fallback).

        Args:
            tensor: (M, K) contiguous BF16/F32 input.
            offs: (num_groups,) int32 end-offsets for each group along dim 0.
            output_dtype: target FP8 dtype.
            output_scale_dim: number of scale slots per group in output.
            round_scales_to_power_of_2: round scales down to nearest power of 2.

        Returns:
            fp8_data: (M, K) FP8 tensor, row-major.
            scales_flat: (output_scale_dim * num_groups,) float32 scale tensor.
        """
        assert tensor.ndim == 2 and tensor.is_contiguous(), (
            "triton_fp8_tensorwise_per_group_quantize requires 2D contiguous input"
        )
        M, K = tensor.shape
        numel = M * K
        n_groups = offs.numel()
        assert n_groups <= 16, (
            f"n_groups={n_groups} exceeds kernel MAX_GROUPS=16"
        )

        tl_output_dtype = FP8_DTYPE_MAP[output_dtype]
        fp8_max = torch.finfo(output_dtype).max
        fp8_min = torch.finfo(output_dtype).min
        input_dtype_max = torch.finfo(tensor.dtype).max

        fp8_out = torch.empty(M, K, dtype=output_dtype, device=tensor.device)

        # --- Pass 1: per-group amax ---
        group_amax = torch.zeros(
            n_groups, dtype=torch.float32, device=tensor.device,
        )

        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        _fp8_per_group_amax_kernel[grid](
            tensor,
            group_amax,
            offs,
            numel,
            K,
            n_groups,
            INPUT_DTYPE_MAX=input_dtype_max,
            MAX_GROUPS=16,
        )

        # --- Scale computation on tiny (n_groups,) tensor ---
        scales = (
            fp8_max / torch.clamp(group_amax.to(torch.float64), min=EPS)
        ).to(torch.float32)
        if round_scales_to_power_of_2:
            scales = torch.exp2(torch.floor(torch.log2(scales)))

        # --- Pass 2: quantize ---
        _fp8_per_group_quantize_kernel[grid](
            tensor,
            fp8_out,
            scales,
            offs,
            numel,
            K,
            n_groups,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            INPUT_DTYPE_MAX=input_dtype_max,
            OUTPUT_DTYPE=tl_output_dtype,
            MAX_GROUPS=16,
        )

        # Expand per-group scales to flat format for _scaled_grouped_mm.
        scales_flat = (
            scales
            .unsqueeze(1)
            .expand(-1, output_scale_dim)
            .contiguous()
            .view(-1)
        )

        return fp8_out, scales_flat

else:
    triton_fp8_tensorwise_per_group_quantize = None
