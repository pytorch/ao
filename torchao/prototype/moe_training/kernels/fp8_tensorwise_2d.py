# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused Triton kernels for tensorwise FP8 quantization of 2D tensors.

Two-pass approach:
  - Pass 1 (_fp8_tensorwise_2d_amax_kernel): one sequential read of the input
    tensor, computing fused nan_to_num + abs + global max via tl.atomic_max.
  - Pass 2 (_fp8_tensorwise_2d_quantize_kernel): one sequential read of the
    input tensor, computing scale inline from the amax result, then writing the
    FP8 output with fused nan_to_num + scale + clamp + cast.

Compared to the PyTorch fallback (~15 ATen kernel launches per call):
    nan_to_num  abs  reduce  bf16→f32  clamp  reciprocal  mul
    log2  floor  exp2  bf16_copy  multiply  clamp  fp8_copy  expand

This kernel collapses them to 3 GPU launches (amax + quantize + scale broadcast),
cutting ROCm HIP dispatch overhead by ~12 launches × ~60 µs each ≈ 720 µs/call.
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

    # ROCm MI300X benefits from large blocks (high memory bandwidth, many CUs).
    # CUDA uses smaller blocks with more pipeline stages.
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

    @triton.autotune(configs=_configs, key=["numel"])
    @triton.jit
    def _fp8_tensorwise_2d_amax_kernel(
        x_ptr,
        amax_ptr,               # *float32 scalar, pre-zeroed by caller
        numel: int,
        INPUT_DTYPE_MAX: tl.constexpr,  # max finite value of the input dtype
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Pass 1: fused nan_to_num + abs + global amax.

        Each block reduces BLOCK_SIZE elements to a local max, then accumulates
        into the global amax via tl.atomic_max (hardware float atomics on CDNA3).
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # nan_to_num: NaN → 0, ±inf → ±INPUT_DTYPE_MAX (matches torch.nan_to_num)
        x = tl.where(x != x, 0.0, x)                          # NaN → 0
        x = tl.where(x > INPUT_DTYPE_MAX, INPUT_DTYPE_MAX, x)  # +inf → max
        x = tl.where(x < -INPUT_DTYPE_MAX, -INPUT_DTYPE_MAX, x)  # -inf → -max

        local_max = tl.max(tl.abs(x))

        # AMD GPUs: relaxed semantics give better performance for reduction atomics.
        if tl.constexpr(torch.version.hip is not None):
            tl.atomic_max(amax_ptr, local_max, sem="relaxed")
        else:
            tl.atomic_max(amax_ptr, local_max)

    @triton.autotune(configs=_configs, key=["numel"])
    @triton.jit
    def _fp8_tensorwise_2d_quantize_kernel(
        x_ptr,
        out_ptr,                # *FP8 output, same shape as x
        amax_ptr,               # *float32 scalar amax written by pass 1
        scale_out_ptr,          # *float32 scalar scale output (written by block 0)
        numel: int,
        fp8_max: tl.constexpr,
        fp8_min: tl.constexpr,
        INPUT_DTYPE_MAX: tl.constexpr,
        EPS: tl.constexpr,
        ROUND_POW2: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Pass 2: compute scale from precomputed amax, then apply
        fused nan_to_num + scale + clamp + FP8 cast.

        The scale is computed inline in registers (fp8_max / max(amax, EPS) with
        optional power-of-2 rounding), avoiding the ~6 separate scalar ATen kernels
        that amax_to_scale + _round_scale_down_to_power_of_2 would normally launch.
        """
        pid = tl.program_id(0)

        # Load global amax (scalar, hits L1 on all but the first block).
        amax = tl.load(amax_ptr).to(tl.float32)
        scale = fp8_max / tl.maximum(amax, EPS)
        if ROUND_POW2:
            scale = tl.exp2(tl.floor(tl.log2(scale)))

        # Block 0 writes the scale so the caller can build scale_expanded.
        if pid == 0:
            tl.store(scale_out_ptr, scale)

        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # nan_to_num: must match pass 1 so scaled values are consistent.
        x = tl.where(x != x, 0.0, x)
        x = tl.where(x > INPUT_DTYPE_MAX, INPUT_DTYPE_MAX, x)
        x = tl.where(x < -INPUT_DTYPE_MAX, -INPUT_DTYPE_MAX, x)

        # Saturate to FP8 range then cast (mirrors to_fp8_saturated).
        x_clamped = tl.clamp(x * scale, fp8_min, fp8_max)
        tl.store(out_ptr + offs, x_clamped.to(OUTPUT_DTYPE), mask=mask)

    def triton_fp8_tensorwise_quantize_2d(
        tensor: torch.Tensor,
        output_dtype: torch.dtype,
        round_scales_to_power_of_2: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused two-pass FP8 tensorwise quantization of a 2D contiguous tensor.

        GPU launches: amax kernel + quantize kernel + scale broadcast = 3 total.
        The fallback PyTorch path requires ~15 separate ATen kernel launches.

        Args:
            tensor: (M, K) contiguous BF16 or F32 input tensor.
            output_dtype: target FP8 dtype.
            round_scales_to_power_of_2: if True, round scale down to nearest
                power of 2 (compiled into kernel as tl.constexpr).

        Returns:
            fp8_data: (M, K) FP8 tensor, row-major layout.
            scale_expanded: (M,) float32 tensor, all values equal to the single
                tensorwise scale (format required by _scaled_grouped_mm).
        """
        assert tensor.ndim == 2 and tensor.is_contiguous(), (
            "triton_fp8_tensorwise_quantize_2d requires a 2D contiguous input tensor"
        )
        M, K = tensor.shape
        numel = M * K

        tl_output_dtype = FP8_DTYPE_MAP[output_dtype]
        fp8_max = torch.finfo(output_dtype).max
        fp8_min = torch.finfo(output_dtype).min
        input_dtype_max = torch.finfo(tensor.dtype).max

        fp8_out = torch.empty(M, K, dtype=output_dtype, device=tensor.device)
        # amax_buf: pre-zeroed so atomic_max accumulates from 0 (identity for max-abs).
        amax_buf = torch.zeros(1, dtype=torch.float32, device=tensor.device)
        # scale_buf: written by block 0 of the quantize kernel.
        scale_buf = torch.empty(1, dtype=torch.float32, device=tensor.device)

        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

        # Pass 1: reduce to global amax.
        _fp8_tensorwise_2d_amax_kernel[grid](
            tensor,
            amax_buf,
            numel,
            INPUT_DTYPE_MAX=input_dtype_max,
        )

        # Pass 2: compute scale inline, write FP8 output and scalar scale.
        _fp8_tensorwise_2d_quantize_kernel[grid](
            tensor,
            fp8_out,
            amax_buf,
            scale_buf,
            numel,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            INPUT_DTYPE_MAX=input_dtype_max,
            EPS=EPS,
            ROUND_POW2=round_scales_to_power_of_2,
            OUTPUT_DTYPE=tl_output_dtype,
        )

        # Broadcast the scalar scale to (M,) for _scaled_grouped_mm.
        # scale_buf is (1,); expand + contiguous is a single ~384 KB fill kernel.
        scale_expanded = scale_buf.expand(M).contiguous()

        return fp8_out, scale_expanded

else:
    triton_fp8_tensorwise_quantize_2d = None
