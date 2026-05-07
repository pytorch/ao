# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Low-level Triton kernel for FP8 quantization of 3D column-major tensors.

The active TensorWise grouped GEMM path computes a single global amax in the
caller, broadcasts it across experts, and uses the dual-layout kernel here to
materialize both RHS layouts needed by forward and backward grouped GEMMs.
"""

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

    if torch.version.hip is not None:
        _tensorwise_3d_configs = [
            triton.Config(
                {"BLOCK_SIZE_K": bk, "BLOCK_SIZE_N": bn},
                num_warps=warps,
                num_stages=2,
            )
            for bk in [128, 256]
            for bn in [64, 128]
            for warps in [4, 8]
        ]
    else:
        _tensorwise_3d_configs = [
            triton.Config(
                {"BLOCK_SIZE_K": bk, "BLOCK_SIZE_N": bn},
                num_warps=warps,
                num_stages=4,
            )
            for bk in [128, 256]
            for bn in [64, 128]
            for warps in [4, 8]
        ]

    # Quantize once and write both grouped-GEMM RHS layouts.
    @triton.autotune(configs=_tensorwise_3d_configs, key=["K", "N"])
    @triton.jit
    def _fp8_tensorwise_3d_dual_layout_quantize_kernel(
        input_ptr,
        stride_input_e: tl.int64,
        stride_input_k,
        stride_input_n,
        output_fwd_ptr,
        stride_output_fwd_e: tl.int64,
        stride_output_fwd_k,
        stride_output_fwd_n,
        output_rhs_ptr,
        stride_output_rhs_e: tl.int64,
        stride_output_rhs_n,
        stride_output_rhs_k,
        expert_amax_ptr,  # (E,) float32
        fwd_inv_scales_ptr,  # (E, N) float32 inverse scale output
        rhs_inv_scales_ptr,  # (E, K) float32 inverse scale output
        E: int,
        K: int,
        N: int,
        fp8_dtype_min: tl.constexpr,
        fp8_dtype_max: tl.constexpr,
        output_dtype: tl.constexpr,
        ROUND_POW2: tl.constexpr,
        INPUT_DTYPE_MAX: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        EPS: tl.constexpr,
    ):
        expert_idx = tl.program_id(0)
        n_block_idx = tl.program_id(1)

        n_offs = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        amax = tl.load(expert_amax_ptr + expert_idx).to(tl.float32)
        scale = fp8_dtype_max / tl.maximum(amax, EPS)
        if ROUND_POW2:
            scale = tl.exp2(tl.floor(tl.log2(scale)))
        inv_scale = 1.0 / scale

        fwd_scale_offs = expert_idx * N + n_offs
        tl.store(fwd_inv_scales_ptr + fwd_scale_offs, inv_scale, mask=n_mask)

        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offs = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K

            if n_block_idx == 0:
                rhs_scale_offs = expert_idx * K + k_offs
                tl.store(rhs_inv_scales_ptr + rhs_scale_offs, inv_scale, mask=k_mask)

            input_offs = (
                expert_idx * stride_input_e
                + k_offs[:, None] * stride_input_k
                + n_offs[None, :] * stride_input_n
            )
            mask = k_mask[:, None] & n_mask[None, :]
            vals = tl.load(input_ptr + input_offs, mask=mask, other=0.0).to(
                tl.float32
            )

            vals = tl.where(vals != vals, 0.0, vals)
            vals = tl.where(vals > INPUT_DTYPE_MAX, INPUT_DTYPE_MAX, vals)
            vals = tl.where(vals < -INPUT_DTYPE_MAX, -INPUT_DTYPE_MAX, vals)

            scaled_vals = vals * scale
            clamped_vals = tl.minimum(
                tl.maximum(scaled_vals, fp8_dtype_min), fp8_dtype_max
            ).to(output_dtype)

            fwd_offs = (
                expert_idx * stride_output_fwd_e
                + k_offs[:, None] * stride_output_fwd_k
                + n_offs[None, :] * stride_output_fwd_n
            )
            tl.store(output_fwd_ptr + fwd_offs, clamped_vals, mask=mask)

            rhs_offs = (
                expert_idx * stride_output_rhs_e
                + n_offs[None, :] * stride_output_rhs_n
                + k_offs[:, None] * stride_output_rhs_k
            )
            tl.store(output_rhs_ptr + rhs_offs, clamped_vals, mask=mask)

else:
    pass
