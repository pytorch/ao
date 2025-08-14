# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of blockwise fp8 GEMM operations using torch._scaled_mm
as an alternative to custom Triton kernels.

This implementation uses block-by-block processing with torch._scaled_mm to maintain
blockwise scaling precision, providing accurate results comparable to the Triton kernels.
While torch._scaled_mm doesn't natively support arbitrary blockwise scaling, the 
block-by-block approach preserves the precision benefits of blockwise quantization.
"""

from typing import Tuple

import torch



def blockwise_fp8_scaled_mm_1x128_128x128(
    a: torch.Tensor,  # (M, K) in fp8
    a_s: torch.Tensor,  # (M, K // block_size) reciprocals of scales
    b: torch.Tensor,  # (K, N) in fp8, column-major
    b_s: torch.Tensor,  # (K // block_size, N // block_size) reciprocals of scales
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise fp8 GEMM using torch._scaled_mm instead of Triton kernel.

    This implementation uses the advanced block-by-block approach to better 
    preserve blockwise scaling precision compared to simple row/column expansion.

    Args:
        a: Input tensor (M, K) in fp8, row-major
        a_s: Input scales (M, K // block_size), reciprocals
        b: Weight tensor (K, N) in fp8, column-major layout
        b_s: Weight scales (K // block_size, N // block_size), reciprocals
        block_size: Block size for quantization (must be 128)

    Returns:
        Output tensor (M, N) in bfloat16
    """
    assert block_size == 128, "Only block_size=128 is supported"
    assert a.is_contiguous(), "Input tensor a must be contiguous (row-major)"
    assert not b.is_contiguous(), "Weight tensor b must be column-major"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scales must be contiguous"

    # Use the advanced implementation by default for better accuracy
    return blockwise_fp8_scaled_mm_advanced_1x128_128x128(a, a_s, b, b_s, block_size)


def blockwise_fp8_scaled_mm_1x128_128x1(
    a: torch.Tensor,  # (M, K) in fp8
    a_s: torch.Tensor,  # (M, K // block_size) reciprocals of scales
    b: torch.Tensor,  # (K, N) in fp8, column-major
    b_s: torch.Tensor,  # (K // block_size, N) reciprocals of scales
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise fp8 GEMM for backward pass using torch._scaled_mm.

    This variant is used when B has (128 x 1) scaling granularity.
    Uses block-by-block processing to preserve blockwise precision.

    Args:
        a: Input tensor (M, K) in fp8, row-major
        a_s: Input scales (M, K // block_size), reciprocals
        b: Weight tensor (K, N) in fp8, column-major layout
        b_s: Weight scales (K // block_size, N), reciprocals
        block_size: Block size for quantization (must be 128)

    Returns:
        Output tensor (M, N) in bfloat16
    """
    assert block_size == 128, "Only block_size=128 is supported"
    assert a.is_contiguous(), "Input tensor a must be contiguous (row-major)"
    assert not b.is_contiguous(), "Weight tensor b must be column-major"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scales must be contiguous"

    M, K = a.size()
    N = b.size(1)
    k_blocks = K // block_size

    # Initialize output
    output = torch.zeros(M, N, dtype=torch.bfloat16, device=a.device)

    # Process each K-block separately to preserve blockwise scaling
    for k_idx in range(k_blocks):
        k_start = k_idx * block_size
        k_end = k_start + block_size

        # Extract K-block from inputs
        a_block = a[:, k_start:k_end].contiguous()  # (M, block_size)
        a_scale_block = a_s[:, k_idx : k_idx + 1]  # (M, 1)
        
        b_block = b[k_start:k_end, :].contiguous()  # (block_size, N)
        b_scale_block = b_s[k_idx : k_idx + 1, :]  # (1, N)

        # Compute this block's contribution using torch._scaled_mm
        block_output = torch._scaled_mm(
            a_block,  # (M, block_size)
            b_block,  # (block_size, N)
            scale_a=a_scale_block,  # (M, 1)
            scale_b=b_scale_block,  # (1, N)
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        # Accumulate into output
        output += block_output

    return output


def blockwise_fp8_scaled_mm_advanced_1x128_128x128(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Advanced blockwise fp8 GEMM that preserves blockwise scaling precision.

    This implementation processes the computation block-by-block to maintain
    the full precision of blockwise scaling, providing the most accurate
    approximation to the Triton kernel using torch._scaled_mm.

    Args:
        a: Input tensor (M, K) in fp8, row-major
        a_s: Input scales (M, K // block_size), reciprocals
        b: Weight tensor (K, N) in fp8, column-major layout
        b_s: Weight scales (K // block_size, N // block_size), reciprocals
        block_size: Block size for quantization (must be 128)

    Returns:
        Output tensor (M, N) in bfloat16
    """
    assert block_size == 128, "Only block_size=128 is supported"
    assert a.is_contiguous(), "Input tensor a must be contiguous (row-major)"
    assert not b.is_contiguous(), "Weight tensor b must be column-major"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scales must be contiguous"

    M, K = a.size()
    N = b.size(1)

    k_blocks = K // block_size
    n_blocks = N // block_size

    # Initialize output
    output = torch.zeros(M, N, dtype=torch.bfloat16, device=a.device)

    # Process each (K_block, N_block) tile separately to preserve blockwise scaling
    for k_idx in range(k_blocks):
        k_start = k_idx * block_size
        k_end = k_start + block_size

        # Extract K-block from inputs
        a_block = a[:, k_start:k_end].contiguous()  # (M, block_size)
        a_scale_block = a_s[:, k_idx : k_idx + 1]  # (M, 1)

        for n_idx in range(n_blocks):
            n_start = n_idx * block_size
            n_end = n_start + block_size

            # Extract (K_block, N_block) from b
            b_block = b[
                k_start:k_end, n_start:n_end
            ].contiguous()  # (block_size, block_size)
            b_scale_block = b_s[
                k_idx : k_idx + 1, n_idx : n_idx + 1
            ]  # (1, 1) -> scalar

            # Compute this block's contribution using torch._scaled_mm
            block_output = torch._scaled_mm(
                a_block,  # (M, block_size)
                b_block,  # (block_size, block_size)
                scale_a=a_scale_block,  # (M, 1)
                scale_b=b_scale_block,  # (1, 1)
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )

            # Accumulate into output
            output[:, n_start:n_end] += block_output

    return output


# Convenience wrapper functions to match the Triton kernel interface
def blockwise_fp8_gemm_scaled_mm_1x128_128x128(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Wrapper function that matches the Triton kernel interface.
    
    Uses the advanced block-by-block implementation for maximum accuracy.
    """
    return blockwise_fp8_scaled_mm_1x128_128x128(a, a_s, b, b_s, block_size)


def blockwise_fp8_gemm_scaled_mm_1x128_128x1(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Wrapper function that matches the Triton kernel interface."""
    return blockwise_fp8_scaled_mm_1x128_128x1(a, a_s, b, b_s, block_size)
