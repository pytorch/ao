# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of blockwise fp8 GEMM operations using torch._scaled_mm native blockwise scaling.

This implementation leverages the native blockwise scaling support in torch._scaled_mm
available with CUDA 12.9+, providing optimal performance through direct CUTLASS kernel usage.

Based on PyTorch's native support for ScalingType.BlockWise128x128 and other blockwise modes,
this avoids the uncoalesced memory access issues present in custom Triton kernels.
"""

from typing import Tuple

import torch
import warnings



def _check_cuda_version_for_native_blockwise():
    """Check if CUDA version supports native blockwise scaling in torch._scaled_mm."""
    try:
        # Check if we're running with CUDA 12.9+
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return False
        
        major, minor = map(int, cuda_version.split(".")[:2])
        return major > 12 or (major == 12 and minor >= 9)
    except:
        return False


def _outer_dim_major(t: torch.Tensor) -> torch.Tensor:
    """Ensure a 2D scale tensor is outer-dim-major (stride(0) == 1).

    PyTorch's native blockwise scaled GEMM expects 1x128 scales to be
    outer-dim-major. The idiom `t.t().contiguous().t()` preserves shape
    while flipping strides to make the outer dimension contiguous.
    """
    if t.ndim != 2:
        return t
    # Already outer-dim-major if stride(0) == 1
    if t.stride(0) == 1:
        return t
    return t.t().contiguous().t()


def blockwise_fp8_scaled_mm_1x128_128x128(
    a: torch.Tensor,  # (M, K) in fp8
    a_s: torch.Tensor,  # (M, K // block_size) reciprocals of scales
    b: torch.Tensor,  # (K, N) in fp8, column-major
    b_s: torch.Tensor,  # (K // block_size, N // block_size) reciprocals of scales
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise fp8 GEMM using torch._scaled_mm with native blockwise scaling when available.

    This implementation attempts to use native blockwise scaling support in torch._scaled_mm
    with CUDA 12.9+. Falls back to block-by-block processing if native support is unavailable.

    Args:
        a: Input tensor (M, K) in fp8, row-major
        a_s: Input scales (M, K // block_size), reciprocals (will be inverted)
        b: Weight tensor (K, N) in fp8, column-major layout
        b_s: Weight scales (K // block_size, N // block_size), reciprocals (will be inverted)
        block_size: Block size for quantization (must be 128)

    Returns:
        Output tensor (M, N) in bfloat16
    """
    assert block_size == 128, "Only block_size=128 is supported"
    assert a.dtype == torch.float8_e4m3fn, f"Input a must be fp8_e4m3fn, got {a.dtype}"
    assert b.dtype == torch.float8_e4m3fn, f"Input b must be fp8_e4m3fn, got {b.dtype}"
    
    # Convert reciprocal scales back to regular scales for torch._scaled_mm
    scale_a = 1.0 / a_s
    scale_b = 1.0 / b_s

    # For 1x128 on LHS, scales must be outer-dim-major (see PyTorch test_matmul_cuda.py)
    scale_a = _outer_dim_major(scale_a)
    
    # Try native blockwise scaling first (requires CUDA 12.9+)
    if _check_cuda_version_for_native_blockwise():
        try:
            # Use native blockwise scaling with torch._scaled_mm
            # This should dispatch to the CUTLASS kernel with native blockwise support
            return torch._scaled_mm(
                a,  # (M, K) fp8, row-major
                b,  # (K, N) fp8, column-major - torch._scaled_mm should handle layout
                scale_a=scale_a,  # (M, K // 128) blockwise scales for input
                scale_b=scale_b,  # (K // 128, N // 128) blockwise scales for weight
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
        except Exception as e:
            warnings.warn(
                f"Native blockwise scaling failed: {e}. Falling back to block-by-block processing. "
                f"For optimal performance, ensure CUDA 12.9+ and compatible PyTorch version.",
                RuntimeWarning
            )
    
    # Fallback: block-by-block processing to emulate blockwise scaling
    # This preserves the blockwise precision but may be slower than native implementation
    return _blockwise_fp8_scaled_mm_fallback_1x128_128x128(a, scale_a, b, scale_b, block_size)


def _blockwise_fp8_scaled_mm_fallback_1x128_128x128(
    a: torch.Tensor,
    scale_a: torch.Tensor,
    b: torch.Tensor,
    scale_b: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Fallback implementation using block-by-block torch._scaled_mm calls.
    
    This emulates blockwise scaling by processing the computation in blocks,
    preserving the precision benefits while remaining compatible with older CUDA versions.
    """
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
        a_scale_block = scale_a[:, k_idx : k_idx + 1]  # (M, 1)
        
        for n_idx in range(n_blocks):
            n_start = n_idx * block_size
            n_end = n_start + block_size
            
            # Extract (K_block, N_block) from b
            b_block = b[k_start:k_end, n_start:n_end].contiguous()  # (block_size, block_size)
            b_scale_block = scale_b[k_idx : k_idx + 1, n_idx : n_idx + 1]  # (1, 1)
            
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


def blockwise_fp8_scaled_mm_1x128_128x1(
    a: torch.Tensor,  # (M, K) in fp8
    a_s: torch.Tensor,  # (M, K // block_size) reciprocals of scales
    b: torch.Tensor,  # (K, N) in fp8, column-major
    b_s: torch.Tensor,  # (K // block_size, N) reciprocals of scales
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise fp8 GEMM for backward pass using torch._scaled_mm with native scaling when available.

    This variant is used when B has (128 x 1) scaling granularity, corresponding
    to PyTorch's native ScalingType.BlockWise1x128 support.

    Args:
        a: Input tensor (M, K) in fp8, row-major
        a_s: Input scales (M, K // block_size), reciprocals (will be inverted)
        b: Weight tensor (K, N) in fp8, column-major layout
        b_s: Weight scales (K // block_size, N), reciprocals (will be inverted)
        block_size: Block size for quantization (must be 128)

    Returns:
        Output tensor (M, N) in bfloat16
    """
    assert block_size == 128, "Only block_size=128 is supported"
    assert a.dtype == torch.float8_e4m3fn, f"Input a must be fp8_e4m3fn, got {a.dtype}"
    assert b.dtype == torch.float8_e4m3fn, f"Input b must be fp8_e4m3fn, got {b.dtype}"
    
    # Convert reciprocal scales back to regular scales for torch._scaled_mm
    scale_a = 1.0 / a_s
    scale_b = 1.0 / b_s

    # For 1x128 on LHS and 128x1 on RHS, scales must be outer-dim-major
    # Ref: PyTorch test_matmul_cuda.py::test_scaled_mm_vs_emulated_block_wise
    scale_a = _outer_dim_major(scale_a)
    scale_b = _outer_dim_major(scale_b)
    
    # Try native blockwise scaling first (requires CUDA 12.9+)
    if _check_cuda_version_for_native_blockwise():
        try:
            # Use native blockwise scaling with torch._scaled_mm
            # This uses BlockWise1x128 scaling for the weight tensor
            return torch._scaled_mm(
                a,  # (M, K) fp8, row-major
                b,  # (K, N) fp8, column-major - torch._scaled_mm should handle layout
                scale_a=scale_a,  # (M, K // 128) blockwise scales for input
                scale_b=scale_b,  # (K // 128, N) blockwise scales for weight (128x1 scaling)
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
        except Exception as e:
            warnings.warn(
                f"Native blockwise scaling failed: {e}. Falling back to block-by-block processing. "
                f"For optimal performance, ensure CUDA 12.9+ and compatible PyTorch version.",
                RuntimeWarning
            )
    
    # Fallback: block-by-block processing to emulate blockwise scaling
    return _blockwise_fp8_scaled_mm_fallback_1x128_128x1(a, scale_a, b, scale_b, block_size)


def _blockwise_fp8_scaled_mm_fallback_1x128_128x1(
    a: torch.Tensor,
    scale_a: torch.Tensor,
    b: torch.Tensor,
    scale_b: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Fallback implementation for 128x1 scaling using block-by-block processing.
    """
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
        a_scale_block = scale_a[:, k_idx : k_idx + 1]  # (M, 1)
        
        b_block = b[k_start:k_end, :].contiguous()  # (block_size, N)
        b_scale_block = scale_b[k_idx : k_idx + 1, :]  # (1, N)
        
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
    
    Uses native torch._scaled_mm with blockwise scaling for optimal performance.
    """
    return blockwise_fp8_scaled_mm_1x128_128x128(a, a_s, b, b_s, block_size)


def blockwise_fp8_gemm_scaled_mm_1x128_128x1(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Wrapper function that matches the Triton kernel interface.
    
    Uses native torch._scaled_mm with blockwise scaling for optimal performance.
    """
    return blockwise_fp8_scaled_mm_1x128_128x1(a, a_s, b, b_s, block_size)
