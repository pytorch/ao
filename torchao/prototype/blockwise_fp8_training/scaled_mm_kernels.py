# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of blockwise fp8 GEMM operations using torch._scaled_mm
as an alternative to custom Triton kernels.
"""

from typing import Tuple
import torch


def _prepare_blockwise_scales_for_scaled_mm(
    a_scales: torch.Tensor, 
    b_scales: torch.Tensor,
    a_shape: Tuple[int, int],
    b_shape: Tuple[int, int],
    block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare blockwise scales for torch._scaled_mm.
    
    torch._scaled_mm supports:
    - Tensor-wise scaling (scalar scales)
    - Row-wise scaling for A (scale shape: [M, 1])  
    - Column-wise scaling for B (scale shape: [1, N])
    
    For blockwise scaling, we need to broadcast/reshape the scales appropriately.
    """
    M, K = a_shape
    K_b, N = b_shape
    assert K == K_b, f"Inner dimensions must match: {K} != {K_b}"
    
    # Convert blockwise scales to row/column-wise for torch._scaled_mm
    
    # A scales: (M, K // block_size) -> (M, 1) by averaging across K blocks
    # This is a simplification - ideally we'd want row-wise scaling per block
    a_scales_rowwise = a_scales.mean(dim=1, keepdim=True)
    
    # B scales: (K // block_size, N // block_size) -> (1, N) by averaging across K blocks  
    # This is also a simplification
    b_scales_colwise = b_scales.mean(dim=0, keepdim=True)
    if b_scales_colwise.shape[1] != N // block_size:
        # Need to expand to full N dimension
        b_scales_expanded = b_scales_colwise.repeat(1, block_size).view(1, -1)[:, :N]
    else:
        b_scales_expanded = b_scales_colwise.repeat(1, block_size).view(1, -1)[:, :N]
    
    return a_scales_rowwise, b_scales_expanded


def blockwise_fp8_scaled_mm_1x128_128x128(
    a: torch.Tensor,      # (M, K) in fp8
    a_s: torch.Tensor,    # (M, K // block_size) reciprocals of scales  
    b: torch.Tensor,      # (K, N) in fp8, column-major
    b_s: torch.Tensor,    # (K // block_size, N // block_size) reciprocals of scales
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise fp8 GEMM using torch._scaled_mm instead of Triton kernel.
    
    This is a simplified implementation that approximates blockwise scaling
    using row-wise and column-wise scaling supported by torch._scaled_mm.
    
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
    
    # Prepare scales for torch._scaled_mm
    a_scales_rowwise, b_scales_colwise = _prepare_blockwise_scales_for_scaled_mm(
        a_s, b_s, (M, K), (K, N), block_size
    )
    
    # torch._scaled_mm expects b to be (K, N) and contiguous for column-major
    # Our b is already in the right shape but not contiguous due to column-major layout
    b_for_mm = b.contiguous()
    
    # Use torch._scaled_mm with row-wise scaling for a and column-wise for b
    output = torch._scaled_mm(
        a,                    # (M, K) 
        b_for_mm,            # (K, N)
        scale_a=a_scales_rowwise,  # (M, 1)
        scale_b=b_scales_colwise,  # (1, N) 
        out_dtype=torch.bfloat16,
        use_fast_accum=True,  # Enable fast accumulation for better performance
    )
    
    return output


def blockwise_fp8_scaled_mm_1x128_128x1(
    a: torch.Tensor,      # (M, K) in fp8
    a_s: torch.Tensor,    # (M, K // block_size) reciprocals of scales
    b: torch.Tensor,      # (K, N) in fp8, column-major  
    b_s: torch.Tensor,    # (K // block_size, N) reciprocals of scales
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise fp8 GEMM for backward pass using torch._scaled_mm.
    
    This variant is used when B has (128 x 1) scaling granularity.
    
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
    
    # For this scaling pattern, we need to handle (K//block_size, N) scales for B
    # Convert to column-wise scaling by averaging across K dimension
    a_scales_rowwise = a_s.mean(dim=1, keepdim=True)  # (M, 1)
    b_scales_colwise = b_s.mean(dim=0, keepdim=True)  # (1, N)
    
    # torch._scaled_mm expects b to be contiguous
    b_for_mm = b.contiguous() 
    
    # Use torch._scaled_mm
    output = torch._scaled_mm(
        a,                    # (M, K)
        b_for_mm,            # (K, N) 
        scale_a=a_scales_rowwise,  # (M, 1)
        scale_b=b_scales_colwise,  # (1, N)
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    return output


def blockwise_fp8_scaled_mm_advanced_1x128_128x128(
    a: torch.Tensor,
    a_s: torch.Tensor, 
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Advanced blockwise fp8 GEMM that preserves more of the blockwise scaling precision.
    
    Since torch._scaled_mm doesn't natively support arbitrary blockwise scaling,
    this implementation breaks down the computation into multiple _scaled_mm calls
    and combines the results to better approximate true blockwise scaling.
    """
    assert block_size == 128, "Only block_size=128 is supported"
    
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
        a_scale_block = a_s[:, k_idx:k_idx+1]       # (M, 1) 
        
        for n_idx in range(n_blocks):
            n_start = n_idx * block_size  
            n_end = n_start + block_size
            
            # Extract (K_block, N_block) from b
            b_block = b[k_start:k_end, n_start:n_end].contiguous()  # (block_size, block_size)
            b_scale_block = b_s[k_idx:k_idx+1, n_idx:n_idx+1]      # (1, 1) -> scalar
            
            # Compute this block's contribution using torch._scaled_mm
            block_output = torch._scaled_mm(
                a_block,           # (M, block_size)
                b_block,           # (block_size, block_size) 
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
    use_advanced: bool = False,
) -> torch.Tensor:
    """
    Wrapper function that matches the Triton kernel interface.
    
    Args:
        use_advanced: If True, uses the advanced implementation that better
                     preserves blockwise scaling at the cost of more computation.
    """
    if use_advanced:
        return blockwise_fp8_scaled_mm_advanced_1x128_128x128(
            a, a_s, b, b_s, block_size
        )
    else:
        return blockwise_fp8_scaled_mm_1x128_128x128(
            a, a_s, b, b_s, block_size  
        )


def blockwise_fp8_gemm_scaled_mm_1x128_128x1(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor, 
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Wrapper function that matches the Triton kernel interface."""
    return blockwise_fp8_scaled_mm_1x128_128x1(a, a_s, b, b_s, block_size)