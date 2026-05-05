# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-tensor FP8 quantization kernels for Q, K, V.

Input/output format: [B, H, S, D].
Produces a single scalar descale per tensor (shape [1, 1, 1, 1]) for cuDNN.
Supports GQA (different head counts for Q vs K/V).

Uses a flat grid with atomic max reduction for Phase 1, eliminating the
separate reduce kernel and partial-max buffer needed by per-head quantization.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from torchao.prototype.attention.quantization.triton_hadamard_utils import (
    _compute_num_chunks,
)
from torchao.prototype.attention.quantization.triton_qkv_quantization import (
    single_phase2_kernel,
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def per_tensor_phase1_kernel(
    # Input tensor [B, H, S, D]
    x_ptr,
    # Output: single global max [1], atomically updated
    global_max_ptr,
    # Input strides
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    # Dimensions
    B,
    H,
    S,
    D,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute per-tensor absmax using a flat grid with atomic reduction.

    Grid: (num_blocks,) — tuned for GPU occupancy, NOT tied to B*H*S.
    Each block processes a strided portion of the entire tensor and
    atomically updates a single global max.
    """
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    total_elements = B * H * S * D

    x_max = 0.0

    for idx_start in range(pid * BLOCK_SIZE, total_elements, num_blocks * BLOCK_SIZE):
        offs = idx_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements

        # Convert flat index to (b, h, s, d) coordinates
        d = offs % D
        remaining = offs // D
        s = remaining % S
        remaining = remaining // S
        h = remaining % H
        b = remaining // H

        ptr_offset = b * stride_b + h * stride_h + s * stride_s + d * stride_d
        x_val = tl.load(x_ptr + ptr_offset, mask=mask, other=0.0).to(tl.float32)
        x_max = tl.maximum(x_max, tl.max(tl.abs(x_val)))

    # Single atomic update per block after local reduction
    tl.atomic_max(global_max_ptr, x_max)


@triton.jit
def compute_scale_descale_kernel(
    global_max_ptr,  # [1]
    scale_ptr,  # [1]
    descale_ptr,  # [1]
):
    """
    Compute scale and descale from a single global max.

    Grid: (1,)
    """
    x_max = tl.load(global_max_ptr)
    FP8_MAX = 448.0
    eps = 1e-12
    tl.store(scale_ptr, tl.where(x_max > eps, FP8_MAX / x_max, 1.0))
    tl.store(descale_ptr, tl.where(x_max > eps, x_max / FP8_MAX, 1.0))


def triton_fp8_per_tensor_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Per-tensor FP8 quantization for Q, K, V tensors.

    Each tensor gets a single scalar scale/descale (shape [1, 1, 1, 1])
    for use with cuDNN per-tensor FP8 attention.

    Args:
        q: Query tensor of shape [B, H_q, S_q, D] in bf16/fp16
        k: Key tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        v: Value tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        num_chunks: Number of chunks to split S dimension into (for Phase 2).

    Returns:
        q_fp8: Quantized query, shape [B, H_q, S_q, D] in fp8
        k_fp8: Quantized key, shape [B, H_kv, S_kv, D] in fp8
        v_fp8: Quantized value, shape [B, H_kv, S_kv, D] in fp8
        q_descale: Query descale factor, shape [1, 1, 1, 1] in fp32
        k_descale: Key descale factor, shape [1, 1, 1, 1] in fp32
        v_descale: Value descale factor, shape [1, 1, 1, 1] in fp32
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert k.shape == v.shape
    assert q.shape[0] == k.shape[0]
    assert q.shape[3] == k.shape[3]
    assert q.shape[1] % k.shape[1] == 0

    B, H_q, S_q, D = q.shape
    H_kv = k.shape[1]
    S_kv = k.shape[2]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Flat grid for Phase 1: num_blocks tuned to GPU occupancy
    props = torch.cuda.get_device_properties(q.device)
    num_blocks = props.multi_processor_count * 4

    # Chunk computation for Phase 2 (still uses (B, H, chunks) grid)
    if num_chunks is None:
        q_num_chunks = _compute_num_chunks(q.device, B, H_q, S_q)
        kv_num_chunks = _compute_num_chunks(k.device, B, H_kv, S_kv)
    else:
        q_num_chunks = num_chunks
        kv_num_chunks = num_chunks
    q_chunk_size = (S_q + q_num_chunks - 1) // q_num_chunks
    kv_chunk_size = (S_kv + kv_num_chunks - 1) // kv_num_chunks

    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)

    # Global max buffers: zero-initialized for atomic_max
    q_global_max = torch.zeros(1, dtype=torch.float32, device=q.device)
    k_global_max = torch.zeros(1, dtype=torch.float32, device=q.device)
    v_global_max = torch.zeros(1, dtype=torch.float32, device=q.device)

    # One scale/descale per tensor
    q_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(1, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(1, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(1, dtype=torch.float32, device=q.device)

    # Phase 1: flat-grid absmax with atomic reduction
    per_tensor_phase1_kernel[(num_blocks,)](
        q,
        q_global_max,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        B,
        H_q,
        S_q,
        D,
    )
    per_tensor_phase1_kernel[(num_blocks,)](
        k,
        k_global_max,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        B,
        H_kv,
        S_kv,
        D,
    )
    per_tensor_phase1_kernel[(num_blocks,)](
        v,
        v_global_max,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        B,
        H_kv,
        S_kv,
        D,
    )

    # Compute scale/descale from global max
    compute_scale_descale_kernel[(1,)](q_global_max, q_scale, q_descale)
    compute_scale_descale_kernel[(1,)](k_global_max, k_scale, k_descale)
    compute_scale_descale_kernel[(1,)](v_global_max, v_scale, v_descale)

    # Phase 2: quantize using global scale
    # H_scale=0 so all blocks read scale_ptr[0] regardless of pid_b/pid_h
    q_grid = (B, H_q, q_num_chunks)
    kv_grid = (B, H_kv, kv_num_chunks)

    single_phase2_kernel[q_grid](
        q,
        q_fp8,
        q_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        S_q,
        D,
        H_q,
        q_chunk_size,
        0,
        H_q,  # H_scale=0, groups=H_q: index = pid_b*0 + pid_h//H_q = 0
    )
    single_phase2_kernel[kv_grid](
        k,
        k_fp8,
        k_scale,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        S_kv,
        D,
        H_kv,
        kv_chunk_size,
        0,
        H_kv,  # H_scale=0, groups=H_kv: index = 0
    )
    single_phase2_kernel[kv_grid](
        v,
        v_fp8,
        v_scale,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        S_kv,
        D,
        H_kv,
        kv_chunk_size,
        0,
        H_kv,  # H_scale=0, groups=H_kv: index = 0
    )

    # Reshape descales to [1, 1, 1, 1] for cuDNN per-tensor format
    q_descale = q_descale.view(1, 1, 1, 1)
    k_descale = k_descale.view(1, 1, 1, 1)
    v_descale = v_descale.view(1, 1, 1, 1)

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
