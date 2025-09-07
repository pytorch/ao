# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
import triton
import triton.language as tl

EPS = 1e-12

FP8_DTYPE_MAP = {
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

block_sizes_n = [128]  # large dim (output_features)
block_sizes_k = [128]  # small dim (input_features)
num_warps = [4]
num_stages = [4]
atomic_kernel_configs_2D = [
    triton.Config(
        {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k},
        num_warps=warps,
        num_stages=stages,
    )
    for block_size_n in block_sizes_n
    for block_size_k in block_sizes_k
    for warps in num_warps
    for stages in num_stages
]


@torch.library.custom_op("torchao::triton_fp8_rowwise_transpose_rhs", mutates_args={})
def triton_fp8_rowwise_3d_transpose_rhs(
    hp_tensor: torch.Tensor,  # (E, K, N)
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hp_tensor.ndim == 3, "input tensor must be 3D"

    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

    fp8_dtype_min = torch.finfo(output_dtype).min
    fp8_dtype_max = torch.finfo(output_dtype).max

    e, k, n = hp_tensor.shape

    # allocate on-device buffers for output and scales
    # output shape = input.transpose(-2, -1).shape = (E, N, K) in column major layout
    output_buffer = torch.empty(
        (e, n, k), dtype=output_dtype, device=hp_tensor.device
    ).as_strided((e, n, k), (n * k, 1, n))

    scales_buffer = torch.full(
        (e, k), float("inf"), dtype=torch.float32, device=hp_tensor.device
    )

    # parallelize across experts, and for each expert, parallelize across rows and cols
    grid = lambda meta: (
        e,
        triton.cdiv(k, meta["BLOCK_SIZE_K"]),
        triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )

    # compute scales
    _triton_fp8_rowwise_3d_transpose_scales_rhs_kernel[grid](
        hp_tensor,
        hp_tensor.stride(0),
        hp_tensor.stride(1),
        hp_tensor.stride(2),
        scales_buffer,
        scales_buffer.stride(0),
        scales_buffer.stride(1),
        e,
        n,
        k,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
        EPS=EPS,
    )

    # perform casting
    _triton_fp8_rowwise_3d_transpose_cast_rhs_kernel[grid](
        hp_tensor,
        hp_tensor.stride(0),
        hp_tensor.stride(1),
        hp_tensor.stride(2),
        output_buffer,
        output_buffer.stride(0),
        output_buffer.stride(1),
        output_buffer.stride(2),
        scales_buffer,
        scales_buffer.stride(0),
        scales_buffer.stride(1),
        e,
        n,
        k,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        tl_output_dtype,
    )
    return output_buffer, scales_buffer


@triton_fp8_rowwise_3d_transpose_rhs.register_fake
def _fake_triton_fp8_rowwise_3d_transpose_rhs(
    hp_tensor: torch.Tensor,  # (E, K, N)
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hp_tensor.ndim == 3, "input tensor must be 3D"
    e, k, n = hp_tensor.shape
    output_buffer = torch.empty(
        (e, n, k), dtype=output_dtype, device=hp_tensor.device
    ).as_strided((e, n, k), (n * k, 1, n))

    scales_buffer = torch.empty((e, k), dtype=torch.float32, device=hp_tensor.device)
    return output_buffer, scales_buffer


@triton.autotune(configs=atomic_kernel_configs_2D, key=["K", "N"])
@triton.jit
def _triton_fp8_rowwise_3d_transpose_scales_rhs_kernel(
    input_ptr,
    stride_input_dim0: tl.int64,
    stride_input_dim1,
    stride_input_dim2,
    scales_ptr,
    stride_scales_dim0: int,
    stride_scales_dim1,
    E: int,
    N: int,
    K: int,
    fp8_dtype_min: tl.constexpr,
    fp8_dtype_max: tl.constexpr,
    input_dtype: tl.constexpr,
    round_scales_to_power_of_2: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EPS: tl.constexpr,
):
    # parallelize across experts, rows, and cols
    expert_idx = tl.program_id(0)
    k_block_idx = tl.program_id(1)
    n_block_idx = tl.program_id(2)

    # compute offsets for each dimension
    k_offs = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    n_offs = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # load block of input data, shape (K, N)
    input_offs = (
        expert_idx * stride_input_dim0
        + k_offs[:, None] * stride_input_dim1
        + (n_offs[None, :] * stride_input_dim2)
    )
    input_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
    input_data = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)

    # In a normal torch implementation, we should transpose the tensor then compute the amax
    # along the dim1 (N), to compute colwise scales for a RHS operand of a scaled grouped gemm:
    #    input_data = input_data.transpose(-2,-1) # (E, K, N) -> (E, N, K)
    #    amaxes = input_data.abs().max(dim=1) # (E, N, K) -> (E, 1, K)
    #
    # Here, we are reading a (K, N) chunk for a given E, and computing the amax along the dim=1 (N)
    # to compute an equivalent scale of shape (K,) for this chunk of the expert.
    # We then use atomic min to compute the final scale for these logical columns of the transposed tensor.
    #
    # Later, we will use this scale to cast the same (K,N) input chunk to fp8 and transpose it to (N, K) before
    # writing it to the output tensor.
    #    ((K, N) * (K, 1))^T = (N, K)
    amaxes = tl.max(tl.abs(input_data), axis=1).to(tl.float64)  # (K,)
    scales = (fp8_dtype_max / tl.clamp(amaxes, min=EPS, max=float("inf"))).to(
        tl.float32
    )
    if round_scales_to_power_of_2:
        scales = tl.exp2(tl.floor(tl.log2(scales)))

    # compute global scales using atomics with local scales - shape (1, K)
    scales_offs = (
        expert_idx[:, None] * stride_scales_dim0 + k_offs[None, :] * stride_scales_dim1
    )
    scales_mask = k_offs[None, :] < K
    tl.atomic_min(scales_ptr + scales_offs, scales[None, :], mask=scales_mask)


@triton.autotune(configs=atomic_kernel_configs_2D, key=["num_elements"])
@triton.jit
def _triton_fp8_rowwise_3d_transpose_cast_rhs_kernel(
    input_ptr,
    stride_input_dim0: tl.int64,
    stride_input_dim1,
    stride_input_dim2,
    output_ptr,
    stride_output_dim0: tl.int64,
    stride_output_dim1,
    stride_output_dim2,
    scales_ptr,
    stride_scales_dim0: int,
    stride_scales_dim1,
    E: int,
    N: int,
    K: int,
    fp8_dtype_min: tl.constexpr,
    fp8_dtype_max: tl.constexpr,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # parallelize across experts, rows, and cols
    expert_idx = tl.program_id(0)
    k_block_idx = tl.program_id(1)
    n_block_idx = tl.program_id(2)

    # compute offsets for each dimension
    k_offs = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    n_offs = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # load block of input data for this expert - shape (K, N)
    input_offs = (
        expert_idx * stride_input_dim0
        + k_offs[:, None] * stride_input_dim1
        + (n_offs[None, :] * stride_input_dim2)
    )
    input_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
    input_data = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)
    input_data = input_data.trans(1, 0)  # (K, N) -> (N, K)

    # load global scales for this block of the given expert - shape (1, K)
    scales_offs = (
        expert_idx[:, None] * stride_scales_dim0 + k_offs[None, :] * stride_scales_dim1
    )
    scales_mask = k_offs[None, :] < K
    scales = tl.load(scales_ptr + scales_offs, mask=scales_mask, other=0.0)

    # transpose data and apply scales - shape (N,K) * (1,K) = (N,K)
    output_data = tl.clamp(
        input_data * scales, min=fp8_dtype_min, max=fp8_dtype_max
    ).to(output_dtype)

    # store transpose and store output data - shape (N, K)
    output_offs = (
        expert_idx * stride_output_dim0
        + n_offs[:, None] * stride_output_dim1
        + (k_offs[None, :] * stride_output_dim2)
    )
    output_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
    tl.store(output_ptr + output_offs, output_data, mask=output_mask)


block_sizes_n = [
    64,
]  # large dim (output_features)
block_sizes_k = [128]  # small dim (input_features)
num_warps = [8]
num_stages = [6]
reduction_kernel_configs_2D = [
    triton.Config(
        {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k},
        num_warps=warps,
        num_stages=stages,
    )
    for block_size_n in block_sizes_n
    for block_size_k in block_sizes_k
    for warps in num_warps
    for stages in num_stages
]


@triton.autotune(configs=reduction_kernel_configs_2D, key=["K", "N"])
@triton.jit
def _triton_fp8_rowwise_3d_transpose_rhs_fused_reduction_kernel(
    input_ptr,
    stride_input_dim0: tl.int64,
    stride_input_dim1,
    stride_input_dim2,
    output_ptr,
    stride_output_dim0: tl.int64,
    stride_output_dim1,
    stride_output_dim2,
    scales_ptr,
    stride_scales_dim0: int,
    stride_scales_dim1,
    E: int,
    N: int,
    K: int,
    fp8_dtype_min: tl.constexpr,
    fp8_dtype_max: tl.constexpr,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    round_scales_to_power_of_2: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EPS: tl.constexpr,
):
    # This kernel parallelizes across experts and K blocks
    # Each program computes scales for one K block of one expert
    expert_idx = tl.program_id(0)
    k_block_idx = tl.program_id(1)

    # Compute K offsets for this block
    k_offs = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offs < K

    # Initialize row maxes for this K block
    row_maxes = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float64) - float("inf")

    # First pass: compute row-wise maximum absolute values across all N
    for n_block_start in range(0, N, BLOCK_SIZE_N):
        n_offs = n_block_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        # Load block of input data - shape (K, N)
        input_offs = (
            expert_idx * stride_input_dim0
            + k_offs[:, None] * stride_input_dim1
            + n_offs[None, :] * stride_input_dim2
        )
        input_mask = k_mask[:, None] & n_mask[None, :]
        input_data = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)

        # Compute row-wise max for this N block
        block_row_maxes = tl.max(tl.abs(input_data), axis=1)

        # Update running maxes
        row_maxes = tl.maximum(row_maxes, block_row_maxes)

    # Convert row maxes to scales
    clamped_maxes = tl.clamp(row_maxes, min=EPS, max=float("inf"))
    scales = (fp8_dtype_max / clamped_maxes.to(tl.float64)).to(tl.float32)

    if round_scales_to_power_of_2:
        scales = tl.exp2(tl.floor(tl.log2(scales)))

    # Store computed scales for this K block
    scales_offs = expert_idx * stride_scales_dim0 + k_offs * stride_scales_dim1
    tl.store(scales_ptr + scales_offs, scales, mask=k_mask)

    # Second pass: apply scales and transpose data for output
    for n_block_start in range(0, N, BLOCK_SIZE_N):
        n_offs = n_block_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        # Load block of input data - shape (K, N)
        input_offs = (
            expert_idx * stride_input_dim0
            + k_offs[:, None] * stride_input_dim1
            + n_offs[None, :] * stride_input_dim2
        )
        input_mask = k_mask[:, None] & n_mask[None, :]
        input_data = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)

        # Transpose data: (K, N) -> (N, K)
        input_data_transposed = input_data.trans(1, 0)

        # Apply scales: (N, K) * (1, K) = (N, K)
        scaled_data = input_data_transposed * scales[None, :]

        # Clamp and cast to output dtype
        output_data = tl.clamp(scaled_data, min=fp8_dtype_min, max=fp8_dtype_max).to(
            output_dtype
        )

        # Store transposed output - shape (N, K)
        output_offs = (
            expert_idx * stride_output_dim0
            + n_offs[:, None] * stride_output_dim1
            + k_offs[None, :] * stride_output_dim2
        )
        output_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(output_ptr + output_offs, output_data, mask=output_mask)


@torch.library.custom_op(
    "torchao::triton_fp8_rowwise_transpose_rhs_fused", mutates_args={}
)
def triton_fp8_rowwise_3d_transpose_rhs_fused_reduction(
    hp_tensor: torch.Tensor,  # (E, K, N)
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Equivalent fused Triton kernel to triton_fp8_rowwise_3d_transpose_rhs that uses
    reduction to calculate rowwise scales instead of atomic operations.

    This kernel fuses the scale computation and casting into a single kernel,
    avoiding the need for atomic operations by using reduction operations.
    """
    assert hp_tensor.ndim == 3, "input tensor must be 3D"

    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

    fp8_dtype_min = torch.finfo(output_dtype).min
    fp8_dtype_max = torch.finfo(output_dtype).max

    e, k, n = hp_tensor.shape

    # allocate on-device buffers for output and scales
    # output shape = input.transpose(-2, -1).shape = (E, N, K) in column major layout
    output_buffer = torch.empty(
        (e, n, k), dtype=output_dtype, device=hp_tensor.device
    ).as_strided((e, n, k), (n * k, 1, n))

    scales_buffer = torch.empty((e, k), dtype=torch.float32, device=hp_tensor.device)

    # Use a grid that parallelizes across experts and K blocks
    # Each program handles one K block of one expert
    grid = lambda meta: (e, triton.cdiv(k, meta["BLOCK_SIZE_K"]), 1)

    # Single fused kernel that computes scales using reduction and performs casting
    _triton_fp8_rowwise_3d_transpose_rhs_fused_reduction_kernel[grid](
        hp_tensor,
        hp_tensor.stride(0),
        hp_tensor.stride(1),
        hp_tensor.stride(2),
        output_buffer,
        output_buffer.stride(0),
        output_buffer.stride(1),
        output_buffer.stride(2),
        scales_buffer,
        scales_buffer.stride(0),
        scales_buffer.stride(1),
        e,
        n,
        k,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        tl_output_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
        EPS=EPS,
    )

    return output_buffer, scales_buffer


@triton_fp8_rowwise_3d_transpose_rhs_fused_reduction.register_fake
def _fake_triton_fp8_rowwise_3d_transpose_rhs_fused_reduction(
    hp_tensor: torch.Tensor,  # (E, K, N)
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hp_tensor.ndim == 3, "input tensor must be 3D"
    e, k, n = hp_tensor.shape
    output_buffer = torch.empty(
        (e, n, k), dtype=output_dtype, device=hp_tensor.device
    ).as_strided((e, n, k), (n * k, 1, n))

    scales_buffer = torch.empty((e, k), dtype=torch.float32, device=hp_tensor.device)
    return output_buffer, scales_buffer
