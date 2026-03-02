# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


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

    atomic_kernel_configs_2D = [
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=4,
        )
    ]

    @torch.library.custom_op(
        "torchao::triton_fp8_rowwise_transpose_rhs", mutates_args={}
    )
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

        scales_buffer = torch.empty(
            (e, k), dtype=torch.float32, device=hp_tensor.device
        )
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
            expert_idx[:, None] * stride_scales_dim0
            + k_offs[None, :] * stride_scales_dim1
        )
        scales_mask = k_offs[None, :] < K
        # AMD GPUs need relaxed semantics for better performance
        if tl.constexpr(torch.version.hip is not None):
            tl.atomic_min(
                scales_ptr + scales_offs,
                scales[None, :],
                mask=scales_mask,
                sem="relaxed",
            )
        else:
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
            expert_idx[:, None] * stride_scales_dim0
            + k_offs[None, :] * stride_scales_dim1
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

    reduction_kernel_configs_2D = [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=8,
            num_stages=6,
        )
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
            output_data = tl.clamp(
                scaled_data, min=fp8_dtype_min, max=fp8_dtype_max
            ).to(output_dtype)

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

        scales_buffer = torch.empty(
            (e, k), dtype=torch.float32, device=hp_tensor.device
        )

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

        scales_buffer = torch.empty(
            (e, k), dtype=torch.float32, device=hp_tensor.device
        )
        return output_buffer, scales_buffer

    # ── Autotune configs for 2D fused rowwise scale+cast kernel ─────────
    #
    # This kernel fuses the 3-kernel FP8 quantization sequence used in the
    # forward pass of _Float8GroupedMM into a single kernel launch:
    #   Original: tensor_to_scale() + A * scales + to_fp8_saturated()
    #   Fused:    single two-pass kernel (absmax reduction + scale-and-cast)
    #
    # The kernel uses a two-pass approach over each row:
    #   Pass 1: Compute per-row absmax (reduction over K dimension)
    #   Pass 2: Apply scale and cast to FP8 (elementwise over K dimension)
    # The second pass benefits from L2 cache reuse since the same data was
    # just loaded in pass 1, avoiding a full re-read from HBM.
    #
    # Grid: one program per row (M programs total), each iterating over K
    # in blocks of BLOCK_SIZE_K. This is efficient because:
    #   - Each row's scale is independent (no cross-row synchronization)
    #   - K dimension (hidden dim, e.g., 2048) fits well in a few blocks
    #   - Row-major input means contiguous K-dimension access
    if torch.version.hip is not None:
        fused_2d_kernel_configs = [
            triton.Config(
                {"BLOCK_SIZE_K": block_size_k},
                num_warps=warps,
                num_stages=stages,
            )
            for block_size_k in [128, 256]
            for warps in [4, 8]
            for stages in [2]
        ]
    else:
        fused_2d_kernel_configs = [
            triton.Config(
                {"BLOCK_SIZE_K": 128},
                num_warps=4,
                num_stages=4,
            )
        ]

    @triton.autotune(configs=fused_2d_kernel_configs, key=["K"])
    @triton.jit
    def _triton_fp8_rowwise_2d_fused_scale_and_cast_kernel(
        input_ptr,
        stride_input_row: tl.int64,
        stride_input_col,
        output_ptr,
        stride_output_row: tl.int64,
        stride_output_col,
        scales_ptr,
        M: int,
        K: int,
        fp8_dtype_min: tl.constexpr,
        fp8_dtype_max: tl.constexpr,
        input_dtype: tl.constexpr,
        output_dtype: tl.constexpr,
        round_scales_to_power_of_2: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """
        Fused kernel that computes per-row absmax scale and casts a 2D tensor
        to FP8 in a single kernel launch.

        This replaces three separate kernel launches:
          1. tensor_to_scale: reduction over K to find per-row absmax, then
             compute scale = FP8_MAX / absmax
          2. A * scales: elementwise multiply input by scale
          3. to_fp8_saturated: clamp to FP8 range and cast dtype

        Two-pass approach (one program per row):
          Pass 1: Iterate over K in blocks to compute row-wise absmax.
          Pass 2: Iterate again to multiply by scale, clamp, and cast to FP8.

        The second pass reads the same data that was just loaded in pass 1,
        which is likely still in L2 cache, reducing HBM traffic compared to
        running 3 separate kernels that each read the full tensor.

        Args:
            input_ptr: Pointer to input tensor of shape (M, K).
            stride_input_row/col: Strides for input tensor (supports any layout).
            output_ptr: Pointer to output FP8 tensor of shape (M, K).
            stride_output_row/col: Strides for output tensor.
            scales_ptr: Pointer to output scales of shape (M,), one per row.
            M: Number of rows.
            K: Number of columns (reduction dimension for absmax).
            fp8_dtype_min/max: Min/max representable values in target FP8 dtype.
            round_scales_to_power_of_2: If true, round scale down to nearest
                power of 2 for hardware-friendly scaling.
            BLOCK_SIZE_K: Number of K elements processed per iteration (autotuned).
            EPS: Small epsilon to avoid division by zero in scale computation.
        """
        row_idx = tl.program_id(0)

        # ── Pass 1: compute row-wise maximum absolute value ──
        # Iterate over the K dimension in blocks, tracking the running max
        # of absolute values across all blocks for this row.
        row_amax: tl.float32 = 0.0

        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offs = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K

            input_offs = row_idx * stride_input_row + k_offs * stride_input_col
            vals = tl.load(input_ptr + input_offs, mask=k_mask, other=0.0)

            block_amax = tl.max(tl.abs(vals))
            row_amax = tl.maximum(row_amax, block_amax)

        # ── Compute scale: scale = FP8_MAX / absmax ──
        # This maps the row's dynamic range into the FP8 representable range.
        # Use float64 for the division to maintain precision, then convert back.
        row_amax = tl.maximum(row_amax, EPS)
        scale = fp8_dtype_max / row_amax.to(tl.float64)
        scale = scale.to(tl.float32)

        # Optionally round to power of 2 for hardware-friendly scaling.
        # Power-of-2 scales can be applied as exponent additions rather than
        # multiplications, which is more efficient on some hardware.
        if round_scales_to_power_of_2:
            scale = tl.exp2(tl.floor(tl.log2(scale)))

        tl.store(scales_ptr + row_idx, scale)

        # ── Pass 2: apply scale and cast to FP8 ──
        # Re-read the same row data (likely in L2 cache from pass 1),
        # multiply by the computed scale, clamp to FP8 range, and store.
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offs = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K

            input_offs = row_idx * stride_input_row + k_offs * stride_input_col
            vals = tl.load(input_ptr + input_offs, mask=k_mask, other=0.0)

            scaled_vals = vals.to(tl.float32) * scale
            clamped_vals = tl.clamp(
                scaled_vals, min=fp8_dtype_min, max=fp8_dtype_max
            ).to(output_dtype)

            output_offs = row_idx * stride_output_row + k_offs * stride_output_col
            tl.store(output_ptr + output_offs, clamped_vals, mask=k_mask)

    @torch.library.custom_op(
        "torchao::triton_fp8_rowwise_2d_scale_and_cast", mutates_args={}
    )
    def triton_fp8_rowwise_2d_scale_and_cast(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused scale computation and FP8 cast for 2D row-major tensors.

        Replaces the 3-kernel sequence:
            1. tensor_to_scale(A, axiswise_dim=-1)
            2. A_scaled = A.to(float32) * A_scales
            3. A_fp8 = to_fp8_saturated(A_scaled, fp8_dtype)

        With a single Triton kernel that computes per-row absmax, scale, and
        FP8 cast in two passes (benefiting from L2 cache reuse).

        Args:
            hp_tensor: Input tensor of shape (M, K) in float32 or bfloat16.
            output_dtype: Target FP8 dtype.
            round_scales_to_power_of_2: Whether to round scales to nearest power of 2.

        Returns:
            Tuple of (fp8_data, scales):
                - fp8_data: shape (M, K) in output_dtype, row-major.
                - scales: shape (M, 1) in float32 (forward scales: FP8_MAX / amax).
        """
        assert hp_tensor.ndim == 2, "input tensor must be 2D"

        tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
        tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

        fp8_dtype_min = torch.finfo(output_dtype).min
        fp8_dtype_max = torch.finfo(output_dtype).max

        m, k = hp_tensor.shape

        output_buffer = torch.empty(
            (m, k), dtype=output_dtype, device=hp_tensor.device
        )
        scales_buffer = torch.empty(
            (m,), dtype=torch.float32, device=hp_tensor.device
        )

        grid = lambda meta: (m,)

        _triton_fp8_rowwise_2d_fused_scale_and_cast_kernel[grid](
            hp_tensor,
            hp_tensor.stride(0),
            hp_tensor.stride(1),
            output_buffer,
            output_buffer.stride(0),
            output_buffer.stride(1),
            scales_buffer,
            m,
            k,
            fp8_dtype_min,
            fp8_dtype_max,
            tl_input_dtype,
            tl_output_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
            EPS=EPS,
        )

        return output_buffer, scales_buffer.unsqueeze(-1)

    @triton_fp8_rowwise_2d_scale_and_cast.register_fake
    def _fake_triton_fp8_rowwise_2d_scale_and_cast(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hp_tensor.ndim == 2, "input tensor must be 2D"
        m, k = hp_tensor.shape
        output_buffer = torch.empty(
            (m, k), dtype=output_dtype, device=hp_tensor.device
        )
        scales_buffer = torch.empty(
            (m, 1), dtype=torch.float32, device=hp_tensor.device
        )
        return output_buffer, scales_buffer

else:

    def triton_fp8_rowwise_3d_transpose_rhs(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_fp8_rowwise_3d_transpose_rhs requires torch 2.7.0+ and triton installed"
        )

    def triton_fp8_rowwise_3d_transpose_rhs_fused_reduction(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_fp8_rowwise_3d_transpose_rhs_fused_reduction requires torch 2.7.0+ and triton installed"
        )

    def triton_fp8_rowwise_2d_scale_and_cast(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_fp8_rowwise_2d_scale_and_cast requires torch 2.7.0+ and triton installed"
        )
