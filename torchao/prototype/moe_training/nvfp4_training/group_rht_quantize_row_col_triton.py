# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""TE-compatible graph-safe Triton GroupRHT kernel and custom op.

Packs a BF16 matrix ``A`` plus per-group row counts into rowwise flat buffers
and columnwise per-group views over one flat columnwise buffer. The wrapper
forwards single-element views of a caller-owned Philox state so stochastic
rounding stays CUDA-graph-safe (no host RNG).

The NVFP4 quantization epilogue helpers (``_nvfp4_quantize``, ``_pack_fp4``,
``_swizzle_scales``, ``_store_scales_swizzle``) are reused from
``nvfp4_training.hadamard_utils``; only the grouped index lookup is local to
this kernel.
"""

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

BLOCK_M = 128
BLOCK_N = 128
SAME_BOTH_DIMS = 0
VARYING_FIRST_DIM = 1


if torch_version_at_least("2.10.0") and has_triton():
    from typing import Optional, Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        _nvfp4_quantize,
        _pack_fp4,
        _store_scales_swizzle,
        _swizzle_scales,
    )
    from torchao.utils import is_sm_at_least_100

    @triton.jit
    def _group_idx_from_range(
        token_offset,
        group_range_ptr,
        num_tensors: tl.constexpr,
    ):
        group_idx = 0
        for i in range(num_tensors):
            start = tl.load(group_range_ptr + i)
            if token_offset >= start:
                group_idx = i
        return group_idx

    @triton.jit
    def _group_row_col_rht_gemm_triton_kernel(
        A,
        RHT,
        QA_base,
        SFA_base,
        group_range,
        amax_row,
        amax_col,
        QD,
        SFD,
        col_seed_base_ptr,
        col_offset_base_ptr,
        row_seed_base_ptr,
        row_offset_base_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        GROUP_RANGE_IS_ELEMENT_OFFSETS: tl.constexpr,
        STOCHASTIC_ROUNDING: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Grouped fused RHT columnwise and direct rowwise NVFP4 quantization."""
        tile_id = tl.program_id(0)
        num_tiles_token = tl.cdiv(M, BLOCK_M)
        num_tiles_hidden = tl.cdiv(N, BLOCK_N)
        pid_m = tile_id // num_tiles_hidden
        pid_n = tile_id - pid_m * num_tiles_hidden

        if SHAPE_REP == 0:
            group_idx = pid_m // (num_tiles_token // num_tensors)
        else:
            token_offset = pid_m * BLOCK_M
            lookup_offset = token_offset
            if GROUP_RANGE_IS_ELEMENT_OFFSETS:
                lookup_offset = token_offset * N
            group_idx = _group_idx_from_range(lookup_offset, group_range, num_tensors)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(A + offs_m[:, None] * N + offs_n[None, :])

        hadamard_offsets = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        hadamard = tl.load(RHT + hadamard_offsets)

        colwise_global_amax = tl.load(amax_col + group_idx)
        a_t = tl.trans(a)
        a_t_reshape = tl.reshape(a_t, [BLOCK_N * BLOCK_M // 16, 16])
        a_t_rht = tl.dot(a_t_reshape, hadamard)
        a_t_rht = a_t_rht.to(tl.bfloat16)

        col_scale, col_scaled = _nvfp4_quantize(
            a_t_rht, colwise_global_amax, BLOCK_N, BLOCK_M
        )
        col_fp4 = _pack_fp4(
            col_scaled,
            BLOCK_N,
            BLOCK_M,
            STOCHASTIC_ROUNDING,
            col_seed_base_ptr,
            col_offset_base_ptr,
            tile_id,
        )

        col_swizzled = _swizzle_scales(col_scale, BLOCK_N, BLOCK_M)
        _store_scales_swizzle(
            col_swizzled,
            SFD,
            pid_n,
            pid_m,
            N,
            M,
            BLOCK_N,
            BLOCK_M,
        )
        outer = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        packed_inner = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
        packed_offsets = outer[:, None] * (M // 2) + packed_inner[None, :]
        tl.store(QD + packed_offsets, col_fp4)

        rowwise_global_amax = tl.load(amax_row + group_idx)
        row_scale, row_scaled = _nvfp4_quantize(
            a, rowwise_global_amax, BLOCK_M, BLOCK_N
        )
        row_fp4 = _pack_fp4(
            row_scaled,
            BLOCK_M,
            BLOCK_N,
            STOCHASTIC_ROUNDING,
            row_seed_base_ptr,
            row_offset_base_ptr,
            tile_id,
        )

        row_swizzled = _swizzle_scales(row_scale, BLOCK_M, BLOCK_N)
        _store_scales_swizzle(
            row_swizzled,
            SFA_base,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
        )
        outer = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        packed_inner = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        packed_offsets = outer[:, None] * (N // 2) + packed_inner[None, :]
        tl.store(QA_base + packed_offsets, row_fp4)

    def _validate_common_inputs(A: torch.Tensor, B: torch.Tensor) -> None:
        if not isinstance(A, torch.Tensor):
            raise TypeError("A must be a torch.Tensor")
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got {A.ndim}D")
        if A.dtype != torch.bfloat16:
            raise ValueError("A.dtype must be torch.bfloat16")
        if not A.is_contiguous():
            raise ValueError("A must be row-major (contiguous)")
        if not isinstance(B, torch.Tensor):
            raise TypeError("B must be a torch.Tensor")
        if B.ndim != 2:
            raise ValueError(f"B must be 2D, got {B.ndim}D")
        if B.dtype != torch.bfloat16:
            raise ValueError("B.dtype must be torch.bfloat16")
        if B.shape != (16, 16):
            raise ValueError(f"B must have shape (16, 16), got {tuple(B.shape)}")
        if A.shape[1] % BLOCK_N != 0:
            raise ValueError("A.shape[1] must be divisible by 128")
        if not A.is_cuda:
            raise ValueError("A must be on CUDA")
        if not B.is_cuda:
            raise ValueError("B must be on CUDA")
        if B.device != A.device:
            raise ValueError("B must be on the same device as A")
        if torch.cuda.is_available() and not is_sm_at_least_100():
            raise NotImplementedError(
                "GroupRHT Triton kernel requires SM100 (Blackwell) for FP4 conversion."
            )

    def _validate_graph_amax(
        amax: Optional[torch.Tensor],
        name: str,
        num_tensors: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(amax, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if amax.dtype != torch.float32:
            raise ValueError(f"{name}.dtype must be torch.float32")
        if not amax.is_cuda or amax.device != device:
            raise ValueError(f"{name} must be on the same device as A")
        if amax.numel() < num_tensors:
            raise ValueError(f"{name} must have at least num_tensors elements")
        return amax

    def _validate_rng_state(
        rng_state: Optional[torch.Tensor],
        device: torch.device,
        enable_stochastic_rounding: bool,
    ) -> Optional[torch.Tensor]:
        """Validate the caller-owned Philox state used for graph-safe stochastic rounding.

        When SR is enabled, ``rng_state`` is an int64 CUDA tensor laid out as
        ``[col_seed, col_offset, row_seed, row_offset]``. The caller owns advancement of
        these values across CUDA-graph replays (torchao seed-plumbing); the wrapper only
        forwards single-element views of them, so it performs no host RNG and stays graph-safe.
        """
        if not enable_stochastic_rounding:
            return None
        if not isinstance(rng_state, torch.Tensor):
            raise TypeError(
                "rng_state must be a torch.Tensor when enable_stochastic_rounding is True"
            )
        if rng_state.dtype != torch.int64:
            raise ValueError("rng_state.dtype must be torch.int64")
        if not rng_state.is_cuda or rng_state.device != device:
            raise ValueError("rng_state must be on the same device as A")
        if rng_state.numel() < 4:
            raise ValueError(
                "rng_state must have at least 4 elements "
                "[col_seed, col_offset, row_seed, row_offset]"
            )
        return rng_state

    @torch.library.custom_op(
        "torchao::triton_group_rht_quantize_row_col", mutates_args=()
    )
    def triton_group_rht_quantize_row_col(
        A: torch.Tensor,
        B: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        a_global_amax: Optional[torch.Tensor],
        d_global_amax: Optional[torch.Tensor],
        rng_state: Optional[torch.Tensor],
        enable_stochastic_rounding: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Grouped fused RHT columnwise + direct rowwise NVFP4 E2M1 quantization.

        ``A`` is the pre-packed ``torch.cat(tensors, dim=0)`` buffer; ``offsets`` is a
        1D int64 CUDA tensor of cumulative element offsets (``first_dim * hidden_size``)
        marking each group's start. ``B`` is the (16, 16) bfloat16 RHT matrix.

        Returns ``(qa_base, sfa, qd, sfd)``. Both scale tensors carry swizzled bytes
        reinterpreted to their logical 2D shapes.

        Stochastic rounding is CUDA-graph-safe: ``rng_state`` is an int64 CUDA tensor
        ``[col_seed, col_offset, row_seed, row_offset]`` whose advancement the caller owns;
        the op only forwards single-element views, performing no host RNG.
        """
        _validate_common_inputs(A, B)
        rng_state = _validate_rng_state(rng_state, A.device, enable_stochastic_rounding)
        if packed_sequence_length != A.shape[0]:
            raise ValueError("packed_sequence_length must match A.shape[0]")
        if hidden_size != A.shape[1]:
            raise ValueError("hidden_size must match A.shape[1]")
        if packed_sequence_length % BLOCK_M != 0:
            raise ValueError("packed_sequence_length must be divisible by 128")
        if hidden_size % BLOCK_N != 0:
            raise ValueError("hidden_size must be divisible by 128")
        if num_tensors <= 0:
            raise ValueError("num_tensors must be greater than 0")
        if shape_rep not in (SAME_BOTH_DIMS, VARYING_FIRST_DIM):
            raise ValueError(
                "graph-safe TE GroupRHT only supports SAME_BOTH_DIMS or VARYING_FIRST_DIM"
            )
        if shape_rep == SAME_BOTH_DIMS:
            if packed_sequence_length % num_tensors != 0:
                raise ValueError(
                    "packed_sequence_length must be divisible by num_tensors "
                    "for SAME_BOTH_DIMS"
                )
            if (packed_sequence_length // num_tensors) % BLOCK_M != 0:
                raise ValueError(
                    "SAME_BOTH_DIMS group row count must be divisible by 128"
                )
        if not isinstance(offsets, torch.Tensor):
            raise TypeError("offsets must be a torch.Tensor")
        if offsets.ndim != 1:
            raise ValueError(f"offsets must be 1D, got {offsets.ndim}D")
        if offsets.dtype != torch.int64:
            raise ValueError("offsets.dtype must be torch.int64")
        if not offsets.is_cuda:
            raise ValueError("offsets must be on CUDA")
        if offsets.device != A.device:
            raise ValueError("offsets must be on the same device as A")
        if offsets.numel() < num_tensors:
            raise ValueError("offsets must have at least num_tensors elements")
        if shape_rep == VARYING_FIRST_DIM:
            torch.ops.aten._assert_async.msg(
                torch.all(offsets[:num_tensors] % (BLOCK_M * hidden_size) == 0),
                "VARYING_FIRST_DIM offsets must align group boundaries to 128-row tiles",
            )

        qa_base = torch.empty(
            (packed_sequence_length, hidden_size // 2),
            dtype=torch.uint8,
            device=A.device,
        )
        row_amax = _validate_graph_amax(
            a_global_amax, "a_global_amax", num_tensors, A.device
        )
        sfa_storage = torch.empty(
            (packed_sequence_length // 128, hidden_size // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=A.device,
        )
        sfa_return = sfa_storage.view(packed_sequence_length, hidden_size // 16)

        qd = torch.empty(
            (hidden_size, packed_sequence_length // 2),
            dtype=torch.uint8,
            device=A.device,
        )
        col_amax = _validate_graph_amax(
            d_global_amax, "d_global_amax", num_tensors, A.device
        )
        sfd_storage = torch.empty(
            (hidden_size // 128, packed_sequence_length // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=A.device,
        )
        sfd_return = sfd_storage.view(hidden_size, packed_sequence_length // 16)

        if enable_stochastic_rounding:
            col_seed_base = rng_state[0:1]
            col_offset_base = rng_state[1:2]
            row_seed_base = rng_state[2:3]
            row_offset_base = rng_state[3:4]
        else:
            col_seed_base = 0
            col_offset_base = 0
            row_seed_base = 0
            row_offset_base = 0

        m, n = A.shape
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_row_col_rht_gemm_triton_kernel[grid](
            A,
            B,
            qa_base,
            sfa_storage,
            offsets,
            row_amax,
            col_amax,
            qd,
            sfd_storage,
            col_seed_base,
            col_offset_base,
            row_seed_base,
            row_offset_base,
            m,
            n,
            num_tensors=num_tensors,
            GROUP_RANGE_IS_ELEMENT_OFFSETS=True,
            STOCHASTIC_ROUNDING=enable_stochastic_rounding,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=3,
        )
        return qa_base, sfa_return, qd, sfd_return

    @triton_group_rht_quantize_row_col.register_fake
    def _(
        A,
        B,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        a_global_amax,
        d_global_amax,
        rng_state,
        enable_stochastic_rounding,
    ):
        qa_base = A.new_empty(
            (packed_sequence_length, hidden_size // 2), dtype=torch.uint8
        )
        sfa = A.new_empty(
            (packed_sequence_length, hidden_size // 16), dtype=torch.float8_e4m3fn
        )
        qd = A.new_empty((hidden_size, packed_sequence_length // 2), dtype=torch.uint8)
        sfd = A.new_empty(
            (hidden_size, packed_sequence_length // 16), dtype=torch.float8_e4m3fn
        )

        return qa_base, sfa, qd, sfd

else:

    def triton_group_rht_quantize_row_col(
        A: torch.Tensor,
        B: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        a_global_amax,
        d_global_amax,
        rng_state,
        enable_stochastic_rounding: bool,
    ):
        raise NotImplementedError(
            "triton_group_rht_quantize_row_col requires torch 2.10.0+ and triton installed"
        )
