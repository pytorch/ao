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
``nvfp4_training.hadamard_utils``. Shared grouped validation and index lookup
live in ``nvfp4_training.group_hadamard_utils``.
"""

from typing import Optional

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

if torch_version_at_least("2.10.0") and has_triton():
    from typing import List, Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
        BLOCK_M,
        BLOCK_N,
        _get_group_idx_binary,
        _validate_grouped_hadamard_inputs,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        _device_key,
        _nvfp4_quantize,
        _pack_fp4,
        _store_scales_swizzle,
        _swizzle_scales,
        get_rht_matrix,
    )

    # BLOCK_M/BLOCK_N are held at 128 (group offsets are only 128-aligned, so a
    # larger row tile could straddle two experts and apply the wrong amax). Only
    # num_warps/num_stages are autotuned; the shipped default (w8/s3) is in the set,
    # so autotune never regresses. Measured ~1.4x from num_warps=4 (register-heavy
    # quantize body over-subscribes at 8 warps). Body is straight-line (no tl.range),
    # so num_stages is purely the launch-time pipeliner and the grid is unaffected.
    _GROUP_QUANTIZE_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for ns in (2, 3, 4)
        for nw in (4, 8)
    ]

    @triton.autotune(
        configs=_GROUP_QUANTIZE_CONFIGS,
        key=["M", "N", "STOCHASTIC_ROUNDING"],
    )
    @triton.jit
    def _group_rht_quantize_row_col_kernel(
        a_ptr,
        b_ptr,
        qa_ptr,
        sfa_ptr,
        offsets_ptr,
        global_amax_row_ptr,
        global_amax_col_ptr,
        qa_t_ptr,
        sfa_t_ptr,
        col_seed_base_ptr,
        col_offset_base_ptr,
        row_seed_base_ptr,
        row_offset_base_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        STOCHASTIC_ROUNDING: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        logical_packed_length_ptr,
    ):
        """Grouped fused RHT columnwise and direct rowwise NVFP4 quantization."""
        VARYING_FIRST_DIM: tl.constexpr = 1

        tile_idx = tl.program_id(0)
        num_tiles_token = tl.cdiv(M, BLOCK_M)
        num_tiles_hidden = tl.cdiv(N, BLOCK_N)
        pid_m = tile_idx // num_tiles_hidden
        pid_n = tile_idx - pid_m * num_tiles_hidden

        if SHAPE_REP == VARYING_FIRST_DIM:
            token_offset = pid_m * BLOCK_M
            group_idx = _get_group_idx_binary(
                token_offset,
                offsets_ptr,
                num_tensors,
            )
        else:
            group_idx = pid_m // (num_tiles_token // num_tensors)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        logical_packed_length = tl.load(logical_packed_length_ptr)
        a = tl.load(
            a_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=offs_m[:, None] < logical_packed_length,
            other=0.0,
        )

        rht_offsets = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        hadamard = tl.load(b_ptr + rht_offsets)

        colwise_global_amax = tl.load(global_amax_col_ptr + group_idx)
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
            tile_idx,
        )

        col_swizzled = _swizzle_scales(col_scale, BLOCK_N, BLOCK_M)
        _store_scales_swizzle(
            col_swizzled,
            sfa_t_ptr,
            pid_n,
            pid_m,
            N,
            M,
            BLOCK_N,
            BLOCK_M,
        )
        outer_t = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        packed_inner_t = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
        packed_offsets_t = outer_t[:, None] * (M // 2) + packed_inner_t[None, :]
        tl.store(qa_t_ptr + packed_offsets_t, col_fp4)

        rowwise_global_amax = tl.load(global_amax_row_ptr + group_idx)
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
            tile_idx,
        )

        row_swizzled = _swizzle_scales(row_scale, BLOCK_M, BLOCK_N)
        _store_scales_swizzle(
            row_swizzled,
            sfa_ptr,
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
        tl.store(qa_ptr + packed_offsets, row_fp4)

    def _validate_graph_amax(
        amax: torch.Tensor,
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
        sign_vector: List[int],
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        a_global_amax: torch.Tensor,
        d_global_amax: torch.Tensor,
        rng_state: Optional[torch.Tensor],
        enable_stochastic_rounding: bool,
        logical_packed_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Grouped fused RHT columnwise + direct rowwise NVFP4 E2M1 quantization.

        ``A`` is the pre-packed capacity buffer; ``offsets`` is a 1D int32 CUDA
        tensor of cumulative row-end offsets, one per group. ``logical_packed_length``
        is the valid padded row count; rows after it are ignored. ``sign_vector``
        selects the cached 16x16 RHT matrix used by the Triton kernel.

        Returns ``(qa_base, sfa, qd, sfd)``. Both scale tensors carry swizzled bytes
        reinterpreted to their logical 2D shapes.

        Stochastic rounding is CUDA-graph-safe: ``rng_state`` is an int64 CUDA tensor
        ``[col_seed, col_offset, row_seed, row_offset]`` whose advancement the caller owns;
        the op only forwards single-element views, performing no host RNG.
        """
        B = get_rht_matrix(
            tuple(sign_vector), _device_key(A.device), torch.bfloat16, 16
        )
        _validate_grouped_hadamard_inputs(
            A,
            B,
            offsets,
            num_tensors,
            packed_sequence_length,
            hidden_size,
            shape_rep,
            logical_packed_length,
        )
        rng_state = _validate_rng_state(rng_state, A.device, enable_stochastic_rounding)

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
        if logical_packed_length is None:
            logical_packed_length = offsets[-1:]
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_rht_quantize_row_col_kernel[grid](
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
            STOCHASTIC_ROUNDING=enable_stochastic_rounding,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            logical_packed_length_ptr=logical_packed_length,
        )
        return qa_base, sfa_return, qd, sfd_return

    @triton_group_rht_quantize_row_col.register_fake
    def _(
        A,
        sign_vector,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        a_global_amax,
        d_global_amax,
        rng_state,
        enable_stochastic_rounding,
        logical_packed_length=None,
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
        sign_vector: list[int],
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        a_global_amax: torch.Tensor,
        d_global_amax: torch.Tensor,
        rng_state,
        enable_stochastic_rounding: bool,
        logical_packed_length: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(
            "triton_group_rht_quantize_row_col requires torch 2.10.0+ and triton installed"
        )
