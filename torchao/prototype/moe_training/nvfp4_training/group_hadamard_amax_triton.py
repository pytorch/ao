"""Grouped Randomized Hadamard Transform (RHT) global amax reduction.

Grouped analog of ``triton_rht_amax`` (hadamard_amax_triton.py); corresponds to
TransformerEngine's ``nvte_group_hadamard_transform_amax_graph_safe``. Computes,
per expert group of a row-concatenated packed tensor, the post-RHT columnwise amax
and the raw rowwise amax without materializing the transformed output.

The pure-torch reference oracle lives in the test file. Semantics match the
non-grouped ``triton_rht_amax`` (single sign vector):
``col_amax[g] = max|RHT(A_g.T)|`` and ``row_amax[g] = max|A_g|``.
"""

import torch
import torch.nn.functional as F
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

_DEFAULT_SCALING_TYPE = (
    int(F.ScalingType.TensorWise) if hasattr(F, "ScalingType") else 0
)

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
        _compute_pid,
        _device_key,
        get_rht_matrix,
        prepare_for_cuda_graph,
    )

    # Below this average rows-per-group the tiled kernel wins (too few tiles per
    # group to amortize the persistent prologue); above it the per-group-CTA
    # persistent kernel recovers single-tensor bandwidth. Measured crossover ~1K.
    _PERSISTENT_MIN_AVG_ROWS = 1024

    # Autotune space for the persistent kernel (mirrors the single-tensor kernel's
    # tile set). 64x128 typically wins; 128x128 has 2x the register footprint.
    _PERSISTENT_TILE_SHAPES = [(128, 32), (128, 64), (128, 128), (64, 64), (64, 128)]
    _PERSISTENT_CONFIGS = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_STAGES": ns},
            num_warps=nw,
            num_stages=ns,
        )
        for (bm, bn) in _PERSISTENT_TILE_SHAPES
        for ns in (3, 4)
        for nw in (4, 8)
    ]

    @triton.jit
    def _atomic_max_2d(values, output_ptr, group_idx):
        amax = tl.max(tl.max(values, axis=1), axis=0)
        amax_has_nan = tl.max(
            tl.max((values != values).to(tl.int32), axis=1),
            axis=0,
        )
        amax = tl.where(amax_has_nan != 0, float("nan"), amax)
        tl.atomic_max(output_ptr + group_idx, amax.to(tl.float32))

    @triton.jit
    def _group_rht_amax_triton_kernel(
        a_ptr,
        b_ptr,
        offsets_ptr,
        global_amax_row_ptr,
        global_amax_col_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
        logical_packed_length_ptr,
    ):
        """Grouped RHT columnwise and direct rowwise amax reduction."""
        VARYING_FIRST_DIM: tl.constexpr = 1

        num_tiles_token = tl.cdiv(M, BLOCK_M)
        num_tiles_hidden = tl.cdiv(N, BLOCK_N)
        tile_idx = tl.program_id(0)
        token_tile_idx = tile_idx // num_tiles_hidden
        hidden_tile_idx = tile_idx - token_tile_idx * num_tiles_hidden

        if SHAPE_REP == VARYING_FIRST_DIM:
            group_idx = _get_group_idx_binary(
                token_tile_idx * BLOCK_M,
                offsets_ptr,
                num_tensors,
            )
        else:
            group_idx = token_tile_idx // (num_tiles_token // num_tensors)

        offsets_m = token_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = hidden_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        logical_packed_length = tl.load(logical_packed_length_ptr)
        a = tl.load(
            a_ptr + offsets_m[:, None] * N + offsets_n[None, :],
            mask=offsets_m[:, None] < logical_packed_length,
            other=0.0,
        )

        rht_offsets = (
            tl.arange(0, RHT_SIZE)[:, None] * RHT_SIZE + tl.arange(0, RHT_SIZE)[None, :]
        )
        hadamard = tl.load(b_ptr + rht_offsets)
        a_t = tl.trans(a)
        a_t_reshape = tl.reshape(a_t, [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
        a_t_rht = tl.dot(a_t_reshape, hadamard).to(tl.bfloat16)

        _atomic_max_2d(
            tl.abs(a_t_rht),
            global_amax_col_ptr,
            group_idx,
        )
        _atomic_max_2d(
            tl.abs(a.to(tl.float32)),
            global_amax_row_ptr,
            group_idx,
        )

    @triton.autotune(configs=_PERSISTENT_CONFIGS, key=["M", "N"])
    @triton.jit
    def _pergroup_cta_rht_amax_kernel(
        a_ptr,
        b_ptr,
        offsets_ptr,
        global_amax_row_ptr,
        global_amax_col_ptr,
        M,
        N,
        CTAS_PER_GROUP: tl.constexpr,
        RHT_SIZE: tl.constexpr,
        GROUP_SIZE_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        """Persistent grouped RHT amax: each CTA is bound to ONE expert group.

        Generalizes the single-tensor _hadamard_amax_kernel: CTAS_PER_GROUP CTAs
        cooperate over a group's tiles, each keeping an elementwise cumulative max
        (no in-loop reduction, so warp_specialize stays on) and doing ONE atomic per
        CTA. Groups are read purely from ``offsets`` (cumulative row-ends), so this
        is correct for both SAME_BOTH_DIMS and VARYING_FIRST_DIM. Recovers
        single-tensor bandwidth for large groups.
        """
        a_desc = tl.make_tensor_descriptor(
            a_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[RHT_SIZE, RHT_SIZE],
            strides=[RHT_SIZE, 1],
            block_shape=[RHT_SIZE, RHT_SIZE],
        )
        hadamard = b_desc.load([0, 0])

        pid = tl.program_id(0)
        g = pid // CTAS_PER_GROUP
        local = pid % CTAS_PER_GROUP

        # Masked load avoids the OOB read of offsets_ptr[-1] when g == 0.
        g_start = tl.load(offsets_ptr + g - 1, mask=g > 0, other=0)
        g_end = tl.load(offsets_ptr + g)
        num_pid_m = tl.cdiv(g_end - g_start, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_N * num_pid_m
        num_tiles = num_pid_m * num_pid_n

        cum_col = tl.zeros((BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE), dtype=tl.float32)
        cum_row = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for tile_id in tl.range(
            local,
            num_tiles,
            CTAS_PER_GROUP,
            flatten=False,
            warp_specialize=True,
            num_stages=NUM_STAGES,
        ):
            pid_n, pid_m = _compute_pid(
                tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N
            )
            a = a_desc.load([g_start + pid_m * BLOCK_M, pid_n * BLOCK_N])
            a_t = tl.trans(a)
            a_t_r = tl.reshape(a_t, [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
            a_t_rht = tl.dot(a_t_r, hadamard).to(tl.bfloat16)
            cum_col = tl.maximum(
                cum_col, tl.abs(a_t_rht), propagate_nan=tl.PropagateNan.ALL
            )
            cum_row = tl.maximum(
                cum_row, tl.abs(a.to(tl.float32)), propagate_nan=tl.PropagateNan.ALL
            )

        col = tl.max(tl.max(cum_col, axis=1), axis=0)
        col_nan = tl.max(tl.max((cum_col != cum_col).to(tl.int32), axis=1), axis=0)
        col = tl.where(col_nan != 0, float("nan"), col)
        tl.atomic_max(global_amax_col_ptr + g, col.to(tl.float32))

        row = tl.max(tl.max(cum_row, axis=1), axis=0)
        row_nan = tl.max(tl.max((cum_row != cum_row).to(tl.int32), axis=1), axis=0)
        row = tl.where(row_nan != 0, float("nan"), row)
        tl.atomic_max(global_amax_row_ptr + g, row.to(tl.float32))

    @torch.library.custom_op("torchao::triton_group_rht_amax", mutates_args=())
    def triton_group_rht_amax(
        A: torch.Tensor,
        sign_vector: List[int],
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        scaling_type: int = int(F.ScalingType.TensorWise),
        logical_packed_length: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-group RHT columnwise amax and raw rowwise amax (grouped, graph-safe).

        Args:
            A: packed (sum_M, N) bfloat16 tensor, row-major. Groups are concatenated
                along the row dimension; each group's M must be divisible by 16 and
                N divisible by 128.
            sign_vector: Sign vector used to construct the cached 16x16 RHT matrix.
            offsets: int32 cumulative row-end offsets, one per group.
            num_tensors: number of expert groups.
            packed_sequence_length: allocated row capacity of A.
            hidden_size: number of columns in A.
            shape_rep: grouped shape representation.
            scaling_type: int encoding of F.ScalingType. Only TensorWise is supported.
            logical_packed_length: one-element int32 CUDA tensor containing the
                valid padded row count. Rows beyond it are storage capacity only.

        Returns:
            Tuple of (col_amax, row_amax), each (num_tensors,) float32:
              - col_amax[g] = max(abs(RHT(A_g.T))).
              - row_amax[g] = max(abs(A_g)).
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
        if scaling_type != int(F.ScalingType.TensorWise):
            raise ValueError(
                f"scaling_type={scaling_type!r} is not supported; "
                "only ScalingType.TensorWise is implemented."
            )
        row_amax = torch.zeros(
            (num_tensors,),
            dtype=torch.float32,
            device=A.device,
        )
        col_amax = torch.zeros(
            (num_tensors,),
            dtype=torch.float32,
            device=A.device,
        )

        m, n = A.shape
        rht_size = B.shape[0]
        if logical_packed_length is None:
            logical_packed_length = offsets[-1:]

        # Fast path: for large groups the per-group-CTA persistent kernel recovers
        # single-tensor bandwidth (~2x the tiled kernel). It bins CTAs by group, so
        # it needs at least one CTA per group; below _PERSISTENT_MIN_AVG_ROWS the
        # tiled kernel wins. Group membership is read from offsets, valid for both
        # shape_reps.
        num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
        if num_tensors <= num_sms and (m // num_tensors) >= _PERSISTENT_MIN_AVG_ROWS:
            ctas_per_group = num_sms // num_tensors
            workspace = prepare_for_cuda_graph(
                A.device, sign_vectors=(tuple(sign_vector),)
            )
            triton.set_allocator(
                lambda size, align, stream: workspace[: max(size, 1)]
            )
            try:
                _pergroup_cta_rht_amax_kernel[(num_tensors * ctas_per_group,)](
                    A,
                    B,
                    offsets,
                    row_amax,
                    col_amax,
                    m,
                    n,
                    CTAS_PER_GROUP=ctas_per_group,
                    RHT_SIZE=rht_size,
                    GROUP_SIZE_N=8,
                )
            finally:
                triton.set_allocator(None)
            return col_amax, row_amax

        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_rht_amax_triton_kernel[grid](
            A,
            B,
            offsets,
            row_amax,
            col_amax,
            m,
            n,
            num_tensors=num_tensors,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=rht_size,
            logical_packed_length_ptr=logical_packed_length,
            num_warps=8,
            num_stages=3,
        )
        return col_amax, row_amax

    @triton_group_rht_amax.register_fake
    def _(
        A,
        sign_vector,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        scaling_type=int(F.ScalingType.TensorWise),
        logical_packed_length=None,
    ):
        col_amax = A.new_empty((num_tensors,), dtype=torch.float32)
        row_amax = A.new_empty((num_tensors,), dtype=torch.float32)
        return col_amax, row_amax

else:

    def triton_group_rht_amax(
        A: torch.Tensor,
        sign_vector: list[int],
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        scaling_type: int = _DEFAULT_SCALING_TYPE,
        logical_packed_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_rht_amax requires torch 2.10.0+ and triton installed"
        )
