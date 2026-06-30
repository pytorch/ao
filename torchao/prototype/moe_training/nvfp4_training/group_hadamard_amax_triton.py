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
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
        BLOCK_M,
        BLOCK_N,
        _get_group_idx_binary,
        _validate_grouped_hadamard_inputs,
    )
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
                token_tile_idx * BLOCK_M * N,
                offsets_ptr,
                num_tensors,
            )
        else:
            group_idx = token_tile_idx // (num_tiles_token // num_tensors)

        offsets_m = token_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = hidden_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(a_ptr + offsets_m[:, None] * N + offsets_n[None, :])

        rht_offsets = (
            tl.arange(0, RHT_SIZE)[:, None] * RHT_SIZE
            + tl.arange(0, RHT_SIZE)[None, :]
        )
        hadamard = tl.load(b_ptr + rht_offsets)
        a_t = tl.trans(a)
        a_t_reshape = tl.reshape(
            a_t, [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE]
        )
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

    @torch.library.custom_op("torchao::triton_group_rht_amax", mutates_args=())
    def triton_group_rht_amax(
        A: torch.Tensor,
        B: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        scaling_type: int = int(F.ScalingType.TensorWise),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-group RHT columnwise amax and raw rowwise amax (grouped, graph-safe).

        Args:
            A: packed (sum_M, N) bfloat16 tensor, row-major. Groups are concatenated
                along the row dimension; each group's M must be divisible by 16 and
                N divisible by 128.
            B: (16, 16) bfloat16 RHT matrix.
            offsets: int64 cumulative element offsets for each group start.
            num_tensors: number of expert groups.
            packed_sequence_length: total number of rows in A.
            hidden_size: number of columns in A.
            shape_rep: grouped shape representation.
            scaling_type: int encoding of F.ScalingType. Only TensorWise is supported.

        Returns:
            Tuple of (col_amax, row_amax), each (num_tensors,) float32:
              - col_amax[g] = max(abs(RHT(A_g.T))).
              - row_amax[g] = max(abs(A_g)).
        """
        _validate_grouped_hadamard_inputs(
            A,
            B,
            offsets,
            num_tensors,
            packed_sequence_length,
            hidden_size,
            shape_rep,
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
            num_warps=8,
            num_stages=3,
        )
        return col_amax, row_amax

    @triton_group_rht_amax.register_fake
    def _(
        A,
        B,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        scaling_type=int(F.ScalingType.TensorWise),
    ):
        col_amax = A.new_empty((num_tensors,), dtype=torch.float32)
        row_amax = A.new_empty((num_tensors,), dtype=torch.float32)
        return col_amax, row_amax

else:

    def triton_group_rht_amax(
        A: torch.Tensor,
        B: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        scaling_type: int = _DEFAULT_SCALING_TYPE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_rht_amax requires torch 2.10.0+ and triton installed"
        )
