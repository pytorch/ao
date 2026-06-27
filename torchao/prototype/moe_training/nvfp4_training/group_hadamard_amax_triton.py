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
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        prepare_for_cuda_graph,
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
        NUM_STAGES: tl.constexpr,
    ):
        """Grouped fused RHT columnwise and direct rowwise NVFP4 quantization."""
        # Enum for shape representation; only VARYING_FIRST_DIM and SAME_BOTH_DIMS are supported
        VARYING_FIRST_DIM: tl.constexpr = 1

        # Create TMA descriptors in-kernel from raw pointers and shape/stride
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[16, 16],
            strides=[16, 1],
            block_shape=[16, 16],
        )

        num_tiles_token = tl.cdiv(M, BLOCK_M)
        num_tiles_hidden = tl.cdiv(N, BLOCK_N)
        token_tile_idx = tl.program_id(0)

        if SHAPE_REP == VARYING_FIRST_DIM:
            group_idx = _get_group_idx_binary(
                token_tile_idx * BLOCK_M * N,
                offsets_ptr,
                num_tensors,
            )
        else:
            group_idx = token_tile_idx // (num_tiles_token // num_tensors)

        # Load random hadamard matrix once
        hadamard = b_desc.load([0, 0])

        cumulative_rht_amax = tl.zeros(
            (BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE), dtype=tl.float32
        )
        cumulative_a_amax = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for hidden_tile_idx in tl.range(
            0,
            num_tiles_hidden,
            flatten=False,
            warp_specialize=True,
            num_stages=NUM_STAGES,
        ):
            # Load A (BLOCK_M, BLOCK_N)
            a = a_desc.load([token_tile_idx * BLOCK_M, hidden_tile_idx * BLOCK_N])
            # Transpose A_t (BLOCK_N, BLOCK_M)
            a_t = tl.trans(a)
            # Reshape to A_t_reshape
            a_t_reshape = tl.reshape(a_t, [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
            # Cast to bfloat16 like regular matmul output
            a_t_rht = tl.dot(a_t_reshape, hadamard).to(tl.bfloat16)

            # Update cumulative max at tile level to avoid failing
            # TritonGPUAutomaticWarpSpecialization MLIR pass
            cumulative_rht_amax = tl.maximum(
                cumulative_rht_amax,
                tl.abs(a_t_rht),
                propagate_nan=tl.PropagateNan.ALL,
            )
            cumulative_a_amax = tl.maximum(
                cumulative_a_amax,
                tl.abs(a.to(tl.float32)),
                propagate_nan=tl.PropagateNan.ALL,
            )

        # Get scalar max for this block and update global max with atomic max operation
        _atomic_max_2d(
            cumulative_rht_amax,
            global_amax_col_ptr,
            group_idx,
        )
        _atomic_max_2d(
            cumulative_a_amax,
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

        # Instead of a persistent kernel, divide batch dimension into blocks by BLOCK_M.
        # Since number of tokens per expert is a multiple of BLOCK_M, each block will
        # be fully contained in a single expert group. Each block will handle the
        # entire hidden dimension, so warp specialization will have non-trivial pipeline.
        # A persistent kernel requires global atomic max within a for-loop, which is not
        # supported by Triton. An occupancy based approach requires a single global
        # atomic after the for-loop at end of the kernel.
        grid = (triton.cdiv(m, BLOCK_M),)

        # Prepare a kernel workspace for TMA descriptors.
        if hasattr(triton, "set_allocator"):
            workspace = prepare_for_cuda_graph(A.device)
            triton.set_allocator(lambda size, align, stream: workspace[: max(size, 1)])

        try:
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
                NUM_STAGES=3,
                num_warps=8,
            )
        finally:
            if hasattr(triton, "set_allocator"):
                triton.set_allocator(None)
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
