"""
Token combine: routes tokens from expert-major layout back to source ranks
in their original rank-major order via symmetric memory.

This is the inverse of token dispatch. It is used for both:
- Forward combine: after expert computation, send output projections back
  to source ranks in their original token order.
- Backward dispatch: send gradients from expert-major layout back to the
  ranks that originally dispatched those tokens.

Both operations are identical — read from expert-major (padded) layout on
the local rank and push tokens to source ranks' rank-major (contiguous)
buffers.  This mirrors the SOL's _combine_send / gather_outputs.cu kernel,
which is shared by combine() and dispatch_bwd().
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
    SymmetricMemoryBufferManager,
)


@triton.jit
def _precompute_combine_offsets_kernel(
    all_expert_splits_ptr,
    expert_padded_offsets_ptr,
    read_offsets_ptr,
    write_offsets_ptr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
):
    """
    Precompute read and write offsets for the combine (gather) pass.

    The combine operation reads from the local expert-major buffer and pushes
    tokens back to source ranks in rank-major order.

    Inputs:
        all_expert_splits[src_rank, dst_rank, expert_idx]:
            tokens src_rank sent to expert_idx on dst_rank
        expert_padded_offsets[expert_idx]:
            starting offset for expert_idx in this rank's expert-major buffer

    Outputs:
        read_offsets[src_rank, expert_idx]:
            where in this rank's expert-major buffer to read tokens
            destined for src_rank for expert_idx
        write_offsets[src_rank, expert_idx]:
            where in src_rank's rank-major output buffer to write those tokens
    """
    if tl.program_id(0) == 0:
        for src_rank in range(world_size):
            for expert_idx in range(num_experts_per_rank):
                # --- Read offset ---
                # Position in this rank's expert-major buffer:
                #   expert_padded_offsets[expert_idx]
                #   + tokens from ranks 0..src_rank-1 for this expert on this rank
                read_offset = tl.load(expert_padded_offsets_ptr + expert_idx).to(
                    tl.int64
                )
                for prev_src in range(src_rank):
                    offset = (
                        prev_src * (world_size * num_experts_per_rank)
                        + rank * num_experts_per_rank
                        + expert_idx
                    )
                    read_offset += tl.load(all_expert_splits_ptr + offset)
                tl.store(
                    read_offsets_ptr + src_rank * num_experts_per_rank + expert_idx,
                    read_offset,
                )

                # --- Write offset ---
                # Position in src_rank's rank-major buffer:
                #   sum of all tokens src_rank sent to dst_ranks before this rank
                #   + sum of tokens src_rank sent to experts before expert_idx on this rank
                write_offset = tl.zeros([], dtype=tl.int64)

                # Tokens src_rank sent to all (dst_rank, expert) pairs before this rank
                for prev_dst in range(rank):
                    for e in range(num_experts_per_rank):
                        offset = (
                            src_rank * (world_size * num_experts_per_rank)
                            + prev_dst * num_experts_per_rank
                            + e
                        )
                        write_offset += tl.load(all_expert_splits_ptr + offset)

                # Tokens src_rank sent to experts before expert_idx on this rank
                for prev_e in range(expert_idx):
                    offset = (
                        src_rank * (world_size * num_experts_per_rank)
                        + rank * num_experts_per_rank
                        + prev_e
                    )
                    write_offset += tl.load(all_expert_splits_ptr + offset)

                tl.store(
                    write_offsets_ptr + src_rank * num_experts_per_rank + expert_idx,
                    write_offset,
                )


# ---------------------------------------------------------------------------
# Combine push kernel
# ---------------------------------------------------------------------------


@triton.jit
def _combine_all_to_all_kernel(
    input_ptr,
    all_expert_splits_ptr,
    read_offsets_ptr,
    write_offsets_ptr,
    output_ptrs,  # symmetric memory buffer pointers
    dim: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Push-based combine kernel: routes tokens from expert-major layout back
    to source ranks in rank-major order.

    Each program handles one destination rank (with BLOCKS_PER_REMOTE_RANK
    parallelism within). Reads from the local expert-major buffer and pushes
    tokens to the destination rank's output buffer.
    """
    dst_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    # Get output buffer pointer for dst_rank
    dst_ptr = tl.load(output_ptrs.to(tl.pointer_type(tl.uint64)) + dst_rank).to(
        tl.pointer_type(tl.bfloat16)
    )

    for expert_idx in range(num_experts_per_rank):
        # How many tokens did dst_rank send to this expert on this rank?
        num_tokens = tl.load(
            all_expert_splits_ptr
            + dst_rank * (world_size * num_experts_per_rank)
            + rank * num_experts_per_rank
            + expert_idx
        )

        if num_tokens > 0:
            read_offset = tl.load(
                read_offsets_ptr + dst_rank * num_experts_per_rank + expert_idx
            )
            write_offset = tl.load(
                write_offsets_ptr + dst_rank * num_experts_per_rank + expert_idx
            )

            total_elems = num_tokens * dim
            num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)
            for block_idx in tl.range(num_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_elems
                    data = tl.load(
                        input_ptr + read_offset * dim + offs,
                        mask=mask,
                        other=0.0,
                    )
                    tl.store(
                        dst_ptr + write_offset * dim + offs,
                        data,
                        mask=mask,
                    )


def _token_combine_launcher(
    input: torch.Tensor,
    all_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    num_output_tokens: int,
    dim: int,
    buffers: SymmetricMemoryBufferManager,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
) -> torch.Tensor:
    """
    Launcher for the token combine operation.

    Reads from the local expert-major buffer and pushes tokens back to
    source ranks' output buffers in rank-major order via symmetric memory.

    Used for both:
    - Forward combine (input = expert outputs after GEMM)
    - Backward dispatch (input = gradients in expert-major layout)

    Args:
        input: tensor in expert-major layout (max_rows, dim), bf16
        all_expert_splits: all-gathered expert splits (world_size, world_size, num_experts_per_rank)
        expert_padded_offsets: starting offset for each expert (num_experts_per_rank,)
        num_output_tokens: number of tokens in the output for this rank
        dim: feature dimension
        buffers: buffer manager (holds the output symmetric memory buffer)
        group: process group
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_experts_per_rank = all_expert_splits.shape[2]

    # Ensure output buffer is allocated and rendezvoused
    buffers.ensure_grad_input_buffer(num_output_tokens, dim, input.device, group)
    output = buffers.grad_input[:num_output_tokens]
    output_ptrs = buffers._grad_input_hdl.buffer_ptrs_dev

    # Flatten all_expert_splits for Triton kernel access
    all_expert_splits_flat = all_expert_splits.contiguous().view(-1)

    # Allocate offset tables
    read_offsets = torch.empty(
        world_size,
        num_experts_per_rank,
        dtype=torch.int64,
        device=input.device,
    )
    write_offsets = torch.empty(
        world_size,
        num_experts_per_rank,
        dtype=torch.int64,
        device=input.device,
    )

    # Phase 1: Precompute offsets
    _precompute_combine_offsets_kernel[(1, 1, 1)](
        all_expert_splits_flat,
        expert_padded_offsets,
        read_offsets,
        write_offsets,
        rank=rank,
        world_size=world_size,
        num_experts_per_rank=num_experts_per_rank,
    )

    # Phase 2: Push tokens to destination ranks
    num_blocks = world_size * BLOCKS_PER_REMOTE_RANK
    _combine_all_to_all_kernel[(num_blocks, 1, 1)](
        input,
        all_expert_splits_flat,
        read_offsets,
        write_offsets,
        output_ptrs,
        dim=dim,
        num_experts_per_rank=num_experts_per_rank,
        rank=rank,
        world_size=world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output


class SynclessTokenCombine(torch.autograd.Function):
    """Autograd function for token combine (inverse of token dispatch).

    Routes tokens from expert-major layout back to source ranks in their
    original rank-major order via symmetric memory push writes.

    Used for both:
    - Forward combine: after expert computation, send output projections back
    - Backward dispatch: send gradients from expert-major layout back
    """

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        input: torch.Tensor,
        all_expert_splits: torch.Tensor,
        expert_padded_offsets: torch.Tensor,
        num_output_tokens: int,
        group: dist.ProcessGroup = dist.group.WORLD,
        buffer_manager: SymmetricMemoryBufferManager = None,
    ):
        """
        Routes tokens from expert-major layout back to source ranks in
        rank-major order.

        Args:
            input: bf16 tensor in expert-major layout (max_rows, dim).
            all_expert_splits: all-gathered expert splits
                (world_size, world_size, num_experts_per_rank).
            expert_padded_offsets: starting offset for each expert
                (num_experts_per_rank,).
            num_output_tokens: number of tokens in the output for this rank.
            group: process group to scope the collective.
            buffer_manager: optional buffer manager for reusing buffers.
        """
        from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
            get_buffer_manager,
        )

        buffers = buffer_manager or get_buffer_manager()
        dim = input.shape[1]

        output = _token_combine_launcher(
            input=input,
            all_expert_splits=all_expert_splits,
            expert_padded_offsets=expert_padded_offsets,
            num_output_tokens=num_output_tokens,
            dim=dim,
            buffers=buffers,
            group=group,
        )

        ctx.all_expert_splits = all_expert_splits
        ctx.expert_padded_offsets = expert_padded_offsets
        ctx.num_input_tokens = input.shape[0]
        ctx.dim = dim
        ctx.group = group
        ctx.buffer_manager = buffers

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("backward support not yet implemented")


# Alias
token_combine = SynclessTokenCombine.apply
