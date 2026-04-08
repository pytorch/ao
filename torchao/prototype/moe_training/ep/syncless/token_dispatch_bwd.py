import torch
import torch.distributed as dist
import triton
import triton.language as tl

from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
    SymmetricMemoryBufferManager,
)


# ---------------------------------------------------------------------------
# Precompute backward offsets kernel
# ---------------------------------------------------------------------------


@triton.jit
def _precompute_backward_offsets_kernel(
    all_expert_splits_ptr,
    expert_padded_offsets_ptr,
    read_offsets_ptr,
    write_offsets_ptr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
):
    """
    Precompute read and write offsets for the backward (combine) pass.

    The backward reverses the forward's routing: each rank reads from its own
    expert-major grad_output and pushes gradients back to source ranks in
    rank-major order.

    Inputs:
        all_expert_splits[src_rank, dst_rank, expert_idx]:
            tokens src_rank sent to expert_idx on dst_rank
        expert_padded_offsets[expert_idx]:
            starting offset for expert_idx in this rank's expert-major buffer

    Outputs:
        read_offsets[src_rank, expert_idx]:
            where in this rank's grad_output to read tokens from src_rank for expert_idx
        write_offsets[src_rank, expert_idx]:
            where in src_rank's grad_input to write those tokens
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
                # Position in src_rank's rank-major input buffer:
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
# Backward push kernel
# ---------------------------------------------------------------------------


@triton.jit
def _grad_all_to_all_kernel(
    grad_output_ptr,
    all_expert_splits_ptr,
    read_offsets_ptr,
    write_offsets_ptr,
    grad_input_ptrs,  # symmetric memory buffer pointers
    dim: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Push-based backward kernel: reverses the forward dispatch routing.

    Each program handles one source rank (with BLOCKS_PER_REMOTE_RANK
    parallelism within). Reads bf16 gradients from local expert-major
    grad_output and pushes them to the source rank's grad_input buffer
    in rank-major order.
    """
    src_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    # Get grad_input buffer pointer for src_rank
    dst_ptr = tl.load(grad_input_ptrs.to(tl.pointer_type(tl.uint64)) + src_rank).to(
        tl.pointer_type(tl.bfloat16)
    )

    for expert_idx in range(num_experts_per_rank):
        # How many tokens did src_rank send to this expert on this rank?
        num_tokens = tl.load(
            all_expert_splits_ptr
            + src_rank * (world_size * num_experts_per_rank)
            + rank * num_experts_per_rank
            + expert_idx
        )

        if num_tokens > 0:
            read_offset = tl.load(
                read_offsets_ptr + src_rank * num_experts_per_rank + expert_idx
            )
            write_offset = tl.load(
                write_offsets_ptr + src_rank * num_experts_per_rank + expert_idx
            )

            total_elems = num_tokens * dim
            num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)
            for block_idx in tl.range(num_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_elems
                    data = tl.load(
                        grad_output_ptr + read_offset * dim + offs,
                        mask=mask,
                        other=0.0,
                    )
                    tl.store(
                        dst_ptr + write_offset * dim + offs,
                        data,
                        mask=mask,
                    )


# ---------------------------------------------------------------------------
# Backward launcher
# ---------------------------------------------------------------------------


def _token_dispatch_backward_launcher(
    grad_output: torch.Tensor,
    all_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    num_input_tokens: int,
    dim: int,
    buffers: SymmetricMemoryBufferManager,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
) -> torch.Tensor:
    """
    Launcher for the backward pass of token dispatch.

    Reverses the forward routing: reads bf16 gradients from the local
    expert-major grad_output buffer and pushes them back to source ranks'
    grad_input buffers in rank-major order via symmetric memory.

    Args:
        grad_output: gradient tensor in expert-major layout (max_output_rows, dim), bf16
        all_expert_splits: all-gathered expert splits (world_size, world_size, num_experts_per_rank)
        expert_padded_offsets: starting offset for each expert (num_experts_per_rank,)
        num_input_tokens: number of tokens in the original forward input for this rank
        dim: feature dimension
        buffers: buffer manager (holds the grad_input symmetric memory buffer)
        group: process group
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_experts_per_rank = all_expert_splits.shape[2]

    # Ensure grad_input buffer is allocated and rendezvoused
    buffers.ensure_grad_input_buffer(num_input_tokens, dim, grad_output.device, group)
    grad_input = buffers.grad_input[:num_input_tokens]
    grad_input_ptrs = buffers._grad_input_hdl.buffer_ptrs_dev

    # Flatten all_expert_splits for Triton kernel access
    all_expert_splits_flat = all_expert_splits.contiguous().view(-1)

    # Allocate offset tables
    read_offsets = torch.empty(
        world_size,
        num_experts_per_rank,
        dtype=torch.int64,
        device=grad_output.device,
    )
    write_offsets = torch.empty(
        world_size,
        num_experts_per_rank,
        dtype=torch.int64,
        device=grad_output.device,
    )

    # Phase 1: Precompute offsets
    _precompute_backward_offsets_kernel[(1, 1, 1)](
        all_expert_splits_flat,
        expert_padded_offsets,
        read_offsets,
        write_offsets,
        rank=rank,
        world_size=world_size,
        num_experts_per_rank=num_experts_per_rank,
    )

    # Phase 2: Push gradients to source ranks
    num_blocks = world_size * BLOCKS_PER_REMOTE_RANK
    _grad_all_to_all_kernel[(num_blocks, 1, 1)](
        grad_output,
        all_expert_splits_flat,
        read_offsets,
        write_offsets,
        grad_input_ptrs,
        dim=dim,
        num_experts_per_rank=num_experts_per_rank,
        rank=rank,
        world_size=world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return grad_input
