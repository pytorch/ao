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
from torch._C._distributed_c10d import (
    _resolve_process_group,
)
from torch.library import triton_op, wrap_triton

from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
    SymmetricMemoryBufferManager,
)


class SynclessTokenCombine(torch.autograd.Function):
    """Autograd function for token combine (inverse of token dispatch).

    Routes tokens from expert-major layout back to source ranks in their
    original rank-major order via symmetric memory push writes.

    Used for both:
    - Forward combine: after expert computation, send output projections back
    - Backward dispatch: send gradients from expert-major layout back
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        all_expert_splits: torch.Tensor,
        expert_padded_offsets: torch.Tensor,
        num_output_tokens: int,
        group: dist.ProcessGroup = dist.group.WORLD,
        buffer_manager: SymmetricMemoryBufferManager = None,
        token_alignment: int = 128,
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
            token_alignment: expert token group alignment (default 128).
        """
        from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
            get_buffer_manager,
        )

        buffers = buffer_manager or get_buffer_manager()
        dim = input.shape[1]

        # Ensure buffer is allocated before passing individual attributes
        buffers.ensure_bf16_buffer(dim, input.device, group)

        # Use pre-allocated output buffer and device pointers
        output = buffers.bf16_buffer[:num_output_tokens]

        _token_combine_launcher(
            input=input,
            all_expert_splits=all_expert_splits,
            expert_padded_offsets=expert_padded_offsets,
            num_output_tokens=num_output_tokens,
            dim=dim,
            output=output,
            output_dev_ptrs=buffers._bf16_buffer_hdl.buffer_ptrs_dev,
            group_name=group.group_name,
        )

        ctx.all_expert_splits = all_expert_splits
        ctx.expert_padded_offsets = expert_padded_offsets
        ctx.num_input_tokens = input.shape[0]
        ctx.dim = dim
        ctx.group = group
        ctx.buffer_manager = buffers
        ctx.token_alignment = token_alignment

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse the forward combine routing.

        Reads bf16 gradients from the local rank-major grad_output and
        pushes them back to expert-major layout on destination ranks via
        symmetric memory (bf16 dispatch without MXFP8 quantization).
        """
        # Ensure buffer is allocated before passing individual attributes
        group = _resolve_process_group(ctx.group.group_name)
        ctx.buffer_manager.ensure_bf16_buffer(ctx.dim, grad_output.device, group)

        grad_input = ctx.buffer_manager.bf16_buffer[: ctx.num_input_tokens]

        _token_combine_bwd_launcher(
            input=grad_output,
            all_expert_splits=ctx.all_expert_splits,
            expert_padded_offsets=ctx.expert_padded_offsets,
            num_output_tokens=ctx.num_input_tokens,
            dim=ctx.dim,
            output=grad_input,
            output_dev_ptrs=ctx.buffer_manager._bf16_buffer_hdl.buffer_ptrs_dev,
            group_name=ctx.group.group_name,
            token_alignment=ctx.token_alignment,
        )
        return grad_input, None, None, None, None, None, None


# Alias
token_combine = SynclessTokenCombine.apply


# Forward combine: expert-major back to rank-major
@triton_op("torchao::token_combine", mutates_args={"output"})
def _token_combine_launcher(
    input: torch.Tensor,
    all_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    num_output_tokens: int,
    dim: int,
    output: torch.Tensor,
    output_dev_ptrs: int,
    group_name: str = "WORLD",
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
        output_buffer: the allocated bf16 symmetric memory buffer
        output_dev_ptrs: device pointers to remote ranks' buffers
        group_name: name of the process group ("WORLD" for default)
    """
    # Look up process group by name
    group = _resolve_process_group(group_name)

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_experts_per_rank = all_expert_splits.shape[2]

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
    wrap_triton(_precompute_combine_offsets_kernel)[(1, 1, 1)](
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
    wrap_triton(_combine_all_to_all_kernel)[(num_blocks, 1, 1)](
        input,
        all_expert_splits_flat,
        read_offsets,
        write_offsets,
        output_dev_ptrs,
        dim=dim,
        num_experts_per_rank=num_experts_per_rank,
        rank=rank,
        world_size=world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
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


@triton_op("torchao::token_combine_bwd", mutates_args={"output"})
def _token_combine_bwd_launcher(
    input: torch.Tensor,
    all_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    num_output_tokens: int,
    dim: int,
    output: torch.Tensor,
    output_dev_ptrs: int,
    group_name: str = "WORLD",
    token_alignment: int = 128,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
) -> None:
    """
    Launcher for the bf16 dispatch operation (combine backward).

    Reads from this rank's rank-major grad_output and pushes tokens to
    destination ranks' expert-major grad_input buffers via symmetric memory.

    Args:
        input: bf16 tensor in rank-major layout (num_tokens, dim)
        all_expert_splits: all-gathered expert splits
            (world_size, world_size, num_experts_per_rank)
        expert_padded_offsets: starting offset for each expert on this rank
            (num_experts_per_rank,)
        num_output_tokens: number of rows in the expert-major output
        dim: feature dimension
        output_buffer: the allocated bf16 symmetric memory buffer
        output_dev_ptrs: device pointers to remote ranks' buffers
        group_name: name of the process group ("WORLD" for default)
        token_alignment: expert token group alignment (default 128)
    """
    # Look up process group by name
    group = _resolve_process_group(group_name)

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_experts_per_rank = all_expert_splits.shape[2]

    # TODO: do we need the contiguous calls here?
    # input_expert_splits: tokens this rank sent to each (dst_rank, expert_idx)
    input_expert_splits = all_expert_splits[rank, :, :].contiguous()

    # output_expert_splits: tokens each src_rank sent to each expert on THIS rank
    output_expert_splits = all_expert_splits[:, rank, :].contiguous()

    # Compute write offsets for dispatch direction
    all_expert_splits_flat = all_expert_splits.contiguous().view(-1)
    write_offsets = torch.empty(
        world_size,
        num_experts_per_rank,
        dtype=torch.int64,
        device=input.device,
    )
    wrap_triton(_precompute_combine_bwd_write_offsets_kernel)[(1, 1, 1)](
        all_expert_splits_flat,
        write_offsets,
        rank=rank,
        world_size=world_size,
        num_experts_per_rank=num_experts_per_rank,
        TOKEN_ALIGNMENT=token_alignment,
    )

    # Push grad_output to expert-major layout on destination ranks
    num_blocks = world_size * BLOCKS_PER_REMOTE_RANK
    wrap_triton(_token_combine_bwd_kernel)[(num_blocks, 1, 1)](
        input,
        input_expert_splits.view(-1),
        write_offsets,
        output_dev_ptrs,
        expert_padded_offsets,
        output_expert_splits.view(-1),
        dim=dim,
        num_experts_per_rank=num_experts_per_rank,
        rank=rank,
        world_size=world_size,
        TOKEN_ALIGNMENT=token_alignment,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )


# combine backward: rank-major to expert-major
@triton.jit
def _precompute_combine_bwd_write_offsets_kernel(
    all_expert_splits_ptr,
    write_offsets_ptr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    TOKEN_ALIGNMENT: tl.constexpr,
):
    """
    Precompute write offsets for the bf16 dispatch (combine backward).

    write_offsets[dst_rank, expert_idx] = where this rank writes in
    dst_rank's expert-major buffer for expert_idx.
    """
    if tl.program_id(0) == 0:
        for dst_rank in range(world_size):
            for expert_idx in range(num_experts_per_rank):
                # Expert base offset on dst_rank: sum of padded sizes for
                # all previous experts on dst_rank.
                expert_base_offset = tl.zeros([], dtype=tl.int64)
                for prev_expert in range(expert_idx):
                    prev_expert_total = tl.zeros([], dtype=tl.int64)
                    for src_rank in range(world_size):
                        offset = (
                            src_rank * (world_size * num_experts_per_rank)
                            + dst_rank * num_experts_per_rank
                            + prev_expert
                        )
                        prev_expert_total += tl.load(all_expert_splits_ptr + offset)
                    padded_size = (
                        (prev_expert_total + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT
                    ) * TOKEN_ALIGNMENT
                    expert_base_offset += padded_size

                # Within-expert offset: tokens from ranks before this rank
                within_expert_offset = tl.zeros([], dtype=tl.int64)
                for prev_src in range(rank):
                    offset = (
                        prev_src * (world_size * num_experts_per_rank)
                        + dst_rank * num_experts_per_rank
                        + expert_idx
                    )
                    within_expert_offset += tl.load(all_expert_splits_ptr + offset)

                write_offset = expert_base_offset + within_expert_offset
                tl.store(
                    write_offsets_ptr + dst_rank * num_experts_per_rank + expert_idx,
                    write_offset,
                )


@triton.jit
def _token_combine_bwd_kernel(
    input_ptr,
    input_expert_splits_ptr,
    write_offsets_ptr,
    output_ptrs,
    expert_padded_offsets_ptr,
    output_expert_splits_ptr,
    dim: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    TOKEN_ALIGNMENT: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Push-based bf16 dispatch kernel for combine backward.

    Reads from this rank's rank-major grad_output and pushes tokens to
    destination ranks' expert-major grad_input buffers via symmetric memory.
    Also zero-fills padding regions in this rank's own expert-major buffer.
    """
    dst_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    dst_ptr = tl.load(output_ptrs.to(tl.pointer_type(tl.uint64)) + dst_rank).to(
        tl.pointer_type(tl.bfloat16)
    )

    # Compute base read offset for this dst_rank
    base_local_read_offset = tl.zeros([], dtype=tl.int64)
    for prev_dst_rank in range(dst_rank):
        for e in range(num_experts_per_rank):
            base_local_read_offset += tl.load(
                input_expert_splits_ptr + prev_dst_rank * num_experts_per_rank + e
            )

    local_read_offset = base_local_read_offset

    for expert_idx in range(num_experts_per_rank):
        my_tokens_to_expert = tl.load(
            input_expert_splits_ptr + dst_rank * num_experts_per_rank + expert_idx
        )

        if my_tokens_to_expert > 0:
            write_offset = tl.load(
                write_offsets_ptr + dst_rank * num_experts_per_rank + expert_idx
            )

            total_elems = my_tokens_to_expert * dim
            num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)
            for block_idx in tl.range(num_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_elems
                    data = tl.load(
                        input_ptr + local_read_offset * dim + offs,
                        mask=mask,
                        other=0.0,
                    )
                    tl.store(
                        dst_ptr + write_offset * dim + offs,
                        data,
                        mask=mask,
                    )

        local_read_offset += my_tokens_to_expert

    # Zero-fill padding regions in own expert-major buffer
    if dst_rank == rank and block_offset == 0:
        own_output_ptr = tl.load(output_ptrs.to(tl.pointer_type(tl.uint64)) + rank).to(
            tl.pointer_type(tl.bfloat16)
        )

        for expert_idx in range(num_experts_per_rank):
            actual_tokens = tl.zeros([], dtype=tl.int64)
            for src_rank in range(world_size):
                tokens = tl.load(
                    output_expert_splits_ptr
                    + src_rank * num_experts_per_rank
                    + expert_idx
                )
                actual_tokens += tokens

            expert_start = tl.load(expert_padded_offsets_ptr + expert_idx)
            actual_end = expert_start + actual_tokens

            if expert_idx < num_experts_per_rank - 1:
                padded_end = tl.load(expert_padded_offsets_ptr + expert_idx + 1)
            else:
                padded_size = (
                    (actual_tokens + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT
                ) * TOKEN_ALIGNMENT
                padded_end = expert_start + padded_size

            if actual_end < padded_end:
                padding_tokens = padded_end - actual_end
                total_padding_elems = padding_tokens * dim
                num_padding_blocks = tl.cdiv(total_padding_elems, BLOCK_SIZE)
                for block_idx in tl.range(num_padding_blocks):
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_padding_elems
                    tl.store(
                        own_output_ptr + actual_end * dim + offs,
                        tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16),
                        mask=mask,
                    )
