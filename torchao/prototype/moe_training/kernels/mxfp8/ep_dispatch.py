import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from torchao.prototype.moe_training.kernels.mxfp8 import (
    EPBufferManager,
    get_buffer_manager,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0


class MXFP8SynclessAllToAllExpertMajor(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        input: torch.Tensor,
        input_rank_splits: torch.Tensor,
        input_expert_splits: torch.Tensor,
        max_output_rows_per_rank: int,
        group: dist.ProcessGroup = dist.group.WORLD,
        buffer_manager: EPBufferManager = None,
    ):
        """
        Performs dynamic MXFP8 quantization along dim0, then an on-device all-to-all operation
        using Triton + Symmetric Memory, writing tokens directly to expert-major layout on target
        ranks for consumption by a MXFP8 Grouped GEMM.

        Implementation is syncless (i.e., no device to host syncs).

        Args:
            input: input float8_e4m3fn tensor with data for all ranks concatenated.
            input_scales: float8_e8m0fnu scales for the input tensor.
            input_rank_splits: input splits of shape (group.world_size,)
            max_output_rows_per_rank: maximum output rows/tokens per rank.
            input_expert_splits: per-expert token counts per destination rank, shape (world_size, num_experts_per_rank).
                input_expert_splits[i, j] = number of tokens this rank is sending to expert j on rank i.
                Will be exchanged during all-to-all to provide per-expert metadata at destination.
            group: process group to scope the collective.
            buffer_manager: optional buffer manager for reusing buffers across layers.
        """
        assert input.dtype in (torch.float32, torch.bfloat16)

        # Get or create buffer manager
        buffers = buffer_manager or get_buffer_manager()
        buffers.max_output_rows_per_rank = max_output_rows_per_rank

        block_size = 32
        input_data, input_scales = triton_to_mxfp8_dim0(
            input,
            inner_block_size=block_size,
            scaling_mode="rceil",
        )

        input_scales = input_scales.view(torch.uint8)

        num_experts_per_rank = input_expert_splits.shape[1]
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        # Allocate symmetric memory buffers for output data and scales (destinations for push writes)
        if buffers.output is None:
            buffers.output = symm_mem.empty(
                buffers.max_output_rows_per_rank,
                *input_data.shape[1:],
                dtype=input_data.dtype,
                device=input_data.device,
            )
        if buffers.output_scales is None:
            buffers.output_scales = symm_mem.empty(
                buffers.max_output_rows_per_rank,
                *input_scales.shape[1:],
                dtype=input_scales.dtype,
                device=input_scales.device,
            )

        # All-gather expert_splits so all ranks have global view
        # Shape: (world_size, world_size, num_experts_per_rank)
        # all_expert_splits[src_rank, dst_rank, expert_idx] = tokens src_rank sends to expert_idx on dst_rank
        all_expert_splits = torch.empty(
            world_size * world_size,
            num_experts_per_rank,
            dtype=input_expert_splits.dtype,
            device=input_expert_splits.device,
        )
        dist.all_gather_into_tensor(
            all_expert_splits,
            input_expert_splits,
            group=group,
        )
        all_expert_splits = all_expert_splits.view(
            world_size, world_size, num_experts_per_rank
        )

        # Allocate buffers for metadata
        expert_padded_offsets = torch.empty(
            num_experts_per_rank, dtype=torch.int64, device=input_data.device
        )

        # Compute output_rank_splits and output_expert_splits from all_expert_splits
        # output_rank_splits[src_rank] = total tokens src_rank sends to this rank
        # output_expert_splits[src_rank, expert_idx] = tokens src_rank sends to expert_idx on this rank
        output_expert_splits = all_expert_splits[:, rank, :]
        output_rank_splits = output_expert_splits.sum(dim=1)

        # Allocate write_offsets buffer
        # write_offsets[dst_rank, expert_idx] = where this rank writes for (dst_rank, expert_idx)
        write_offsets = torch.empty(
            world_size,
            num_experts_per_rank,
            dtype=torch.int64,
            device=input_data.device,
        )

        # Push input to remote output buffers
        _mxfp8_ep_dispatch_launcher(
            input_data,
            input_scales,
            input_rank_splits,
            input_expert_splits,
            all_expert_splits,
            write_offsets,
            buffers.output,
            buffers.output_scales,
            output_rank_splits,
            output_expert_splits,
            expert_padded_offsets,
            group=group,
        )

        hp_dtype = input.dtype

        # Wrap output as MXTensor so autograd sees it as orig_dtype/bf16 (backward will receive bf16 gradients)
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        mx_output = MXTensor(
            buffers.output,
            buffers.output_scales.view(torch.float8_e8m0fnu),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            orig_dtype=hp_dtype,
            kernel_preference=None,
            act_quant_kwargs=None,
            is_swizzled_scales=False,
        )

        return (
            mx_output,
            output_rank_splits,
            output_expert_splits,
            expert_padded_offsets,
        )

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_rank_splits,
        grad_expert_splits,
        grad_expert_padded_offsets,
    ):
        raise NotImplementedError("backward support not yet implemented")


# Alias
mxfp8_ep_dispatch = MXFP8SynclessAllToAllExpertMajor.apply


# Triton launcher function for push model
def _mxfp8_ep_dispatch_launcher(
    input_data: torch.Tensor,
    input_scales: torch.Tensor,
    input_rank_splits: torch.Tensor,
    input_expert_splits: torch.Tensor,
    all_expert_splits: torch.Tensor,
    write_offsets: torch.Tensor,
    output: torch.Tensor,
    output_scales: torch.Tensor,
    output_rank_splits: torch.Tensor,
    output_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
):
    """
    Push model launcher: each rank pushes its data to remote output buffers.

    Args:
        input_data: local quantized data (tokens, dim)
        input_scales: local scales (tokens, scale_dim)
        input_rank_splits: tokens per rank (world_size,)
        input_expert_splits: tokens this rank sends per expert per dst_rank (world_size, num_experts_per_rank)
        all_expert_splits: all-gathered expert splits (world_size, world_size, num_experts_per_rank)
        write_offsets: precomputed write offsets (world_size, num_experts_per_rank)
        output: symmetric memory buffer for output data
        output_scales: symmetric memory buffer for output scales
        output_rank_splits: output rank splits
        output_expert_splits: output expert splits
        expert_padded_offsets: expert padded offsets
        group: process group
    """
    assert input_data.dim() == 2, f"{input_data.shape}"
    assert output.dim() == 2, f"{output.shape}"
    assert output.shape[1] == input_data.shape[1]

    # Setup symmetric memory for output buffers (where remote ranks will push writes)
    output_hdl = symm_mem.rendezvous(output, group=group)
    output_scales_hdl = symm_mem.rendezvous(output_scales, group=group)

    output_ptrs = output_hdl.buffer_ptrs_dev
    output_scales_ptrs = output_scales_hdl.buffer_ptrs_dev

    dim = output.shape[1]
    scale_dim = input_scales.shape[-1]
    num_experts_per_rank = input_expert_splits.shape[-1]
    rank = output_hdl.rank
    world_size = output_hdl.world_size

    # Phase 1: Precompute write offsets and expert_padded_offsets
    _precompute_push_write_offsets_kernel[(1, 1, 1)](
        all_expert_splits,
        output_expert_splits,
        write_offsets,
        expert_padded_offsets,
        rank=rank,
        world_size=world_size,
        num_experts_per_rank=num_experts_per_rank,
    )

    # Phase 2: Push data to remote ranks
    num_blocks = world_size * BLOCKS_PER_REMOTE_RANK

    _mxfp8_all_to_all_expert_major_kernel[(num_blocks, 1, 1)](
        input_data,
        input_scales,
        input_expert_splits,
        write_offsets,
        output_ptrs,
        output_scales_ptrs,
        dim=dim,
        scale_dim=scale_dim,
        num_experts_per_rank=num_experts_per_rank,
        rank=rank,
        world_size=world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    # Phase 3: Zero-fill padding regions after data transfer
    # Calculate which ranges need to be zeroed (padding regions within experts)
    begin_offs, end_offs = _compute_padding_ranges(
        output_expert_splits, expert_padded_offsets, num_experts_per_rank, world_size
    )

    _zero_fill_ranges_kernel[(num_experts_per_rank, 1, 1)](
        output,
        output_scales,
        begin_offs,
        end_offs,
        dim=dim,
        scale_dim=scale_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return output


@triton.jit
def _mxfp8_all_to_all_expert_major_kernel(
    input_data_ptr,
    input_scales_ptr,
    input_expert_splits_ptr,
    write_offsets_ptr,
    output_ptrs,  # sym mem buf
    output_scales_ptrs,  # sym mem buf
    dim: tl.constexpr,
    scale_dim: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Push model kernel: each rank pushes its data to remote output buffers."""

    dst_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    # Get output buffer pointers for dst_rank
    dst_output_ptr = tl.load(output_ptrs.to(tl.pointer_type(tl.uint64)) + dst_rank).to(
        tl.pointer_type(tl.float8e4nv)
    )

    dst_output_scales_ptr = tl.load(
        output_scales_ptrs.to(tl.pointer_type(tl.uint64)) + dst_rank
    ).to(tl.pointer_type(tl.uint8))

    # Compute base read offset for this dst_rank (sum of tokens sent to previous ranks)
    base_local_read_offset = tl.zeros([], dtype=tl.int64)
    for prev_dst_rank in range(dst_rank):
        for e in range(num_experts_per_rank):
            base_local_read_offset += tl.load(
                input_expert_splits_ptr + prev_dst_rank * num_experts_per_rank + e
            )

    local_read_offset = base_local_read_offset

    # Push data to dst_rank for each expert
    for expert_idx in range(num_experts_per_rank):
        # How many tokens am I sending to this expert on dst_rank?
        my_tokens_to_expert = tl.load(
            input_expert_splits_ptr + dst_rank * num_experts_per_rank + expert_idx
        )

        if my_tokens_to_expert > 0:
            # Get precomputed write offset
            write_offset = tl.load(
                write_offsets_ptr + dst_rank * num_experts_per_rank + expert_idx
            )

            # Copy data
            total_elems = my_tokens_to_expert * dim
            num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)
            for block_idx in tl.range(num_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_elems
                    # Read from my input
                    data = tl.load(
                        input_data_ptr + local_read_offset * dim + offs,
                        mask=mask,
                        other=0.0,
                    )
                    # Write to dst_rank's output
                    tl.store(
                        dst_output_ptr + write_offset * dim + offs, data, mask=mask
                    )

            # Copy scales
            total_scales = my_tokens_to_expert * scale_dim
            num_scale_blocks = tl.cdiv(total_scales, BLOCK_SIZE)
            for block_idx in tl.range(num_scale_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_scales
                    # Read from my input scales
                    data = tl.load(
                        input_scales_ptr + local_read_offset * scale_dim + offs,
                        mask=mask,
                        other=0.0,
                    )
                    # Write to dst_rank's output scales
                    tl.store(
                        dst_output_scales_ptr + write_offset * scale_dim + offs,
                        data,
                        mask=mask,
                    )

        # Update local read offset for next expert
        local_read_offset += my_tokens_to_expert


def _compute_padding_ranges(
    output_expert_splits, expert_padded_offsets, num_experts_per_rank, world_size
):
    """Compute ranges that need to be zero-filled (padding regions within experts)."""
    # Calculate actual tokens per expert (sum across all source ranks)
    actual_tokens_per_expert = output_expert_splits.sum(
        dim=0
    )  # shape: (num_experts_per_rank,)

    # Where each expert's actual data ends
    actual_end_offsets = expert_padded_offsets + actual_tokens_per_expert

    # Where each expert's padded region ends = start of next expert (from expert_padded_offsets)
    # For last expert, compute its padded end
    if num_experts_per_rank > 1:
        padded_end_offsets = torch.cat(
            [
                expert_padded_offsets[1:],  # Start of next expert for all but last
                expert_padded_offsets[-1:]
                + ((actual_tokens_per_expert[-1] + 31) // 32)
                * 32,  # Last expert padded end
            ]
        )
    else:
        padded_end_offsets = (
            expert_padded_offsets + ((actual_tokens_per_expert + 31) // 32) * 32
        )

    return actual_end_offsets, padded_end_offsets


@triton.jit
def _zero_fill_ranges_kernel(
    output_ptr,
    output_scales_ptr,
    begin_offs_ptr,
    end_offs_ptr,
    dim: tl.constexpr,
    scale_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero-fill specified ranges in output buffer."""
    range_idx = tl.program_id(0)

    # Load range bounds
    begin_offset = tl.load(begin_offs_ptr + range_idx)
    end_offset = tl.load(end_offs_ptr + range_idx)

    # Only proceed if there's a valid range to fill (begin < end)
    if begin_offset < end_offset:
        # Zero-fill data range
        total_elems = (end_offset - begin_offset) * dim
        num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)

        for block_idx in tl.range(num_blocks):
            offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < total_elems
            tl.store(
                output_ptr + begin_offset * dim + offs,
                tl.zeros([BLOCK_SIZE], dtype=tl.float8e4nv),
                mask=mask,
            )

        # Zero-fill scales range
        total_scale_elems = (end_offset - begin_offset) * scale_dim
        num_scale_blocks = tl.cdiv(total_scale_elems, BLOCK_SIZE)

        for block_idx in tl.range(num_scale_blocks):
            offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < total_scale_elems
            tl.store(
                output_scales_ptr + begin_offset * scale_dim + offs,
                tl.zeros([BLOCK_SIZE], dtype=tl.uint8),
                mask=mask,
            )


@triton.jit
def _precompute_push_write_offsets_kernel(
    all_expert_splits_ptr,
    output_expert_splits_ptr,
    write_offsets_ptr,
    expert_padded_offsets_ptr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
):
    """
    Precompute write offsets for push model.

    Inputs:
        all_expert_splits[src_rank, dst_rank, expert_idx] = tokens src_rank sends to expert_idx on dst_rank
        output_expert_splits[src_rank, expert_idx] = tokens src_rank sends to expert_idx on this rank

    Outputs:
        write_offsets[dst_rank, expert_idx] = where this rank writes for (dst_rank, expert_idx)
        expert_padded_offsets[expert_idx] = starting offset for expert_idx in this rank's output buffer
    """
    # Each program handles all computations (single kernel launch)
    if tl.program_id(0) == 0:
        # Compute expert_padded_offsets for this rank
        cumulative_offset = tl.zeros([], dtype=tl.int64)
        for expert_idx in range(num_experts_per_rank):
            # Store starting offset for this expert
            tl.store(expert_padded_offsets_ptr + expert_idx, cumulative_offset)

            # Sum tokens from all source ranks for this expert
            total_tokens = tl.zeros([], dtype=tl.int64)
            for src_rank in range(world_size):
                tokens = tl.load(
                    output_expert_splits_ptr
                    + src_rank * num_experts_per_rank
                    + expert_idx
                )
                total_tokens += tokens

            # Pad to multiple of 32
            padded_size = ((total_tokens + 31) // 32) * 32
            cumulative_offset += padded_size

        # Compute write_offsets for this rank (where it writes to each dst_rank)
        for dst_rank in range(world_size):
            for expert_idx in range(num_experts_per_rank):
                # Step 1: Expert base offset on dst_rank
                expert_base_offset = tl.zeros([], dtype=tl.int64)
                for prev_expert in range(expert_idx):
                    # Total tokens for prev_expert on dst_rank
                    prev_expert_total = tl.zeros([], dtype=tl.int64)
                    for src_rank in range(world_size):
                        offset = (
                            src_rank * (world_size * num_experts_per_rank)
                            + dst_rank * num_experts_per_rank
                            + prev_expert
                        )
                        tokens = tl.load(all_expert_splits_ptr + offset)
                        prev_expert_total += tokens
                    # Pad to multiple of 32
                    padded_size = ((prev_expert_total + 31) // 32) * 32
                    expert_base_offset += padded_size

                # Step 2: Within-expert offset (tokens from ranks 0..rank-1)
                within_expert_offset = tl.zeros([], dtype=tl.int64)
                for src_rank in range(rank):
                    offset = (
                        src_rank * (world_size * num_experts_per_rank)
                        + dst_rank * num_experts_per_rank
                        + expert_idx
                    )
                    tokens = tl.load(all_expert_splits_ptr + offset)
                    within_expert_offset += tokens

                # Store final write offset
                write_offset = expert_base_offset + within_expert_offset
                tl.store(
                    write_offsets_ptr + dst_rank * num_experts_per_rank + expert_idx,
                    write_offset,
                )
