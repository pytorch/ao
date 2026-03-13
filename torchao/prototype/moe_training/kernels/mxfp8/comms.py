import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from torch.distributed._functional_collectives import (
    all_to_all_single,
)

from torchao.prototype.moe_training.kernels.triton_utils import (
    blockwise_barrier,
    sync_threads,
)
from torchao.prototype.mx_formats.kernels import (
    triton_mxfp8_dequant_dim0,
    triton_to_mxfp8_dim0,
)
from torchao.prototype.mx_formats.mx_tensor import to_dtype, to_mx


# This performs dynamic mxfp8 quantization of the input tensor,
# followed by an on-device all-to-all-v operation as determined by the input_splits, implented via Triton + PyTorch symmetric memory.
# This kernel is an extension of the original bf16 version here:
# https://github.com/pytorch/torchtitan/blob/476a965f93432f4f1681bc1bac064d689a2d0cec/torchtitan/experiments/deepseek_v3/symm_mem_recipes/triton_on_device_all_to_all_v.py#L1
class MXFP8OnDeviceAllToAllV(torch.autograd.Function):
    # A symmetric memory buffer for exchanging input rows/tokens during forward
    input_sym_mem_buf = None

    # A symmetric memory for exchanging scales during both forward and backward
    scales_sym_mem_buf = None

    # A symmetric memory for exchanging split sizes during both forward and backward
    input_splits_sym_mem_buf = None

    # A symmetric memory for exchanging per-expert token counts during both forward and backward
    expert_splits_sym_mem_buf = None

    # A symmetric memory buffer holding the grad_output during backward
    grad_out_sym_mem_buf = None

    # Maximum output length (need to be set before use of MXFP8OnDeviceAllToAllV)
    max_output_rows_per_rank = None

    # A preallocated buffer for holding the grad_input, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_buf = None

    # A preallocated buffer for holding the grad_input scales, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_scales_buf = None

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        input: torch.Tensor,
        input_splits: torch.Tensor,
        max_output_rows_per_rank: int,
        expert_splits_per_rank: torch.Tensor,
        group: dist.ProcessGroup = dist.group.WORLD,
    ):
        """
        Args:
            input: input float8_e4m3fn tensor with data for all ranks concatenated.
            input_scales: float8_e8m0fnu scales for the input tensor.
            input_splits: input splits of shape (group.world_size,)
            max_output_rows_per_rank: maximum output rows/tokens per rank.
            expert_splits_per_rank: per-expert token counts per destination rank, shape (world_size, num_experts_per_rank).
                expert_splits_per_rank[i, j] = number of tokens this rank is sending to expert j on rank i.
                Will be exchanged during all-to-all to provide per-expert metadata at destination.
            group: process group to scope the collective.
        """
        assert input.dtype in (torch.float32, torch.bfloat16)

        MXFP8OnDeviceAllToAllV.max_output_rows_per_rank = max_output_rows_per_rank

        # Quantize input
        block_size = 32
        to_mx_c = torch.compile(to_mx)
        input_scales, input_data = to_mx_c(
            input,
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # Triton doesn't support float8_e8m0fnu yet, view as uint8
        input_scales = input_scales.view(torch.uint8)

        # Initialize sym mem buffer for float8 e4m3 input data (one time only)
        if MXFP8OnDeviceAllToAllV.input_sym_mem_buf is None:
            MXFP8OnDeviceAllToAllV.input_sym_mem_buf = symm_mem.empty(
                MXFP8OnDeviceAllToAllV.max_output_rows_per_rank,
                *input_data.shape[1:],
                dtype=input_data.dtype,
                device=input_data.device,
            )

        # Initialize symm mem buffer for float8 e8m0 scales (one time only)
        if MXFP8OnDeviceAllToAllV.scales_sym_mem_buf is None:
            MXFP8OnDeviceAllToAllV.scales_sym_mem_buf = symm_mem.empty(
                MXFP8OnDeviceAllToAllV.max_output_rows_per_rank,
                *input_scales.shape[1:],
                dtype=input_scales.dtype,
                device=input_scales.device,
            )

        # Initialize input splits buffer (one time only)
        if MXFP8OnDeviceAllToAllV.input_splits_sym_mem_buf is None:
            MXFP8OnDeviceAllToAllV.input_splits_sym_mem_buf = symm_mem.empty(
                *input_splits.shape,
                dtype=input_splits.dtype,
                device=input_splits.device,
            )

        # Initialize expert splits buffer (one time only)
        if MXFP8OnDeviceAllToAllV.expert_splits_sym_mem_buf is None:
            MXFP8OnDeviceAllToAllV.expert_splits_sym_mem_buf = symm_mem.empty(
                *expert_splits_per_rank.shape,
                dtype=expert_splits_per_rank.dtype,
                device=expert_splits_per_rank.device,
            )

        # Copy quantized data, scales, and output splits to symm mem buffers
        MXFP8OnDeviceAllToAllV.input_sym_mem_buf.narrow(
            0, 0, input_data.shape[0]
        ).copy_(input_data)

        # Copy input splits to symm mem buffer
        MXFP8OnDeviceAllToAllV.input_splits_sym_mem_buf.copy_(input_splits)

        # Copy expert splits to symm mem buffer
        MXFP8OnDeviceAllToAllV.expert_splits_sym_mem_buf.copy_(expert_splits_per_rank)

        # Copy input scales to symm mem buffer
        MXFP8OnDeviceAllToAllV.scales_sym_mem_buf.narrow(
            0, 0, input_scales.shape[0]
        ).copy_(input_scales)

        # Allocate buffers for output data, scales, and splits.
        output = input_data.new_empty(
            MXFP8OnDeviceAllToAllV.max_output_rows_per_rank, *input_data.shape[1:]
        )
        output_scales = input_scales.new_empty(
            MXFP8OnDeviceAllToAllV.max_output_rows_per_rank, *input_scales.shape[1:]
        )
        output_splits = torch.empty_like(input_splits)
        output_expert_splits = torch.empty_like(expert_splits_per_rank)

        # Padded end offsets for each local expert group. These can be passed
        # directly as grouped-mm offsets.
        num_experts_per_rank = expert_splits_per_rank.shape[1]
        padded_group_end_offsets = torch.empty(
            num_experts_per_rank, dtype=torch.int64, device=input_data.device
        )

        # Shuffle input to output
        _mxfp8_on_device_all_to_all_v(
            MXFP8OnDeviceAllToAllV.input_sym_mem_buf,
            MXFP8OnDeviceAllToAllV.scales_sym_mem_buf,
            MXFP8OnDeviceAllToAllV.input_splits_sym_mem_buf,
            MXFP8OnDeviceAllToAllV.expert_splits_sym_mem_buf,
            output,
            output_scales,
            output_splits,
            output_expert_splits,
            padded_group_end_offsets,
            group=group,
        )

        # Dequantize output
        lowp_dtype = output.dtype
        hp_dtype = input.dtype
        to_dtype_c = torch.compile(to_dtype)
        hp_output = to_dtype_c(
            output,
            output_scales.view(torch.float8_e8m0fnu),
            lowp_dtype,
            block_size,
            hp_dtype,
        )

        # Saving for backward: we need the original sender layout metadata locally,
        # plus the gathered per-expert metadata to invert the grouped-by-expert layout.
        ctx.group = group
        ctx.input_shape = input_data.shape
        ctx.input_scales_shape = input_scales.shape
        ctx.hp_dtype = hp_dtype
        ctx.max_output_rows_per_rank = max_output_rows_per_rank
        ctx.save_for_backward(
            input_splits, expert_splits_per_rank, output_expert_splits
        )
        return (
            hp_output,
            output_splits,
            output_expert_splits,
            padded_group_end_offsets,
        )

    @staticmethod
    @torch.compiler.disable
    def backward(
        ctx, grad_output, grad_splits, grad_expert_splits, grad_expert_padded_offsets
    ):
        """
        Backward is implemented as a shuffle of the output's gradients to the input.
        Args:
            `grad_output`: output's gradients passed from the downstream.
            `grad_splits`: unused.
            `grad_expert_splits`: unused.
            `grad_expert_padded_offsets`: unused.
        """
        # In backward, grad_output arrives grouped by local expert with per-group
        # padding. We need to invert that layout before the reverse all-to-all.
        input_splits, input_expert_splits, grad_output_expert_splits = ctx.saved_tensors

        # Initialize grad_output sym mem buffer (one time only)
        if MXFP8OnDeviceAllToAllV.grad_out_sym_mem_buf is None:
            MXFP8OnDeviceAllToAllV.grad_out_sym_mem_buf = symm_mem.empty(
                MXFP8OnDeviceAllToAllV.max_output_rows_per_rank,
                *grad_output.shape[1:],
                dtype=torch.float8_e4m3fn,
                device=grad_output.device,
            )

        # Quantize grad_output
        block_size = 32
        to_mx_c = torch.compile(to_mx)
        grad_out_scales, grad_out_data = to_mx_c(
            grad_output,
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # Triton doesn't support float8_e8m0fnu yet, view as uint8
        grad_out_scales = grad_out_scales.view(torch.uint8)

        # Copy in float8 grad out data to a symm mem buffer
        MXFP8OnDeviceAllToAllV.grad_out_sym_mem_buf.narrow(
            0, 0, grad_out_data.shape[0]
        ).copy_(grad_out_data)

        # Copy in grad out e8m0 scales to symm mem buffer
        MXFP8OnDeviceAllToAllV.scales_sym_mem_buf.narrow(
            0, 0, grad_out_scales.shape[0]
        ).copy_(grad_out_scales)

        # Copy in expert splits to symm mem buffer
        MXFP8OnDeviceAllToAllV.expert_splits_sym_mem_buf.copy_(
            grad_output_expert_splits
        )

        # Allocate buffers for grad_input data and scales if necessary.
        if MXFP8OnDeviceAllToAllV.grad_input_buf is None:
            MXFP8OnDeviceAllToAllV.grad_input_buf = grad_out_data.new_empty(
                ctx.input_shape[0],
                *ctx.input_shape[1:],
            )

        if MXFP8OnDeviceAllToAllV.grad_input_scales_buf is None:
            MXFP8OnDeviceAllToAllV.grad_input_scales_buf = torch.empty(
                ctx.input_scales_shape[0],
                *ctx.input_scales_shape[1:],
                dtype=grad_out_scales.dtype,
                device=grad_out_scales.device,
            )
        _mxfp8_on_device_all_to_all_v_bwd(
            MXFP8OnDeviceAllToAllV.grad_out_sym_mem_buf,
            MXFP8OnDeviceAllToAllV.scales_sym_mem_buf,
            MXFP8OnDeviceAllToAllV.expert_splits_sym_mem_buf,
            input_splits,
            input_expert_splits,
            MXFP8OnDeviceAllToAllV.grad_input_buf,
            MXFP8OnDeviceAllToAllV.grad_input_scales_buf,
            group=ctx.group,
        )

        # Dequantize grad_input
        lowp_dtype = grad_out_data.dtype
        to_dtype_c = torch.compile(to_dtype)
        grad_input_hp = to_dtype_c(
            MXFP8OnDeviceAllToAllV.grad_input_buf,
            MXFP8OnDeviceAllToAllV.grad_input_scales_buf.view(torch.float8_e8m0fnu),
            lowp_dtype,
            block_size,
            ctx.hp_dtype,
        )
        return grad_input_hp[: ctx.input_shape[0]], None, None, None, None


# Alias
mxfp8_on_device_all_to_all_v = MXFP8OnDeviceAllToAllV.apply


# Triton launcher function
def _mxfp8_on_device_all_to_all_v(
    input: torch.Tensor,
    input_scales: torch.Tensor,
    input_splits: torch.Tensor,
    expert_splits: torch.Tensor,
    output: torch.Tensor,
    output_scales: torch.Tensor,
    output_splits: torch.Tensor,
    output_expert_splits: torch.Tensor,
    padded_group_end_offsets: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
):
    assert input.dim() == 2, f"{input.shape}"
    assert output.dim() == 2, f"{output.shape}"
    assert output.shape[1] == input.shape[1]

    # Prepare symmetric memory managed buffers for input, input_splits, input_scales, and expert_splits.
    # - `input` shape (tokens, dim) -> to a sym mem managed buffer of shape (num_ranks, tokens, dim)
    # - `input_splits` shape (num_ranks,) -> to a sym mem managed buffer of shape (num_ranks, num_ranks)`
    # - `input_scales` shape (tokens, dim//block_size) -> to a sym mem managed buffer of shape (num_ranks, tokens, dim//block_size)
    # - `expert_splits` shape (num_ranks, num_experts_per_rank) -> to a sym mem managed buffer of shape (num_ranks, num_ranks, num_experts_per_rank)
    input_hdl = symm_mem.rendezvous(input, group=group)
    input_splits_hdl = symm_mem.rendezvous(input_splits, group=group)
    input_scales_hdl = symm_mem.rendezvous(input_scales, group=group)
    expert_splits_hdl = symm_mem.rendezvous(expert_splits, group=group)

    input_ptrs = input_hdl.buffer_ptrs_dev
    input_splits_ptrs = input_splits_hdl.buffer_ptrs_dev
    input_scales_ptrs = input_scales_hdl.buffer_ptrs_dev
    expert_splits_ptrs = expert_splits_hdl.buffer_ptrs_dev
    signal_pad_ptrs = input_hdl.signal_pad_ptrs_dev
    dim = output.shape[1]
    dim_scaling_groups = input_scales.shape[-1]
    num_experts_per_rank = expert_splits.shape[-1]
    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK

    _mxfp8_all_to_all_v_kernel[(num_blocks, 1, 1)](
        input_ptrs,
        input_scales_ptrs,
        input_splits_ptrs,
        expert_splits_ptrs,
        output,
        output_scales,
        output_splits,
        output_expert_splits,
        padded_group_end_offsets,
        signal_pad_ptrs,
        dim=dim,
        dim_scaling_groups=dim_scaling_groups,
        num_experts_per_rank=num_experts_per_rank,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output


@triton.jit
def _mxfp8_all_to_all_v_kernel(
    input_ptrs,
    input_scales_ptrs,
    input_splits_ptr,
    expert_splits_ptr,
    output_ptr,
    output_scales_ptr,
    output_splits_ptr,
    output_expert_splits_ptr,
    padded_group_end_offsets_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,
    dim_scaling_groups: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    # ===== PHASE 1: Metadata Exchange and Offset Calculation =====
    # Only block 0 performs the metadata exchange and computes padded offsets
    if tl.program_id(0) == 0:
        # Exchange expert_splits from all remote ranks
        expert_splits_ptrs = expert_splits_ptr.to(tl.pointer_type(tl.uint64))

        # Accumulate tokens per local expert across all remote ranks
        expert_offsets = tl.arange(0, num_experts_per_rank)

        for remote_r in range(world_size):
            # Get pointer to remote rank's expert_splits tensor
            remote_expert_splits_ptr = tl.load(expert_splits_ptrs + remote_r).to(
                tl.pointer_type(tl.int64)
            )
            # expert_splits[remote_r, rank, :] contains tokens remote_r is sending to our local experts
            remote_expert_splits_ptr = (
                remote_expert_splits_ptr + rank * num_experts_per_rank
            )

            # Load expert splits from this remote rank
            remote_expert_split_values = tl.load(
                remote_expert_splits_ptr + expert_offsets
            )

            # Store to output_expert_splits[remote_r, :]
            output_expert_splits_offset = remote_r * num_experts_per_rank
            tl.store(
                output_expert_splits_ptr + output_expert_splits_offset + expert_offsets,
                remote_expert_split_values,
            )

        # Compute padded end offsets: round up to multiple of 32 and cumsum.
        # We compute total tokens per expert by summing across all remote ranks
        cumulative_offset = tl.zeros([], dtype=tl.int64)
        for expert_idx in range(num_experts_per_rank):
            # Sum tokens for this expert across all remote ranks
            # output_expert_splits[remote_r, expert_idx] is at offset remote_r * num_experts_per_rank + expert_idx
            expert_tokens_total = tl.zeros([], dtype=tl.int64)
            for remote_r in range(world_size):
                expert_tokens_from_remote = tl.load(
                    output_expert_splits_ptr
                    + remote_r * num_experts_per_rank
                    + expert_idx
                )
                expert_tokens_total += expert_tokens_from_remote

            # Round up tokens for this expert to multiple of 32
            padded_tokens: tl.int64 = ((expert_tokens_total + 31) // 32) * 32

            # Zero the padding rows for this expert so downstream grouped mm can
            # consume them directly as dummy rows without reading uninitialized data.
            padding_rows = padded_tokens - expert_tokens_total
            if padding_rows > 0:
                padding_row_offset = cumulative_offset + expert_tokens_total
                total_padding_elems = padding_rows * dim
                num_padding_blocks = tl.cdiv(total_padding_elems, BLOCK_SIZE)
                for block_idx in tl.range(num_padding_blocks):
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_padding_elems
                    tl.store(
                        output_ptr + padding_row_offset * dim + offs,
                        0.0,
                        mask=mask,
                    )

                total_padding_scales = padding_rows * dim_scaling_groups
                num_padding_scale_blocks = tl.cdiv(total_padding_scales, BLOCK_SIZE)
                for block_idx in tl.range(num_padding_scale_blocks):
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_padding_scales
                    tl.store(
                        output_scales_ptr
                        + padding_row_offset * dim_scaling_groups
                        + offs,
                        0,
                        mask=mask,
                    )

            # Update cumulative offset for next expert
            cumulative_offset = cumulative_offset + padded_tokens
            tl.store(padded_group_end_offsets_ptr + expert_idx, cumulative_offset)

    # Barrier to ensure metadata exchange is complete before data transfer
    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")

    # ===== PHASE 2: Data Transfer with Padding =====
    # Transfer data expert-by-expert from remote_rank to local padded expert regions

    # One thread block per rank will update output_splits
    if block_offset == 0:
        # Calculate total rows from this remote rank
        split_sizes_ptrs_typed = input_splits_ptr.to(tl.pointer_type(tl.uint64))
        remote_rank_input_splits_ptr = tl.load(split_sizes_ptrs_typed + remote_rank).to(
            tl.pointer_type(tl.int64)
        )
        num_rows_from_remote = tl.load(remote_rank_input_splits_ptr + rank)
        tl.store(output_splits_ptr + remote_rank, num_rows_from_remote)

    # Get base input pointer for this remote rank
    input_base_ptr = tl.load(
        input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.float8e4nv))
    input_scales_base_ptr = tl.load(
        input_scales_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.uint8))

    # Calculate input starting offset for this rank's data on remote_rank
    split_sizes_ptrs_typed = input_splits_ptr.to(tl.pointer_type(tl.uint64))
    remote_rank_input_splits_ptr = tl.load(split_sizes_ptrs_typed + remote_rank).to(
        tl.pointer_type(tl.int64)
    )
    rank_offsets = tl.arange(0, world_size)
    remote_split_sizes_prefix = tl.load(
        remote_rank_input_splits_ptr + rank_offsets, mask=rank_offsets < rank, other=0
    )
    input_row_offset = tl.sum(remote_split_sizes_prefix)

    # Get expert splits for this remote rank
    expert_splits_ptrs_typed = expert_splits_ptr.to(tl.pointer_type(tl.uint64))
    remote_expert_splits_ptr = tl.load(expert_splits_ptrs_typed + remote_rank).to(
        tl.pointer_type(tl.int64)
    )
    remote_expert_splits_ptr = remote_expert_splits_ptr + rank * num_experts_per_rank

    # Process each expert's data from this remote rank
    for expert_idx in range(num_experts_per_rank):
        expert_tokens = tl.load(remote_expert_splits_ptr + expert_idx)

        if expert_tokens > 0:
            expert_output_offset = tl.zeros([], dtype=tl.int64)
            for prev_expert in range(expert_idx):
                prev_expert_total = tl.zeros([], dtype=tl.int64)
                for sender_rank in range(world_size):
                    sender_expert_splits_ptr = tl.load(
                        expert_splits_ptrs_typed + sender_rank
                    ).to(tl.pointer_type(tl.int64))
                    sender_expert_splits_ptr = (
                        sender_expert_splits_ptr + rank * num_experts_per_rank
                    )
                    prev_expert_total += tl.load(sender_expert_splits_ptr + prev_expert)
                expert_output_offset += ((prev_expert_total + 31) // 32) * 32

            for prev_remote_rank in range(remote_rank):
                prev_remote_expert_splits_ptr = tl.load(
                    expert_splits_ptrs_typed + prev_remote_rank
                ).to(tl.pointer_type(tl.int64))
                prev_remote_expert_splits_ptr = (
                    prev_remote_expert_splits_ptr + rank * num_experts_per_rank
                )
                expert_output_offset += tl.load(
                    prev_remote_expert_splits_ptr + expert_idx
                )

            # Get cumulative offset within this remote rank's data for this expert
            # Add up tokens from previous experts from this same remote rank
            cumulative_tokens_before = tl.zeros([], dtype=tl.int64)
            for prev_expert in range(expert_idx):
                cumulative_tokens_before += tl.load(
                    remote_expert_splits_ptr + prev_expert
                )

            # Calculate actual input/output pointers
            input_ptr = (
                input_base_ptr + (input_row_offset + cumulative_tokens_before) * dim
            )
            output_ptr_expert = output_ptr + expert_output_offset * dim

            input_scale_ptr = (
                input_scales_base_ptr
                + (input_row_offset + cumulative_tokens_before) * dim_scaling_groups
            )
            output_scale_ptr_expert = (
                output_scales_ptr + expert_output_offset * dim_scaling_groups
            )

            # Copy data for this expert
            total_elems = expert_tokens * dim
            num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)
            for block_idx in tl.range(num_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_elems
                    data = tl.load(input_ptr + offs, mask=mask, other=0.0)
                    tl.store(output_ptr_expert + offs, data, mask=mask)

            # Copy scales for this expert
            total_scales = expert_tokens * dim_scaling_groups
            num_scale_blocks = tl.cdiv(total_scales, BLOCK_SIZE)
            for block_idx in tl.range(num_scale_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_scales
                    data = tl.load(input_scale_ptr + offs, mask=mask, other=0.0)
                    tl.store(output_scale_ptr_expert + offs, data, mask=mask)

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    return


def _mxfp8_on_device_all_to_all_v_bwd(
    input: torch.Tensor,
    input_scales: torch.Tensor,
    expert_splits: torch.Tensor,
    local_input_splits: torch.Tensor,
    local_input_expert_splits: torch.Tensor,
    output: torch.Tensor,
    output_scales: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
):
    assert input.dim() == 2, f"{input.shape}"
    assert output.dim() == 2, f"{output.shape}"
    assert output.shape[1] == input.shape[1]

    input_hdl = symm_mem.rendezvous(input, group=group)
    input_scales_hdl = symm_mem.rendezvous(input_scales, group=group)
    expert_splits_hdl = symm_mem.rendezvous(expert_splits, group=group)

    input_ptrs = input_hdl.buffer_ptrs_dev
    input_scales_ptrs = input_scales_hdl.buffer_ptrs_dev
    expert_splits_ptrs = expert_splits_hdl.buffer_ptrs_dev
    signal_pad_ptrs = input_hdl.signal_pad_ptrs_dev
    dim = output.shape[1]
    dim_scaling_groups = output_scales.shape[-1]
    num_experts_per_rank = local_input_expert_splits.shape[-1]
    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK

    _mxfp8_all_to_all_v_bwd_kernel[(num_blocks, 1, 1)](
        input_ptrs,
        input_scales_ptrs,
        expert_splits_ptrs,
        local_input_splits,
        local_input_expert_splits,
        output,
        output_scales,
        signal_pad_ptrs,
        dim=dim,
        dim_scaling_groups=dim_scaling_groups,
        num_experts_per_rank=num_experts_per_rank,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output


@triton.jit
def _mxfp8_all_to_all_v_bwd_kernel(
    input_ptrs,
    input_scales_ptrs,
    expert_splits_ptrs,
    local_input_splits_ptr,
    local_input_expert_splits_ptr,
    output_ptr,
    output_scales_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,
    dim_scaling_groups: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    input_base_ptr = tl.load(
        input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.float8e4nv))
    input_scales_base_ptr = tl.load(
        input_scales_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.uint8))

    output_rank_offset = tl.zeros([], dtype=tl.int64)
    for prev_remote_rank in range(remote_rank):
        output_rank_offset += tl.load(local_input_splits_ptr + prev_remote_rank)

    local_expert_row_offset = remote_rank * num_experts_per_rank
    remote_expert_splits_ptr = tl.load(
        expert_splits_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.int64))

    remote_expert_start_offset = tl.zeros([], dtype=tl.int64)
    output_expert_offset = tl.zeros([], dtype=tl.int64)
    for expert_idx in range(num_experts_per_rank):
        expert_tokens = tl.load(
            local_input_expert_splits_ptr + local_expert_row_offset + expert_idx
        )

        remote_expert_tokens_total = tl.zeros([], dtype=tl.int64)
        for sender_rank in range(world_size):
            remote_expert_tokens_total += tl.load(
                remote_expert_splits_ptr
                + sender_rank * num_experts_per_rank
                + expert_idx
            )

        if expert_tokens > 0:
            remote_rank_prefix_for_expert = tl.zeros([], dtype=tl.int64)
            for prev_sender_rank in range(rank):
                remote_rank_prefix_for_expert += tl.load(
                    remote_expert_splits_ptr
                    + prev_sender_rank * num_experts_per_rank
                    + expert_idx
                )

            input_ptr = (
                input_base_ptr
                + (remote_expert_start_offset + remote_rank_prefix_for_expert) * dim
            )
            output_ptr_expert = (
                output_ptr + (output_rank_offset + output_expert_offset) * dim
            )

            input_scale_ptr = (
                input_scales_base_ptr
                + (remote_expert_start_offset + remote_rank_prefix_for_expert)
                * dim_scaling_groups
            )
            output_scale_ptr_expert = (
                output_scales_ptr
                + (output_rank_offset + output_expert_offset) * dim_scaling_groups
            )

            total_elems = expert_tokens * dim
            num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)
            for block_idx in tl.range(num_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_elems
                    data = tl.load(input_ptr + offs, mask=mask, other=0.0)
                    tl.store(output_ptr_expert + offs, data, mask=mask)

            total_scales = expert_tokens * dim_scaling_groups
            num_scale_blocks = tl.cdiv(total_scales, BLOCK_SIZE)
            for block_idx in tl.range(num_scale_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_scales
                    data = tl.load(input_scale_ptr + offs, mask=mask, other=0.0)
                    tl.store(output_scale_ptr_expert + offs, data, mask=mask)

        remote_expert_start_offset += ((remote_expert_tokens_total + 31) // 32) * 32
        output_expert_offset += expert_tokens

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    return


@triton.jit
def _exchange_row_offsets(
    split_sizes_ptrs,
    local_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """
    Returns:
    - `input_offset_for_remote_rank`:
    - `output_offset_for_remote_rank`:
    - `num_rows`:
    """
    # split_sizes_ptr points to 2d tensor of stacked input split size vectors (one per rank). Example:
    # rank 0 = [30, 10, 10, 20]
    # rank 1 = [20, 20, 10, 20]
    split_sizes_ptrs = split_sizes_ptrs.to(tl.pointer_type(tl.uint64))

    # Get pointer to remote rank's input_split_sizes tensor.
    remote_rank_input_splits_ptr = tl.load(split_sizes_ptrs + remote_rank).to(
        tl.pointer_type(tl.int64)
    )

    # num_rows_to_read is the specific number of tokens to read from remote_rank.
    num_rows_to_read = tl.load(remote_rank_input_splits_ptr + local_rank)

    # Calculate starting offset in symm mem buf to read data from remote_rank for this local_rank.
    #
    # Do this by computing prefix sum of remote split offsets prev ranks.
    # Ex. remote_rank split sizes = [10, 20, 30]
    # For local rank 1, masked load = [10, 0, 0]
    # Starting offset = sum([10, 0, 0]) = 10
    offsets = tl.arange(0, world_size)
    remote_split_sizes_prefix = tl.load(
        remote_rank_input_splits_ptr + offsets, mask=offsets < local_rank, other=0
    )
    input_offset_for_remote_rank = tl.sum(remote_split_sizes_prefix)

    # Calculate offset in local output buffer to start writing data to, for data coming from the remote_rank to this local_rank.
    #
    # We add `offsets` arange to get a set of pointers to the start of each row (rank) in the split_sizes matrix.
    # Then, we add the local rank to each pointer, incrementing it colwise to reach the value for this local rank.
    # Each ptrs now all point to how many tokens/rows that device has for local rank.
    #
    # torch equivalent: split_sizes_matrix[:, rank]
    ptr_to_each_rank_split_sizes = tl.load(split_sizes_ptrs + offsets).to(
        tl.pointer_type(tl.int64)
    )
    output_split_sizes_ptrs = ptr_to_each_rank_split_sizes + local_rank
    output_split_sizes = tl.load(
        output_split_sizes_ptrs, mask=offsets < remote_rank, other=0
    )
    output_offset_for_remote_rank = tl.sum(output_split_sizes)

    return input_offset_for_remote_rank, output_offset_for_remote_rank, num_rows_to_read


class ToMXFP8AllToAllVDequant(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: dist.ProcessGroup = dist.group.WORLD,
    ):
        """
        Dynamically quantizes input to mxfp8, performs all-to-all, then dequantizes output back to original precision.
        Requires d2h sync to get input_splits and output_splits on host, as required by torch.distributed.all_to_all_single API.
        Uses RCEIL scaling mode for quantization.
        """
        # Quantize input
        block_size = 32
        input_data, input_scales = triton_to_mxfp8_dim0(
            input,
            inner_block_size=block_size,
        )

        # Dispatch data (async)
        output_data = all_to_all_single(
            input_data,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )

        # Dispatch scales (async)
        output_scales = all_to_all_single(
            input_scales.view(torch.uint8),  # NCCL cannot handle float8_e8m0fnu yet
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )

        # Explicitly wait since the a2a ops are async
        output_scales = torch.ops._c10d_functional.wait_tensor(output_scales)
        output_data = torch.ops._c10d_functional.wait_tensor(output_data)

        # Dequantize output
        hp_dtype = input.dtype
        triton_hp_output = triton_mxfp8_dequant_dim0(
            output_data,
            output_scales,
            hp_dtype,
            block_size,
        )
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        return triton_hp_output

    @staticmethod
    def backward(ctx, grad_output_hp):
        """
        Backward is implemented as a shuffle of the output's gradients to the input.
        Args:
            `grad_output_hp`: high precision output gradient passed from upstream
        """
        # In backward, mxfp8_all_to_all_v input is `grad_output`, and output is `grad_input`.
        # Input splits are the output splits from forward (and vice-versa).
        input_splits, output_splits = ctx.input_splits, ctx.output_splits

        # Quantize grad_output
        block_size = 32
        grad_out_data, grad_out_scales = triton_to_mxfp8_dim0(
            grad_output_hp,
            inner_block_size=block_size,
        )

        # Dispatch data (async)
        grad_input_data = all_to_all_single(
            grad_out_data,
            output_split_sizes=input_splits,
            input_split_sizes=output_splits,
            group=ctx.group,
        )

        # Dispatch scales (async)
        grad_input_scales = all_to_all_single(
            grad_out_scales.view(torch.uint8),  # NCCL cannot handle float8_e8m0fnu yet
            output_split_sizes=input_splits,
            input_split_sizes=output_splits,
            group=ctx.group,
        )

        # Explicitly wait since the a2a ops are async
        grad_input_data = torch.ops._c10d_functional.wait_tensor(grad_input_data)
        grad_input_scales = torch.ops._c10d_functional.wait_tensor(grad_input_scales)

        hp_dtype = grad_output_hp.dtype
        grad_input_hp = triton_mxfp8_dequant_dim0(
            grad_input_data,
            grad_input_scales,
            hp_dtype,
            block_size,
        )
        return grad_input_hp, None, None, None


# Alias
to_mxfp8_a2a_dequant = ToMXFP8AllToAllVDequant.apply
