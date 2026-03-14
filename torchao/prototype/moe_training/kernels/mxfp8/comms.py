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
class MXFP8SynclessAllToAllExpertMajor(torch.autograd.Function):
    # A symmetric memory buffer for exchanging input rows/tokens during forward
    input_sym_mem_buf = None

    # A symmetric memory for exchanging scales during both forward and backward
    scales_sym_mem_buf = None

    # A symmetric memory for exchanging split sizes during both forward and backward
    input_rank_splits_sym_mem_buf = None

    # A symmetric memory for exchanging per-expert token counts during both forward and backward
    input_expert_splits_sym_mem_buf = None

    # A symmetric memory buffer holding the grad_output during backward
    grad_out_sym_mem_buf = None

    # Maximum output length (need to be set before use of MXFP8SynclessAllToAllExpertMajor)
    max_output_rows_per_rank = None

    # A preallocated buffer for holding the grad_input, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_buf = None

    # A preallocated buffer for holding the grad_input scales, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_scales_buf = None

    # A preallocated buffer for holding the grad_input splits, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_splits_buf = None

    # A preallocated buffer for holding the grad_input expert splits, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_expert_splits_buf = None

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        input: torch.Tensor,
        input_rank_splits: torch.Tensor,
        input_expert_splits: torch.Tensor,
        max_output_rows_per_rank: int,
        group: dist.ProcessGroup = dist.group.WORLD,
    ):
        """
        Args:
            input: input float8_e4m3fn tensor with data for all ranks concatenated.
            input_scales: float8_e8m0fnu scales for the input tensor.
            input_rank_splits: input splits of shape (group.world_size,)
            max_output_rows_per_rank: maximum output rows/tokens per rank.
            input_expert_splits: per-expert token counts per destination rank, shape (world_size, num_experts_per_rank).
                input_expert_splits[i, j] = number of tokens this rank is sending to expert j on rank i.
                Will be exchanged during all-to-all to provide per-expert metadata at destination.
            group: process group to scope the collective.
        """
        assert input.dtype in (torch.float32, torch.bfloat16)

        # Enable symm mem for the group if not already enabled
        if not symm_mem.is_symm_mem_enabled_for_group(group):
            symm_mem.enable_symm_mem_for_group(group)

        MXFP8SynclessAllToAllExpertMajor.max_output_rows_per_rank = (
            max_output_rows_per_rank
        )

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
        if MXFP8SynclessAllToAllExpertMajor.input_sym_mem_buf is None:
            MXFP8SynclessAllToAllExpertMajor.input_sym_mem_buf = symm_mem.empty(
                MXFP8SynclessAllToAllExpertMajor.max_output_rows_per_rank,
                *input_data.shape[1:],
                dtype=input_data.dtype,
                device=input_data.device,
            )

        # Initialize symm mem buffer for float8 e8m0 scales (one time only)
        if MXFP8SynclessAllToAllExpertMajor.scales_sym_mem_buf is None:
            MXFP8SynclessAllToAllExpertMajor.scales_sym_mem_buf = symm_mem.empty(
                MXFP8SynclessAllToAllExpertMajor.max_output_rows_per_rank,
                *input_scales.shape[1:],
                dtype=input_scales.dtype,
                device=input_scales.device,
            )

        # Initialize input rank splits buffer (one time only)
        if MXFP8SynclessAllToAllExpertMajor.input_rank_splits_sym_mem_buf is None:
            MXFP8SynclessAllToAllExpertMajor.input_rank_splits_sym_mem_buf = symm_mem.empty(
                *input_rank_splits.shape,
                dtype=input_rank_splits.dtype,
                device=input_rank_splits.device,
            )

        # Initialize expert splits buffer (one time only)
        if MXFP8SynclessAllToAllExpertMajor.input_expert_splits_sym_mem_buf is None:
            MXFP8SynclessAllToAllExpertMajor.input_expert_splits_sym_mem_buf = symm_mem.empty(
                *input_expert_splits.shape,
                dtype=input_expert_splits.dtype,
                device=input_expert_splits.device,
            )

        # Copy quantized data, scales, and output splits to symm mem buffers
        MXFP8SynclessAllToAllExpertMajor.input_sym_mem_buf.narrow(
            0, 0, input_data.shape[0]
        ).copy_(input_data)

        # Copy input rank splits to symm mem buffer
        MXFP8SynclessAllToAllExpertMajor.input_rank_splits_sym_mem_buf.copy_(input_rank_splits)

        # Copy input expert splits to symm mem buffer
        MXFP8SynclessAllToAllExpertMajor.input_expert_splits_sym_mem_buf.copy_(
            input_expert_splits
        )

        # Copy input scales to symm mem buffer
        MXFP8SynclessAllToAllExpertMajor.scales_sym_mem_buf.narrow(
            0, 0, input_scales.shape[0]
        ).copy_(input_scales)

        # Allocate buffers for output data, scales, and splits.
        output = input_data.new_empty(
            MXFP8SynclessAllToAllExpertMajor.max_output_rows_per_rank,
            *input_data.shape[1:],
        )
        output_scales = input_scales.new_empty(
            MXFP8SynclessAllToAllExpertMajor.max_output_rows_per_rank,
            *input_scales.shape[1:],
        )
        output_rank_splits = torch.empty_like(input_rank_splits)
        output_expert_splits = torch.empty_like(input_expert_splits)

        # Allocate buffer for padded expert offsets (one offset per expert)
        num_experts_per_rank = input_expert_splits.shape[1]
        expert_padded_offsets = torch.empty(
            num_experts_per_rank, dtype=torch.int64, device=input_data.device
        )

        # Shuffle input to output
        _mxfp8_syncless_all_to_all_expert_major_launcher(
            MXFP8SynclessAllToAllExpertMajor.input_sym_mem_buf,
            MXFP8SynclessAllToAllExpertMajor.scales_sym_mem_buf,
            MXFP8SynclessAllToAllExpertMajor.input_rank_splits_sym_mem_buf,
            MXFP8SynclessAllToAllExpertMajor.input_expert_splits_sym_mem_buf,
            output,
            output_scales,
            output_rank_splits,
            output_expert_splits,
            expert_padded_offsets,
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

        # Saving for backward: output splits in forward is the input splits in backward
        ctx.group = group
        ctx.input_shape = input_data.shape
        ctx.input_scales_shape = input_scales.shape
        ctx.expert_splits_shape = input_expert_splits.shape
        ctx.hp_dtype = hp_dtype
        ctx.max_output_rows_per_rank = max_output_rows_per_rank
        ctx.save_for_backward(
            output_rank_splits, output_expert_splits, expert_padded_offsets
        )
        # Return full expert-major layout with padding
        return (
            hp_output,
            output_rank_splits,
            output_expert_splits,
            expert_padded_offsets,
        )

    @staticmethod
    @torch.compiler.disable
    def backward(
        ctx, grad_output, grad_rank_splits, grad_expert_splits, grad_expert_padded_offsets
    ):
        """
        Backward is implemented as a shuffle of the output's gradients to the input.
        Args:
            `grad_output`: output's gradients passed from the downstream.
            `grad_rank_splits`: unused.
            `grad_expert_splits`: unused.
            `grad_expert_padded_offsets`: unused.
        """
        # In backward, mxfp8_all_to_all_v input is `grad_output`, and output is `grad_input`.
        grad_output_rank_splits, grad_output_expert_splits, _ = ctx.saved_tensors

        # Initialize grad_output sym mem buffer (one time only)
        if MXFP8SynclessAllToAllExpertMajor.grad_out_sym_mem_buf is None:
            MXFP8SynclessAllToAllExpertMajor.grad_out_sym_mem_buf = symm_mem.empty(
                MXFP8SynclessAllToAllExpertMajor.max_output_rows_per_rank,
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
        MXFP8SynclessAllToAllExpertMajor.grad_out_sym_mem_buf.narrow(
            0, 0, grad_out_data.shape[0]
        ).copy_(grad_out_data)

        # Copy in grad out e8m0 scales to symm mem buffer
        MXFP8SynclessAllToAllExpertMajor.scales_sym_mem_buf.narrow(
            0, 0, grad_out_scales.shape[0]
        ).copy_(grad_out_scales)

        # Copy in rank splits to symm mem buffer
        MXFP8SynclessAllToAllExpertMajor.input_rank_splits_sym_mem_buf.copy_(
            grad_output_rank_splits
        )

        # Copy in expert splits to symm mem buffer
        MXFP8SynclessAllToAllExpertMajor.input_expert_splits_sym_mem_buf.copy_(
            grad_output_expert_splits
        )

        # Allocate buffers for grad_input data, scales, and splits if necessary
        if MXFP8SynclessAllToAllExpertMajor.grad_input_buf is None:
            MXFP8SynclessAllToAllExpertMajor.grad_input_buf = grad_out_data.new_empty(
                ctx.max_output_rows_per_rank,
                *ctx.input_shape[1:],
            )

        if MXFP8SynclessAllToAllExpertMajor.grad_input_scales_buf is None:
            MXFP8SynclessAllToAllExpertMajor.grad_input_scales_buf = torch.empty(
                ctx.max_output_rows_per_rank,
                *ctx.input_scales_shape[1:],
                dtype=grad_out_scales.dtype,
                device=grad_out_scales.device,
            )
        if MXFP8SynclessAllToAllExpertMajor.grad_input_splits_buf is None:
            MXFP8SynclessAllToAllExpertMajor.grad_input_splits_buf = torch.empty_like(
                grad_output_rank_splits
            )
        if MXFP8SynclessAllToAllExpertMajor.grad_input_expert_splits_buf is None:
            MXFP8SynclessAllToAllExpertMajor.grad_input_expert_splits_buf = torch.empty(
                *ctx.expert_splits_shape,
                dtype=grad_output_expert_splits.dtype,
                device=grad_output_expert_splits.device,
            )

        # Allocate buffer for padded expert offsets in backward
        num_experts_per_rank = grad_output_expert_splits.shape[1]
        grad_expert_padded_offsets = torch.empty(
            num_experts_per_rank, dtype=torch.int64, device=grad_output.device
        )

        # Shuffle gradients back to the input
        _mxfp8_syncless_all_to_all_expert_major_launcher(
            MXFP8SynclessAllToAllExpertMajor.grad_out_sym_mem_buf,  # input
            MXFP8SynclessAllToAllExpertMajor.scales_sym_mem_buf,  # input scales
            MXFP8SynclessAllToAllExpertMajor.input_rank_splits_sym_mem_buf,  # input rank splits
            MXFP8SynclessAllToAllExpertMajor.input_expert_splits_sym_mem_buf,  # input expert splits
            MXFP8SynclessAllToAllExpertMajor.grad_input_buf,  # output
            MXFP8SynclessAllToAllExpertMajor.grad_input_scales_buf,  # output scales
            MXFP8SynclessAllToAllExpertMajor.grad_input_splits_buf,  # output rank splits
            MXFP8SynclessAllToAllExpertMajor.grad_input_expert_splits_buf,  # output expert splits
            grad_expert_padded_offsets,  # expert padded offsets
            group=ctx.group,
        )

        # Dequantize grad_input
        lowp_dtype = grad_out_data.dtype
        to_dtype_c = torch.compile(to_dtype)
        grad_input_hp = to_dtype_c(
            MXFP8SynclessAllToAllExpertMajor.grad_input_buf,
            MXFP8SynclessAllToAllExpertMajor.grad_input_scales_buf.view(
                torch.float8_e8m0fnu
            ),
            lowp_dtype,
            block_size,
            ctx.hp_dtype,
        )
        tokens_on_device_after_a2a_bwd = (
            MXFP8SynclessAllToAllExpertMajor.grad_input_splits_buf.sum()
        )
        return grad_input_hp[:tokens_on_device_after_a2a_bwd], None, None, None, None


# Alias
mxfp8_syncless_all_to_all_expert_major = MXFP8SynclessAllToAllExpertMajor.apply


# Triton launcher function
def _mxfp8_syncless_all_to_all_expert_major_launcher(
    input: torch.Tensor,
    input_scales: torch.Tensor,
    input_rank_splits: torch.Tensor,
    input_expert_splits: torch.Tensor,
    output: torch.Tensor,
    output_scales: torch.Tensor,
    output_rank_splits: torch.Tensor,
    output_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
):
    assert input.dim() == 2, f"{input.shape}"
    assert output.dim() == 2, f"{output.shape}"
    assert output.shape[1] == input.shape[1]

    # Prepare symmetric memory managed buffers for input, input_rank_splits, input_scales, and input_expert_splits.
    # - `input` shape (tokens, dim) -> to a sym mem managed buffer of shape (num_ranks, tokens, dim)
    # - `input_rank_splits` shape (num_ranks,) -> to a sym mem managed buffer of shape (num_ranks, num_ranks)`
    # - `input_scales` shape (tokens, dim//block_size) -> to a sym mem managed buffer of shape (num_ranks, tokens, dim//block_size)
    # - `input_expert_splits` shape (num_ranks, num_experts_per_rank) -> to a sym mem managed buffer of shape (num_ranks, num_ranks, num_experts_per_rank)
    input_hdl = symm_mem.rendezvous(input, group=group)
    input_rank_splits_hdl = symm_mem.rendezvous(input_rank_splits, group=group)
    input_scales_hdl = symm_mem.rendezvous(input_scales, group=group)
    input_expert_splits_hdl = symm_mem.rendezvous(input_expert_splits, group=group)

    input_ptrs = input_hdl.buffer_ptrs_dev
    input_rank_splits_ptrs = input_rank_splits_hdl.buffer_ptrs_dev
    input_scales_ptrs = input_scales_hdl.buffer_ptrs_dev
    input_expert_splits_ptrs = input_expert_splits_hdl.buffer_ptrs_dev
    signal_pad_ptrs = input_hdl.signal_pad_ptrs_dev
    dim = output.shape[1]
    scale_dim = input_scales.shape[-1]
    num_experts_per_rank = input_expert_splits.shape[-1]
    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK

    _mxfp8_all_to_all_expert_major_kernel[(num_blocks, 1, 1)](
        input_ptrs,
        input_scales_ptrs,
        input_rank_splits_ptrs,
        input_expert_splits_ptrs,
        output,
        output_scales,
        output_rank_splits,
        output_expert_splits,
        expert_padded_offsets,
        signal_pad_ptrs,
        dim=dim,
        scale_dim=scale_dim,
        num_experts_per_rank=num_experts_per_rank,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output


@triton.jit
def _mxfp8_all_to_all_expert_major_kernel(
    input_ptrs,
    input_scales_ptrs,
    input_rank_splits_ptr,
    input_expert_splits_ptr,
    output_ptr,
    output_scales_ptr,
    output_rank_splits_ptr,
    output_expert_splits_ptr,
    expert_padded_offsets_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,
    scale_dim: tl.constexpr,
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
        input_expert_splits_ptrs = input_expert_splits_ptr.to(tl.pointer_type(tl.uint64))

        # Accumulate tokens per local expert across all remote ranks
        expert_offsets = tl.arange(0, num_experts_per_rank)

        for remote_r in range(world_size):
            # Get pointer to remote rank's expert_splits tensor
            remote_expert_splits_ptr = tl.load(input_expert_splits_ptrs + remote_r).to(
                tl.pointer_type(tl.int64)
            )
            # input_expert_splits[remote_r, rank, :] contains tokens remote_r is sending to our local experts
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

        # Compute local padded starting offsets for each expert on this rank,
        # using the output_expert_splits metadata we just computed above.
        cumulative_offset = tl.zeros(
            [], dtype=tl.int64
        )  # annoyingly only way to init a int64 scalar 0 in triton?

        for expert_idx in range(num_experts_per_rank):
            # Store the starting offset for this expert
            tl.store(expert_padded_offsets_ptr + expert_idx, cumulative_offset)

            # Get total tokens for this expert across all remote ranks
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

            # Update cumulative offset for next expert
            cumulative_offset = cumulative_offset + padded_tokens

    # Barrier to ensure metadata exchange is complete before data transfer
    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")

    # ===== PHASE 2: Data Transfer with Padding =====
    # Transfer data expert-by-expert from the target remote_rank to local padded expert regions

    # One thread block per rank will update output_rank_splits
    if block_offset == 0:
        # Calculate total rows from this remote rank
        rank_splits_ptrs_typed = input_rank_splits_ptr.to(tl.pointer_type(tl.uint64))
        remote_rank_input_rank_splits_ptr = tl.load(rank_splits_ptrs_typed + remote_rank).to(
            tl.pointer_type(tl.int64)
        )
        num_rows_from_remote = tl.load(remote_rank_input_rank_splits_ptr + rank)
        tl.store(output_rank_splits_ptr + remote_rank, num_rows_from_remote)

    # Get base input pointer for this remote rank
    input_base_ptr = tl.load(
        input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.float8e4nv))

    input_scales_base_ptr = tl.load(
        input_scales_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.uint8))

    # Calculate input starting offset for this rank's data on remote_rank
    rank_splits_ptrs_typed = input_rank_splits_ptr.to(tl.pointer_type(tl.uint64))
    remote_rank_input_rank_splits_ptr = tl.load(rank_splits_ptrs_typed + remote_rank).to(
        tl.pointer_type(tl.int64)
    )
    rank_offsets = tl.arange(0, world_size)
    remote_rank_split_sizes_prefix = tl.load(
        remote_rank_input_rank_splits_ptr + rank_offsets, mask=rank_offsets < rank, other=0
    )
    input_row_offset = tl.sum(remote_rank_split_sizes_prefix)

    # Get expert splits for this remote rank
    input_expert_splits_ptrs_typed = input_expert_splits_ptr.to(tl.pointer_type(tl.uint64))
    remote_expert_splits_ptr = tl.load(input_expert_splits_ptrs_typed + remote_rank).to(
        tl.pointer_type(tl.int64)
    )
    remote_expert_splits_ptr = remote_expert_splits_ptr + rank * num_experts_per_rank

    # Process each expert's data from this remote rank
    for expert_idx in range(num_experts_per_rank):
        expert_tokens = tl.load(remote_expert_splits_ptr + expert_idx)

        if expert_tokens > 0:
            # Get output offset for this expert (with padding). This will be part of determining where we write to.
            expert_padded_start_offset = tl.load(expert_padded_offsets_ptr + expert_idx)

            # Determines where we write to locally.
            # After the expert padded start offset, we need to skip by tokens written from other remote ranks for this expert
            # Add tokens that previous remote ranks sent to this expert.
            prior_tokens_from_other_ranks_for_expert = tl.zeros([], dtype=tl.int64)
            for prev_remote in range(remote_rank):
                prev_remote_tokens_to_expert = tl.load(
                    output_expert_splits_ptr
                    + prev_remote * num_experts_per_rank
                    + expert_idx
                )
                prior_tokens_from_other_ranks_for_expert += prev_remote_tokens_to_expert

            # Determine where we read from on the remote rank.
            # Get cumulative offset on the remote rank's data for this expert.
            # Add up tokens from previous experts on that remote rank.
            prior_tokens_on_remote_rank_for_expert = tl.zeros([], dtype=tl.int64)
            for prev_expert in range(expert_idx):
                prior_tokens_on_remote_rank_for_expert += tl.load(
                    remote_expert_splits_ptr + prev_expert
                )

            # Calculate actual input/output pointers
            input_ptr = (
                input_base_ptr
                + (input_row_offset + prior_tokens_on_remote_rank_for_expert) * dim
            )
            output_ptr_expert = (
                output_ptr
                + (
                    expert_padded_start_offset
                    + prior_tokens_from_other_ranks_for_expert
                )
                * dim
            )

            input_scale_ptr = (
                input_scales_base_ptr
                + (input_row_offset + prior_tokens_on_remote_rank_for_expert)
                * scale_dim
            )
            output_scale_ptr_expert = (
                output_scales_ptr
                + (
                    expert_padded_start_offset
                    + prior_tokens_from_other_ranks_for_expert
                )
                * scale_dim
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
            total_scales = expert_tokens * scale_dim
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
