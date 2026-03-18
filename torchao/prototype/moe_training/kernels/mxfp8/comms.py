import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from torchao.prototype.moe_training.kernels.triton_utils import (
    blockwise_barrier,
    sync_threads,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx


class MXFP8BufferManager:
    """Manages reusable buffers for MXFP8 all-to-all operations across MoE layers."""

    def __init__(self):
        # Forward buffers - symmetric memory
        self.input_sym_mem_buf = None  # e4m3 data
        self.scales_sym_mem_buf = None  # e8m0 scales
        self.input_rank_splits_sym_mem_buf = None  # shape (ep_degree,)
        self.input_expert_splits_sym_mem_buf = (
            None  # shape (ep_degree, experts_per_rank)
        )

        # Forward buffers - output
        self.output = None  # e4m3 output data
        self.output_scales = None  # e8m0 output scales

        # Backward buffers - symmetric memory
        self.grad_out_sym_mem_buf = (
            None  # upstream grad is bf16, shape (ep_degree, output_data.shape[-1])
        )

        # Backward buffers - grad input
        self.grad_input_buf = None  #
        self.grad_input_splits_buf = None
        self.grad_input_expert_splits_buf = None

        # Configuration
        self.max_output_rows_per_rank = None

    def reset(self):
        """Clear all buffers (useful for testing or changing configs)."""
        self.__init__()


# Module-level singleton for buffer management
_default_buffer_manager = None


def get_buffer_manager():
    """Get the default buffer manager, creating it if necessary."""
    global _default_buffer_manager
    if _default_buffer_manager is None:
        _default_buffer_manager = MXFP8BufferManager()
    return _default_buffer_manager


# This performs dynamic mxfp8 quantization of the input tensor,
# followed by an on-device all-to-all-v operation as determined by the input_splits, implented via Triton + PyTorch symmetric memory.
# This kernel is an extension of the original bf16 version here:
# https://github.com/pytorch/torchtitan/blob/476a965f93432f4f1681bc1bac064d689a2d0cec/torchtitan/experiments/deepseek_v3/symm_mem_recipes/triton_on_device_all_to_all_v.py#L1
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
        buffer_manager: MXFP8BufferManager = None,
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
            buffer_manager: optional buffer manager for reusing buffers across layers.
        """
        assert input.dtype in (torch.float32, torch.bfloat16)

        # Get or create buffer manager
        buffers = buffer_manager or get_buffer_manager()

        # Enable symm mem for the group if not already enabled
        if not symm_mem.is_symm_mem_enabled_for_group(group):
            symm_mem.enable_symm_mem_for_group(group)

        buffers.max_output_rows_per_rank = max_output_rows_per_rank

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
        if buffers.input_sym_mem_buf is None:
            buffers.input_sym_mem_buf = symm_mem.empty(
                buffers.max_output_rows_per_rank,
                *input_data.shape[1:],
                dtype=input_data.dtype,
                device=input_data.device,
            )

        # Initialize symm mem buffer for float8 e8m0 scales (one time only)
        if buffers.scales_sym_mem_buf is None:
            buffers.scales_sym_mem_buf = symm_mem.empty(
                buffers.max_output_rows_per_rank,
                *input_scales.shape[1:],
                dtype=input_scales.dtype,
                device=input_scales.device,
            )

        # Initialize input rank splits buffer (one time only)
        if buffers.input_rank_splits_sym_mem_buf is None:
            buffers.input_rank_splits_sym_mem_buf = symm_mem.empty(
                *input_rank_splits.shape,
                dtype=input_rank_splits.dtype,
                device=input_rank_splits.device,
            )

        # Initialize expert splits buffer (one time only)
        if buffers.input_expert_splits_sym_mem_buf is None:
            buffers.input_expert_splits_sym_mem_buf = symm_mem.empty(
                *input_expert_splits.shape,
                dtype=input_expert_splits.dtype,
                device=input_expert_splits.device,
            )

        # Allocate buffers for output data, scales. This alloccates huge overallocateed buffers once,
        # to be shared between all MoE layers.
        if buffers.output is None:
            buffers.output = torch.zeros(
                buffers.max_output_rows_per_rank,
                *input_data.shape[1:],
                dtype=input_data.dtype,
                device=input_data.device,
            )
        if buffers.output_scales is None:
            buffers.output_scales = torch.zeros(
                buffers.max_output_rows_per_rank,
                *input_scales.shape[1:],
                dtype=input_scales.dtype,
                device=input_scales.device,
            )

        # Copy quantized data, scales, and output splits to symm mem buffers
        buffers.input_sym_mem_buf.narrow(0, 0, input_data.shape[0]).copy_(input_data)

        # Copy input rank splits to symm mem buffer
        buffers.input_rank_splits_sym_mem_buf.copy_(input_rank_splits)

        # Copy input expert splits to symm mem buffer
        buffers.input_expert_splits_sym_mem_buf.copy_(input_expert_splits)

        # Copy input scales to symm mem buffer
        buffers.scales_sym_mem_buf.narrow(0, 0, input_scales.shape[0]).copy_(
            input_scales
        )

        # Allocate buffer for padded expert offsets (one offset per expert)
        num_experts_per_rank = input_expert_splits.shape[1]
        expert_padded_offsets = torch.empty(
            num_experts_per_rank, dtype=torch.int64, device=input_data.device
        )
        output_rank_splits = torch.empty_like(input_rank_splits)
        output_expert_splits = torch.empty_like(input_expert_splits)

        # Shuffle input to output
        _mxfp8_syncless_all_to_all_expert_major_launcher(
            buffers.input_sym_mem_buf,
            buffers.scales_sym_mem_buf,
            buffers.input_rank_splits_sym_mem_buf,
            buffers.input_expert_splits_sym_mem_buf,
            buffers.output,
            buffers.output_scales,
            output_rank_splits,
            output_expert_splits,
            expert_padded_offsets,
            group=group,
        )

        hp_dtype = input.dtype

        # Wrap output as MXTensor so autograd sees it as bf16 (backward will receive bf16 gradients)
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

        # Saving for backward: output splits in forward is the input splits in backward
        ctx.group = group
        ctx.input_shape = input_data.shape
        ctx.input_scales_shape = input_scales.shape
        ctx.expert_splits_shape = input_expert_splits.shape
        ctx.hp_dtype = hp_dtype
        ctx.max_output_rows_per_rank = max_output_rows_per_rank
        ctx.buffers = buffers
        ctx.save_for_backward(
            output_rank_splits, output_expert_splits, expert_padded_offsets
        )

        # Return MXTensor (autograd sees as bf16) + metadata
        return (
            mx_output,
            output_rank_splits,
            output_expert_splits,
            expert_padded_offsets,
        )

    @staticmethod
    @torch.compiler.disable
    def backward(
        ctx,
        grad_output,
        grad_rank_splits,
        grad_expert_splits,
        grad_expert_padded_offsets,
    ):
        """
        Backward is implemented as a shuffle of the output's gradients to the input.
        Uses bf16 directly without quantization/dequantization.
        Args:
            `grad_output`: output's gradients passed from the downstream in bf16 (from MXTensor).
            `grad_rank_splits`: unused.
            `grad_expert_splits`: unused.
            `grad_expert_padded_offsets`: unused.
        """
        grad_output_rank_splits, grad_output_expert_splits, _ = ctx.saved_tensors
        buffers = ctx.buffers
        assert grad_output.dtype == ctx.hp_dtype, (
            f"Expected grad_output dtype {ctx.hp_dtype}, got {grad_output.dtype}"
        )

        # Initialize grad_output bf16 sym mem buffer (one time only)
        if buffers.grad_out_sym_mem_buf is None:
            buffers.grad_out_sym_mem_buf = symm_mem.empty(
                buffers.max_output_rows_per_rank,
                *grad_output.shape[1:],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

        # Copy grad_output to symm mem buffer (no quantization)
        buffers.grad_out_sym_mem_buf.narrow(0, 0, grad_output.shape[0]).copy_(
            grad_output
        )

        # Copy in rank splits to symm mem buffer
        buffers.input_rank_splits_sym_mem_buf.copy_(grad_output_rank_splits)

        # Copy in expert splits to symm mem buffer
        buffers.input_expert_splits_sym_mem_buf.copy_(grad_output_expert_splits)

        # Allocate buffers for grad_input and splits if necessary
        if buffers.grad_input_buf is None:
            buffers.grad_input_buf = torch.empty(
                ctx.max_output_rows_per_rank,
                *grad_output.shape[1:],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

        if buffers.grad_input_splits_buf is None:
            buffers.grad_input_splits_buf = torch.empty_like(grad_output_rank_splits)
        if buffers.grad_input_expert_splits_buf is None:
            buffers.grad_input_expert_splits_buf = torch.empty(
                *ctx.expert_splits_shape,
                dtype=grad_output_expert_splits.dtype,
                device=grad_output_expert_splits.device,
            )

        # Allocate buffer for padded expert offsets in backward
        num_experts_per_rank = grad_output_expert_splits.shape[1]
        grad_expert_padded_offsets_buf = torch.empty(
            num_experts_per_rank, dtype=torch.int64, device=grad_output.device
        )

        # Shuffle gradients back to the input using bf16 kernel
        _bf16_syncless_all_to_all_expert_major_launcher(
            buffers.grad_out_sym_mem_buf,  # input
            buffers.input_rank_splits_sym_mem_buf,  # input rank splits
            buffers.input_expert_splits_sym_mem_buf,  # input expert splits
            buffers.grad_input_buf,  # output
            buffers.grad_input_splits_buf,  # output rank splits
            buffers.grad_input_expert_splits_buf,  # output expert splits
            grad_expert_padded_offsets_buf,  # expert padded offsets
            group=ctx.group,
        )

        # Return grad_input (no dequantization needed)
        orig_input_rows = ctx.input_shape[0]
        return (
            buffers.grad_input_buf[:orig_input_rows],
            None,
            None,
            None,
            None,
            None,
        )


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


# Triton launcher for bf16 backward pass (no quantization)
def _bf16_syncless_all_to_all_expert_major_launcher(
    input: torch.Tensor,
    input_rank_splits: torch.Tensor,
    input_expert_splits: torch.Tensor,
    output: torch.Tensor,
    output_rank_splits: torch.Tensor,
    output_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
):
    """Bf16 version without scale handling for backward pass."""
    assert input.dim() == 2, f"{input.shape}"
    assert output.dim() == 2, f"{output.shape}"
    assert output.shape[1] == input.shape[1]

    input_hdl = symm_mem.rendezvous(input, group=group)
    input_rank_splits_hdl = symm_mem.rendezvous(input_rank_splits, group=group)
    input_expert_splits_hdl = symm_mem.rendezvous(input_expert_splits, group=group)

    input_ptrs = input_hdl.buffer_ptrs_dev
    input_rank_splits_ptrs = input_rank_splits_hdl.buffer_ptrs_dev
    input_expert_splits_ptrs = input_expert_splits_hdl.buffer_ptrs_dev
    signal_pad_ptrs = input_hdl.signal_pad_ptrs_dev
    dim = output.shape[1]
    num_experts_per_rank = input_expert_splits.shape[-1]
    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK

    _bf16_all_to_all_expert_major_kernel[(num_blocks, 1, 1)](
        input_ptrs,
        input_rank_splits_ptrs,
        input_expert_splits_ptrs,
        output,
        output_rank_splits,
        output_expert_splits,
        expert_padded_offsets,
        signal_pad_ptrs,
        dim=dim,
        num_experts_per_rank=num_experts_per_rank,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output


@triton.jit
def _compute_expert_metadata_and_offsets(
    input_expert_splits_ptr,
    output_expert_splits_ptr,
    expert_padded_offsets_ptr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
):
    """Phase 1: Exchange expert_splits from all ranks and compute padded offsets."""
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
        remote_expert_split_values = tl.load(remote_expert_splits_ptr + expert_offsets)

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
                output_expert_splits_ptr + remote_r * num_experts_per_rank + expert_idx
            )
            expert_tokens_total += expert_tokens_from_remote

        # Round up tokens for this expert to multiple of 32
        padded_tokens: tl.int64 = ((expert_tokens_total + 31) // 32) * 32

        # Update cumulative offset for next expert
        cumulative_offset = cumulative_offset + padded_tokens


@triton.jit
def _transfer_expert_data(
    remote_rank: tl.constexpr,
    block_offset: tl.constexpr,
    input_ptrs,
    input_scales_ptrs,
    input_rank_splits_ptr,
    input_expert_splits_ptr,
    output_ptr,
    output_scales_ptr,
    output_rank_splits_ptr,
    output_expert_splits_ptr,
    expert_padded_offsets_ptr,
    dim: tl.constexpr,
    scale_dim: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 2: Transfer data expert-by-expert from remote_rank to local padded expert regions."""
    # One thread block per rank will update output_rank_splits
    if block_offset == 0:
        # Calculate total rows from this remote rank
        rank_splits_ptrs_typed = input_rank_splits_ptr.to(tl.pointer_type(tl.uint64))
        remote_rank_input_rank_splits_ptr = tl.load(
            rank_splits_ptrs_typed + remote_rank
        ).to(tl.pointer_type(tl.int64))
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
    remote_rank_input_rank_splits_ptr = tl.load(
        rank_splits_ptrs_typed + remote_rank
    ).to(tl.pointer_type(tl.int64))
    rank_offsets = tl.arange(0, world_size)
    remote_rank_split_sizes_prefix = tl.load(
        remote_rank_input_rank_splits_ptr + rank_offsets,
        mask=rank_offsets < rank,
        other=0,
    )
    input_row_offset = tl.sum(remote_rank_split_sizes_prefix)

    # Get expert splits for this remote rank
    input_expert_splits_ptrs_typed = input_expert_splits_ptr.to(
        tl.pointer_type(tl.uint64)
    )
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


@triton.jit
def _transfer_expert_data_bf16(
    remote_rank: tl.constexpr,
    block_offset: tl.constexpr,
    input_ptrs,
    input_rank_splits_ptr,
    input_expert_splits_ptr,
    output_ptr,
    output_rank_splits_ptr,
    output_expert_splits_ptr,
    expert_padded_offsets_ptr,
    dim: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 2: Transfer bf16 data expert-by-expert (no scales)."""
    # One thread block per rank will update output_rank_splits
    if block_offset == 0:
        rank_splits_ptrs_typed = input_rank_splits_ptr.to(tl.pointer_type(tl.uint64))
        remote_rank_input_rank_splits_ptr = tl.load(
            rank_splits_ptrs_typed + remote_rank
        ).to(tl.pointer_type(tl.int64))
        num_rows_from_remote = tl.load(remote_rank_input_rank_splits_ptr + rank)
        tl.store(output_rank_splits_ptr + remote_rank, num_rows_from_remote)

    # Get base input pointer for this remote rank
    input_base_ptr = tl.load(
        input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank
    ).to(tl.pointer_type(tl.bfloat16))

    # Calculate input starting offset for this rank's data on remote_rank
    rank_splits_ptrs_typed = input_rank_splits_ptr.to(tl.pointer_type(tl.uint64))
    remote_rank_input_rank_splits_ptr = tl.load(
        rank_splits_ptrs_typed + remote_rank
    ).to(tl.pointer_type(tl.int64))
    rank_offsets = tl.arange(0, world_size)
    remote_rank_split_sizes_prefix = tl.load(
        remote_rank_input_rank_splits_ptr + rank_offsets,
        mask=rank_offsets < rank,
        other=0,
    )
    input_row_offset = tl.sum(remote_rank_split_sizes_prefix)

    # Get expert splits for this remote rank
    input_expert_splits_ptrs_typed = input_expert_splits_ptr.to(
        tl.pointer_type(tl.uint64)
    )
    remote_expert_splits_ptr = tl.load(input_expert_splits_ptrs_typed + remote_rank).to(
        tl.pointer_type(tl.int64)
    )
    remote_expert_splits_ptr = remote_expert_splits_ptr + rank * num_experts_per_rank

    # Process each expert's data from this remote rank
    for expert_idx in range(num_experts_per_rank):
        expert_tokens = tl.load(remote_expert_splits_ptr + expert_idx)

        if expert_tokens > 0:
            expert_padded_start_offset = tl.load(expert_padded_offsets_ptr + expert_idx)

            # Calculate write offset
            prior_tokens_from_other_ranks_for_expert = tl.zeros([], dtype=tl.int64)
            for prev_remote in range(remote_rank):
                prev_remote_tokens_to_expert = tl.load(
                    output_expert_splits_ptr
                    + prev_remote * num_experts_per_rank
                    + expert_idx
                )
                prior_tokens_from_other_ranks_for_expert += prev_remote_tokens_to_expert

            # Calculate read offset
            prior_tokens_on_remote_rank_for_expert = tl.zeros([], dtype=tl.int64)
            for prev_expert in range(expert_idx):
                prior_tokens_on_remote_rank_for_expert += tl.load(
                    remote_expert_splits_ptr + prev_expert
                )

            # Calculate input/output pointers
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

            # Copy data for this expert
            total_elems = expert_tokens * dim
            num_blocks = tl.cdiv(total_elems, BLOCK_SIZE)
            for block_idx in tl.range(num_blocks):
                if block_idx % BLOCKS_PER_REMOTE_RANK == block_offset:
                    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < total_elems
                    data = tl.load(input_ptr + offs, mask=mask, other=0.0)
                    tl.store(output_ptr_expert + offs, data, mask=mask)


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

    # PHASE 1: exhange input expert splits between ranks, and compute local padded token group offsets.
    if tl.program_id(0) == 0:
        _compute_expert_metadata_and_offsets(
            input_expert_splits_ptr,
            output_expert_splits_ptr,
            expert_padded_offsets_ptr,
            rank,
            world_size,
            num_experts_per_rank,
        )

    # Barrier to ensure metadata exchange is complete before data transfer
    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="acq_rel")

    # PHASE 2: transfer tokens destined for local experts, iterating expert by expert,
    # gathering tokens from other remote ranks.
    _transfer_expert_data(
        remote_rank,
        block_offset,
        input_ptrs,
        input_scales_ptrs,
        input_rank_splits_ptr,
        input_expert_splits_ptr,
        output_ptr,
        output_scales_ptr,
        output_rank_splits_ptr,
        output_expert_splits_ptr,
        expert_padded_offsets_ptr,
        dim,
        scale_dim,
        num_experts_per_rank,
        rank,
        world_size,
        BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE,
    )

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    return


@triton.jit
def _bf16_all_to_all_expert_major_kernel(
    input_ptrs,
    input_rank_splits_ptr,
    input_expert_splits_ptr,
    output_ptr,
    output_rank_splits_ptr,
    output_expert_splits_ptr,
    expert_padded_offsets_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,
    num_experts_per_rank: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bf16 kernel for backward pass without quantization."""
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    # PHASE 1: exchange input expert splits between ranks, and compute local padded token group offsets.
    if tl.program_id(0) == 0:
        _compute_expert_metadata_and_offsets(
            input_expert_splits_ptr,
            output_expert_splits_ptr,
            expert_padded_offsets_ptr,
            rank,
            world_size,
            num_experts_per_rank,
        )

    # Barrier to ensure metadata exchange is complete before data transfer
    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="acq_rel")

    # PHASE 2: transfer tokens destined for local experts, iterating expert by expert,
    # gathering tokens from other remote ranks.
    _transfer_expert_data_bf16(
        remote_rank,
        block_offset,
        input_ptrs,
        input_rank_splits_ptr,
        input_expert_splits_ptr,
        output_ptr,
        output_rank_splits_ptr,
        output_expert_splits_ptr,
        expert_padded_offsets_ptr,
        dim,
        num_experts_per_rank,
        rank,
        world_size,
        BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE,
    )

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    return
