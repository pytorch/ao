from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import ceil_div


def torch_to_blocked_per_group_2d(
    x_scales: Tensor, group_offs: Tensor, K: int, block_size: int = 32
) -> Tuple[Tensor, Tensor]:
    """
    Convert scales to blocked format for a 2D tensor (input activations / token groups)

    Args:
        x_scales: Tensor with per group scales in blocked format concatenated into one tensor.
        group_offs: Tensor of shape (num_groups,) which contains the end index of each group along the Mg dimension.
        Mg: total size of all groups summed together
        K: K dim size

    Returns:
        blocked_scales: Tensor
        start_row_after_padding: Tensor of shape (num_groups,) which contains the start row after padding for each group.
    """

    assert x_scales.ndim == 2, "x_scales must be 2D"
    assert block_size == 32, "Only block_size=32 is supported for now"
    blocked_scales_list = []
    start_row_after_padding_list = [0]
    group_start_idx = 0
    for i, group_end_idx in enumerate(group_offs.tolist()):
        group_size = group_end_idx - group_start_idx
        prev_start_row_after_padding = start_row_after_padding_list[i]
        if group_size == 0:
            start_row_after_padding_list.append(prev_start_row_after_padding)
            continue

        # Convert group scales to blocked format
        group_scales = x_scales[group_start_idx:group_end_idx]
        group_scales_blocked = to_blocked(group_scales)
        blocked_scales_list.append(group_scales_blocked)

        # Calculate the start row after padding
        scaling_groups_per_row = K // block_size
        rows_for_group = group_scales_blocked.numel() // scaling_groups_per_row
        new_start_row = prev_start_row_after_padding + rows_for_group
        start_row_after_padding_list.append(new_start_row)

        # Update next group start index
        group_start_idx = group_end_idx

    blocked_scales = torch.cat(blocked_scales_list, dim=0).contiguous()
    blocked_scales = blocked_scales.reshape(-1, K // 32)
    start_row_after_padding = torch.tensor(
        start_row_after_padding_list, device=x_scales.device, dtype=torch.int64
    )
    return blocked_scales, start_row_after_padding


def torch_to_blocked_per_group_2d2d_lhs(
    x_scales: Tensor, group_offs: Tensor, block_size: int = 32
) -> Tuple[Tensor, Tensor]:
    """
    Convert scales to blocked format for a 2D tensor (input activations) when scaling along the contraction dimension.

    Args:
        x_scales: Tensor with per group scales in blocked format concatenated into one tensor.
        group_offs: Tensor of shape (num_groups,) which contains the end index of each group along the total_k dimension.
        total_K: total size of all groups summed together

    Returns:
        blocked_scales: Tensor
        start_row_after_padding: Tensor of shape (num_groups,) which contains the start row after padding for each group.
    """
    assert x_scales.ndim == 2, "x_scales must be 2D"
    assert block_size == 32, "Only block_size=32 is supported for now"
    blocked_scales_list = []
    start_col_after_padding_list = [0]
    group_start_idx = 0
    for i, group_end_idx in enumerate(group_offs.tolist()):
        group_size = group_end_idx - group_start_idx
        prev_start_row_after_padding = start_col_after_padding_list[i]
        if group_size == 0:
            start_col_after_padding_list.append(prev_start_row_after_padding)
            continue

        # Convert group scales to blocked format
        group_scales = x_scales[:, group_start_idx:group_end_idx]
        group_scales_blocked = to_blocked(group_scales)
        cols_after_padding = ceil_div(group_size, 4) * 4
        blocked_scales_list.append(group_scales_blocked.reshape(-1, cols_after_padding))

        # Calculate the start row after padding
        new_start_col = prev_start_row_after_padding + cols_after_padding
        start_col_after_padding_list.append(new_start_col)

        # Update next group start index
        group_start_idx = group_end_idx

    blocked_scales = torch.cat(blocked_scales_list, dim=1)
    start_cols_after_padding = torch.tensor(
        start_col_after_padding_list, device=x_scales.device, dtype=torch.int64
    )
    return blocked_scales, start_cols_after_padding


def torch_to_blocked_per_group_3d(weight_scales: Tensor) -> Tensor:
    """
    Convert scales to blocked format for each group for a 3D tensor (expert weights)

    Args:
        scales: Tensor of shape (E, N, K//block_size)
        group_offs: Tensor of shape (num_groups,) which contains the end index of each group along the
    """

    blocked_scales_list = []
    num_groups = weight_scales.shape[0]
    for i in range(num_groups):
        group_scales = weight_scales[i]
        group_scales_blocked = to_blocked(group_scales)
        blocked_scales_list.append(group_scales_blocked)
    weight_scales_blocked = torch.stack(blocked_scales_list, dim=0).contiguous()
    weight_scales_blocked = weight_scales_blocked.reshape(num_groups, -1)
    return weight_scales_blocked


def compute_per_group_blocked_scale_offsets(offsets: torch.Tensor):
    """
    Rounds each integer in a 1D PyTorch tensor up to the nearest multiple of 128.

    Args:
    offsets: A 1D PyTorch tensor of integers in ascending sorted order, representing the end index of each group along the Mg dimension.

    Returns:
        - group_sizes: A 1D PyTorch tensor of integers representing the size of each group.
        - starting_row_after_padding: 1D integer tensor representing the starting row after padding each to blocked format.
    """
    # Calculate group sizes
    zero = torch.tensor([0], dtype=offsets.dtype, device=offsets.device)
    group_sizes = torch.diff(offsets, prepend=zero).to(torch.int64)

    # Round each group size up to the nearest multiple of 128
    rounded_group_sizes = ceil_div(group_sizes, 128) * 128

    # Calculate the starting row after padding for each group
    starting_row_after_padding = torch.cumsum(rounded_group_sizes, dim=0)

    # Must start with 0
    starting_row_after_padding = torch.cat([zero, starting_row_after_padding])
    return group_sizes, starting_row_after_padding


def compute_per_group_blocked_scale_offsets_2d2d_lhs(offsets: torch.Tensor):
    """
    Performs round_up(x, 4) on each element in a 1D offsets tensor,
    to compute the starting offsets of each group after scaling along the contraction dimension.

    Args:
        offsets: A 1D PyTorch tensor of integers in ascending sorted order, representing the end index of each group along the Mg dimension.

    Returns:
        - starting_row_after_padding: 1D integer tensor representing the starting row after padding each to blocked format.
    """
    # Calculate group sizes
    zero = torch.tensor([0], dtype=offsets.dtype, device=offsets.device)
    group_sizes = torch.diff(offsets, prepend=zero).to(torch.int64)

    # After scaling with block_size 32, each group size up to the nearest multiple of 4
    rounded_group_sizes = ceil_div(group_sizes, 4) * 4

    # Calculate the starting row after padding for each group
    starting_col_after_padding = torch.cumsum(rounded_group_sizes, dim=0)

    # Must start with 0
    starting_col_after_padding = torch.cat([zero, starting_col_after_padding])
    return group_sizes, starting_col_after_padding


def triton_mx_block_rearrange_per_group_2d(
    scales_tensor: torch.Tensor,
    input_group_end_offsets: torch.Tensor,
    output_group_start_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Rearranges an E8M0 tensor scale to block-scaled swizzle format.
    This format is suitable for Tmem as described in NVIDIA documentation:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    Args:
        scales_tensor: Input tensor containing e8m0 scales for each logical group of a target tensor.
        input_group_end_offsets: tensor of int32 values representing group end indexes for the input scales
        output_group_start_offsets: tensor of int32 values representing pre-computed group start indexes after blocked format padding
    Returns:
        - Rearranged tensor in block-scaled swizzle format
    """
    assert scales_tensor.ndim == 2, "scales tensor must be 2d"
    assert scales_tensor.element_size() == 1, (
        "Expected element size to be 1 byte (8 bits)"
    )
    rows, cols = scales_tensor.shape
    num_groups = input_group_end_offsets.numel()

    # Final offset is the total number of rows in the tensor
    padded_rows = output_group_start_offsets[-1]

    num_col_blocks = ceil_div(cols, 4)
    padded_cols = num_col_blocks * 4
    output = scales_tensor.new_empty((padded_rows, padded_cols))

    # Output block stride for the rearranged format
    BLOCK_ROWS, BLOCK_COLS = 128, 4
    output_stride_per_block = BLOCK_ROWS * BLOCK_COLS
    output_stride_per_row_of_blocks = (
        BLOCK_ROWS * BLOCK_COLS * (padded_cols // BLOCK_COLS)
    )

    # We parallelize per group and per col block.
    # Rows per group is variable so we just loop through row blocks per group, per col block.
    grid = lambda META: (
        num_groups,
        num_col_blocks,
    )
    triton_scale_swizzle_per_group_2d[grid](
        # Input scales
        scales_tensor.view(torch.uint8),
        scales_tensor.stride(0),
        scales_tensor.stride(1),
        rows,
        cols,
        num_groups,
        # Original offsets (to read from)
        input_group_end_offsets,
        # Output scales tensor and group offsets after padding (to write to)
        output.view(torch.uint8),
        output.stride(0),
        output_group_start_offsets,
        output_stride_per_block,
        output_stride_per_row_of_blocks,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )
    return output


@triton.jit
def triton_scale_swizzle_per_group_2d(
    scales_ptr,  # (M, K//block_size)
    scales_stride_dim0,
    scales_stride_dim1,
    scale_rows,
    scale_cols,
    num_groups,
    orig_offsets,  # (num_groups,)
    output_scales_ptr,
    output_scales_stride_dim0,
    output_scales_group_offsets,  # (num_groups,)
    output_stride_per_block,
    output_stride_per_row_of_blocks,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    group_pid = tl.program_id(0)
    block_col_pid = tl.program_id(1)
    # Input scales row range for this group
    input_group_start_row = tl.load(
        orig_offsets + group_pid - 1, mask=group_pid > 0, other=0
    )
    input_group_end_row = tl.load(
        orig_offsets + group_pid, mask=group_pid < num_groups, other=0
    )
    # Output scales start row we will begin writing to
    output_group_start_row = tl.load(
        output_scales_group_offsets + group_pid, mask=group_pid < num_groups, other=0
    )
    # Calculate destination indices for each row and col in block swizzled layout.
    # We can reuse this swizzle transformation on each block of data we read.
    row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
    col_offs = tl.arange(0, BLOCK_COLS)[None, :]
    r_div_32 = row_offs // 32
    r_mod_32 = row_offs % 32

    # Rearrange to (32, 4, 4) then to final (32, 16) coordinates
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + col_offs

    # Flatten
    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))

    # For this group and col block, we iterate through row blocks, reading (BLOCK_ROWS, BLOCK_COLS) from the input scales.
    # We track how many row blocks we have iterated through.
    block_row_id = 0
    current_start_row = input_group_start_row

    # TODO: Investigate if it is possible and beneficial to parallelize along
    # row blocks as well, and get rid of this loop.
    while current_start_row < input_group_end_row:
        # Read block of input scales
        block_row_offs = current_start_row + row_offs
        block_col_offs = block_col_pid * BLOCK_COLS + col_offs
        block_offs = (
            block_row_offs * scales_stride_dim0 + block_col_offs * scales_stride_dim1
        )
        mask = (block_row_offs < input_group_end_row) & (block_col_offs < scale_cols)
        input_scales = tl.load(scales_ptr + block_offs, mask=mask, other=0.0)
        scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))
        # Calculate block offset using provided output block stride
        output_block_offsets = (
            output_group_start_row * output_scales_stride_dim0
            + (block_row_id * output_stride_per_row_of_blocks)
            + (block_col_pid * output_stride_per_block)
        )
        # Apply swizzling for write to gmem
        tl.store(
            output_scales_ptr + output_block_offsets + dest_indices_flat,
            scales_flat,
        )
        # Update row block id to next block
        block_row_id += 1
        current_start_row += BLOCK_ROWS


def triton_mx_block_rearrange_per_group_3d(scale_tensor: torch.Tensor) -> torch.Tensor:
    """
    Rearranges an E8M0 tensor scale to block-scaled swizzle format.

    This format is suitable for Tmem as described in NVIDIA documentation:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        scale_tensor: Input tensor in row-major format with 8-bit elements

    Returns:
        Rearranged tensor in block-scaled swizzle format
    """
    assert scale_tensor.ndim == 3, "scales tensor must be 3d"
    assert scale_tensor.element_size() == 1, (
        "Expected element size to be 1 byte (8 bits)"
    )

    num_groups, rows, cols = scale_tensor.shape
    input_stride_dim0 = scale_tensor.stride(0)
    input_stride_dim1 = scale_tensor.stride(1)
    input_stride_dim2 = scale_tensor.stride(2)

    # Calculate blocks needed and allocate output tensor
    num_row_blocks = triton.cdiv(rows, 128)
    num_col_blocks = triton.cdiv(cols, 4)
    padded_rows = num_row_blocks * 128
    padded_cols = num_col_blocks * 4
    output = scale_tensor.new_empty((num_groups, padded_rows * padded_cols))
    output_stride_dim0 = output.stride(0)

    # We probably want handle multiple blocks per tile but for now keep it simple
    BLOCK_ROWS, BLOCK_COLS = 128, 4

    # Output block stride for the rearranged format
    output_block_stride = BLOCK_ROWS * BLOCK_COLS * (padded_cols // BLOCK_COLS)

    grid = lambda META: (
        num_groups,
        num_row_blocks,
        num_col_blocks,
    )

    triton_scale_swizzle_per_group_3d[grid](
        scale_tensor.view(torch.uint8),
        input_stride_dim0,
        input_stride_dim1,
        input_stride_dim2,
        output.view(torch.uint8),
        output_stride_dim0,
        output_block_stride,
        rows,
        cols,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )

    return output


@triton.jit
def triton_scale_swizzle_per_group_3d(
    input_ptr,
    input_stride_dim0,
    input_stride_dim1,
    input_stride_dim2,
    output_ptr,
    output_stride_dim0,
    output_block_stride,
    scale_rows,
    scale_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_group = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_col = tl.program_id(2)

    # Update base pointers based on this group id
    input_ptr += pid_group * input_stride_dim0
    output_ptr += pid_group * output_stride_dim0

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    # Calculate starting row and column for this tile
    start_row = pid_row * BLOCK_ROWS
    start_col = pid_col * BLOCK_COLS
    global_rows = start_row + rows
    global_cols = start_col + cols

    mask = (global_rows < scale_rows) & (global_cols < scale_cols)

    input_scales = tl.load(
        input_ptr + global_rows * input_stride_dim1 + global_cols * input_stride_dim2,
        mask=mask,
        other=0.0,
    )

    r_div_32 = rows // 32
    r_mod_32 = rows % 32

    # 2) Rearrange to (32, 4, 4) then to final (32, 16) coordinates
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

    # Flatten
    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))
    scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

    # Calculate block offset using provided output block stride
    LOCAL_NUMEL = BLOCK_ROWS * BLOCK_COLS
    block_offset = pid_col * LOCAL_NUMEL + (pid_row * output_block_stride)

    tl.store(
        output_ptr + block_offset + dest_indices_flat,
        scales_flat,
    )


def triton_mx_block_rearrange_per_group_2d2d_lhs(
    scales_tensor: torch.Tensor,
    input_group_end_offsets: torch.Tensor,
    output_group_start_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Rearranges an E8M0 tensor scale to block-scaled swizzle format on a per group basis,
    where the groups are along the contraction dimension of the GEMM.

    This format is suitable for Tmem as described in NVIDIA documentation:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        scales_tensor: Input tensor containing e8m0 scales for each logical group of a target tensor.
        input_group_end_offsets: tensor of int32 values representing group end indexes for the input scales
        output_group_start_offsets: tensor of int32 values representing pre-computed group start indexes after blocked format padding
    Returns:
        - Rearranged tensor in block-scaled swizzle format
    """
    assert scales_tensor.ndim == 2, "scales tensor must be 2d"
    assert scales_tensor.element_size() == 1, (
        "Expected element size to be 1 byte (8 bits)"
    )
    rows, cols = scales_tensor.shape
    # Calculate blocks needed
    num_groups = input_group_end_offsets.numel()
    num_row_blocks = ceil_div(rows, 128)
    padded_rows = num_row_blocks * 128

    # output_group_start_offsets always starts with 0 and ends with the total number of cols
    padded_cols = output_group_start_offsets[-1]
    output = scales_tensor.new_empty((padded_rows, padded_cols))

    # Output block stride for the rearranged format
    BLOCK_ROWS, BLOCK_COLS = 128, 4
    output_stride_per_block = BLOCK_ROWS * BLOCK_COLS
    num_col_blocks = padded_cols // BLOCK_COLS
    output_stride_per_row_of_blocks = output_stride_per_block * num_col_blocks

    # We parallelize per group and per row block.
    # Cols per group is variable, so we just loop through col blocks for each group.
    grid = lambda META: (
        num_groups,
        num_row_blocks,
    )
    triton_scale_swizzle_per_group_2d2d_lhs[grid](
        # Input scales
        scales_tensor.view(torch.uint8),
        scales_tensor.stride(0),
        scales_tensor.stride(1),
        rows,
        cols,
        num_groups,
        # Original offsets (to read from)
        input_group_end_offsets,
        # Output scales tensor and group offsets after padding (to write to)
        output.view(torch.uint8),
        output.stride(0),
        output.stride(1),
        output_group_start_offsets,
        output_stride_per_block,
        output_stride_per_row_of_blocks,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )
    return output


@triton.jit
def triton_scale_swizzle_per_group_2d2d_lhs(
    scales_ptr,  # (K, total_M//block_size)
    scales_stride_dim0,
    scales_stride_dim1,
    scale_rows,
    scale_cols,
    num_groups,
    orig_offsets,  # (num_groups,)
    output_scales_ptr,
    output_scales_stride_dim0,
    output_scales_stride_dim1,
    output_scales_group_offsets,  # (num_groups,)
    output_stride_per_block,
    output_stride_per_row_of_blocks_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    DEBUG: tl.constexpr = True,
):
    group_pid = tl.program_id(0)
    block_row_pid = tl.program_id(1)

    # Input scales row range for this group
    input_group_start_col = tl.load(
        orig_offsets + group_pid - 1, mask=group_pid > 0, other=0
    )
    input_group_end_col = tl.load(
        orig_offsets + group_pid, mask=group_pid < num_groups, other=0
    )
    # Output scales start row we will begin writing to
    output_group_start_col = tl.load(
        output_scales_group_offsets + group_pid, mask=group_pid < num_groups, other=0
    )

    out_stride_per_row_of_blocks = tl.load(output_stride_per_row_of_blocks_ptr)

    # Calculate destination indices for each row and col in block swizzled layout.
    # We can reuse this swizzle transformation on each block of data we read.
    row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
    col_offs = tl.arange(0, BLOCK_COLS)[None, :]
    r_div_32 = row_offs // 32
    r_mod_32 = row_offs % 32

    # Rearrange to (32, 4, 4) then to final (32, 16) coordinates
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + col_offs

    # Flatten
    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))

    # For this group and row block, we iterate through col blocks, reading (BLOCK_ROWS, BLOCK_COLS) from the input scales.
    # We track how many col blocks we have iterated through.
    curr_input_start_col = input_group_start_col
    curr_out_start_col_block = output_group_start_col // BLOCK_COLS
    while curr_input_start_col < input_group_end_col:
        # Read block of input scales
        block_row_offs = block_row_pid * BLOCK_ROWS + row_offs
        block_col_offs = curr_input_start_col + col_offs
        block_offs = (
            block_row_offs * scales_stride_dim0 + block_col_offs * scales_stride_dim1
        )
        mask = (block_row_offs < scale_rows) & (block_col_offs < input_group_end_col)
        input_scales = tl.load(scales_ptr + block_offs, mask=mask, other=0.0)
        scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

        # Calculate block offset using provided output block stride
        tgt_row_off = block_row_pid * out_stride_per_row_of_blocks
        tgt_col_off = curr_out_start_col_block * output_stride_per_block

        output_block_offsets = tgt_row_off + tgt_col_off
        if DEBUG:
            tl.device_print("block_row_pid: ", block_row_pid)
            tl.device_print("group_pid: ", group_pid)
            tl.device_print("tgt_row_block", block_row_pid)
            tl.device_print("tgt_col_block", curr_out_start_col_block)
            tl.device_print("tgt_row_off: ", tgt_row_off)
            tl.device_print("tgt_col_off: ", tgt_col_off)
            tl.device_print("global_off:", tgt_row_off + tgt_col_off)

        # Apply swizzling for write to gmem
        tl.store(
            output_scales_ptr + output_block_offsets + dest_indices_flat,
            scales_flat,
        )

        # Advance to next col block
        curr_input_start_col += BLOCK_COLS
        curr_out_start_col_block += 1
