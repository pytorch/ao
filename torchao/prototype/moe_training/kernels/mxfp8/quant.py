from typing import Tuple

import torch
from torch import Tensor
from torch.utils._triton import has_triton

from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import (
    ceil_div,
    is_cuda_version_at_least,
    is_sm_at_least_100,
    torch_version_at_least,
)


def torch_to_blocked_2d_M_groups(
    x_scales: Tensor, group_offs: Tensor, block_size: int = 32
) -> Tuple[Tensor, Tensor]:
    """
    Convert scales to blocked format for a 2D tensor (input activations / token groups),
    where groups are along the total_M dimension (rows).

    Args:
        x_scales: Tensor with per group scales in blocked format concatenated into one tensor.
        group_offs: Tensor of shape (num_groups,) which contains the end index of each group along the total_M dimension.
        total_M: total size of all groups summed together
        K: K dim size

    Returns:
        blocked_scales: Tensor
        start_row_after_padding: Tensor of shape (num_groups,) which contains the start row after padding for each group.
    """

    assert x_scales.ndim == 2, "x_scales must be 2D"
    assert block_size == 32, "Only block_size=32 is supported for now"
    total_M, scale_cols = x_scales.shape
    num_groups = group_offs.shape[0]

    # Each group will require a variable amount of padding, so to avoid d2h sync causing by iterating over each group,
    # the Triton kernenl will use an upper bound of adding 128 padding rows to each group.
    # (This torch impl is used as a reference for correctness, so we must match the triton kernel's impl).
    total_M_padded = total_M + num_groups * 128
    blocked_scales = x_scales.new_zeros(total_M_padded, scale_cols)
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

        # Calculate the start row after padding
        rows_for_group = group_scales_blocked.numel() // scale_cols
        new_start_row = prev_start_row_after_padding + rows_for_group
        start_row_after_padding_list.append(new_start_row)

        # Write output to subtensor
        group_rows_padded = ceil_div(group_size, 128) * 128
        blocked_scales[
            prev_start_row_after_padding : prev_start_row_after_padding
            + group_rows_padded,
            :,
        ] = group_scales_blocked.reshape(-1, scale_cols)

        # Update next group start index
        group_start_idx = group_end_idx

    start_row_after_padding = torch.tensor(
        start_row_after_padding_list, device=x_scales.device, dtype=torch.int64
    )
    return blocked_scales, start_row_after_padding


def torch_to_blocked_2d_K_groups(
    x_scales: Tensor, group_offs: Tensor, block_size: int = 32
) -> Tuple[Tensor, Tensor]:
    """
    Convert scales to blocked format for a 2D tensor (input activations),
    when groups are along the scaled (K) dimension.

    Args:
        x_scales: Tensor with per group scales in blocked format concatenated into one tensor.
        group_offs: Tensor of shape (num_groups,) which contains the end index of each group along the total_k dimension.
        total_K: total size of all groups summed together

    Returns:
        blocked_scales: Tensor
        start_row_after_padding: Tensor of shape (num_groups,) which contains the start row after padding for each group.
    """
    # TODO: add diagram of this transformation to the docs, and link in the docstring
    assert x_scales.ndim == 2, "x_scales must be 2D"
    assert block_size == 32, "Only block_size=32 is supported for now"
    M, total_K = x_scales.shape
    padded_M = ceil_div(M, 128) * 128
    num_groups = group_offs.shape[0]

    # Each group will require a variable amount of padding, so to avoid d2h sync causing by iterating over each group,
    total_K_padded = total_K + num_groups * 4
    blocked_scales = x_scales.new_zeros(padded_M, total_K_padded)

    # Flattened view for easier indexing when writing to subregions of memory
    blocked_scales_flat = blocked_scales.view(-1)

    BLOCK_ROWS, BLOCK_COLS = 128, 4
    output_stride_per_block = BLOCK_ROWS * BLOCK_COLS  # 512

    start_col_after_padding_list = [0]
    group_start_idx = 0
    for i, group_end_idx in enumerate(group_offs.tolist()):
        group_size = group_end_idx - group_start_idx
        prev_start_col_after_padding = start_col_after_padding_list[i]
        if group_size == 0:
            start_col_after_padding_list.append(prev_start_col_after_padding)
            continue

        # Convert group scales to blocked format
        group_scales = x_scales[:, group_start_idx:group_end_idx]
        group_scales_blocked = to_blocked(group_scales)
        cols_after_padding = ceil_div(group_size, 4) * 4

        num_row_blocks = ceil_div(M, 128)
        num_col_blocks = cols_after_padding // 4

        # Reshape blocked scales from flattened format to (num_row_blocks, num_col_blocks, ...)
        # so we can write each SF tile to its output buffer individually.
        group_scales_reshaped = group_scales_blocked.view(
            num_row_blocks, num_col_blocks, -1
        )
        out_group_base_offset = prev_start_col_after_padding * padded_M

        # For each SF tile, write to the output tensor
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                block_data = group_scales_reshaped[row_block, col_block]

                stride_per_row_of_blocks_in_group = (
                    num_col_blocks * output_stride_per_block
                )
                offset_in_group = (
                    row_block * stride_per_row_of_blocks_in_group
                    + col_block * output_stride_per_block
                )
                final_offset = out_group_base_offset + offset_in_group

                # flattened (512,) for (128,4) sf tile
                block_flat = block_data.reshape(-1)
                blocked_scales_flat[
                    final_offset : final_offset + output_stride_per_block
                ] = block_flat

        # Calculate the start col after padding
        new_start_col = prev_start_col_after_padding + cols_after_padding
        start_col_after_padding_list.append(new_start_col)

        # Update next group start index
        group_start_idx = group_end_idx

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


def compute_blocked_scale_offsets_for_M_groups(offsets: torch.Tensor):
    """
    Given a 1D tensor of input group offsets along the total_M dimension (rows),
    compute the starting row offset of the scales for each group after padding to blocked format.

    In effect, this rrounds each integer in a 1D PyTorch tensor up to the nearest multiple of 128.

    Args:
        - offsets: A 1D PyTorch tensor of integers in ascending sorted order, representing the end index of each group along the total_M dimension.

    Returns:
        - group_sizes: A 1D PyTorch tensor of integers representing the size of each group.
        - starting_row_after_padding: 1D integer tensor representing the starting row after padding each to blocked format.
    """
    # Calculate group sizes
    zero = torch.zeros(1, dtype=offsets.dtype, device=offsets.device)
    group_sizes = torch.diff(offsets, prepend=zero)

    # Round each group size up to the nearest multiple of 128
    rounded_group_sizes = ceil_div(group_sizes, 128) * 128

    # Calculate the starting row after padding for each group
    starting_row_after_padding = torch.cumsum(rounded_group_sizes, dim=0)

    # Must start with 0
    starting_row_after_padding = torch.cat([zero, starting_row_after_padding])
    return group_sizes, starting_row_after_padding


def compute_blocked_scale_offsets_for_K_groups(
    scale_group_offsets: torch.Tensor, block_size: int = 32
):
    """
    Performs round_up(x, 4) on each element in a 1D offsets tensor,
    to compute the starting offsets of each group after scaling along the contraction dimension.

    Args:
        offsets: A 1D PyTorch tensor of integers in ascending sorted order, representing the end index of each group along the total_M dimension.

    Returns:
        - starting_col_after_padding: 1D integer tensor representing the starting row after padding each to blocked format.
    """
    # Calculate group sizes
    zero = torch.zeros(
        1, dtype=scale_group_offsets.dtype, device=scale_group_offsets.device
    )
    group_sizes = torch.diff(scale_group_offsets, prepend=zero)

    # After scaling with block_size 32, each group size is rounded up to the nearest multiple of 4
    rounded_group_sizes = ceil_div(group_sizes, 4) * 4

    # Calculate the starting row after padding for each group
    starting_col_after_padding = torch.cumsum(rounded_group_sizes, dim=0)

    # Must start with 0
    starting_col_after_padding = torch.cat([zero, starting_col_after_padding])
    return group_sizes, starting_col_after_padding


if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton_op("torchao::triton_mx_block_rearrange_2d_M_groups", mutates_args={})
    def triton_mx_block_rearrange_2d_M_groups(
        scales_tensor: torch.Tensor,
        input_group_end_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rearranges an E8M0 tensor scale to block-scaled swizzle format,
        where groups are along the total_M dimension (rows).

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
        num_groups = input_group_end_offsets.shape[0]

        # Final offset is the total number of rows in the tensor.
        # Padding needing per group is variable/data dependent, so we just pad each group by
        # the upper bound of 128 rows to avoid a d2h sync caused by iterating over each group.
        padded_rows = rows + num_groups * 128

        num_col_blocks = ceil_div(cols, 4)
        padded_cols = num_col_blocks * 4
        output = scales_tensor.new_zeros((padded_rows, padded_cols))

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
        wrap_triton(triton_scale_swizzle_M_groups)[grid](
            # Input scales
            scales_tensor.view(torch.uint8),
            scales_tensor.stride(0),
            scales_tensor.stride(1),
            rows,
            cols,
            # Original offsets (to read from)
            input_group_end_offsets,
            # Output scales tensor and group offsets after padding (to write to)
            output.view(torch.uint8),
            output.stride(0),
            output_stride_per_block,
            output_stride_per_row_of_blocks,
            num_groups=num_groups,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        )
        return output

    @triton.jit
    def triton_scale_swizzle_M_groups(
        scales_ptr,  # (M, K//block_size)
        scales_stride_dim0,
        scales_stride_dim1,
        scale_rows,
        scale_cols,
        orig_offsets,  # (num_groups,)
        output_scales_ptr,
        output_scales_stride_dim0,
        output_stride_per_block,
        output_stride_per_row_of_blocks,
        num_groups: tl.constexpr,
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

        # Calculate this group's start row after blocked format padding, by doing a prefix sum
        # of each previous group's padded size.
        output_group_start_row = _blocked_group_start_idx(
            group_pid, orig_offsets, num_groups, 128
        )

        # Calculate destination indices for each row and col in block swizzled layout.
        # We can reuse this swizzle transformation on each block of data we read.
        row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
        col_offs = tl.arange(0, BLOCK_COLS)[None, :]

        # Compute desination indices for each elem in block swizzled layout
        dest_indices_flat = _dest_indices_for_block(
            row_offs,
            col_offs,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        )

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
                block_row_offs * scales_stride_dim0
                + block_col_offs * scales_stride_dim1
            )
            mask = (block_row_offs < input_group_end_row) & (
                block_col_offs < scale_cols
            )
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

    @triton_op("torchao::triton_mx_block_rearrange_per_group_3d", mutates_args={})
    def triton_mx_block_rearrange_per_group_3d(
        scale_tensor: torch.Tensor,
    ) -> torch.Tensor:
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

        wrap_triton(triton_scale_swizzle_per_group_3d)[grid](
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

        row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
        col_offs = tl.arange(0, BLOCK_COLS)[None, :]

        # Compute desination offs for each elem in block swizzled layout
        dest_indices_flat = _dest_indices_for_block(
            row_offs,
            col_offs,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        )

        # Calculate starting row and column for this tile
        start_row = pid_row * BLOCK_ROWS
        start_col = pid_col * BLOCK_COLS
        global_rows = start_row + row_offs
        global_cols = start_col + col_offs

        mask = (global_rows < scale_rows) & (global_cols < scale_cols)

        input_scales = tl.load(
            input_ptr
            + global_rows * input_stride_dim1
            + global_cols * input_stride_dim2,
            mask=mask,
            other=0.0,
        )
        scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

        # Calculate block offset using provided output block stride
        LOCAL_NUMEL = BLOCK_ROWS * BLOCK_COLS
        block_offset = pid_col * LOCAL_NUMEL + (pid_row * output_block_stride)

        tl.store(
            output_ptr + block_offset + dest_indices_flat,
            scales_flat,
        )

    @triton_op("torchao::triton_mx_block_rearrange_2d_K_groups", mutates_args={})
    def triton_mx_block_rearrange_2d_K_groups(
        scales_tensor: torch.Tensor,
        input_group_end_offsets: torch.Tensor,
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
        num_groups = input_group_end_offsets.shape[0]
        num_row_blocks = ceil_div(rows, 128)
        padded_rows = num_row_blocks * 128

        # Padding needing per group is variable/data dependent, so we just pad each group by
        # the upper bound of 4 cols to avoid a d2h sync caused by iterating over each group.
        padded_cols = cols + num_groups * 4
        output = scales_tensor.new_zeros((padded_rows, padded_cols))

        # Output block stride for the rearranged format
        BLOCK_ROWS, BLOCK_COLS = 128, 4
        output_stride_per_block = BLOCK_ROWS * BLOCK_COLS

        # We parallelize per group and per row block.
        # Cols per group is variable, so we just loop through col blocks for each group.
        grid = lambda META: (
            num_groups,
            num_row_blocks,
        )
        wrap_triton(triton_scale_swizzle_2d_K_groups)[grid](
            # Input scales
            scales_tensor.view(torch.uint8),
            scales_tensor.stride(0),
            scales_tensor.stride(1),
            rows,
            cols,
            padded_rows,
            input_group_end_offsets,
            output.view(torch.uint8),
            output_stride_per_block,
            num_groups=num_groups,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            DEBUG=False,
        )
        return output

    @triton.jit
    def triton_scale_swizzle_2d_K_groups(
        scales_ptr,  # (M, total_K//block_size)
        scales_stride_dim0,
        scales_stride_dim1,
        scale_rows,
        scale_cols,
        padded_rows,
        orig_offsets,  # (num_groups,)
        output_scales_ptr,
        output_stride_per_block,
        num_groups: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        DEBUG: tl.constexpr = False,
    ):
        group_pid = tl.program_id(0)
        block_row_pid = tl.program_id(1)

        # Input scales row range for this group
        input_group_start_col = tl.load(
            orig_offsets + group_pid - 1, mask=group_pid > 0, other=0
        )
        input_group_end_col = tl.load(orig_offsets + group_pid)

        # Calculate this group's start row after blocked format padding, by doing a prefix sum
        # of each previous group's padded size.
        output_group_start_col = _blocked_group_start_idx(
            group_pid, orig_offsets, num_groups, 4
        )

        row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
        col_offs = tl.arange(0, BLOCK_COLS)[None, :]

        # Compute desination offs for each elem in block swizzled layout
        dest_indices_flat = _dest_indices_for_block(
            row_offs,
            col_offs,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        )

        # For this group and row block, we iterate through col blocks, reading (BLOCK_ROWS, BLOCK_COLS) from the input scales.
        # We track how many col blocks we have iterated through.
        out_group_base_offset = output_group_start_col * padded_rows
        curr_input_start_col = input_group_start_col
        curr_out_start_col_block = 0
        while curr_input_start_col < input_group_end_col:
            # Read block of input scales
            block_row_offs = block_row_pid * BLOCK_ROWS + row_offs
            block_col_offs = curr_input_start_col + col_offs
            block_offs = (
                block_row_offs * scales_stride_dim0
                + block_col_offs * scales_stride_dim1
            )
            mask = (block_row_offs < scale_rows) & (
                block_col_offs < input_group_end_col
            )
            input_scales = tl.load(scales_ptr + block_offs, mask=mask, other=0.0)
            scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

            # Get offset within the group to add to the group's base offset
            num_cols_in_group = input_group_end_col - input_group_start_col
            num_col_blocks_in_group = tl.cdiv(num_cols_in_group, BLOCK_COLS)
            stride_per_row_of_blocks_in_group = (
                num_col_blocks_in_group * output_stride_per_block
            )
            offset_in_group = (
                block_row_pid * stride_per_row_of_blocks_in_group
                + curr_out_start_col_block * output_stride_per_block
            )
            final_offset = out_group_base_offset + offset_in_group

            # Apply swizzling for write to gmem
            tl.store(
                output_scales_ptr + final_offset + dest_indices_flat,
                scales_flat,
            )

            # Advance to next col block
            curr_input_start_col += BLOCK_COLS
            curr_out_start_col_block += 1

    @triton.jit
    def _dest_indices_for_block(
        row_offs,
        col_offs,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        # Calculate destination indices for each row and col in block swizzled layout.
        # We can reuse this swizzle transformation on each block of data we read.
        r_div_32 = row_offs // 32
        r_mod_32 = row_offs % 32

        # Rearrange to (32, 4, 4) then to final (32, 16) coordinates
        dest_indices = r_mod_32 * 16 + r_div_32 * 4 + col_offs

        # Flatten
        dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))
        return dest_indices_flat

    @triton.jit
    def _blocked_group_start_idx(
        group_pid,
        orig_offsets,
        num_groups: tl.constexpr,
        padding_size: tl.constexpr,
    ):
        """Prefix sum to compute the start index of a given group."""
        offsets = tl.load(orig_offsets + tl.arange(0, num_groups))
        prev_offsets = tl.load(
            orig_offsets + tl.arange(0, num_groups) - 1,
            mask=tl.arange(0, num_groups) > 0,
            other=0,
        )
        group_sizes = tl.where(
            tl.arange(0, num_groups) > 0,
            offsets - prev_offsets,
            offsets,
        )
        padded_sizes = tl.cdiv(group_sizes, padding_size) * padding_size
        prefix_mask = tl.arange(0, num_groups) < group_pid
        group_start_idx = tl.sum(tl.where(prefix_mask, padded_sizes, 0))
        return group_start_idx

else:

    def triton_mx_block_rearrange_2d_M_groups(
        scales_tensor: torch.Tensor,
        input_group_end_offsets: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "triton_mx_block_rearrange_2d_M_groups requires torch 2.7.0+ and triton installed"
        )

    def triton_mx_block_rearrange_per_group_3d(
        scale_tensor: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "triton_mx_block_rearrange_per_group_3d requires torch 2.7.0+ and triton installed"
        )

    def triton_mx_block_rearrange_2d_K_groups(
        scales_tensor: torch.Tensor,
        input_group_end_offsets: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "triton_mx_block_rearrange_2d_K_groups requires torch 2.7.0+ and triton installed"
        )


_mxfp8_cuda_kernels_available = (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
)

if _mxfp8_cuda_kernels_available:
    lib = torch.library.Library("torchao", "FRAGMENT")
    lib.define(
        "mxfp8_quantize_3d(Tensor input, int scale_dim_n, str fp8_format, str scaling_mode) -> (Tensor, Tensor)",
        tags=[torch._C.Tag.needs_fixed_stride_order],
    )

    def mxfp8_quantize_cuda_3d(
        x: torch.Tensor,
        block_size: int = 32,
        scaling_mode: str = "floor",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes a 3D tensor of shape (E,N,K) to MXFP8 format, scaling along N.

        This is a high-level wrapper that calls the underlying CUDA kernel via
        torch.ops.torchao.mxfp8_quantize_3d.

        Args:
            x (torch.Tensor): Input tensor to be quantized.
            block_size (int, optional): Block size for quantization. Defaults to 32.
            scaling_mode (str, optional): Scaling mode for quantization. Defaults to "floor".

        Returns:
            torch.Tensor: quantized tensor in column-major layout
            torch.Tensor: scales tensor
        """
        assert x.ndim == 3, "Input tensor must be 3D"
        assert x.dtype in (
            torch.float32,
            torch.bfloat16,
        ), "Input tensor must be float32 or bfloat16"
        return torch.ops.torchao.mxfp8_quantize_3d.default(
            x, block_size, "e4m3", scaling_mode
        )

    @torch.library.register_fake("torchao::mxfp8_quantize_3d")
    def _fake_mxfp8_quantize_3d(
        x: torch.Tensor,
        scale_dim_n: int,
        fp8_format: str,
        scaling_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fake/meta implementation for mxfp8_quantize_3d."""
        assert x.ndim == 3, "Input tensor must be 3D"
        E, N, K = x.shape
        # Quantized tensor is in column major layout
        q_data = x.new_empty(x.shape, dtype=torch.float8_e4m3fn).as_strided(
            x.shape, (N * K, 1, N)
        )
        scales = x.new_empty((E, N // scale_dim_n, K), dtype=torch.float8_e8m0fnu)
        return q_data, scales

    # CUDA kernel for per group blocked layout transform with groups along M
    lib.define(
        "mx_block_rearrange_2d_M_groups(Tensor scales_tensor, Tensor input_group_end_offsets, int chunks_per_tb) -> Tensor",
        tags=[torch._C.Tag.needs_fixed_stride_order],
    )

    def mx_block_rearrange_2d_M_groups_cuda(
        scales_tensor: torch.Tensor,
        input_group_end_offsets: torch.Tensor,
        chunks_per_tb: int = 4,
    ) -> torch.Tensor:
        """
        Rearranges an E8M0 tensor scale to block-scaled swizzle format using CUDA,
        where groups are along the total_M dimension (rows).

        This is a high-level wrapper that calls the underlying CUDA kernel via
        torch.ops.torchao.mx_block_rearrange_2d_M_groups.

        This format is suitable for Tmem as described in NVIDIA documentation:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

        Args:
            scales_tensor: Input tensor containing e8m0 scales for each logical group of a target tensor.
                Must be 2D with dtype uint8 or float8_e8m0fnu.
            input_group_end_offsets: tensor of int32 values representing group end indexes for the input scales.
            chunks_per_tb: Number of 128-row chunks per threadblock (1, 4, 8, or 16)

        Returns:
            Rearranged tensor in block-scaled swizzle format with shape (padded_rows, padded_cols).
        """
        assert scales_tensor.ndim == 2, "scales_tensor must be 2D"
        assert scales_tensor.dtype in (
            torch.uint8,
            torch.float8_e8m0fnu,
        ), "scales_tensor must be uint8 or float8_e8m0fnu"
        assert input_group_end_offsets.dtype == torch.int32, (
            "input_group_end_offsets must be int32"
        )
        assert chunks_per_tb in (1, 4, 8, 16), "chunks_per_tb must be 1, 4, 8, or 16"

        return torch.ops.torchao.mx_block_rearrange_2d_M_groups.default(
            scales_tensor,
            input_group_end_offsets,
            chunks_per_tb,
        )

    @torch.library.register_fake("torchao::mx_block_rearrange_2d_M_groups")
    def _fake_mx_block_rearrange_2d_M_groups_cuda(
        scales_tensor: torch.Tensor,
        input_group_end_offsets: torch.Tensor,
        chunks_per_tb: int,
    ) -> torch.Tensor:
        """Fake/meta implementation for mx_block_rearrange_2d_M_groups."""
        assert scales_tensor.ndim == 2, "scales_tensor must be 2D"
        rows, cols = scales_tensor.shape
        num_groups = input_group_end_offsets.shape[0]

        # Each group is padded to 128 rows upper bound
        BLOCK_ROWS = 128
        BLOCK_COLS = 4
        padded_rows = rows + num_groups * BLOCK_ROWS

        # Columns are padded to multiple of BLOCK_COLS
        num_col_blocks = ceil_div(cols, BLOCK_COLS)
        padded_cols = num_col_blocks * BLOCK_COLS

        return scales_tensor.new_empty((padded_rows, padded_cols))

else:

    def mxfp8_quantize_cuda_3d(
        x: torch.Tensor,
        block_size: int = 32,
        scaling_mode: str = "floor",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "mxfp8_quantize_cuda_3d is not implemented on this device"
        )

    def mx_block_rearrange_2d_M_groups_cuda(
        scales_tensor: torch.Tensor,
        input_group_end_offsets: torch.Tensor,
        chunks_per_tb: int = 8,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "mx_block_rearrange_2d_M_groups_cuda is not implemented on this device"
        )
