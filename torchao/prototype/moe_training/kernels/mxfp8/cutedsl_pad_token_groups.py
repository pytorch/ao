# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch


@functools.cache
def _compile_copy_token_groups_cutedsl(
    num_groups: int,
    element_size: int,
    is_pad: bool,
    row_bytes_aligned: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    NUM_GROUPS = num_groups
    ELEMENT_SIZE = element_size
    IS_PAD = is_pad
    ROW_BYTES_ALIGNED = row_bytes_aligned
    VEC_ELEMS = 16 // ELEMENT_SIZE
    WARP_SIZE = 32
    WARPS_PER_BLOCK = 8
    THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK
    UNROLL = 4
    LANE_STRIDE = WARP_SIZE * VEC_ELEMS
    UNROLL_STRIDE = UNROLL * LANE_STRIDE
    ELEMENT_DTYPE = cutlass.BFloat16 if ELEMENT_SIZE == 2 else cutlass.Float32

    class CopyTokenGroups:
        @cute.jit
        def _find_group(
            self,
            row: cutlass.Int32,
            group_end_offsets: cute.Tensor,
        ):
            group = cutlass.Int32(0)
            keep_looking = cutlass.Int32(1)
            while group < NUM_GROUPS - 1 and keep_looking != 0:
                end = cutlass.Int32((group_end_offsets.iterator + group).load())
                if row >= end:
                    group = group + cutlass.Int32(1)
                else:
                    keep_looking = cutlass.Int32(0)
            return group

        @cute.jit
        def _group_start(
            self,
            group: cutlass.Int32,
            group_end_offsets: cute.Tensor,
        ):
            group_start = cutlass.Int32(0)
            if group > 0:
                group_start = cutlass.Int32(
                    (group_end_offsets.iterator + group - 1).load()
                )
            return group_start

        @cute.jit
        def _padded_start(
            self,
            group: cutlass.Int32,
            group_end_offsets: cute.Tensor,
        ):
            cumulative = cutlass.Int32(0)
            previous = cutlass.Int32(0)
            for g in range(NUM_GROUPS):
                group_end = cutlass.Int32((group_end_offsets.iterator + g).load())
                group_size = group_end - previous
                padded_size = (
                    (group_size + cutlass.Int32(31)) // cutlass.Int32(32)
                ) * cutlass.Int32(32)
                if g < group:
                    cumulative = cumulative + padded_size
                previous = group_end
            return cumulative

        @cute.jit
        def _fill_padded_offsets(
            self,
            tidx: cutlass.Int32,
            group_end_offsets: cute.Tensor,
            padded_group_start_offsets: cute.Tensor,
            padded_group_end_offsets: cute.Tensor,
        ):
            if tidx < NUM_GROUPS:
                group = cutlass.Int32(tidx)
                group_start = self._group_start(group, group_end_offsets)
                group_end = cutlass.Int32((group_end_offsets.iterator + group).load())
                group_size = group_end - group_start
                padded_size = (
                    (group_size + cutlass.Int32(31)) // cutlass.Int32(32)
                ) * cutlass.Int32(32)
                padded_start = self._padded_start(group, group_end_offsets)
                (padded_group_start_offsets.iterator + group).store(padded_start)
                (padded_group_end_offsets.iterator + group).store(
                    padded_start + padded_size
                )

        @cute.jit
        def _copy_vec_or_tail(
            self,
            inputs: cute.Tensor,
            input_offset: cutlass.Int32,
            elem_col: cutlass.Int32,
            output: cute.Tensor,
            output_offset: cutlass.Int32,
            dim: cutlass.Int32,
        ):
            if cutlass.const_expr(ROW_BYTES_ALIGNED):
                value = cute.recast_ptr(
                    (inputs.iterator + input_offset).align(16),
                    dtype=cutlass.Int128,
                ).load()
                cute.recast_ptr(
                    (output.iterator + output_offset).align(16),
                    dtype=cutlass.Int128,
                ).store(value)
            else:
                for i in range(VEC_ELEMS):
                    if elem_col + i < dim:
                        value = (inputs.iterator + input_offset + i).load()
                        (output.iterator + output_offset + i).store(value)

        @cute.kernel
        def kernel(
            self,
            inputs: cute.Tensor,
            group_end_offsets: cute.Tensor,
            padded_group_start_offsets: cute.Tensor,
            padded_group_end_offsets: cute.Tensor,
            output: cute.Tensor,
            num_tokens: cutlass.Int32,
            dim: cutlass.Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block_idx, _, _ = cute.arch.block_idx()
            lane_id = cutlass.Int32(tidx % WARP_SIZE)
            warp_id = cutlass.Int32(tidx // WARP_SIZE)
            if cutlass.const_expr(IS_PAD):
                if block_idx == 0:
                    self._fill_padded_offsets(
                        tidx,
                        group_end_offsets,
                        padded_group_start_offsets,
                        padded_group_end_offsets,
                    )
            row = cutlass.Int32(block_idx * WARPS_PER_BLOCK + warp_id)
            if row < num_tokens:
                group_start = cutlass.Int32(0)
                padded_start = cutlass.Int32(0)
                if lane_id == 0:
                    if cutlass.const_expr(IS_PAD):
                        group = cutlass.Int32(0)
                        previous = cutlass.Int32(0)
                        cumulative = cutlass.Int32(0)
                        keep_looking = cutlass.Int32(1)
                        while group < NUM_GROUPS and keep_looking != 0:
                            group_end = cutlass.Int32(
                                (group_end_offsets.iterator + group).load()
                            )
                            group_size = group_end - previous
                            padded_size = (
                                (group_size + cutlass.Int32(31)) // cutlass.Int32(32)
                            ) * cutlass.Int32(32)
                            if row < group_end:
                                group_start = previous
                                padded_start = cumulative
                                keep_looking = cutlass.Int32(0)
                            else:
                                cumulative = cumulative + padded_size
                                previous = group_end
                                group = group + cutlass.Int32(1)
                    else:
                        group = self._find_group(row, group_end_offsets)
                        group_start = self._group_start(group, group_end_offsets)
                        padded_start = cutlass.Int32(
                            (padded_group_start_offsets.iterator + group).load()
                        )
                group_start = cute.arch.shuffle_sync(group_start, 0)
                padded_start = cute.arch.shuffle_sync(padded_start, 0)
                offset_in_group = row - group_start
                padded_row = padded_start + offset_in_group
                input_row = row
                output_row = padded_row
                if cutlass.const_expr(not IS_PAD):
                    input_row = padded_row
                    output_row = row
                elem_col = lane_id * VEC_ELEMS
                input_offset = input_row * dim + elem_col
                output_offset = output_row * dim + elem_col
                if cutlass.const_expr(ROW_BYTES_ALIGNED):
                    while elem_col + (UNROLL - 1) * LANE_STRIDE + VEC_ELEMS <= dim:
                        vals = []
                        for k in cutlass.range_constexpr(UNROLL):
                            vals.append(
                                cute.recast_ptr(
                                    (
                                        inputs.iterator + input_offset + k * LANE_STRIDE
                                    ).align(16),
                                    dtype=cutlass.Int128,
                                ).load()
                            )
                        for k in cutlass.range_constexpr(UNROLL):
                            cute.recast_ptr(
                                (
                                    output.iterator + output_offset + k * LANE_STRIDE
                                ).align(16),
                                dtype=cutlass.Int128,
                            ).store(vals[k])
                        elem_col = elem_col + UNROLL_STRIDE
                        input_offset = input_offset + UNROLL_STRIDE
                        output_offset = output_offset + UNROLL_STRIDE
                while elem_col < dim:
                    self._copy_vec_or_tail(
                        inputs,
                        input_offset,
                        elem_col,
                        output,
                        output_offset,
                        dim,
                    )
                    elem_col = elem_col + WARP_SIZE * VEC_ELEMS
                    input_offset = input_offset + WARP_SIZE * VEC_ELEMS
                    output_offset = output_offset + WARP_SIZE * VEC_ELEMS

        @cute.jit
        def __call__(
            self,
            inputs: cute.Tensor,
            group_end_offsets: cute.Tensor,
            padded_group_start_offsets: cute.Tensor,
            padded_group_end_offsets: cute.Tensor,
            output: cute.Tensor,
            num_tokens: cutlass.Int32,
            dim: cutlass.Int32,
            num_blocks: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                inputs,
                group_end_offsets,
                padded_group_start_offsets,
                padded_group_end_offsets,
                output,
                num_tokens,
                dim,
            ).launch(
                grid=(num_blocks, 1, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    kernel = CopyTokenGroups()

    n = cute.sym_int()
    d = cute.sym_int()
    out_n = cute.sym_int()
    offs_stride = cute.sym_int()
    assumed_align = 16 if ROW_BYTES_ALIGNED else 1
    fake_input = make_fake_tensor(
        ELEMENT_DTYPE,
        (n, d),
        stride=(d, 1),
        assumed_align=assumed_align,
    )
    fake_output = make_fake_tensor(
        ELEMENT_DTYPE,
        (out_n, d),
        stride=(d, 1),
        assumed_align=assumed_align,
    )
    fake_group_end_offsets = make_fake_tensor(
        cutlass.Int32,
        (NUM_GROUPS,),
        stride=(offs_stride,),
    )
    fake_padded_group_start_offsets = make_fake_tensor(
        cutlass.Int32,
        (NUM_GROUPS,),
        stride=(offs_stride,),
    )
    fake_padded_group_end_offsets = make_fake_tensor(
        cutlass.Int32,
        (NUM_GROUPS,),
        stride=(offs_stride,),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        inputs=fake_input,
        group_end_offsets=fake_group_end_offsets,
        padded_group_start_offsets=fake_padded_group_start_offsets,
        padded_group_end_offsets=fake_padded_group_end_offsets,
        output=fake_output,
        num_tokens=0,
        dim=0,
        num_blocks=1,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def pad_token_groups_cutedsl(
    inputs: torch.Tensor,
    group_end_offsets: torch.Tensor,
    alignment_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert inputs.ndim == 2
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.dtype in (torch.float32, torch.bfloat16)
    assert group_end_offsets.ndim == 1
    assert group_end_offsets.is_cuda
    assert group_end_offsets.dtype == torch.int32
    assert alignment_size == 32
    num_tokens, dim = inputs.shape
    num_groups = group_end_offsets.shape[0]
    assert num_groups <= 32
    padded_group_start_offsets = torch.empty(
        (num_groups,),
        dtype=group_end_offsets.dtype,
        device=group_end_offsets.device,
    )
    padded_group_end_offsets = torch.empty(
        (num_groups,),
        dtype=group_end_offsets.dtype,
        device=group_end_offsets.device,
    )
    output_rows = num_tokens + num_groups * alignment_size
    output_rows = (
        (output_rows + alignment_size - 1) // alignment_size
    ) * alignment_size
    output = torch.zeros(
        (output_rows, dim),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    compiled = _compile_copy_token_groups_cutedsl(
        num_groups,
        inputs.element_size(),
        True,
        dim * inputs.element_size() % 16 == 0,
    )
    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        inputs,
        group_end_offsets,
        padded_group_start_offsets,
        padded_group_end_offsets,
        output,
        int(num_tokens),
        int(dim),
        int((num_tokens + 7) // 8),
        stream,
    )
    return output, padded_group_start_offsets, padded_group_end_offsets


def unpad_token_groups_cutedsl(
    padded_inputs: torch.Tensor,
    group_end_offsets: torch.Tensor,
    padded_group_start_offsets: torch.Tensor,
    num_tokens: int,
    alignment_size: int = 32,
) -> torch.Tensor:
    assert padded_inputs.ndim == 2
    assert padded_inputs.is_cuda
    assert padded_inputs.is_contiguous()
    assert padded_inputs.dtype in (torch.float32, torch.bfloat16)
    assert group_end_offsets.ndim == 1
    assert group_end_offsets.is_cuda
    assert group_end_offsets.dtype == torch.int32
    assert padded_group_start_offsets.ndim == 1
    assert padded_group_start_offsets.is_cuda
    assert padded_group_start_offsets.dtype == torch.int32
    assert alignment_size == 32
    dim = padded_inputs.shape[1]
    num_groups = group_end_offsets.shape[0]
    assert num_groups <= 32
    output = torch.empty(
        (num_tokens, dim),
        dtype=padded_inputs.dtype,
        device=padded_inputs.device,
    )
    compiled = _compile_copy_token_groups_cutedsl(
        num_groups,
        padded_inputs.element_size(),
        False,
        dim * padded_inputs.element_size() % 16 == 0,
    )
    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        padded_inputs,
        group_end_offsets,
        padded_group_start_offsets,
        padded_group_start_offsets,
        output,
        int(num_tokens),
        int(dim),
        int((num_tokens + 7) // 8),
        stream,
    )
    return output
