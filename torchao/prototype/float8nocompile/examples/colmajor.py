import math

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,  # rows
    N: tl.constexpr,  # cols
    BLOCK_SIZE: tl.constexpr,
):
    block_row_id = tl.program_id(0)
    block_col_id = tl.program_id(1)

    row_offs = block_row_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offs = block_col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    in_offs = row_offs[:, None] * N + col_offs[None, :]
    mask_load = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    block_data = tl.load(input_ptr + in_offs, mask=mask_load)

    # tranpose row and column indexes
    print("row offs")
    print(row_offs)
    print("col offs")
    print(col_offs)

    # out_row_offs = col_offs
    # out_col_offs = row_offs
    # out_offs = out_row_offs[:, None] * M + out_col_offs[None, :]
    out_offs = in_offs.trans(1, 0)
    # mask_store = (out_row_offs[:, None] < N) & (out_col_offs[None, :] < M)
    tl.store(output_ptr + out_offs, block_data)  # , mask=mask_store)


def transpose_2d_triton(input_tensor, BLOCK_SIZE=2):
    assert input_tensor.dim() == 2, "input tensor must be 2D"
    M, N = input_tensor.shape

    output_tensor = torch.empty_like(input_tensor)
    grid = (tl.cdiv(M, BLOCK_SIZE), tl.cdiv(N, BLOCK_SIZE))
    transpose_kernel[grid](input_tensor, output_tensor, M=M, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return output_tensor


if __name__ == "__main__":
    M, N = 4, 4

    input = torch.arange(M * N).reshape(M, N).cuda()

    output = transpose_2d_triton(input)
    expected = input.t().contiguous().t()

    print("input memory")
    print(input.storage())
    print()
    print("output memory")
    print(output.storage())
    print()
    print("expected memory")
    print(expected.storage())
    print()

    print("input tensor")
    print(input)

    print("output tensor")
    print(output)

    print("expected tensor")
    print(expected)

    if torch.allclose(output, expected):
        print("Conversion successful! The tensors match.")
    else:
        print("Conversion failed. The tensors do not match.")
