# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# We need to skip before doing any imports which would use triton, since
# triton won't be available on CPU builds
if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


from torchao.prototype.moe_training.kernels.float8_rowwise import (
    triton_fp8_rowwise_3d_transpose_rhs,
    triton_fp8_rowwise_3d_transpose_rhs_fused_reduction,
)
from torchao.prototype.moe_training.kernels.jagged_float8_scales import (
    triton_fp8_per_group_colwise_scales,
    triton_fp8_per_group_rowwise_scales,
)
from torchao.prototype.moe_training.kernels.mxfp8_blocked_scales import (
    compute_per_group_blocked_scale_offsets,
    compute_per_group_blocked_scale_offsets_2d2d_lhs,
    torch_to_blocked_per_group_2d,
    torch_to_blocked_per_group_2d2d_lhs,
    torch_to_blocked_per_group_3d,
    triton_mx_block_rearrange_per_group_2d,
    triton_mx_block_rearrange_per_group_2d2d_lhs,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
    generate_jagged_offs,
    torch_to_3d_rowwise_float8_transpose_rhs,
    torch_to_float8_per_group_colwise,
    torch_to_float8_per_group_rowwise,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.testing.utils import skip_if_rocm


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_row_major_with_jagged_rowwise_scales(round_scales_to_power_of_2: bool):
    # Tests case where rowwise scales are computed for multiple distinct subtensors,
    # with end boundary of each group is determine by their end column indexes (offsets).
    device = "cuda"
    m, k, n_groups = 256, 256, 4
    x = torch.randn(k, m * n_groups, device=device)
    colwise_offs = torch.arange(m, m * n_groups + 1, m, device=device)

    # Torch reference impl
    ref_fp8_data, ref_scales = torch_to_float8_per_group_rowwise(
        x,
        colwise_offs,
        target_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    # Triton kernel
    kernel_fp8_data, kernel_scales = triton_fp8_per_group_rowwise_scales(
        x,
        colwise_offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert not _is_column_major(kernel_fp8_data), "fp8 data is not row major"


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_row_major_with_jagged_rowwise_scales_transpose_method(
    round_scales_to_power_of_2: bool,
):
    # tests case where rowwise scales are computed for multiple distinct subtensors,
    # with end boundary of each group is determine by their end column indexes (offsets).
    device = "cuda"
    m, k, n_groups = 256, 256, 4
    grad_out = torch.randn(m * n_groups, k, device=device)
    colwise_offs = torch.arange(m, m * n_groups + 1, m, device=device)
    grad_out_t = grad_out.t()

    # compute reference with torch impl
    ref_fp8_data, ref_scales = torch_to_float8_per_group_rowwise(
        grad_out_t,
        colwise_offs,
        target_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    # Transpose method requires grad_out to be column major, then we compute per group
    # colwise scales writing to column major, then transpose outputs back to the desired
    # shape and row major format.
    kernel_fp8_data, kernel_scales = triton_fp8_per_group_colwise_scales(
        grad_out.t().contiguous().t(),
        colwise_offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    kernel_fp8_data = kernel_fp8_data.t()  # (mg, n) -> (n, mg)
    kernel_scales = kernel_scales.t()  # (1, n * n_groups) -> (n * n_groups, 1)

    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert not _is_column_major(kernel_fp8_data), "fp8 data is not row major"


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_column_major_with_jagged_colwise_scales(round_scales_to_power_of_2: bool):
    # tests case where colwise scales are computed for multiple distinct subtensors,
    # with end boundary of each group is determine by their end row indexes (offsets).
    device = "cuda"
    m, k, n_groups = 256, 256, 4
    x = torch.randn(m * n_groups, k, device=device).t().contiguous().t()
    rowwise_offs = torch.arange(m, m * n_groups + 1, m, device=device)

    # compute reference with torch impl
    ref_fp8_data, ref_scales = torch_to_float8_per_group_colwise(
        x,
        rowwise_offs,
        target_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    kernel_fp8_data, kernel_scales = triton_fp8_per_group_colwise_scales(
        x,
        rowwise_offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert _is_column_major(kernel_fp8_data), "fp8 data is not column major"


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_fp8_rowwise_3d_transpose_rhs_atomic(round_scales_to_power_of_2: bool):
    device = "cuda"
    experts, n, k = 8, 4 * 5120, 5120

    # Example expert weights as it comes into forward transposed
    torch.manual_seed(0)
    x = torch.randn((experts, n, k), dtype=torch.bfloat16, device=device).transpose(
        -2, -1
    )

    # Compute reference with torch impl
    ref_fp8, ref_scales = torch_to_3d_rowwise_float8_transpose_rhs(
        x,
        target_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    # Torch impl keeps empty scaled dim, so we squeeze it out to be consistent with triton impl
    ref_scales = ref_scales.squeeze(1)

    triton_fp8, triton_scales = triton_fp8_rowwise_3d_transpose_rhs(
        x,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    assert ref_scales.shape == triton_scales.shape, "scale shapes not equal"
    assert ref_scales.stride() == triton_scales.stride(), "scale strides not equal"
    assert torch.allclose(ref_scales, triton_scales, rtol=0, atol=0), "scales not equal"

    assert ref_fp8.shape == triton_fp8.shape, "output shapes not equal"
    assert ref_fp8.stride() == triton_fp8.stride(), "output strides not equal"
    assert torch.allclose(ref_fp8, triton_fp8, rtol=0, atol=0), "fp8 data not equal"


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_fp8_rowwise_3d_transpose_rhs_reduction(round_scales_to_power_of_2: bool):
    device = "cuda"
    experts, n, k = 8, 4 * 5120, 5120

    # Example expert weights as it comes into forward transposed
    torch.manual_seed(0)
    x = torch.randn((experts, n, k), dtype=torch.bfloat16, device=device).transpose(
        -2, -1
    )

    # Compute reference with torch impl
    ref_fp8, ref_scales = torch_to_3d_rowwise_float8_transpose_rhs(
        x,
        target_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    # Torch impl keeps empty scaled dim, so we squeeze it out to be consistent with triton impl
    ref_scales = ref_scales.squeeze(1)

    triton_fp8, triton_scales = triton_fp8_rowwise_3d_transpose_rhs_fused_reduction(
        x,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    assert ref_scales.shape == triton_scales.shape, "scale shapes not equal"
    assert ref_scales.stride() == triton_scales.stride(), "scale strides not equal"
    assert torch.allclose(ref_scales, triton_scales, rtol=0, atol=0), "scales not equal"

    assert ref_fp8.shape == triton_fp8.shape, "output shapes not equal"
    assert ref_fp8.stride() == triton_fp8.stride(), "output strides not equal"
    assert torch.allclose(ref_fp8, triton_fp8, rtol=0, atol=0), "fp8 data not equal"


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize(
    "m,k,n_groups", [(256, 256, 4), (16640, 5120, 16), (16640, 8192, 16)]
)
def test_mxfp8_per_group_blocked_scales_2d(
    m: int,
    k: int,
    n_groups: int,
):
    device = "cuda"
    block_size = 32
    input_data = torch.randn(m, k, device=device)
    e8m0_scales, _ = to_mx(
        input_data, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )
    input_group_offsets = generate_jagged_offs(
        n_groups, m, multiple_of=block_size, device=device
    )

    # torch reference
    ref_out_scales, _ = torch_to_blocked_per_group_2d(
        e8m0_scales, input_group_offsets, k, block_size=block_size
    )

    # triton kernel
    _, output_group_offsets = compute_per_group_blocked_scale_offsets(
        input_group_offsets
    )
    triton_out_scales = triton_mx_block_rearrange_per_group_2d(
        e8m0_scales,
        input_group_offsets,
        output_group_offsets,
    )
    assert torch.allclose(ref_out_scales, triton_out_scales, atol=0, rtol=0), (
        "blocked scales not equal"
    )


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("e,n,k", [(1, 8192, 5120), (2, 8192, 5120), (8, 5120, 8192)])
def test_mxfp8_per_group_blocked_scales_3d(
    e: int,
    n: int,
    k: int,
):
    device = "cuda"
    block_size = 32
    weights = torch.randn(e, n, k // block_size, device=device)
    weight_scales, _ = to_mx(
        weights, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )

    # torch reference
    ref_out_scales = torch_to_blocked_per_group_3d(weight_scales)

    # triton kernel
    triton_out_scales = triton_mx_block_rearrange_per_group_3d(weight_scales)
    assert torch.allclose(ref_out_scales, triton_out_scales, atol=0, rtol=0), (
        "blocked scales not equal"
    )


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("m,total_k,n_groups", [(256, 64, 2)])
def test_mxfp8_per_group_blocked_scales_2d2d_lhs(
    m: int,
    total_k: int,
    n_groups: int,
):
    device = "cuda"
    block_size = 32
    input_data = torch.randn(m, total_k, device=device)
    e8m0_scales, _ = to_mx(
        input_data, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )

    # Generate group end offsets along total_K, then divide by block_size to get scale group end offsets
    input_group_offsets = generate_jagged_offs(
        n_groups, total_k, multiple_of=block_size, device=device
    )
    input_group_offsets //= block_size

    # torch reference
    ref_out_scales, ref_start_cols_after_padding = torch_to_blocked_per_group_2d2d_lhs(
        e8m0_scales,
        input_group_offsets,
    )

    # triton kernel
    _, output_group_offsets = compute_per_group_blocked_scale_offsets_2d2d_lhs(
        input_group_offsets
    )
    assert torch.allclose(output_group_offsets, ref_start_cols_after_padding), (
        "output scale group start offsets not equal"
    )
    triton_out_scales = triton_mx_block_rearrange_per_group_2d2d_lhs(
        e8m0_scales,
        input_group_offsets,
        output_group_offsets,
    )
    breakpoint()
    assert torch.allclose(ref_out_scales, triton_out_scales, atol=0, rtol=0), (
        "blocked scales not equal"
    )
