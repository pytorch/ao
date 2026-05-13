# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# FP8 MoE kernels require FP8-capable hardware (SM 10.x on CUDA, MI300+ on ROCm)
from torchao.utils import is_MI300, is_MI350


def _is_sm_10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


if not (torch.cuda.is_available() and (_is_sm_10x() or is_MI300() or is_MI350())):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM 10.x, MI300, or MI350)",
        allow_module_level=True,
    )

from torchao.float8.config import ScalingGranularity, e4m3_dtype
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.kernels import (
    triton_fp8_colwise_3d_scale_and_cast,
    triton_fp8_rowwise_2d_scale_and_cast,
)
from torchao.prototype.moe_training.kernels.float8_rowwise import (
    triton_fp8_rowwise_3d_transpose_rhs,
    triton_fp8_rowwise_3d_transpose_rhs_fused_reduction,
)
from torchao.prototype.moe_training.kernels.jagged_float8_scales import (
    triton_fp8_per_group_colwise_scales,
    triton_fp8_per_group_rowwise_scales,
)
from torchao.prototype.moe_training.kernels.mxfp8 import (
    fused_pad_token_groups_cuda,
    fused_unpad_token_groups_cuda,
    mx_block_rearrange_2d_M_groups_cuda,
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    mxfp8_quantize_cuda_3d,
    torch_pad_token_groups,
    torch_to_blocked_2d_K_groups,
    torch_to_blocked_2d_M_groups,
    torch_to_blocked_per_group_3d,
    torch_unpad_token_groups,
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_2d_M_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cuda_kernels_available,
    _mxfp8_cutedsl_kernels_available,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
    generate_jagged_offs,
    torch_to_3d_rowwise_float8_transpose_rhs,
    torch_to_float8_per_group_colwise,
    torch_to_float8_per_group_rowwise,
)
from torchao.prototype.mx_formats.kernels import triton_mx_block_rearrange
from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx
from torchao.prototype.mx_formats.utils import from_blocked
from torchao.testing.utils import skip_if_rocm


@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
@skip_if_rocm("jagged rowwise scales kernel vs torch reference mismatch on ROCm")
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
        target_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    # Triton kernel
    kernel_fp8_data, kernel_scales = triton_fp8_per_group_rowwise_scales(
        x,
        colwise_offs,
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert not _is_column_major(kernel_fp8_data), "fp8 data is not row major"


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
        target_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    # Transpose method requires grad_out to be column major, then we compute per group
    # colwise scales writing to column major, then transpose outputs back to the desired
    # shape and row major format.
    kernel_fp8_data, kernel_scales = triton_fp8_per_group_colwise_scales(
        grad_out.t().contiguous().t(),
        colwise_offs,
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    kernel_fp8_data = kernel_fp8_data.t()  # (mg, n) -> (n, mg)
    kernel_scales = kernel_scales.t()  # (1, n * n_groups) -> (n * n_groups, 1)

    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert not _is_column_major(kernel_fp8_data), "fp8 data is not row major"


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
        target_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    kernel_fp8_data, kernel_scales = triton_fp8_per_group_colwise_scales(
        x,
        rowwise_offs,
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert _is_column_major(kernel_fp8_data), "fp8 data is not column major"


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
        target_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    # Torch impl keeps empty scaled dim, so we squeeze it out to be consistent with triton impl
    ref_scales = ref_scales.squeeze(1)

    triton_fp8, triton_scales = triton_fp8_rowwise_3d_transpose_rhs(
        x,
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    assert ref_scales.shape == triton_scales.shape, "scale shapes not equal"
    assert ref_scales.stride() == triton_scales.stride(), "scale strides not equal"
    assert torch.allclose(ref_scales, triton_scales, rtol=0, atol=0), "scales not equal"

    assert ref_fp8.shape == triton_fp8.shape, "output shapes not equal"
    assert ref_fp8.stride() == triton_fp8.stride(), "output strides not equal"
    assert torch.allclose(ref_fp8, triton_fp8, rtol=0, atol=0), "fp8 data not equal"


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
        target_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    # Torch impl keeps empty scaled dim, so we squeeze it out to be consistent with triton impl
    ref_scales = ref_scales.squeeze(1)

    triton_fp8, triton_scales = triton_fp8_rowwise_3d_transpose_rhs_fused_reduction(
        x,
        output_dtype=e4m3_dtype,
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
def test_triton_mx_block_rearrange_2d_M_groups(
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
    ref_out_scales, _ = torch_to_blocked_2d_M_groups(
        e8m0_scales, input_group_offsets, block_size=block_size
    )

    # triton kernel
    triton_out_scales = triton_mx_block_rearrange_2d_M_groups(
        e8m0_scales,
        input_group_offsets,
    )
    assert torch.allclose(ref_out_scales, triton_out_scales, atol=0, rtol=0), (
        "blocked scales not equal"
    )


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize(
    "m,k,n_groups,chunks_per_tb",
    [
        (16640, 2048, 8, 4),
        (16640, 2048, 8, 8),
        (131072, 8192, 32, 16),
        (512, 512, 4, 4),
        (512, 1024, 4, 4),
        (512, 2048, 4, 4),
        (1024, 512, 8, 4),
        (128, 1408, 2, 1),  # dsv3-16b moe intermediate dim
        (256, 1408, 2, 1),
        (32768, 1408, 4, 1),
        (8192, 1536, 4, 1),  # dsv3-236b moe intermediate dim
    ],
)
def test_cuda_mx_block_rearrange_2d_M_groups(
    m: int,
    k: int,
    n_groups: int,
    chunks_per_tb: int,
):
    device = "cuda"
    block_size = 32
    input_data = torch.randn(m, k, device=device)
    e8m0_scales, _ = to_mx(
        input_data, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )
    scale_rows, scale_cols = e8m0_scales.shape

    input_group_offsets = generate_jagged_offs(
        n_groups, m, multiple_of=block_size, device=device
    )

    # torch reference
    ref_out_scales, _ = torch_to_blocked_2d_M_groups(
        e8m0_scales, input_group_offsets, block_size=block_size
    )

    # cuda kernel
    cuda_out_scales = mx_block_rearrange_2d_M_groups_cuda(
        e8m0_scales,
        input_group_offsets,
        chunks_per_tb=chunks_per_tb,
    )
    assert torch.allclose(ref_out_scales, cuda_out_scales, atol=0, rtol=0), (
        f"blocked scales not equal for scale_cols={scale_cols}"
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
@pytest.mark.parametrize("m", [256, 512, 1024, 5120])
@pytest.mark.parametrize("total_k", [512, 1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("n_groups", [1, 4, 8, 16])
def test_triton_mx_block_rearrange_2d_K_groups(
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
    scale_group_offsets = input_group_offsets // block_size

    # torch reference
    ref_out_scales, ref_start_cols_after_padding = torch_to_blocked_2d_K_groups(
        e8m0_scales,
        scale_group_offsets,
    )

    # triton kernel
    triton_out_scales = triton_mx_block_rearrange_2d_K_groups(
        e8m0_scales,
        scale_group_offsets,
    )
    assert torch.equal(ref_out_scales, triton_out_scales), "blocked scales not equal"


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.parametrize("E", (1, 2, 4, 8))
@pytest.mark.parametrize("N", (32, 1536, 5120, 7168, 8192))
@pytest.mark.parametrize("K", (32, 1536, 5120, 7168, 8192))
@pytest.mark.parametrize("input_dtype", (torch.bfloat16,))
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
@pytest.mark.parametrize(
    "variant",
    ("32x1_n", "32x32_n", "32x1_t"),
)
def test_cuda_mx_3d_cutedsl_numerics(E, N, K, input_dtype, scaling_mode, variant):
    if not _mxfp8_cutedsl_kernels_available:
        pytest.skip("mxfp8_quantize_3d is unavailable")

    scaling_mode_str = (
        "floor" if scaling_mode == ScaleCalculationMode.FLOOR else "rceil"
    )
    block_size = 32
    scale_block_dim2 = 32 if variant == "32x32_n" else 1

    # Use disinct incrementing values from 0 to E*M*K-1 to make debugging easier.
    x = (
        torch.arange(0, E * N * K, dtype=input_dtype, device="cuda")
        .reshape(E, N, K)
        .contiguous()
    )
    x_t = x.transpose(-2, -1)

    if variant == "32x1_t":
        s_ref, y_ref = to_mx(
            x.contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scaling_mode,
        )
        y_ref = y_ref.transpose(-2, -1)
        s_ref = s_ref.transpose(-2, -1)

        y_unblocked, s_unblocked = mxfp8_quantize_cuda_3d(
            x_t,
            block_size=block_size,
            scale_block_dim1=block_size,
            scale_block_dim2=1,
            scaling_mode=scaling_mode_str,
            blocked_scale_output=False,
        )
        s_unblocked = s_unblocked.to(s_ref.dtype)
        torch.testing.assert_close(s_unblocked, s_ref, rtol=0, atol=0)
        torch.testing.assert_close(y_unblocked, y_ref, rtol=0, atol=0)
        assert y_unblocked.stride() == y_ref.stride(), (
            "transposed-input unblocked quantized tensor strides do not match"
        )
        y, s = mxfp8_quantize_cuda_3d(
            x_t,
            block_size=block_size,
            scale_block_dim1=block_size,
            scale_block_dim2=1,
            scaling_mode=scaling_mode_str,
            blocked_scale_output=True,
        )
        s_rows, s_cols = x_t.shape[-1], x_t.shape[-2] // block_size
        s_logical = (
            torch.stack(
                [
                    from_blocked(s[e], s_rows, s_cols).view(torch.uint8)
                    for e in range(E)
                ],
                dim=0,
            )
            .view(torch.float8_e8m0fnu)
            .transpose(-2, -1)
            .to(s_ref.dtype)
        )
        torch.testing.assert_close(s_logical, s_ref, rtol=0, atol=0)
        torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
        assert y.stride() == y_ref.stride(), (
            "transposed-input quantized tensor strides do not match"
        )
        return

    if scale_block_dim2 == 1:
        s_ref, y_ref = to_mx(
            x.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scaling_mode,
        )
        y_ref = y_ref.transpose(-2, -1)
        s_ref = s_ref.transpose(-2, -1)
        s_rows, s_cols = K, N // block_size
        undo_scale = (
            lambda scale: from_blocked(scale, s_rows, s_cols)
            .transpose(-2, -1)
            .contiguous()
        )
    else:
        x_tiles = (
            x.view(E, N // block_size, block_size, K // block_size, block_size)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, N // block_size, K // block_size, block_size * block_size)
        )
        s_ref, y_tiles_ref = to_mx(
            x_tiles,
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size * block_size,
            scaling_mode=scaling_mode,
        )
        s_ref = s_ref.squeeze(-1)
        y_ref = (
            y_tiles_ref.view(
                E, N // block_size, K // block_size, block_size, block_size
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, N, K)
        )
        y_ref = y_ref.transpose(-2, -1).contiguous().transpose(-2, -1)
        s_rows, s_cols = K, N // block_size
        undo_scale = lambda scale: from_blocked(scale, s_rows, s_cols)[
            ::block_size
        ].transpose(-2, -1)

    y, s = mxfp8_quantize_cuda_3d(
        x,
        block_size=block_size,
        scale_block_dim1=block_size,
        scale_block_dim2=scale_block_dim2,
        scaling_mode=scaling_mode_str,
        blocked_scale_output=True,
    )
    if scale_block_dim2 == 32:
        s_blocked_full = (
            torch.stack(
                [
                    from_blocked(s[e], s_rows, s_cols).view(torch.uint8)
                    for e in range(E)
                ],
                dim=0,
            )
            .view(torch.float8_e8m0fnu)
            .to(s_ref.dtype)
        )
        s_ref_replicated = s_ref.transpose(-2, -1).repeat_interleave(block_size, dim=1)
        torch.testing.assert_close(s_blocked_full, s_ref_replicated, rtol=0, atol=0)
    s = (
        torch.stack([undo_scale(s[e]).view(torch.uint8) for e in range(E)], dim=0)
        .view(torch.float8_e8m0fnu)
        .to(s_ref.dtype)
    )
    # Check scales
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)

    # Check quantized values
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
    assert y.stride() == y_ref.stride(), "quantized tensor strides do not match"

    y_unblocked, s_unblocked = mxfp8_quantize_cuda_3d(
        x,
        block_size=block_size,
        scale_block_dim1=block_size,
        scale_block_dim2=scale_block_dim2,
        scaling_mode=scaling_mode_str,
        blocked_scale_output=False,
    )
    s_unblocked = s_unblocked.to(s_ref.dtype)
    torch.testing.assert_close(s_unblocked, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(y_unblocked, y_ref, rtol=0, atol=0)
    assert y_unblocked.stride() == y_ref.stride(), (
        "unblocked quantized tensor strides do not match"
    )


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@pytest.mark.parametrize("M", (128, 8192))
@pytest.mark.parametrize("K", (1536, 5120, 7168, 8192))
@pytest.mark.parametrize("input_dtype", (torch.bfloat16,))
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_cuda_mx_dim0_2d_numerics(M, K, input_dtype, scaling_mode):
    scaling_mode_str = scaling_mode.value.lower()
    block_size = 32

    # Use distinct incrementing values from 0 to M*K-1 to make debugging easier.
    x = (
        torch.arange(0, M * K, dtype=input_dtype, device="cuda")
        .reshape(M, K)
        .contiguous()
    )

    # Reference implementation
    s_d0_ref, y_d0_ref = to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )
    s_d0_ref = triton_mx_block_rearrange(s_d0_ref)

    # CuTeDSL kernel implementation
    y_d0, s_d0 = mxfp8_quantize_2d_1x32_cutedsl(
        x,
        block_size=block_size,
        scaling_mode=scaling_mode_str,
    )

    # Check scales
    torch.testing.assert_close(s_d0, s_d0_ref, rtol=0, atol=0)

    # Check quantized values
    torch.testing.assert_close(y_d0, y_d0_ref, rtol=0, atol=0)

    # Verify row-major layout
    assert y_d0.stride() == (K, 1), "quantized tensor should be row-major"
    assert y_d0.stride() == y_d0_ref.stride(), "quantized tensor strides do not match"


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@pytest.mark.parametrize("M", (128, 1024))
@pytest.mark.parametrize("K", (128, 256, 1536, 5120, 7168, 8192))
@pytest.mark.parametrize("input_dtype", (torch.bfloat16,))
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
@pytest.mark.parametrize("blocked_scale_output", [False, True])
def test_cuda_mx_dim1_2d_numerics_32x1(
    M, K, input_dtype, scaling_mode, blocked_scale_output
):
    """Test 32x1 scaling kernel that quantizes along M dimension."""
    scaling_mode_str = scaling_mode.value.lower()
    block_size = 32

    # Ensure M is divisible by block_size for 32x1 scaling
    if M % block_size != 0:
        pytest.skip(
            f"M={M} must be divisible by block_size={block_size} for 32x1 scaling"
        )

    # Use distinct incrementing values from 0 to M*K-1 to make debugging easier.
    x = (
        torch.arange(0, M * K, dtype=input_dtype, device="cuda")
        .reshape(M, K)
        .contiguous()
    )

    # Reference implementation: transpose so M becomes the last dimension,
    # since to_mx scales along the final dimension
    x_t = x.transpose(-2, -1).contiguous()  # Shape: (K, M)
    s_d1_ref, y_d1_ref = to_mx(
        x_t,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )
    # Transpose quantized data back to (M, K) for comparison with kernel output
    y_d1_ref = y_d1_ref.transpose(-2, -1).contiguous()  # Shape back to (M, K)

    if blocked_scale_output:
        from torchao.prototype.mx_formats.utils import to_blocked

        # s_d1_ref is already (K, M//32) from to_mx, same format as kernel now returns
        s_d1_ref = to_blocked(s_d1_ref)

    # CuTeDSL 32x1 kernel implementation
    y_d1, s_d1 = mxfp8_quantize_2d_32x1_cutedsl(
        x,
        block_size=block_size,
        scaling_mode=scaling_mode_str,
        blocked_scale_output=blocked_scale_output,
    )

    # Verify output dimensions - data should not be padded, same as input
    assert y_d1.shape == (M, K), (
        f"Quantized data shape mismatch: expected ({M}, {K}), got {y_d1.shape}"
    )
    # Check scales - compare unblocked formats
    torch.testing.assert_close(s_d1, s_d1_ref, rtol=0, atol=0)

    # Check quantized values - no padding needed for data
    torch.testing.assert_close(y_d1, y_d1_ref, rtol=0, atol=0)


@pytest.mark.skipif(
    not _mxfp8_cuda_kernels_available,
    reason="CUDA kernel requires sm_100 and CUDA 12.8+",
)
@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("num_tokens", [128, 157, 4096, 16392])
@pytest.mark.parametrize("dim", [7168])
@pytest.mark.parametrize("num_groups", [1, 2, 4, 8])
@pytest.mark.parametrize("alignment_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cuda_fused_pad_token_groups(
    num_tokens: int, dim: int, num_groups: int, alignment_size: int, dtype: torch.dtype
):
    """Test fused_pad_token_groups_cuda kernel for padding token groups to alignment."""
    device = "cuda"

    # Create input activations
    inputs = torch.randn(num_tokens, dim, dtype=dtype, device=device)

    # Generate group offsets (end indices for each group)
    group_offsets = generate_jagged_offs(
        num_groups, num_tokens, multiple_of=1, device=device
    )

    # Get reference output
    ref_padded_tokens, ref_padded_start_offsets, ref_padded_offsets = (
        torch_pad_token_groups(inputs, group_offsets, alignment_size)
    )

    # Run CUDA kernel
    kernel_padded_tokens, kernel_padded_start_offsets, kernel_padded_end_offsets = (
        fused_pad_token_groups_cuda(inputs, group_offsets, alignment_size)
    )

    # All implementations now use the same upper bound output size
    # Verify outputs match
    assert torch.allclose(ref_padded_tokens, kernel_padded_tokens, rtol=0, atol=1e-5), (
        "Padded tokens do not match"
    )
    assert torch.equal(ref_padded_start_offsets, kernel_padded_start_offsets), (
        "Padded group start offsets do not match"
    )
    assert torch.equal(ref_padded_offsets, kernel_padded_end_offsets), (
        "Padded group end offsets do not match"
    )


@pytest.mark.skipif(
    not _mxfp8_cuda_kernels_available,
    reason="CUDA kernel requires sm_100 and CUDA 12.8+",
)
@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("num_tokens", [128, 157, 4096])
@pytest.mark.parametrize("dim", [7168])
@pytest.mark.parametrize("num_groups", [1, 2, 4, 8])
@pytest.mark.parametrize("alignment_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cuda_fused_unpad_token_groups(
    num_tokens: int, dim: int, num_groups: int, alignment_size: int, dtype: torch.dtype
):
    """Test fused_unpad_token_groups_cuda kernel for removing padding from token groups."""
    device = "cuda"

    # Create input activations
    inputs = torch.randn(num_tokens, dim, dtype=dtype, device=device)

    # Generate group offsets (end indices for each group)
    group_offsets = generate_jagged_offs(
        num_groups, num_tokens, multiple_of=1, device=device
    )

    # First pad the tokens to create padded inputs
    padded_tokens, padded_group_start_offsets, padded_offsets = torch_pad_token_groups(
        inputs, group_offsets, alignment_size
    )

    # Get reference output using torch implementation
    ref_unpadded_tokens = torch_unpad_token_groups(
        padded_tokens,
        group_offsets,
        padded_group_start_offsets,
        num_tokens,
        alignment_size,
    )

    # Run CUDA kernel
    kernel_unpadded_tokens = fused_unpad_token_groups_cuda(
        padded_tokens,
        group_offsets,
        padded_group_start_offsets,
        num_tokens,
        alignment_size,
    )

    # Verify outputs match
    assert torch.allclose(
        ref_unpadded_tokens, kernel_unpadded_tokens, rtol=0, atol=1e-5
    ), "Unpadded tokens do not match"

    # Verify that unpad correctly reverses pad operation
    assert torch.allclose(inputs, kernel_unpadded_tokens, rtol=0, atol=1e-5), (
        "Unpadded tokens should match original inputs"
    )


@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
@pytest.mark.parametrize(
    "m,k",
    [(128, 5120), (1024, 8192), (4096, 5120)],
)
def test_triton_fp8_rowwise_2d_scale_and_cast(
    m: int, k: int, round_scales_to_power_of_2: bool
):
    device = "cuda"
    float8_dtype = torch.float8_e4m3fn

    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)

    # PyTorch reference: 3-kernel sequence
    ref_scales = tensor_to_scale(
        x,
        float8_dtype,
        scaling_granularity=ScalingGranularity.AXISWISE,
        axiswise_dim=-1,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    ref_fp8 = to_fp8_saturated(x.to(torch.float32) * ref_scales, float8_dtype)

    # Fused Triton kernel
    triton_fp8, triton_scales = triton_fp8_rowwise_2d_scale_and_cast(
        x,
        output_dtype=float8_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    assert ref_fp8.shape == triton_fp8.shape, "fp8 output shapes not equal"
    assert ref_scales.shape == triton_scales.shape, "scale shapes not equal"
    assert torch.allclose(ref_fp8, triton_fp8, rtol=0, atol=0), "fp8 data not equal"
    assert torch.allclose(ref_scales, triton_scales, rtol=0, atol=0), "scales not equal"


@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
@pytest.mark.parametrize(
    "e,n,k",
    [(1, 8192, 5120), (2, 5120, 8192), (8, 8192, 5120)],
)
def test_triton_fp8_colwise_3d_scale_and_cast(
    e: int, n: int, k: int, round_scales_to_power_of_2: bool
):
    device = "cuda"
    float8_dtype = torch.float8_e4m3fn

    torch.manual_seed(0)
    # Allocate (E, N, K) row-major then transpose to (E, K, N) column-major
    # (matches B_t layout in _Float8GroupedMM.forward: strides (K*N, 1, K)).
    x = torch.randn(e, n, k, dtype=torch.bfloat16, device=device).transpose(-2, -1)

    # PyTorch reference: 3-kernel sequence (axiswise along K, keeping N).
    ref_scales = tensor_to_scale(
        x,
        float8_dtype,
        scaling_granularity=ScalingGranularity.AXISWISE,
        axiswise_dim=-2,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    ref_fp8 = to_fp8_saturated(x.to(torch.float32) * ref_scales, float8_dtype)

    # Fused Triton kernel
    triton_fp8, triton_scales = triton_fp8_colwise_3d_scale_and_cast(
        x,
        output_dtype=float8_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )

    assert ref_fp8.shape == triton_fp8.shape, "fp8 output shapes not equal"
    assert ref_scales.shape == triton_scales.shape, "scale shapes not equal"
    assert torch.allclose(ref_fp8, triton_fp8, rtol=0, atol=0), "fp8 data not equal"
    assert torch.allclose(ref_scales, triton_scales, rtol=0, atol=0), "scales not equal"


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@skip_if_rocm("ROCm enablement in progress")
def test_cutedsl_1x32_group_validation_error():
    """Test that 1x32 CuTeDSL kernel raises error for non-128-multiple group sizes."""
    import subprocess
    import sys

    script = """\
import torch
from torchao.prototype.moe_training.kernels.mxfp8.quant import mxfp8_quantize_2d_1x32_cutedsl

M, K = 512, 1024
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
invalid_offsets = torch.tensor([127, 256, 384, 512], dtype=torch.int32, device="cuda")
mxfp8_quantize_2d_1x32_cutedsl(x, block_size=32, scaling_mode="rceil", offs=invalid_offsets)
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode != 0, (
        "Expected subprocess to fail for non-128-multiple group sizes"
    )


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@skip_if_rocm("ROCm enablement in progress")
def test_cutedsl_32x1_group_validation_error():
    """Test that 32x1 CuTeDSL kernel raises error for non-128-multiple group sizes."""
    import subprocess
    import sys

    script = """\
import torch
from torchao.prototype.moe_training.kernels.mxfp8.quant import mxfp8_quantize_2d_32x1_cutedsl

M, K = 512, 1024
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
invalid_offsets = torch.tensor([127, 256, 384, 512], dtype=torch.int32, device="cuda")
mxfp8_quantize_2d_32x1_cutedsl(x, block_size=32, scaling_mode="rceil", offs=invalid_offsets)
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode != 0, (
        "Expected subprocess to fail for non-128-multiple group sizes"
    )


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@skip_if_rocm("ROCm enablement in progress")
def test_cutedsl_kernels_work_with_valid_128_multiple_groups():
    """Test that both CuTeDSL kernels work correctly with valid 128-multiple group sizes."""
    device = "cuda"
    M, K = 512, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # Create valid group offsets (all group sizes are multiples of 128)
    valid_group_sizes = [128, 256, 128]  # All multiples of 128
    valid_offsets = (
        torch.cumsum(torch.tensor(valid_group_sizes), dim=0).to(device).to(torch.int32)
    )

    # Verify all group sizes are multiples of 128
    group_sizes = torch.diff(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int32), valid_offsets])
    )
    assert torch.all(group_sizes % 128 == 0), (
        "Test setup failed: not all group sizes are multiples of 128"
    )

    # Both kernels should work without error
    y_1x32, s_1x32 = mxfp8_quantize_2d_1x32_cutedsl(
        x, block_size=32, scaling_mode="rceil", offs=valid_offsets
    )

    y_32x1, s_32x1 = mxfp8_quantize_2d_32x1_cutedsl(
        x, block_size=32, scaling_mode="rceil", offs=valid_offsets
    )

    # Basic output validation
    assert y_1x32.shape == (M, K)
    assert y_32x1.shape == (M, K)
