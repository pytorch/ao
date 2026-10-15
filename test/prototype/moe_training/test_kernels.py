# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# FP8 MoE kernels require FP8-capable hardware (SM 10.x on CUDA, MI300+ on ROCm)
from torchao.utils import ceil_div, is_MI300, is_MI350


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
    mxfp8_quantize_2d_1x32_32x1_cutedsl,
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_1x32_flydsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    mxfp8_quantize_2d_32x1_flydsl,
    mxfp8_quantize_3d_flydsl,
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
    _mxfp8_flydsl_kernels_available,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
    generate_jagged_offs,
    torch_to_3d_rowwise_float8_transpose_rhs,
    torch_to_float8_per_group_colwise,
    torch_to_float8_per_group_rowwise,
)
from torchao.prototype.mx_formats.config import (
    KernelPreference,
    MXFP8Dim1CastKernelChoice,
)
from torchao.prototype.mx_formats.kernels import triton_mx_block_rearrange
from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx
from torchao.prototype.mx_formats.utils import (
    _to_mxfp8_dim1_kernel_wrapper,
    from_blocked,
    to_blocked,
)
from torchao.testing._mxfp8_test_utils import (
    assert_mxfp8_semantics,
    make_mxfp8_semantic_cases,
)
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
    ("32x1_n", "32x32_n", "32x1_t", "32x32_t"),
)
def test_cuda_mx_3d_cutedsl_numerics(E, N, K, input_dtype, scaling_mode, variant):
    if not _mxfp8_cutedsl_kernels_available:
        pytest.skip("mxfp8_quantize_3d is unavailable")

    scaling_mode_str = (
        "floor" if scaling_mode == ScaleCalculationMode.FLOOR else "rceil"
    )
    block_size = 32
    scale_block_dim2 = 32 if variant in ("32x32_n", "32x32_t") else 1

    # Use disinct incrementing values from 0 to E*M*K-1 to make debugging easier.
    x = (
        torch.arange(0, E * N * K, dtype=input_dtype, device="cuda")
        .reshape(E, N, K)
        .contiguous()
    )
    x_t = x.transpose(-2, -1)

    if variant in ("32x1_t", "32x32_t"):
        if scale_block_dim2 == 1:
            s_ref, y_ref = to_mx(
                x.contiguous(),
                elem_dtype=torch.float8_e4m3fn,
                block_size=block_size,
                scaling_mode=scaling_mode,
            )
            y_ref = y_ref.transpose(-2, -1)
            s_ref = s_ref.transpose(-2, -1)
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
                .transpose(-2, -1)
            )
            s_ref = s_ref.transpose(-2, -1)

        y_unblocked, s_unblocked = mxfp8_quantize_cuda_3d(
            x_t,
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
            "transposed-input unblocked quantized tensor strides do not match"
        )
        y, s = mxfp8_quantize_cuda_3d(
            x_t,
            block_size=block_size,
            scale_block_dim1=block_size,
            scale_block_dim2=scale_block_dim2,
            scaling_mode=scaling_mode_str,
            blocked_scale_output=True,
        )
        s_rows, s_cols = x_t.shape[-1], x_t.shape[-2] // block_size
        s_blocked_full = torch.stack(
            [from_blocked(s[e], s_rows, s_cols).view(torch.uint8) for e in range(E)],
            dim=0,
        ).view(torch.float8_e8m0fnu)
        if scale_block_dim2 == 32:
            s_ref_replicated = s_ref.transpose(-2, -1).repeat_interleave(
                block_size, dim=1
            )
            torch.testing.assert_close(
                s_blocked_full.to(s_ref.dtype), s_ref_replicated, rtol=0, atol=0
            )
            s_logical = (
                s_blocked_full[:, ::block_size, :].transpose(-2, -1).to(s_ref.dtype)
            )
        else:
            s_logical = s_blocked_full.transpose(-2, -1).to(s_ref.dtype)
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


def _make_mxfp8_edge_input(shape, dtype, device, pattern: str) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim

    if pattern == "zeros":
        x = torch.zeros(numel, dtype=torch.float32, device=device)
    elif pattern == "arange_signed":
        x = (torch.arange(numel, dtype=torch.float32, device=device) % 257) - 128
    elif pattern == "wide_dynamic":
        base = torch.arange(numel, dtype=torch.float32, device=device)
        signs = torch.where(base.to(torch.int64) % 2 == 0, 1.0, -1.0)
        magnitudes = torch.pow(2.0, (base % 17) - 8)
        x = signs * magnitudes
    else:
        raise AssertionError(f"unknown pattern: {pattern}")

    return x.reshape(shape).to(dtype)


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@pytest.mark.parametrize(
    "M,K,input_dtype,scaling_mode,stage_count,pattern",
    (
        (128, 128, torch.bfloat16, ScaleCalculationMode.RCEIL, 1, "zeros"),
        (384, 1408, torch.bfloat16, ScaleCalculationMode.FLOOR, 2, "wide_dynamic"),
        (128, 256, torch.float32, ScaleCalculationMode.RCEIL, 1, "arange_signed"),
        (256, 384, torch.float32, ScaleCalculationMode.FLOOR, 2, "wide_dynamic"),
    ),
)
def test_cuda_mx_dim0_2d_cutedsl_edge_inputs(
    M, K, input_dtype, scaling_mode, stage_count, pattern
):
    block_size = 32
    x = _make_mxfp8_edge_input((M, K), input_dtype, "cuda", pattern)

    s_ref, y_ref = to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )
    s_ref = triton_mx_block_rearrange(s_ref)

    y, s = mxfp8_quantize_2d_1x32_cutedsl(
        x,
        block_size=block_size,
        scaling_mode=scaling_mode.value.lower(),
        stage_count=stage_count,
    )

    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
    assert y.stride() == (K, 1)


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@pytest.mark.parametrize(
    "M,K,input_dtype,scaling_mode,stage_count,blocked_scale_output,pattern",
    (
        (128, 128, torch.bfloat16, ScaleCalculationMode.RCEIL, 1, False, "zeros"),
        (
            384,
            1408,
            torch.bfloat16,
            ScaleCalculationMode.FLOOR,
            2,
            True,
            "wide_dynamic",
        ),
        (128, 256, torch.float32, ScaleCalculationMode.RCEIL, 1, True, "arange_signed"),
        (256, 384, torch.float32, ScaleCalculationMode.FLOOR, 2, False, "wide_dynamic"),
    ),
)
def test_cuda_mx_dim1_2d_cutedsl_edge_inputs(
    M, K, input_dtype, scaling_mode, stage_count, blocked_scale_output, pattern
):
    block_size = 32
    x = _make_mxfp8_edge_input((M, K), input_dtype, "cuda", pattern)

    x_t = x.transpose(-2, -1).contiguous()
    s_ref, y_ref = to_mx(
        x_t,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )
    y_ref = y_ref.transpose(-2, -1).contiguous()
    if blocked_scale_output:
        s_ref = to_blocked(s_ref)

    y, s = mxfp8_quantize_2d_32x1_cutedsl(
        x,
        block_size=block_size,
        scaling_mode=scaling_mode.value.lower(),
        stage_count=stage_count,
        blocked_scale_output=blocked_scale_output,
    )

    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
    assert y.stride() == (1, M)


@pytest.mark.parametrize("backend", ("cutedsl", "flydsl"))
@pytest.mark.parametrize("orientation", ("1x32", "32x1"))
@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("scaling_mode", ("floor", "rceil"))
def test_mxfp8_dsl_special_value_semantics(
    backend, orientation, input_dtype, scaling_mode
):
    if backend == "cutedsl" and not _mxfp8_cutedsl_kernels_available:
        pytest.skip("MXFP8 CuTeDSL kernels not available")
    if backend == "flydsl" and not _mxfp8_flydsl_kernels_available:
        pytest.skip("MXFP8 FlyDSL kernels not available")

    cases = make_mxfp8_semantic_cases(input_dtype, scaling_mode, device="cuda")
    num_cases = len(cases.names)
    rows = torch.zeros((128, 128), dtype=input_dtype, device="cuda")
    rows[:num_cases] = cases.inputs.repeat(1, 4)
    x = rows if orientation == "1x32" else rows.t().contiguous()
    if backend == "cutedsl" and orientation == "1x32":
        qdata, blocked_scales = mxfp8_quantize_2d_1x32_cutedsl(
            x, block_size=32, scaling_mode=scaling_mode
        )
        scales = from_blocked(blocked_scales, 128, 4)
    elif backend == "cutedsl":
        qdata, scales = mxfp8_quantize_2d_32x1_cutedsl(
            x,
            block_size=32,
            scaling_mode=scaling_mode,
            blocked_scale_output=False,
        )
    elif orientation == "1x32":
        qdata, scales = mxfp8_quantize_2d_1x32_flydsl(
            x, block_size=32, scaling_mode=scaling_mode
        )
    else:
        qdata, scales = mxfp8_quantize_2d_32x1_flydsl(
            x, block_size=32, scaling_mode=scaling_mode
        )

    if orientation == "32x1":
        qdata = qdata.t()
    for block_idx in range(4):
        assert_mxfp8_semantics(
            qdata[:num_cases, block_idx * 32 : (block_idx + 1) * 32],
            scales[:num_cases, block_idx : block_idx + 1],
            cases,
        )

    # Keep the backend-vs-to_mx comparison in addition to the hardcoded contract.
    ref_scales, ref_qdata = to_mx(
        rows,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=ScaleCalculationMode(scaling_mode),
    )
    assert torch.equal(qdata.view(torch.uint8), ref_qdata.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))


@pytest.mark.parametrize("backend", ("cutedsl", "flydsl"))
@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("scaling_mode", ("floor", "rceil"))
@pytest.mark.parametrize("broadcast", ("rows", "columns"))
def test_mxfp8_dsl_3d_special_value_semantics(
    backend, input_dtype, scaling_mode, broadcast
):
    if backend == "cutedsl" and not _mxfp8_cutedsl_kernels_available:
        pytest.skip("MXFP8 CuTeDSL kernels not available")
    if backend == "flydsl" and not _mxfp8_flydsl_kernels_available:
        pytest.skip("MXFP8 FlyDSL kernels not available")

    cases = make_mxfp8_semantic_cases(input_dtype, scaling_mode, device="cuda")
    # Row-constant tiles exercise the per-lane N reduction; column-constant
    # tiles distribute each case across the subsequent K-lane warp reduction.
    if broadcast == "rows":
        tiles = cases.inputs.unsqueeze(-1).expand(-1, 32, 32)
        expected_tiles = cases.expected_data.unsqueeze(-1).expand(-1, 32, 32)
    else:
        tiles = cases.inputs.unsqueeze(1).expand(-1, 32, 32)
        expected_tiles = cases.expected_data.unsqueeze(1).expand(-1, 32, 32)

    # Pack one 32x32 tile per semantic case along K.
    x = tiles.permute(1, 0, 2).reshape(1, 32, -1).contiguous()
    expected_data = expected_tiles.permute(1, 0, 2).reshape(1, 32, -1)
    expected_scales = cases.expected_scales.T.unsqueeze(0)

    quantize = (
        mxfp8_quantize_cuda_3d if backend == "cutedsl" else mxfp8_quantize_3d_flydsl
    )
    qdata, scales = quantize(
        x,
        block_size=32,
        scale_block_dim1=32,
        scale_block_dim2=32,
        scaling_mode=scaling_mode,
        blocked_scale_output=False,
    )

    actual_scales = scales.view(torch.uint8).cpu()
    actual_data = qdata.view(torch.uint8).cpu()
    for case_idx, case_name in enumerate(cases.names):
        assert torch.equal(
            actual_scales[0, 0, case_idx], expected_scales[0, 0, case_idx]
        ), f"scale mismatch for {broadcast}-broadcast {case_name}"
        case_slice = slice(case_idx * 32, (case_idx + 1) * 32)
        assert torch.equal(
            actual_data[0, :, case_slice], expected_data[0, :, case_slice]
        ), f"data mismatch for {broadcast}-broadcast {case_name}"

    y_ref, s_ref = _mxfp8_3d_edge_reference(
        x, "32x32_n", ScaleCalculationMode(scaling_mode)
    )
    assert torch.equal(qdata.view(torch.uint8), y_ref.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), s_ref.view(torch.uint8))


def _mxfp8_3d_edge_reference(
    x: torch.Tensor,
    variant: str,
    scaling_mode: ScaleCalculationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = 32
    if variant == "32x1_t":
        s_ref, y_ref = to_mx(
            x.contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scaling_mode,
        )
        return y_ref.transpose(-2, -1), s_ref.transpose(-2, -1)

    if variant == "32x1_n":
        s_ref, y_ref = to_mx(
            x.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scaling_mode,
        )
        return y_ref.transpose(-2, -1), s_ref.transpose(-2, -1)

    E, N, K = x.shape
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
    y_ref = (
        y_tiles_ref.view(E, N // block_size, K // block_size, block_size, block_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(E, N, K)
    )
    if variant == "32x32_t":
        return y_ref.transpose(-2, -1), s_ref.squeeze(-1).transpose(-2, -1)

    assert variant == "32x32_n"
    y_ref = y_ref.transpose(-2, -1).contiguous().transpose(-2, -1)
    return y_ref, s_ref.squeeze(-1)


def _mxfp8_3d_logical_scales_from_blocked(
    scales_blocked: torch.Tensor,
    quant_input: torch.Tensor,
    scale_block_dim2: int,
) -> torch.Tensor:
    block_size = 32
    s_rows = quant_input.shape[-1]
    s_cols = quant_input.shape[-2] // block_size
    s_blocked_full = (
        torch.stack(
            [
                from_blocked(scales_blocked[e], s_rows, s_cols).view(torch.uint8)
                for e in range(quant_input.shape[0])
            ],
            dim=0,
        )
        .view(torch.float8_e8m0fnu)
        .to(torch.float32)
    )
    if scale_block_dim2 == 32:
        return s_blocked_full[:, ::block_size, :].transpose(-2, -1).contiguous()
    return s_blocked_full.transpose(-2, -1).contiguous()


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@pytest.mark.parametrize(
    "E,N,K,input_dtype,variant,scaling_mode,stage_count,blocked_scale_output,pattern",
    (
        (
            1,
            32,
            128,
            torch.bfloat16,
            "32x1_n",
            ScaleCalculationMode.RCEIL,
            1,
            False,
            "zeros",
        ),
        (
            2,
            64,
            416,
            torch.bfloat16,
            "32x32_t",
            ScaleCalculationMode.FLOOR,
            2,
            True,
            "wide_dynamic",
        ),
        (
            1,
            32,
            128,
            torch.float32,
            "32x1_t",
            ScaleCalculationMode.RCEIL,
            1,
            True,
            "arange_signed",
        ),
        (
            2,
            96,
            416,
            torch.float32,
            "32x32_n",
            ScaleCalculationMode.FLOOR,
            2,
            False,
            "wide_dynamic",
        ),
    ),
)
def test_cuda_mx_3d_cutedsl_edge_inputs(
    E,
    N,
    K,
    input_dtype,
    variant,
    scaling_mode,
    stage_count,
    blocked_scale_output,
    pattern,
):
    block_size = 32
    x = _make_mxfp8_edge_input((E, N, K), input_dtype, "cuda", pattern)
    quant_input = x.transpose(-2, -1) if variant.endswith("_t") else x
    ref_input = x if not variant.endswith("_t") else quant_input.transpose(-2, -1)
    y_ref, s_ref = _mxfp8_3d_edge_reference(ref_input, variant, scaling_mode)
    scale_block_dim2 = 32 if "32x32" in variant else 1

    y, s = mxfp8_quantize_cuda_3d(
        quant_input,
        block_size=block_size,
        scale_block_dim1=block_size,
        scale_block_dim2=scale_block_dim2,
        scaling_mode=scaling_mode.value.lower(),
        stage_count=stage_count,
        blocked_scale_output=blocked_scale_output,
    )

    if blocked_scale_output:
        s = _mxfp8_3d_logical_scales_from_blocked(
            s,
            quant_input,
            scale_block_dim2,
        ).to(s_ref.dtype)
    else:
        s = s.to(s_ref.dtype)

    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
    assert y.stride() == y_ref.stride()


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


# =============================================================================
# FlyDSL MXFP8 quantize kernels (AMD CDNA3+ via FlyDSL).
#
# AMD counterparts to the cutedsl tests above. Same numerics reference
# (`to_mx` with FLOOR mode), gated on `_mxfp8_flydsl_kernels_available`.
# =============================================================================

# Explicit shape tuples (not full M×K×N×E cross-products) so the suite runs
# in O(seconds). Each list holds a full-tile shape, an edge shape, and a
# K/N *tail* shape whose K is deliberately not a multiple of the kernel tile,
# so the per-lane tail-handling path is exercised against the to_mx reference.
# dtype × scaling_mode is still swept in full because those select distinct
# numeric code paths (bf16/f32 load, floor clamp vs. fused rceil cvt).
#
# Tail divisibility per kernel (all shapes keep K % 32 == 0, the MXFP8 min):
#   1x32: tile K = 2048; tail K=2080 (2080 % 2048 != 0).
#   32x1: tile K =  256; tail K=1408 (DSV3 hidden_dim, 1408 % 256 != 0).
#   3D:   tile K =  256; tail K=1408 (DSV3 hidden_dim, 1408 % 256 != 0).
_FLYDSL_1X32_SHAPES = ((1, 2048), (128, 4096), (64, 2080))  # (M, K)
_FLYDSL_32X1_SHAPES = ((32, 256), (128, 5120), (64, 1408))  # (M, K)
_FLYDSL_3D_SHAPES = ((1, 32, 256), (4, 64, 1024), (2, 64, 1408))  # (E, N, K)
_FLYDSL_3D_CFG_SHAPES = ((8, 256, 256), (2, 64, 1408))  # (E, N, K)


@pytest.mark.skipif(
    not _mxfp8_flydsl_kernels_available,
    reason="MXFP8 FlyDSL kernels not available (requires MI300/MI350 + FlyDSL runtime)",
)
@pytest.mark.parametrize("M,K", _FLYDSL_1X32_SHAPES)
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("scaling_mode", ("floor", "rceil"))
def test_flydsl_mx_dim1_2d_numerics(M, K, input_dtype, scaling_mode):
    """1x32 (K-direction) quantize matches torchao to_mx bit-exactly."""
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=input_dtype, device="cuda") * 30.0).contiguous()

    q_fly, s_fly = mxfp8_quantize_2d_1x32_flydsl(
        x, block_size=32, scaling_mode=scaling_mode
    )

    s_ref, q_ref = to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=ScaleCalculationMode(scaling_mode),
    )
    q_ref_fp8 = q_ref.to(torch.float8_e4m3fn).view(M, K)
    s_ref_u8 = s_ref.view(M, K // 32)

    torch.testing.assert_close(
        q_fly.view(torch.uint8),
        q_ref_fp8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        s_fly.view(torch.uint8),
        s_ref_u8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    assert q_fly.dtype == torch.float8_e4m3fn
    assert q_fly.stride() == (K, 1), "1x32 q_data must be row-major"
    assert s_fly.dtype == torch.float8_e8m0fnu


@pytest.mark.skipif(
    not _mxfp8_flydsl_kernels_available,
    reason="MXFP8 FlyDSL kernels not available",
)
@pytest.mark.parametrize("M,K", _FLYDSL_32X1_SHAPES)
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("scaling_mode", ("floor", "rceil"))
def test_flydsl_mx_dim0_2d_numerics(M, K, input_dtype, scaling_mode):
    """32x1 (M-direction) quantize matches torchao to_mx bit-exactly.

    The reference quantizes ``x.transpose(0, 1)`` along the last dim; the
    kernel emits column-major (M, K) output, so we transpose for comparison.
    """
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=input_dtype, device="cuda") * 30.0).contiguous()

    q_fly, s_fly = mxfp8_quantize_2d_32x1_flydsl(
        x, block_size=32, scaling_mode=scaling_mode
    )

    x_t = x.transpose(0, 1).contiguous()
    s_ref, q_ref = to_mx(
        x_t,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=ScaleCalculationMode(scaling_mode),
    )
    q_ref_fp8 = q_ref.to(torch.float8_e4m3fn).view(K, M)
    s_ref_u8 = s_ref.view(K, M // 32)

    fly_t = q_fly.transpose(0, 1).contiguous()
    torch.testing.assert_close(
        fly_t.view(torch.uint8),
        q_ref_fp8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        s_fly.view(torch.uint8),
        s_ref_u8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    assert q_fly.dtype == torch.float8_e4m3fn
    assert q_fly.stride() == (1, M), "32x1 q_data must be column-major"
    assert s_fly.dtype == torch.float8_e8m0fnu


@pytest.mark.skipif(
    not _mxfp8_flydsl_kernels_available,
    reason="MXFP8 FlyDSL kernels not available",
)
@pytest.mark.parametrize("M", (1, 16, 33))
def test_flydsl_2d_32x1_rejects_misaligned_M(M):
    """32x1 kernel requires M % 32 == 0; ``_pick_layout`` should raise a
    clear AssertionError for any M that is not a multiple of the block size.
    """
    x = torch.randn(M, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="must be a multiple of block_size"):
        mxfp8_quantize_2d_32x1_flydsl(x, block_size=32)


@pytest.mark.skipif(
    not _mxfp8_flydsl_kernels_available,
    reason="MXFP8 FlyDSL kernels not available",
)
@pytest.mark.parametrize("E,N,K", _FLYDSL_3D_SHAPES)
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("scaling_mode", ("floor", "rceil"))
def test_flydsl_mx_dim1_3d_numerics(E, N, K, input_dtype, scaling_mode):
    """3D MoE quantize matches torchao to_mx bit-exactly."""
    torch.manual_seed(0)
    x = (torch.randn(E, N, K, dtype=input_dtype, device="cuda") * 30.0).contiguous()

    q_fly, s_fly = mxfp8_quantize_3d_flydsl(x, block_size=32, scaling_mode=scaling_mode)

    x_t = x.transpose(1, 2).contiguous()
    s_ref, q_ref = to_mx(
        x_t,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=ScaleCalculationMode(scaling_mode),
    )
    q_ref_fp8 = q_ref.to(torch.float8_e4m3fn).view(E, K, N)
    s_ref_u8 = s_ref.view(E, K, N // 32)

    fly_t = q_fly.transpose(1, 2).contiguous()
    torch.testing.assert_close(
        fly_t.view(torch.uint8),
        q_ref_fp8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    s_fly_t = s_fly.transpose(1, 2).contiguous()
    torch.testing.assert_close(
        s_fly_t.view(torch.uint8),
        s_ref_u8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    assert q_fly.dtype == torch.float8_e4m3fn
    assert q_fly.stride() == (N * K, 1, N), "3D q_data must be per-expert col-major"
    assert s_fly.dtype == torch.float8_e8m0fnu


@pytest.mark.skipif(
    not _mxfp8_flydsl_kernels_available,
    reason="MXFP8 FlyDSL kernels not available",
)
# The default-config 3D test above already sweeps (E,N,K,dtype) on
# (sbk=1, bso=False); this test only needs to exercise the
# (scale_block_k × blocked_scale_output) cross-product, so the shape grid is
# just one full-tile shape plus one K-tail shape (K=1408 % 256 != 0) to cover
# the tail path under sbk=32 and blocked_scale_output.
@pytest.mark.parametrize("E,N,K", _FLYDSL_3D_CFG_SHAPES)
@pytest.mark.parametrize("input_dtype", (torch.bfloat16,))
@pytest.mark.parametrize(
    "scale_block_k",
    (1, 32),
    ids=("32x1", "32x32"),
)
@pytest.mark.parametrize("blocked_scale_output", (False, True))
@pytest.mark.parametrize("scaling_mode", ("floor", "rceil"))
def test_amd_mx_3d_flydsl_numerics(
    E, N, K, input_dtype, scale_block_k, blocked_scale_output, scaling_mode
):
    """3D MoE quantize across (scale_block_k, blocked_scale_output,
    scaling_mode) matches the to_mx reference bit-exactly (FlyDSL backend).

    Mirrors test_cuda_mx_3d_cutedsl_numerics.
    """
    block_size = 32
    scale_mode_enum = ScaleCalculationMode(scaling_mode)

    torch.manual_seed(0)
    x = (torch.randn(E, N, K, dtype=input_dtype, device="cuda") * 0.5).contiguous()

    if scale_block_k == 1:
        s_ref_t, y_ref_t = to_mx(
            x.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scale_mode_enum,
        )
        y_ref = y_ref_t.transpose(-2, -1)
        s_ref = s_ref_t.transpose(-2, -1).contiguous()  # (E, N//32, K)
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
            scaling_mode=scale_mode_enum,
        )
        s_ref = s_ref.squeeze(-1)  # (E, N//32, K//32)
        y_ref = (
            y_tiles_ref.view(
                E, N // block_size, K // block_size, block_size, block_size
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, N, K)
        )
        y_ref = y_ref.transpose(-2, -1).contiguous().transpose(-2, -1)

    if blocked_scale_output:
        if scale_block_k == 1:
            s_ref_logical = s_ref.transpose(-2, -1).contiguous()
        else:
            s_ref_logical = s_ref.transpose(-2, -1).repeat_interleave(block_size, dim=1)
        s_expected = torch_to_blocked_per_group_3d(s_ref_logical)
    else:
        s_expected = s_ref

    y, s = mxfp8_quantize_3d_flydsl(
        x,
        block_size=block_size,
        scale_block_dim1=block_size,
        scale_block_dim2=scale_block_k,
        scaling_mode=scaling_mode,
        blocked_scale_output=blocked_scale_output,
    )

    torch.testing.assert_close(
        s.view(torch.uint8), s_expected.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        y.view(torch.uint8), y_ref.view(torch.uint8), rtol=0, atol=0
    )
    assert y.stride() == y_ref.stride(), "quantized tensor strides do not match"
    assert y.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float8_e8m0fnu


# =============================================================================
# Fused 1x32 + 32x1 MXFP8 quantization (cutedsl_quantize_2d_1x32_32x1_fused.py)
#
# The kernel replaces three kernels in a grouped-GEMM backward -- the 1x32
# rowwise cast, the 32x1 colwise cast, and the standalone K-groups scale swizzle
# -- with one pass over the input. to_mx(RCEIL) is the numerical oracle for both
# axes; per-group unblocking separately checks the dynamic column-scale layout.
# =============================================================================

_FUSED_CUTEDSL_REQUIRED = pytest.mark.skipif(
    not (_is_sm_10x() and _mxfp8_cutedsl_kernels_available),
    reason="Requires SM10.x and the CuTeDSL runtime",
)

_FUSED_1X32_32X1_SHAPES = (
    # (total_M, N, num_groups). Both dims must be multiples of 128. Each new
    # triple costs a JIT compile, so this list is deliberately short.
    (512, 256, 4),
    (1024, 512, 8),
    (2048, 512, 4),
    (4096, 1024, 8),
    # (M // 128) * (N // 128) = 64 * 9 = 576 CTAs, past SMALL_GRID_CTAS (512) in
    # _launch -- the kernel switches from 512 to 256 threads/CTA here, a
    # separately-coded path (threads == 256 branches in _kernel) that none of
    # the shapes above reach. Without this shape every other test in this file
    # only exercises the 512-thread branch.
    (8192, 1152, 8),
)


def _fused_1x32_32x1_offs_with_tail(total_M, num_groups, tail, device="cuda"):
    """Group end offsets covering total_M - tail rows, every size a multiple of 128.

    Unlike generate_jagged_offs, this deliberately stops SHORT of total_M so the
    uncovered-tile path is exercised.
    """
    covered = total_M - tail
    assert covered % 128 == 0
    n_blocks = covered // 128
    assert n_blocks >= num_groups
    sizes = [n_blocks // num_groups] * num_groups
    for i in range(n_blocks % num_groups):
        sizes[i] += 1
    ends = torch.cumsum(torch.tensor(sizes, dtype=torch.int32), 0) * 128
    return ends.to(device=device, dtype=torch.int32)


def _fused_1x32_32x1_composition_reference(x, offs):
    """The three kernels the fused kernel replaces, in the order backward runs them.

    Returns the fused kernel's four outputs as raw uint8, in the same order.
    """
    q_row, s_row = mxfp8_quantize_2d_1x32_cutedsl(
        x, block_size=32, scaling_mode="rceil", offs=offs
    )
    mx = _to_mxfp8_dim1_kernel_wrapper(
        x,
        32,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=x.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    s_col = triton_mx_block_rearrange_2d_K_groups(mx.scale, offs // 32)
    return (
        q_row.view(torch.uint8),
        s_row.view(torch.uint8),
        mx.qdata.view(torch.uint8),
        s_col.view(torch.uint8),
    )


def _assert_fused_1x32_32x1_matches_reference(x, offs, actual):
    """Check direct RCEIL bytes after unblocking; FP8 NaN signs are equivalent."""
    m, n = x.shape
    row_scales, row_data = to_mx(x, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL)
    col_scales, col_data = to_mx(
        x.t().contiguous(), torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    expected = (row_data, row_scales, col_data, col_scales.view(torch.uint8).clone())
    logical_row = from_blocked(actual[1].view(torch.uint8), m, n // 32)
    flat_col = actual[3].view(torch.uint8).flatten()
    if offs is None:
        logical_col = from_blocked(flat_col, n, m // 32)
    else:
        logical_col = torch.zeros_like(expected[3])
        start = 0
        for end in offs.tolist():
            if end > start:
                slab = flat_col[(start // 32) * n : (end // 32) * n]
                logical_col[:, start // 32 : end // 32] = from_blocked(
                    slab, n, (end - start) // 32
                )
            start = end
        expected[3][:, start // 32 :].zero_()
        # 128-aligned groups have no internal padding: the consumed prefix is
        # M*N/32 bytes, followed by legacy G*4 scale-column slack.
        assert torch.all(flat_col[m * n // 32 :] == 0), (
            "unused scale slack must be zero"
        )
    unpacked = (actual[0], logical_row, actual[2], logical_col)
    names = ("q_row", "s_row", "q_col", "s_col")
    for name, got, exp in zip(names, unpacked, expected, strict=True):
        got, exp = (
            got.contiguous().view(torch.uint8),
            exp.contiguous().view(torch.uint8),
        )
        if name.startswith("q_"):
            got, exp = (
                torch.where(got == 0xFF, 0x7F, got),
                torch.where(exp == 0xFF, 0x7F, exp),
            )
        torch.testing.assert_close(got, exp, rtol=0, atol=0, msg=f"{name} mismatch")


def _fused_1x32_32x1_randn(*shape, dtype=torch.bfloat16, device="cuda"):
    """Create reproducible inputs without changing process-global RNG state."""
    generator = torch.Generator(device=device).manual_seed(42)
    return torch.randn(*shape, dtype=dtype, device=device, generator=generator)


@_FUSED_CUTEDSL_REQUIRED
@pytest.mark.parametrize("total_M,N,num_groups", _FUSED_1X32_32X1_SHAPES)
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_numerics(total_M, N, num_groups):
    x = _fused_1x32_32x1_randn(total_M, N)
    offs = _fused_1x32_32x1_offs_with_tail(total_M, num_groups, 0)
    out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)

    q_row, s_row, q_col, s_col = out
    assert q_row.shape == (total_M, N)
    assert s_row.shape == (total_M, N // 32)
    assert q_col.shape == (N, total_M)
    assert q_row.stride() == (N, 1)
    assert q_col.stride() == (total_M, 1)
    assert s_col.shape[0] == ceil_div(N, 128) * 128

    _assert_fused_1x32_32x1_matches_reference(x, offs, out)
    expected = _fused_1x32_32x1_composition_reference(x, offs)
    for got, reference in zip(out, expected, strict=True):
        torch.testing.assert_close(got.view(torch.uint8), reference, rtol=0, atol=0)


@_FUSED_CUTEDSL_REQUIRED
@pytest.mark.parametrize("tail", (128, 512, 1024))
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_uncovered_tail(tail):
    """offs[-1] < total_M, which for a dropless MoE dispatcher is the NORM.

    Such a dispatcher sizes this buffer for worst-case per-expert padding and
    fills only a variable prefix, so rows that no group owns are expected rather
    than exceptional. Every offset generator in this file ends at exactly
    total_M, so nothing else here covers this path -- and getting it wrong is not
    a rounding error: the store lands inside group 0's slab and every n-tile CTA
    races on one 512-byte line, which surfaces downstream as grad_norm=inf.
    """
    total_M, N, num_groups = 2048, 512, 4
    covered = total_M - tail
    offs = _fused_1x32_32x1_offs_with_tail(total_M, num_groups, tail)
    assert int(offs[-1]) == covered < total_M

    x = _fused_1x32_32x1_randn(total_M, N)
    out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)
    _assert_fused_1x32_32x1_matches_reference(x, offs, out)


@_FUSED_CUTEDSL_REQUIRED
@pytest.mark.parametrize("scale", (1e-38, 1e-30, 1e30))
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_degenerate_blocks(scale):
    """Blocks whose amax is subnormal enough to land on E8M0 scale byte 0.

    Both halves apply a 2**127 multiplier there, matching the reciprocal
    `254 - scale_biased` of the kernels being replaced. `randn` structurally
    cannot reach this branch, so it has to be constructed: a suite built only on
    random tensors passes while leaving the multiplier untested, which is
    exactly how a 2x error survived here once already.
    """
    total_M, N, num_groups = 512, 256, 4
    x = (_fused_1x32_32x1_randn(total_M, N, dtype=torch.float32) * scale).to(
        torch.bfloat16
    )
    offs = _fused_1x32_32x1_offs_with_tail(total_M, num_groups, 0)
    out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)
    _assert_fused_1x32_32x1_matches_reference(x, offs, out)


@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_mixed_magnitude():
    """Interleaved tiny and huge rows, so degenerate and saturating blocks
    coexist in one launch and neither branch can be tuned for in isolation."""
    total_M, N, num_groups = 512, 256, 4
    x = _fused_1x32_32x1_randn(total_M, N)
    x[::4] = (x[::4].float() * 1e-36).to(torch.bfloat16)
    x[1::4] = (x[1::4].float() * 1e28).to(torch.bfloat16)
    offs = _fused_1x32_32x1_offs_with_tail(total_M, num_groups, 0)
    out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)
    _assert_fused_1x32_32x1_matches_reference(x, offs, out)


@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_dirty_output_buffers():
    """Every output byte must be written, including regions no group covers.

    Uses the destination-passing entry point with buffers pre-filled to 0xFF. A
    fresh torch.empty hides a never-written region whenever the allocator hands
    back zeroed pages, so this is the only test here that can catch it.
    """
    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_quantize_2d_1x32_32x1_fused import (
        quantize_fused,
        s_col_shape,
    )

    total_M, N, num_groups = 2048, 512, 4
    offs = _fused_1x32_32x1_offs_with_tail(total_M, num_groups, 512)
    x = _fused_1x32_32x1_randn(total_M, N)

    s_col_rows, s_col_cols = s_col_shape(total_M, N, num_groups)
    bufs = (
        torch.empty((total_M, N), device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((total_M, N // 32), device="cuda", dtype=torch.uint8),
        torch.empty((N, total_M), device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((s_col_rows, s_col_cols), device="cuda", dtype=torch.uint8),
    )
    for _ in range(3):
        for buf in bufs:
            buf.view(torch.uint8).fill_(0xFF)
        quantize_fused(
            x,
            offs=offs,
            q_row_out=bufs[0],
            s_row_out=bufs[1],
            q_col_out=bufs[2],
            s_col_out=bufs[3],
        )
        _assert_fused_1x32_32x1_matches_reference(x, offs, bufs)


@_FUSED_CUTEDSL_REQUIRED
@pytest.mark.parametrize("axis", ["rowwise", "colwise"])
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_special_value_semantics(axis):
    """Pin both orientations to the independent shared MXFP8 special-value contract."""
    cases = make_mxfp8_semantic_cases(torch.bfloat16, "rceil", device="cuda")
    num_cases = len(cases.names)
    x = torch.zeros((128, 128), dtype=torch.bfloat16, device="cuda")
    if axis == "rowwise":
        x[:num_cases] = cases.inputs.repeat(1, 4)
    else:
        x[:, :num_cases] = cases.inputs.t().repeat(4, 1)
    offs = torch.tensor([128], dtype=torch.int32, device="cuda")

    out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)
    q_data, blocked_scales = out[:2] if axis == "rowwise" else out[2:]

    scales = from_blocked(blocked_scales.flatten()[:512], 128, 4)
    for block_idx in range(4):
        assert_mxfp8_semantics(
            q_data[:num_cases, block_idx * 32 : (block_idx + 1) * 32],
            scales[:num_cases, block_idx : block_idx + 1],
            cases,
        )

    _assert_fused_1x32_32x1_matches_reference(x, offs, out)


@_FUSED_CUTEDSL_REQUIRED
@pytest.mark.parametrize("poison", ("inf", "nan", "inf_and_nan"))
@pytest.mark.parametrize("shape", [(128, 128), (8192, 2048)])
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_inf_and_nan(poison, shape):
    """Both NOSAT references poison nonfinite blocks, including quantized data."""
    # Exercise both the 512-thread small-grid and 256-thread large-grid paths.
    x = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    if poison in ("inf", "inf_and_nan"):
        x[0].fill_(float("inf"))
    if poison in ("nan", "inf_and_nan"):
        x[1].fill_(float("nan"))
    offs = torch.tensor([shape[0]], dtype=torch.int32, device="cuda")

    out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)
    _assert_fused_1x32_32x1_matches_reference(x, offs, out)
    assert int(out[3].view(torch.uint8).reshape(-1)[0]) == 255


@_FUSED_CUTEDSL_REQUIRED
@pytest.mark.parametrize("shape", [(128, 256), (256, 128)])
@pytest.mark.parametrize("blocked_scale_output", [False, True])
def test_mxfp8_32x1_cutedsl_fake_metadata(shape, blocked_scale_output):
    """Fake metadata must match real strides and both scale layout modes."""
    x = _fused_1x32_32x1_randn(*shape)
    torch.library.opcheck(
        torch.ops.torchao.mxfp8_quantize_2d_32x1_cutedsl.default,
        (x,),
        {"blocked_scale_output": blocked_scale_output},
        test_utils=["test_schema", "test_faketensor"],
    )


@_FUSED_CUTEDSL_REQUIRED
@pytest.mark.parametrize("shape", [(128, 256), (256, 128)])
@pytest.mark.parametrize("use_offsets", [False, True])
def test_mxfp8_1x32_32x1_cutedsl_fake_metadata(shape, use_offsets):
    """The fused custom op must expose all four real output layouts."""
    x = _fused_1x32_32x1_randn(*shape)
    offs = (
        torch.tensor([shape[0]], device="cuda", dtype=torch.int32)
        if use_offsets
        else None
    )
    torch.library.opcheck(
        torch.ops.torchao.mxfp8_quantize_2d_1x32_32x1_cutedsl.default,
        (x,),
        {"offs": offs},
        test_utils=["test_schema", "test_faketensor"],
    )


@pytest.mark.parametrize("shape", [(128, 128), (8192, 1152)])
@pytest.mark.parametrize("axis", ["rowwise", "colwise"])
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_scale_zero_boundary(shape, axis):
    """Check exact amax neighbors of the RCEIL scale0/1 boundary on both CTA paths."""
    values = torch.tensor(
        [0x045F, 0x0460, 0x0461, 0x04DF, 0x04E0, 0x04E1],
        device="cuda",
        dtype=torch.uint16,
    ).view(torch.bfloat16)
    m, n = shape
    if axis == "rowwise":
        x = values.repeat(ceil_div(m, len(values)))[:m, None].expand(m, n).contiguous()
    else:
        x = values.repeat(ceil_div(n, len(values)))[None, :n].expand(m, n).contiguous()
    for offs in [None, torch.tensor([m], dtype=torch.int32, device="cuda")]:
        out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)
        _assert_fused_1x32_32x1_matches_reference(x, offs, out)


@pytest.mark.parametrize(
    "offsets,m", [([0, 256, 256, 512], 512), ([512, 512], 512), ([0], 256), ([], 256)]
)
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_empty_groups(offsets, m):
    """Empty experts and an empty offset tensor cover no rows, unlike dense None."""
    x = _fused_1x32_32x1_randn(m, 256)
    offs = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x, offs=offs)
    _assert_fused_1x32_32x1_matches_reference(x, offs, out)


@pytest.mark.parametrize("shape", [(512, 256), (8192, 1152)])
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_dense_layout(shape):
    """Dense mode preserves defined grouped bytes without synthetic offsets or slack."""
    x = _fused_1x32_32x1_randn(*shape)
    dense = mxfp8_quantize_2d_1x32_32x1_cutedsl(x)
    grouped = mxfp8_quantize_2d_1x32_32x1_cutedsl(
        x, offs=torch.tensor([shape[0]], dtype=torch.int32, device="cuda")
    )
    assert dense[3].numel() == x.numel() // 32
    _assert_fused_1x32_32x1_matches_reference(x, None, dense)
    for index, (left, right) in enumerate(zip(dense, grouped, strict=True)):
        right = right.flatten()[: left.numel()] if index == 3 else right.flatten()
        torch.testing.assert_close(
            left.view(torch.uint8).flatten(), right.view(torch.uint8), rtol=0, atol=0
        )


@pytest.mark.parametrize("use_offsets", [False, True])
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_compile(use_offsets):
    """Exercise actual Inductor custom-op/fake contracts with two matrix shapes."""
    compiled = torch.compile(
        mxfp8_quantize_2d_1x32_32x1_cutedsl, fullgraph=True, dynamic=True
    )
    for m in [256, 512]:
        x = _fused_1x32_32x1_randn(m, 256)
        offs = (
            torch.tensor([m], dtype=torch.int32, device="cuda") if use_offsets else None
        )
        _assert_fused_1x32_32x1_matches_reference(x, offs, compiled(x, offs=offs))


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "kwargs,match",
    [({"block_size": 16}, "block_size"), ({"scaling_mode": "floor"}, "rceil")],
)
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_real_fake_api_contract(
    fake, kwargs, match
):
    """New public/fake signatures reject unsupported parameters consistently."""
    from contextlib import nullcontext

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() if fake else nullcontext():
        x = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match=match):
            mxfp8_quantize_2d_1x32_32x1_cutedsl(x, **kwargs)


@pytest.mark.parametrize(
    "name,shape,dtype",
    [
        ("q_row_out", (128, 128), torch.float8_e4m3fn),
        ("s_col_out", (128, 4), torch.uint8),
    ],
)
@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "strides", "alignment", "device"]
)
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_output_contract(
    monkeypatch, name, shape, dtype, invalid
):
    """Invalid destinations fail before any raw aligned GPU write."""
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        cutedsl_quantize_2d_1x32_32x1_fused as kernel,
    )

    x = _fused_1x32_32x1_randn(128, 128)
    output = torch.empty(shape, dtype=dtype, device="cuda")
    if invalid == "shape":
        output = torch.empty((shape[0], shape[1] + 1), dtype=dtype, device="cuda")
    elif invalid == "dtype":
        output = output.float()
    elif invalid == "strides":
        output = torch.empty(shape[::-1], dtype=dtype, device="cuda").t()
    elif invalid == "alignment":
        output = torch.empty(output.numel() + 1, dtype=dtype, device="cuda")[1:].view(
            shape
        )
    else:
        output = output.cpu()
    monkeypatch.setattr(
        kernel, "run", lambda *args: pytest.fail("invalid output reached a GPU launch")
    )
    with pytest.raises(AssertionError, match=name):
        kernel.quantize_fused(x, **{name: output})


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_input_device_stream():
    """Launch on the input's nondefault stream even when another device is current."""
    x = _fused_1x32_32x1_randn(256, 256, device="cuda:1")
    with torch.cuda.device(1):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x.add_(1)
            with torch.cuda.device(0):
                out = mxfp8_quantize_2d_1x32_32x1_cutedsl(x)
        stream.synchronize()
    _assert_fused_1x32_32x1_matches_reference(x, None, out)


@pytest.mark.parametrize("alias", ["input", "another_output", "disjoint_views"])
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_output_overlap(monkeypatch, alias):
    """Overlapping destinations fail before writes; disjoint arena views are valid."""
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        cutedsl_quantize_2d_1x32_32x1_fused as kernel,
    )

    x = _fused_1x32_32x1_randn(128, 128)
    if alias == "input":
        output = x.view(torch.float8_e4m3fn).flatten()[: x.numel()].view(x.shape)
        buffers = {"q_row_out": output}
    elif alias == "another_output":
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        buffers = {"q_row_out": output, "q_col_out": output}
    else:
        arena = torch.empty(
            2 * x.numel() + 2 * x.numel() // 32, dtype=torch.uint8, device=x.device
        )
        buffers, start = {}, 0
        for name, shape, dtype in [
            ("q_row_out", (128, 128), torch.float8_e4m3fn),
            ("s_row_out", (128, 4), torch.uint8),
            ("q_col_out", (128, 128), torch.float8_e4m3fn),
            ("s_col_out", (128, 4), torch.uint8),
        ]:
            size = shape[0] * shape[1]
            buffers[name] = arena[start : start + size].view(dtype).view(shape)
            start += size
        out = kernel.quantize_fused(x, **buffers)
        _assert_fused_1x32_32x1_matches_reference(x, None, out)
        return
    monkeypatch.setattr(
        kernel, "run", lambda *args: pytest.fail("aliased output reached a GPU launch")
    )
    with pytest.raises(AssertionError, match="must not overlap"):
        kernel.quantize_fused(x, **buffers)


def _run_mxfp8_fused_offset_validation(offsets):
    """Run only device validation: no raw stores, async errors isolated in a child."""
    import cuda.bindings.driver as driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
        validate_group_sizes,
    )

    @cute.kernel
    def check(tensor):
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            validate_group_sizes(tensor, max_offset=256)

    @cute.jit
    def launch(tensor, stream):
        check(tensor).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    tensor = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    compiled = cute.compile(
        launch,
        from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True),
        make_fake_stream(),
        options="--enable-tvm-ffi --enable-assertions",
    )
    compiled(tensor, driver.CUstream(int(torch.cuda.current_stream().cuda_stream)))
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "offsets,match",
    [
        ([-128, 256], "Offsets must be nonnegative"),
        ([256, 128], "Offsets must be nondecreasing"),
        ([384], "Offset exceeds input rows"),
        ([0, 128, 128, 256], None),
    ],
)
@_FUSED_CUTEDSL_REQUIRED
def test_mxfp8_quantize_2d_1x32_32x1_cutedsl_offset_bounds(offsets, match):
    """Reject negative, descending and too-large aligned endpoints on the device."""
    import subprocess
    import sys

    script = (
        "from test.prototype.moe_training.test_kernels import _run_mxfp8_fused_offset_validation; _run_mxfp8_fused_offset_validation("
        + repr(offsets)
        + ")"
    )
    process = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    error = process.stdout + process.stderr
    if match is None:
        assert process.returncode == 0, error
    else:
        assert (
            process.returncode != 0
            and f"Assertion Error: {match}" in error
            and "CUDA error:" in error
        ), error
