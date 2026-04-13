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
from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx
from torchao.prototype.mx_formats.utils import from_blocked
from torchao.testing.utils import skip_if_rocm


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
def test_cuda_mx_dim1_3d_numerics(E, N, K, input_dtype, scaling_mode):
    if not _mxfp8_cutedsl_kernels_available:
        pytest.skip("mxfp8_quantize_3d is unavailable")

    scaling_mode_str = (
        "floor" if scaling_mode == ScaleCalculationMode.FLOOR else "rceil"
    )
    block_size = 32

    # Use disinct incrementing values from 0 to E*M*K-1 to make debugging easier.
    x = (
        torch.arange(0, E * N * K, dtype=input_dtype, device="cuda")
        .reshape(E, N, K)
        .contiguous()
    )

    # Reference implementation
    s_d1_ref, y_d1_ref = to_mx(
        # Transpose so N is final dim, since to_mx scales along that dim
        x.transpose(-2, -1).contiguous(),
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )

    # Transpose tensors and scales back so we have effectively
    # quantized input shape (E, N, K) along N
    y_d1_ref = y_d1_ref.transpose(-2, -1)
    s_d1_ref = s_d1_ref.transpose(-2, -1)
    y_d1, s_d1 = mxfp8_quantize_cuda_3d(
        x,
        block_size=block_size,
        scaling_mode=scaling_mode_str,
    )
    s_d1 = torch.stack(
        [
            from_blocked(s_d1[e], K, N // block_size).transpose(-2, -1).contiguous()
            for e in range(E)
        ],
        dim=0,
    ).to(s_d1_ref.dtype)
    # Check scales
    torch.testing.assert_close(s_d1, s_d1_ref, rtol=0, atol=0)

    # Check quantized values
    torch.testing.assert_close(y_d1, y_d1_ref, rtol=0, atol=0)
    assert y_d1.stride() == y_d1_ref.stride(), "quantized tensor strides do not match"


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="MXFP8 requires CUDA SM 10.x",
)
@pytest.mark.skipif(
    not _mxfp8_cutedsl_kernels_available,
    reason="MXFP8 cutedsl kernels not available",
)
@pytest.mark.parametrize("M", (32, 160, 8192))
@pytest.mark.parametrize("K", (32, 96, 1536, 5120, 7168, 8192))
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

    # CuTeDSL kernel implementation
    y_d0, s_d0 = mxfp8_quantize_2d_1x32_cutedsl(
        x,
        block_size=block_size,
        scaling_mode=scaling_mode_str,
    )

    # Convert blocked scales back to reference format
    s_d0 = from_blocked(s_d0, M, K // block_size).to(s_d0_ref.dtype)

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
@pytest.mark.parametrize("M", (32, 128, 160, 1024))
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
    padded_tokens, padded_group_start_offsets, padded_group_end_offsets = (
        torch_pad_token_groups(inputs, group_offsets, alignment_size)
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


# Import silu_mul kernels
from torchao.prototype.moe_training.ep.syncless.silu_mul_kernel import (
    silu_mul_bw,
    silu_mul_fw,
)


def _torch_silu_mul_fw(h13_input: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Reference PyTorch implementation of silu_mul forward (matching kernel precision)."""
    h1, h3 = h13_input[:num_tokens].chunk(2, dim=-1)

    # Better precision: keep everything in float32 until final cast
    h1_f32 = h1.to(torch.float32)
    h3_f32 = h3.to(torch.float32)
    silu_h1_f32 = h1_f32 * torch.sigmoid(h1_f32)
    result = (silu_h1_f32 * h3_f32).to(h1.dtype)

    return result


def _torch_silu_mul_bw(h13_input: torch.Tensor, grad_h: torch.Tensor, num_tokens: int):
    """Reference PyTorch implementation of silu_mul backward (matching kernel precision)."""
    h1, h3 = h13_input[:num_tokens].chunk(2, dim=-1)

    # Recompute forward (must match forward kernel exactly)
    h1_f32 = h1.to(torch.float32)
    h3_f32 = h3.to(torch.float32)
    sig_h1 = torch.sigmoid(h1_f32)
    silu_h1_f32 = h1_f32 * sig_h1
    # Keep everything in float32 until final cast for precision
    h = (silu_h1_f32 * h3_f32).to(h1.dtype)

    # Backward - keep in float32 for gradient precision
    grad_h_valid = grad_h[:num_tokens]
    grad_h_f32 = grad_h_valid.to(torch.float32)
    dsilu = sig_h1 + h1_f32 * sig_h1 * (1.0 - sig_h1)
    grad_h1 = (grad_h_f32 * h3_f32 * dsilu).to(h1.dtype)
    grad_h3 = (grad_h_f32 * silu_h1_f32).to(h1.dtype)

    return h, torch.cat([grad_h1, grad_h3], dim=-1)


@pytest.mark.parametrize("num_tokens", [32, 128, 256, 512])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("sym_mem_buffer_rows", [64, 256, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_silu_mul_fw_kernel(
    num_tokens: int, hidden_dim: int, sym_mem_buffer_rows: int, dtype: torch.dtype
):
    """Test silu_mul forward kernel against PyTorch reference."""
    if sym_mem_buffer_rows < num_tokens:
        pytest.skip("sym_mem_buffer_rows must be >= num_tokens")

    device = "cuda"
    saved_activations_buffer_rows = sym_mem_buffer_rows + 100  # Larger buffer

    # Create test inputs
    input_buffer = torch.randn(
        saved_activations_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )
    saved_activation_buffer_offset = torch.tensor(50, dtype=torch.int64, device=device)
    num_tokens_tensor = torch.tensor(num_tokens, dtype=torch.int64, device=device)

    # Get slice for reference computation
    offset_val = saved_activation_buffer_offset.item()
    h13_slice = input_buffer[offset_val : offset_val + sym_mem_buffer_rows]

    # Reference computation
    ref_output = _torch_silu_mul_fw(h13_slice, num_tokens)

    # Zero-pad to sym_mem_buffer_rows
    ref_padded = torch.zeros(
        sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device
    )
    ref_padded[:num_tokens] = ref_output

    # Kernel computation
    kernel_output = silu_mul_fw(
        input_buffer,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        sym_mem_buffer_rows,
    )

    # Verify shapes
    assert kernel_output.shape == (sym_mem_buffer_rows, hidden_dim), (
        "Output shape mismatch"
    )
    assert kernel_output.dtype == dtype, "Output dtype mismatch"

    # Verify numerics (relaxed tolerances for BF16 + float32 intermediate computations)
    torch.testing.assert_close(kernel_output, ref_padded, rtol=1e-2, atol=1e-3)

    # Verify zero-padding beyond num_tokens
    if num_tokens < sym_mem_buffer_rows:
        assert torch.all(kernel_output[num_tokens:] == 0), (
            "Rows beyond num_tokens should be zero"
        )


@pytest.mark.parametrize("num_tokens", [32, 128, 256, 512])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("sym_mem_buffer_rows", [64, 256, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_silu_mul_bw_kernel(
    num_tokens: int, hidden_dim: int, sym_mem_buffer_rows: int, dtype: torch.dtype
):
    """Test silu_mul backward kernel against PyTorch reference."""
    if sym_mem_buffer_rows < num_tokens:
        pytest.skip("sym_mem_buffer_rows must be >= num_tokens")

    device = "cuda"
    saved_activations_buffer_rows = sym_mem_buffer_rows + 100  # Larger buffer

    # Create test inputs
    h13_buffer = torch.randn(
        saved_activations_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )
    grad_h = torch.randn(sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device)
    saved_activation_buffer_offset = torch.tensor(50, dtype=torch.int64, device=device)
    num_tokens_tensor = torch.tensor(num_tokens, dtype=torch.int64, device=device)

    # Pre-allocate output buffers
    h_out = torch.zeros(sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device)
    grad_h13_out = torch.zeros(
        sym_mem_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )

    # Get slice for reference computation
    offset_val = saved_activation_buffer_offset.item()
    h13_slice = h13_buffer[offset_val : offset_val + sym_mem_buffer_rows]

    # Reference computation
    ref_h, ref_grad_h13 = _torch_silu_mul_bw(h13_slice, grad_h, num_tokens)

    # Zero-pad reference outputs to sym_mem_buffer_rows
    ref_h_padded = torch.zeros(
        sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device
    )
    ref_grad_h13_padded = torch.zeros(
        sym_mem_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )
    ref_h_padded[:num_tokens] = ref_h
    ref_grad_h13_padded[:num_tokens] = ref_grad_h13

    # Kernel computation
    silu_mul_bw(
        h13_buffer,
        grad_h,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        h_out,
        grad_h13_out,
    )

    # Verify shapes
    assert h_out.shape == (sym_mem_buffer_rows, hidden_dim), "h_out shape mismatch"
    assert grad_h13_out.shape == (sym_mem_buffer_rows, 2 * hidden_dim), (
        "grad_h13_out shape mismatch"
    )
    assert h_out.dtype == dtype, "h_out dtype mismatch"
    assert grad_h13_out.dtype == dtype, "grad_h13_out dtype mismatch"

    # Verify numerics (relaxed tolerances for BF16 + float32 intermediate computations)
    torch.testing.assert_close(h_out, ref_h_padded, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(grad_h13_out, ref_grad_h13_padded, rtol=1e-2, atol=1e-3)

    # Verify zero-padding beyond num_tokens
    if num_tokens < sym_mem_buffer_rows:
        assert torch.all(h_out[num_tokens:] == 0), (
            "h_out rows beyond num_tokens should be zero"
        )
        assert torch.all(grad_h13_out[num_tokens:] == 0), (
            "grad_h13_out rows beyond num_tokens should be zero"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_mul_kernels_gradient_correctness(dtype: torch.dtype):
    """Test that silu_mul kernels produce correct gradients via autograd."""
    device = "cuda"
    num_tokens = 64
    hidden_dim = 512
    sym_mem_buffer_rows = 128
    saved_activations_buffer_rows = 256

    # Create test inputs with gradient tracking
    input_buffer = torch.randn(
        saved_activations_buffer_rows,
        2 * hidden_dim,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    saved_activation_buffer_offset = torch.tensor(50, dtype=torch.int64, device=device)
    num_tokens_tensor = torch.tensor(num_tokens, dtype=torch.int64, device=device)

    # Forward pass with kernel
    h_fwd = silu_mul_fw(
        input_buffer,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        sym_mem_buffer_rows,
    )

    # Create gradient for backward
    grad_h = torch.randn_like(h_fwd)

    # Pre-allocate backward outputs
    h_out = torch.zeros_like(h_fwd)
    grad_h13_out = torch.zeros(
        sym_mem_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )

    # Backward pass with kernel
    silu_mul_bw(
        input_buffer,
        grad_h,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        h_out,
        grad_h13_out,
    )

    # Verify forward recomputation matches (relaxed tolerances for BF16)
    torch.testing.assert_close(h_fwd, h_out, rtol=1e-2, atol=1e-3)

    # Reference gradient computation using autograd (matching kernel precision)
    input_buffer_ref = input_buffer.clone().detach().requires_grad_(True)
    offset_val = saved_activation_buffer_offset.item()
    h13_slice = input_buffer_ref[offset_val : offset_val + num_tokens]
    h1, h3 = h13_slice.chunk(2, dim=-1)
    # Match kernel precision: keep everything in float32 until final cast
    h1_f32 = h1.to(torch.float32)
    h3_f32 = h3.to(torch.float32)
    silu_h1_f32 = h1_f32 * torch.sigmoid(h1_f32)
    ref_output = (silu_h1_f32 * h3_f32).to(h1.dtype)
    ref_output.backward(grad_h[:num_tokens])

    # Compare gradients
    expected_grad = torch.zeros_like(input_buffer_ref.grad)
    expected_grad[offset_val : offset_val + num_tokens] = input_buffer_ref.grad[
        offset_val : offset_val + num_tokens
    ]

    # Extract kernel gradients at the correct offset
    kernel_grad_slice = grad_h13_out[:num_tokens]

    torch.testing.assert_close(
        kernel_grad_slice,
        input_buffer_ref.grad[offset_val : offset_val + num_tokens],
        rtol=1e-2,
        atol=1e-3,
    )
