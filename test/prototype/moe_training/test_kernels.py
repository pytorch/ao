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
    triton_mxfp8_dispatch_and_quantize,
    triton_mxfp8_pad_and_quantize,
    triton_mxfp8_quantize_dim0_dim1,
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

from .testing_utils import generate_split_sizes


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


# --------------------------------------------------------------------------- #
# Fused MoE dispatch + MXFP8 quantize + blocked-scale rearrange (ao#4184).
# --------------------------------------------------------------------------- #


def _reference_pad_quantize_rearrange(
    x: torch.Tensor,
    group_offsets: torch.Tensor,
    scaling_mode_str: str,
    alignment: int,
):
    """Reference 3-stage pipeline the fused kernel replaces. Uses upper-bound
    sizing (same allocation strategy as ``fused_pad_token_groups_cuda``) to
    match the fused kernel output shape."""
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    padded, padded_start, padded_end = torch_pad_token_groups(
        x, group_offsets, alignment_size=alignment
    )
    qdata_ref, scales_ref = triton_to_mxfp8_dim0(
        padded, inner_block_size=32, scaling_mode=scaling_mode_str
    )
    blocked_ref = triton_mx_block_rearrange_2d_M_groups(
        scales_ref, padded_end.to(torch.int32)
    )
    return qdata_ref, blocked_ref, padded_start, padded_end


def _assert_blocked_scales_equal(
    fused_blocked_flat: torch.Tensor,
    ref_blocked_2d: torch.Tensor,
    padded_M: int,
    k_blocks: int,
) -> None:
    """Blocked scale tensors contain per-group "gap" bytes that our single-pass
    kernel legitimately leaves uninitialized, so compare in the canonical
    unblocked view via from_blocked, which only looks at the non-gap cells."""
    padded_cols = ref_blocked_2d.shape[1]
    # The fused kernel allocates exactly (padded_M * padded_scale_cols,) with
    # no inter-group gaps, while the reference allocates (padded_M +
    # num_groups*128, padded_cols). Since alignment=128 and groups are
    # multiples of 128, the first padded_M rows of ref contain all non-gap
    # bytes. Reshape fused-flat to (padded_M, padded_cols) for comparison.
    fused_view = fused_blocked_flat.view(padded_M, padded_cols)
    ref_view = ref_blocked_2d[:padded_M]
    fused_canonical = from_blocked(
        fused_view.reshape(-1).view(torch.uint8), padded_M, k_blocks
    )
    ref_canonical = from_blocked(
        ref_view.reshape(-1).view(torch.uint8), padded_M, k_blocks
    )
    assert torch.equal(fused_canonical, ref_canonical), (
        "fused blocked scales differ from reference (non-gap cells)"
    )


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="requires CUDA SM 10.x (blocked scale GEMM hw)",
)
@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize(
    "num_tokens,k",
    [
        (256, 128),
        (512, 256),
        (1024, 2048),
        (4096, 2048),
        (4096, 5120),
        (8192, 5120),
        (8192, 7168),
    ],
)
@pytest.mark.parametrize("num_groups", [1, 2, 4, 8])
@pytest.mark.parametrize("scaling_mode_str", ["rceil", "floor"])
def test_triton_mxfp8_pad_and_quantize_numerics(
    num_tokens: int,
    k: int,
    num_groups: int,
    scaling_mode_str: str,
):
    """Fused pad+quantize+blocked-scales == pad_token_groups + triton_to_mxfp8_dim0
    + triton_mx_block_rearrange_2d_M_groups, bit-exactly on non-gap cells."""
    device = "cuda"
    alignment = 128
    torch.manual_seed(42)

    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device=device)
    group_offsets = generate_jagged_offs(
        num_groups, num_tokens, multiple_of=1, device=device
    ).to(torch.int32)

    qdata_ref, blocked_ref, padded_start_ref, padded_end_ref = (
        _reference_pad_quantize_rearrange(x, group_offsets, scaling_mode_str, alignment)
    )

    qdata_fused, blocked_fused, padded_start_fused, padded_end_fused = (
        triton_mxfp8_pad_and_quantize(x, group_offsets, scaling_mode=scaling_mode_str)
    )

    assert torch.equal(padded_start_ref, padded_start_fused), (
        "padded group start offsets differ"
    )
    assert torch.equal(padded_end_ref, padded_end_fused), (
        "padded group end offsets differ"
    )

    # Both the fused kernel and the reference use upper-bound sizing
    # (``num_tokens + num_groups * alignment`` rounded up to alignment), so the
    # full qdata tensors should match bit-exactly, including the trailing
    # zero-padding rows beyond the actual padded end.
    padded_M_ub = qdata_ref.shape[0]
    assert qdata_fused.shape[0] >= padded_M_ub, (
        f"fused qdata rows {qdata_fused.shape[0]} < reference rows {padded_M_ub}"
    )
    assert qdata_fused.shape[1] == k
    # Compare the first padded_M_ub rows of the fused output against the full
    # reference (handles the case where the fused kernel aligns the output up
    # to BLOCK_ROWS beyond the strict upper bound).
    assert torch.equal(
        qdata_fused[:padded_M_ub].view(torch.uint8), qdata_ref.view(torch.uint8)
    ), "fused fp8 data differs from reference"

    _assert_blocked_scales_equal(
        blocked_fused, blocked_ref, padded_M_ub, k // 32
    )


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="requires CUDA SM 10.x (blocked scale GEMM hw)",
)
@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize(
    "unpermuted_M,padded_M,k",
    [
        (512, 768, 128),
        (1024, 1280, 2048),
        (4096, 4352, 2048),
        (4096, 4608, 5120),
        (8192, 8576, 5120),
        (16384, 16896, 7168),
    ],
)
@pytest.mark.parametrize("scaling_mode_str", ["rceil", "floor"])
def test_triton_mxfp8_dispatch_and_quantize_numerics(
    unpermuted_M: int,
    padded_M: int,
    k: int,
    scaling_mode_str: str,
):
    """EP-style arbitrary permutation: build a random src_indices with -1
    sentinels, and check bit-exact match to the explicit reference pipeline
    (vstack zero row -> index -> quantize -> rearrange)."""
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    device = "cuda"
    alignment = 128
    assert padded_M % alignment == 0
    assert padded_M > unpermuted_M
    torch.manual_seed(123)

    x = torch.randn(unpermuted_M, k, dtype=torch.bfloat16, device=device)

    # Build src_indices with -1 sentinels mixed in. Layout groups into ~4
    # sub-blocks of padded_M / 4 rows each; within each sub-block, place
    # a variable number of valid rows followed by padding. This mirrors the
    # shape of `permuted_indices` produced by generate_permute_indices.
    src = torch.full((padded_M,), -1, dtype=torch.int32, device=device)
    num_sub = 4
    sub_rows = padded_M // num_sub
    # Sub-block valid counts must sum to <= unpermuted_M.
    valid_counts = [unpermuted_M // num_sub] * num_sub
    valid_counts[-1] = unpermuted_M - sum(valid_counts[:-1])
    perm = torch.randperm(unpermuted_M, device=device).to(torch.int32)
    cur_src = 0
    for g in range(num_sub):
        start = g * sub_rows
        cnt = valid_counts[g]
        src[start : start + cnt] = perm[cur_src : cur_src + cnt]
        cur_src += cnt

    qdata_fused, blocked_fused = triton_mxfp8_dispatch_and_quantize(
        x, src, scaling_mode=scaling_mode_str
    )

    # Reference: gather with -1 sentinel -> quantize -> rearrange.
    x_plus_zero = torch.vstack((x, x.new_zeros((1, k))))
    safe_src = torch.where(
        src >= 0, src.to(torch.int64), torch.tensor(unpermuted_M, device=device)
    )
    gathered = x_plus_zero[safe_src]
    qdata_ref, scales_ref = triton_to_mxfp8_dim0(
        gathered, inner_block_size=32, scaling_mode=scaling_mode_str
    )

    # For the blocked rearrange reference we need group offsets; since we
    # want to compare the full padded_M we treat it as one big "group" of
    # padded_M rows (alignment=128, so padded_M itself is a valid group end).
    one_group_offsets = torch.tensor([padded_M], dtype=torch.int32, device=device)
    blocked_ref = triton_mx_block_rearrange_2d_M_groups(
        scales_ref, one_group_offsets
    )

    assert torch.equal(
        qdata_fused.view(torch.uint8), qdata_ref.view(torch.uint8)
    ), "fused fp8 data differs from gather+quantize reference"
    _assert_blocked_scales_equal(
        blocked_fused, blocked_ref, padded_M, k // 32
    )


@pytest.mark.skipif(
    not _is_sm_10x(),
    reason="requires CUDA SM 10.x (blocked scale GEMM hw)",
)
@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize(
    "M,N",
    [
        (128, 128),
        (256, 512),
        (1024, 2048),
        (4096, 2048),
        (8192, 2048),
        (16384, 2048),
        (2048, 5120),
    ],
)
@pytest.mark.parametrize("scaling_mode_str", ["rceil", "floor"])
def test_triton_mxfp8_quantize_dim0_dim1_numerics(
    M: int, N: int, scaling_mode_str: str
):
    """Fused dim0+dim1 MXFP8 quantization with blocked scales should be
    bit-exactly equivalent to the decoupled 4-kernel reference pipeline:

        qdata0, scales0_rm = triton_to_mxfp8_dim0(x, 32, mode)
        qdata1_t, scales1_rm = triton_to_mxfp8_dim1(x, 32, mode)
        scales0_blocked = triton_mx_block_rearrange_2d_M_groups(scales0_rm, [M])
        scales1_blocked = triton_mx_block_rearrange_2d_M_groups(scales1_rm, [N])
    """
    from torchao.prototype.mx_formats.kernels import (
        triton_to_mxfp8_dim0,
        triton_to_mxfp8_dim1,
    )

    device = "cuda"
    torch.manual_seed(2024)
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    # Fused kernel under test.
    qdata0_fused, qdata1_t_fused, scales0_fused, scales1_fused = (
        triton_mxfp8_quantize_dim0_dim1(x, scaling_mode=scaling_mode_str)
    )

    # Reference dim0 pipeline: quantize along N, then blocked-rearrange.
    qdata0_ref, scales0_ref_rm = triton_to_mxfp8_dim0(
        x, inner_block_size=32, scaling_mode=scaling_mode_str
    )
    one_group_offsets_m = torch.tensor([M], dtype=torch.int32, device=device)
    scales0_blocked_ref = triton_mx_block_rearrange_2d_M_groups(
        scales0_ref_rm, one_group_offsets_m
    )

    # Reference dim1 pipeline: quantize along M (returns transposed data),
    # then blocked-rearrange.
    qdata1_t_ref, scales1_ref_rm = triton_to_mxfp8_dim1(
        x, inner_block_size=32, scaling_mode=scaling_mode_str
    )
    one_group_offsets_n = torch.tensor([N], dtype=torch.int32, device=device)
    scales1_blocked_ref = triton_mx_block_rearrange_2d_M_groups(
        scales1_ref_rm, one_group_offsets_n
    )

    # qdata_dim0: (M, N) row-major e4m3 - bit-exact parity.
    assert qdata0_fused.shape == (M, N)
    assert qdata0_fused.dtype == torch.float8_e4m3fn
    assert torch.equal(
        qdata0_fused.view(torch.uint8), qdata0_ref.view(torch.uint8)
    ), "fused dim0 fp8 data differs from triton_to_mxfp8_dim0 reference"

    # qdata_dim1_t: (N, M) row-major e4m3 - bit-exact parity vs.
    # transposed dim1 reference. ``triton_to_mxfp8_dim1`` returns its data
    # as a ``.t()`` view of an (N, M) row-major column-major-of-x tensor,
    # so calling ``.t()`` on the reference peels the view back to the raw
    # (N, M) row-major storage we produce.
    assert qdata1_t_fused.shape == (N, M)
    assert qdata1_t_fused.dtype == torch.float8_e4m3fn
    qdata1_t_ref_rowmajor = qdata1_t_ref.t().contiguous()
    assert torch.equal(
        qdata1_t_fused.view(torch.uint8),
        qdata1_t_ref_rowmajor.view(torch.uint8),
    ), "fused dim1-transpose fp8 data differs from triton_to_mxfp8_dim1 reference"

    # Blocked scales for dim0: compare via from_blocked canonical view
    # (128x4 blocks may have gap cells that are legitimately uninitialized).
    _assert_blocked_scales_equal(scales0_fused, scales0_blocked_ref, M, N // 32)
    # Blocked scales for dim1 live in an (N, M/32) logical tensor.
    _assert_blocked_scales_equal(scales1_fused, scales1_blocked_ref, N, M // 32)


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
    device = "cuda"
    M, K = 512, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    num_groups = 4

    # Generate group sizes and force at least one to be invalid
    group_sizes = generate_split_sizes(num_groups, M, device)
    if group_sizes[0] % 128 == 0:
        group_sizes[0] = group_sizes[0] - 1  # Make it not a multiple of 128
        group_sizes[1] = group_sizes[1] + 1  # Compensate to maintain total sum

    invalid_offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)

    # Test should raise RuntimeError due to device assertion failure with specific message
    with pytest.raises(
        RuntimeError,
        match=r"unspecified launch failure",
    ):
        _ = mxfp8_quantize_2d_1x32_cutedsl(
            x, block_size=32, scaling_mode="rceil", offs=invalid_offsets
        )
        # Force synchronization to ensure device error propagates
        torch.cuda.synchronize()


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
    device = "cuda"
    M, K = 512, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    num_groups = 4

    # Generate group sizes and force at least one to be invalid
    group_sizes = generate_split_sizes(num_groups, M, device)
    if group_sizes[0] % 128 == 0:
        group_sizes[0] = group_sizes[0] - 1  # Make it not a multiple of 128
        group_sizes[1] = group_sizes[1] + 1  # Compensate to maintain total sum

    invalid_offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)

    # Test should raise RuntimeError due to device assertion failure with specific message
    with pytest.raises(RuntimeError, match=r"unspecified launch failure"):
        _ = mxfp8_quantize_2d_32x1_cutedsl(
            x, block_size=32, scaling_mode="rceil", offs=invalid_offsets
        )
        # Force synchronization to ensure device error propagates
        torch.cuda.synchronize()


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
    valid_offsets = torch.cumsum(
        torch.tensor(valid_group_sizes, dtype=torch.int32), dim=0
    ).to(device)

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
