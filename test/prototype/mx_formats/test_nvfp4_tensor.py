# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025, NVIDIA CORPORATION.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchao.prototype.mx_formats.constants import (
    F4_E2M1_MAX,
)
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4MMConfig,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
    unpack_uint4,
)
from torchao.prototype.mx_formats.utils import ceil_div
from torchao.quantization.utils import compute_error
from torchao.testing.utils import skip_if_rocm
from torchao.utils import (
    is_sm_at_least_100,
    torch_version_at_least,
)

torch.manual_seed(2)

if not torch_version_at_least("2.8.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


@pytest.mark.parametrize(
    "dtype,shape,use_per_tensor_scale",
    [
        (torch.bfloat16, (32, 64), False),
        (torch.float32, (64, 128), False),
        (torch.bfloat16, (128, 256), False),
        (torch.bfloat16, (64, 128), True),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="torch.compile requires PyTorch 2.8+"
)
def test_nvfp4_reconstruction(dtype, shape, use_per_tensor_scale):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    if use_per_tensor_scale:
        tensor_amax = torch.max(torch.abs(x))
        scale = per_tensor_amax_to_scale(tensor_amax)
    else:
        scale = None

    x_nvfp4 = NVFP4Tensor.to_nvfp4(x, per_tensor_scale=scale)
    x_reconstructed = x_nvfp4.to_dtype(dtype)

    def assert_sqnr_gt_threshold(orig, new, threshold):
        sqnr = compute_error(orig, new)
        if torch.all(torch.isnan(sqnr)):
            # if both operands are full of zeroes, sqnr is nan and this is ok
            # test for this explicitly
            assert torch.all(orig == 0) and torch.all(new == 0)
        else:
            assert sqnr >= threshold

    reconstructed_amax = x_nvfp4.get_hp_scales().view(shape[0], -1, 1) * F4_E2M1_MAX
    max_abs = torch.amax(
        torch.abs(x.reshape(shape[0], -1, x_nvfp4._block_size)), dim=-1
    ).unsqueeze(-1)

    assert_sqnr_gt_threshold(max_abs, reconstructed_amax, 30.0)
    assert_sqnr_gt_threshold(x, x_reconstructed, 8.0)

    assert x.shape == x_reconstructed.shape, (
        f"Shape mismatch: {x.shape} vs {x_reconstructed.shape}"
    )
    assert x.dtype == x_reconstructed.dtype, (
        f"Dtype mismatch: {x.dtype} vs {x_reconstructed.dtype}"
    )

    x_nvfp4_t = x_nvfp4.t()
    x_reconstructed_t = x_nvfp4_t.to_dtype(dtype)
    assert_sqnr_gt_threshold(x.t(), x_reconstructed_t, 8.0)

    assert x.t().shape == x_reconstructed_t.shape, (
        f"Transpose shape mismatch: {x.t().shape} vs {x_reconstructed_t.shape}"
    )
    assert x.t().dtype == x_reconstructed_t.dtype, (
        f"Transpose dtype mismatch: {x.t().dtype} vs {x_reconstructed_t.dtype}"
    )


@pytest.mark.parametrize("is_swizzled_scales", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 64),
        (16, 32),
        (64, 128),
        (384, 128),
    ],
)
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nvfp4_swizzled_scales_construction(is_swizzled_scales, shape):
    """
    Test that NVFP4Tensor can be constructed with swizzled scales and
    that the _is_swizzled_scales flag is set correctly.
    """

    M, K = shape
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=is_swizzled_scales)
    assert tensor._is_swizzled_scales == is_swizzled_scales
    reconstructed = tensor.to_dtype(torch.bfloat16)
    assert reconstructed.shape == data.shape


@pytest.mark.parametrize(
    "slice_dim,slice_spec",
    [
        # Row slicing - must align with 128-row boundaries
        # pytest.param(0, slice(0, 128), id="slice_rows[0:128]"),
        # pytest.param(0, slice(128, 256), id="slice_rows[128:256]"),
        # Column slicing - must align with 64-column boundaries (4 scale columns * 16 block_size)
        pytest.param(1, slice(0, 64), id="slice_cols[0:64]"),
        pytest.param(1, slice(64, 128), id="slice_cols[64:128]"),
        pytest.param(1, slice(0, 128), id="slice_cols[0:128]_full_width"),
        # Test tensor parallelism patterns (half splits)
        pytest.param(1, slice(0, 2048), id="slice_cols[0:2048]_tp_first_half"),
        pytest.param(1, slice(2048, 4096), id="slice_cols[2048:4096]_tp_second_half"),
        # Test quarter splits
        pytest.param(1, slice(0, 1024), id="slice_cols[0:1024]_quarter"),
        pytest.param(1, slice(1024, 2048), id="slice_cols[1024:2048]_quarter"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_slicing(slice_dim, slice_spec):
    """
    Test that slicing works correctly with swizzled scales and maintains
    the swizzled state in the output tensor.
    """

    # Use larger tensor sizes that align with swizzled requirements
    if slice_dim == 0:
        # For row slicing, need at least 256 rows to test 128-row boundaries
        M, K = 256, 4096
    else:
        # For column slicing, need multiples of 64 columns for alignment
        M, K = 128, 4096

    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)
    # tensor.to_dtype(torch.bfloat16)
    assert tensor._is_swizzled_scales == True

    print(
        "before",
        tensor.shape,
        tensor.qdata.shape,
        tensor._scale_e4m3.shape,
        tensor._scale_e4m3.is_contiguous(),
    )
    print(tensor._scale_e4m3[0:128, 0:8])
    if slice_dim == 0:
        sliced_tensor = tensor[slice_spec, :]
    else:
        sliced_tensor = tensor[:, slice_spec]
    print(
        "after",
        sliced_tensor.shape,
        sliced_tensor.qdata.shape,
        sliced_tensor._scale_e4m3.shape,
        tensor._scale_e4m3.is_contiguous(),
    )
    print(tensor._scale_e4m3[0:128, 0:8])

    # Verify sliced tensor maintains swizzled state
    assert sliced_tensor._is_swizzled_scales == True

    # Verify sliced tensor can be dequantized
    sliced_reconstructed = sliced_tensor.to_dtype(torch.bfloat16)

    # Compare with direct slicing of original data
    original_reconstructed = tensor.to_dtype(torch.bfloat16)
    if slice_dim == 0:
        expected = original_reconstructed[slice_spec, :]
    else:
        expected = original_reconstructed[:, slice_spec]

    torch.testing.assert_close(sliced_reconstructed, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "slice_dim,slice_spec,expected_error",
    [
        # Row slicing with misaligned boundaries
        pytest.param(
            0,
            slice(0, 100),
            "Row slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_row_end",
        ),
        pytest.param(
            0,
            slice(50, 150),
            "Row slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_row_start",
        ),
        # Column slicing with misaligned boundaries
        pytest.param(
            1,
            slice(0, 32),
            "Column slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_col_32",
        ),
        pytest.param(
            1,
            slice(16, 80),
            "Column slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_col_start",
        ),
        pytest.param(
            1,
            slice(0, 100),
            "Column slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_col_end",
        ),
        # Odd column boundaries (FP4 packing requirement)
        pytest.param(
            1,
            slice(1, 65),
            "start index to be a multiple of 64, got 1",
            id="odd_start",
        ),
        pytest.param(
            1,
            slice(0, 65),
            " multiple of 64 or equal to tensor size 4096, got 65",
            id="odd_end",
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_slicing_errors(slice_dim, slice_spec, expected_error):
    """
    Test that slicing raises appropriate errors for misaligned boundaries.
    """

    M, K = 256, 4096
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    with pytest.raises(RuntimeError, match=expected_error):
        if slice_dim == 0:
            _ = tensor[slice_spec, :]
        else:
            _ = tensor[:, slice_spec]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_view_semantics():
    """
    Test that slicing maintains proper view semantics where possible.
    """

    M, K = 256, 4096
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    # Test row slicing (should maintain views)
    sliced_tensor = tensor[0:128, :]

    # Test that the sliced tensor shares storage with original for data
    # (Note: scales might not share storage due to swizzled layout complexity)
    assert sliced_tensor.qdata.data_ptr() == tensor.qdata.data_ptr()

    # Test full-width column slicing (should maintain views)
    full_width_slice = tensor[:, 0:K]
    assert full_width_slice._scale_e4m3.data_ptr() == tensor._scale_e4m3.data_ptr()
    assert full_width_slice.qdata.data_ptr() == tensor.qdata.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_serialization():
    """
    Test that tensor flatten/unflatten preserves the swizzled scales state.
    """

    M, K = 32, 64
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Create tensor with swizzled scales
    original_tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    # Test serialization
    tensor_list, ctx = original_tensor.__tensor_flatten__()

    # Verify swizzled flag is preserved in context
    assert "_is_swizzled_scales" in ctx
    assert ctx["_is_swizzled_scales"] == True

    # Test deserialization
    inner_tensors = {}
    for name in tensor_list:
        inner_tensors[name] = getattr(original_tensor, name)

    reconstructed_tensor = NVFP4Tensor.__tensor_unflatten__(
        inner_tensors, ctx, None, None
    )

    # Verify the swizzled state is preserved
    assert reconstructed_tensor._is_swizzled_scales == True

    # Verify functionality is preserved
    original_dq = original_tensor.to_dtype(torch.bfloat16)
    reconstructed_dq = reconstructed_tensor.to_dtype(torch.bfloat16)

    torch.testing.assert_close(original_dq, reconstructed_dq, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_get_scales_method():
    """
    Test that the get_scales() method correctly unswizzles scales when needed.
    """

    M, K = 32, 64
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Create tensors with both storage methods
    regular_tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=False)
    swizzled_tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    # Get scales from both tensors and verify they are equal
    regular_scales = regular_tensor.get_hp_scales()
    swizzled_scales = swizzled_tensor.get_hp_scales()
    torch.testing.assert_close(regular_scales, swizzled_scales, atol=0.0, rtol=0.0)

    # Verify scales have the expected shape
    expected_shape = (M, K // 16)
    assert regular_scales.shape == expected_shape
    assert swizzled_scales.shape == expected_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "M", [128, 256, 512, 1024, 100, 200, 384], ids=lambda m: f"M{m}"
)
@pytest.mark.parametrize("N", [64, 128, 256, 512, 32, 96, 160], ids=lambda n: f"N{n}")
@pytest.mark.parametrize(
    "use_per_tensor_scale", [False, True], ids=["block_scale", "tensor_scale"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="requires sm100+ for raw intrinsics"
)
@torch.no_grad()
def test_triton_nvfp4_quantize_equivalence(M, N, use_per_tensor_scale, dtype):
    """Test that Triton and PyTorch NVFP4 quantization produce equivalent results."""

    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=dtype, device="cuda")

    per_tensor_scale = None
    if use_per_tensor_scale:
        per_tensor_scale = per_tensor_amax_to_scale(torch.amax(torch.abs(x)))

    nvfp4_pt = NVFP4Tensor.to_nvfp4(
        x.clone(),
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )

    nvfp4_triton = NVFP4Tensor.to_nvfp4(
        x.clone(),
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=True,
    )

    torch.testing.assert_close(
        nvfp4_pt._scale_e4m3.flatten(), nvfp4_triton._scale_e4m3.flatten()
    )
    pt_unpacked = unpack_uint4(nvfp4_pt.qdata)
    triton_unpacked = unpack_uint4(nvfp4_triton.qdata)
    torch.testing.assert_close(
        pt_unpacked,
        triton_unpacked,
        atol=0,
        rtol=0,
    )

    x_pt_dequant = nvfp4_pt.to_dtype(dtype)
    x_triton_dequant = nvfp4_triton.to_dtype(dtype)

    sqnr = compute_error(x_pt_dequant, x_triton_dequant)
    SQNR_THRESHOLD = 40.0

    assert sqnr >= SQNR_THRESHOLD, (
        f"SQNR {sqnr:.2f} < {SQNR_THRESHOLD} for M={M}, N={N}, "
        f"use_per_tensor_scale={use_per_tensor_scale}, dtype={dtype}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.parametrize("use_gelu", [True, False])
@pytest.mark.parametrize(
    "mm_config", [NVFP4MMConfig.DYNAMIC, NVFP4MMConfig.WEIGHT_ONLY]
)
@pytest.mark.parametrize("compile", [False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("inpt_dtype", [torch.bfloat16, torch.float32])
# @pytest.mark.parametrize("use_triton_kernel", [True, False])
@pytest.mark.parametrize("use_triton_kernel", [False])
@pytest.mark.parametrize(
    "shapes",
    [
        (128, 64, 256),
        (256, 128, 512),
        (157, 64, 256),
        (128, 96, 256),
        (128, 160, 256),
        (64, 64, 256),
        (200, 192, 256),
    ],
    ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}",
)
@torch.no_grad()
@skip_if_rocm("ROCm float4 gemm require gfx950")
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="CUDA capability >= 10.0 required for fp4"
)
def test_nvfp4_matmul_with_amax(
    use_gelu: bool,
    mm_config: NVFP4MMConfig,
    compile: bool,
    bias: bool,
    inpt_dtype: torch.dtype,
    use_triton_kernel: bool,
    shapes: tuple,
):
    # DYNAMIC mode requires SM100+, but WEIGHT_ONLY works on older GPUs
    if mm_config == NVFP4MMConfig.DYNAMIC and not is_sm_at_least_100():
        pytest.skip("CUDA capability >= 10.0 required for DYNAMIC float4 gemm")

    if bias and inpt_dtype == torch.float32:
        pytest.xfail("Bias is not supported when module weight is in fp32")

    if mm_config == NVFP4MMConfig.WEIGHT_ONLY and compile:
        pytest.skip("TODO: NVFP4MMConfig.WEIGHT_ONLY currently errors w/ compile")

    m, k, n = shapes

    # Create activation tensor
    if use_gelu:
        x = torch.randn(m, k, dtype=inpt_dtype, device="cuda")
        A = torch.nn.functional.gelu(x)
    else:
        A = torch.randn(m, k, dtype=inpt_dtype, device="cuda")

    B = torch.randn(n, k, dtype=inpt_dtype, device="cuda")
    bias_tensor = torch.randn(n, dtype=inpt_dtype, device="cuda") if bias else None

    # Compute reference
    C_ref = F.linear(A, B, bias_tensor)

    a_scale = per_tensor_amax_to_scale(torch.amax(torch.abs(A)))
    b_scale = per_tensor_amax_to_scale(torch.amax(torch.abs(B)))
    act_quant_kwargs = None
    if mm_config == NVFP4MMConfig.DYNAMIC:
        act_quant_kwargs = QuantizeTensorToNVFP4Kwargs()
    A_nvfp4 = NVFP4Tensor.to_nvfp4(
        A,
        per_tensor_scale=a_scale,
        is_swizzled_scales=True,
        use_triton_kernel=use_triton_kernel,
    )
    B_nvfp4 = NVFP4Tensor.to_nvfp4(
        B,
        per_tensor_scale=b_scale,
        is_swizzled_scales=True,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=act_quant_kwargs,
    )

    func = torch.compile(F.linear, fullgraph=True) if compile else F.linear

    C_nvfp4 = func(A_nvfp4, B_nvfp4, bias_tensor)
    assert C_nvfp4.dtype == inpt_dtype, (
        f"Got {C_nvfp4.dtype} for inpt_dtype={inpt_dtype}"
    )

    sqnr = compute_error(C_ref, C_nvfp4)
    SQNR_THRESHOLD = 16.0
    assert sqnr >= SQNR_THRESHOLD, (
        f"SQNR {sqnr:.2f} < {SQNR_THRESHOLD}, use_gelu={use_gelu}, mm_config={mm_config}, compile={compile}, bias={bias}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_to_copy():
    x = NVFP4Tensor.to_nvfp4(torch.randn((32, 128))).cuda()
    y = torch.ops.aten._to_copy(x, dtype=torch.bfloat16)
    assert torch.equal(x.qdata, y.qdata)
    assert torch.equal(x._scale_e4m3, y._scale_e4m3)
    assert x._per_tensor_scale is None
    assert y._per_tensor_scale is None
    assert x._act_per_tensor_scale is None
    assert y._act_per_tensor_scale is None
    assert x._block_size == y._block_size
    assert x.use_triton_kernel == y.use_triton_kernel
    assert x.act_quant_kwargs == y.act_quant_kwargs
    assert x.dtype == torch.float32
    assert y.dtype == torch.bfloat16


@pytest.mark.parametrize("transpose", [False, True])
# @pytest.mark.parametrize("transpose", [True])
# @pytest.mark.parametrize("transpose", [False])
@pytest.mark.parametrize("use_triton_kernel", [False, True])
# @pytest.mark.parametrize("use_triton_kernel", [False])
# @pytest.mark.parametrize("use_triton_kernel", [True])
@pytest.mark.parametrize("is_swizzled_scales", [False, True])
# @pytest.mark.parametrize("is_swizzled_scales", [True])
@pytest.mark.parametrize(
    "mk",
    (
        (128, 64),
        (128 + 16, 64),
        (128, 64 + 16),
        (128 + 16, 64 + 16),
    ),
)
# @pytest.mark.parametrize("mk", ((128 + 16, 64),))
def test_scale_shape_matches_qdata(
    transpose, use_triton_kernel, is_swizzled_scales, mk
):
    if use_triton_kernel and not is_swizzled_scales:
        pytest.skip("triton kernel requires swizzled scales")

    M, K = mk

    block_size = 16

    # TODO(this PR): test larger tensors that don't exactly map to (128, 64) tiles,
    # to test the padding logic
    # context: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    x_hp = torch.randn(M, K, device="cuda")
    x = NVFP4Tensor.to_nvfp4(
        x_hp, is_swizzled_scales=is_swizzled_scales, use_triton_kernel=use_triton_kernel
    )

    m_dim, k_dim = 0, 1
    if transpose:
        x_hp = x_hp.t()
        x = x.t()
        m_dim, k_dim = 1, 0

    orig_m = x_hp.shape[m_dim]
    expected_padded_m = orig_m
    if is_swizzled_scales:
        expected_padded_m = ceil_div(orig_m, 128) * 128
    actual_padded_m = x._scale_e4m3.shape[m_dim]
    assert expected_padded_m == actual_padded_m, (
        f"incompatible padded shape for dim {m_dim}: {x.shape} and {x._scale_e4m3.shape}"
    )

    orig_k = x_hp.shape[k_dim]
    expected_padded_k = orig_k // block_size
    if is_swizzled_scales:
        expected_padded_k = ceil_div(orig_k // block_size, 4) * 4
    actual_padded_k = x._scale_e4m3.shape[k_dim]

    assert expected_padded_k == actual_padded_k, (
        f"incompatible padded shape for dim {k_dim}: {x.shape} and {x._scale_e4m3.shape}"
    )
