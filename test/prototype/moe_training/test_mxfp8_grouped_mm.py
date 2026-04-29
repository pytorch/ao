# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn import functional as F

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.utils import (
    is_MI300,
    is_MI350,
    is_sm_at_least_90,
    is_sm_version,
    torch_version_at_least,
)

if not (
    torch_version_at_least("2.7.0")
    and torch.cuda.is_available()
    and (is_sm_at_least_90() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mx_block_rearrange_2d_M_groups_cuda,
    mxfp8_quantize_cuda_3d,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _emulated_mxfp8_scaled_grouped_mm_2d_2d,
    _emulated_mxfp8_scaled_grouped_mm_2d_3d,
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import (
    _to_mxfp8_per_group_colwise,
    _to_mxfp8_per_group_rowwise,
    generate_jagged_offs,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_mx
from torchao.quantization.quantize_.common import KernelPreference
from torchao.testing.utils import skip_if_rocm

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(
    not is_sm_version(10, 0),
    reason="3D MXFP8 quantization requires SM100",
)
@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8, 16))
@pytest.mark.parametrize("scale_block_k", (1, 32), ids=("32x1_n", "32x32_n"))
@pytest.mark.parametrize(
    "scale_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_emulate_mxfp8_grouped_gemm_2d_3d(
    M, K, N, num_experts, scale_block_k, scale_mode
):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)
    offs_ref = offs.clone()

    # Quantize inputs to mxpf8 for emulated mxfp8 scaled grouped mm
    block_size = 32
    x_scale, x_fp8 = to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=scale_mode,
    )
    x_ref = (
        x_fp8.reshape(M, K // block_size, block_size).to(torch.bfloat16)
        * x_scale.unsqueeze(-1).to(torch.bfloat16)
    ).reshape(M, K)

    if scale_block_k == 1:
        w_scale, w_fp8 = to_mx(
            w,
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scale_mode,
        )
    else:
        w_fp8, w_scale = mxfp8_quantize_cuda_3d(
            w,
            block_size=block_size,
            scale_block_n=block_size,
            scale_block_k=scale_block_k,
            scaling_mode=scale_mode.value.lower(),
            blocked_scale_output=False,
        )
        w_scale = w_scale.repeat_interleave(block_size, dim=1)
    w_ref = (
        w_fp8.reshape(num_experts, N, K // block_size, block_size).to(torch.bfloat16)
        * w_scale.unsqueeze(-1).to(torch.bfloat16)
    ).reshape(num_experts, N, K)

    ref_out = torch._grouped_mm(
        x_ref, w_ref.transpose(-2, -1), offs=offs_ref, out_dtype=torch.bfloat16
    )
    out = _emulated_mxfp8_scaled_grouped_mm_2d_3d(
        x_fp8, x_scale, w_fp8, w_scale, offs=offs, out_dtype=torch.bfloat16
    )

    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"


@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(
    not is_sm_version(10, 0),
    reason="3D MXFP8 quantization and MXFP8 grouped GEMM require SM100",
)
@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8, 16))
@pytest.mark.parametrize("scale_block_k", (1, 32), ids=("32x1_n", "32x32_n"))
@pytest.mark.parametrize(
    "scale_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_mxfp8_grouped_gemm_2d_3d(M, K, N, num_experts, scale_block_k, scale_mode):
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)
    offs_ref = offs.clone()

    # Real SM100 grouped MM: 1x32-scaled A @ 3D-quantized B -> grad_input.
    block_size = 32
    grad_out_fp8, grad_out_scale = triton_to_mxfp8_dim0(
        grad_out,
        inner_block_size=block_size,
        scaling_mode=scale_mode.value.lower(),
    )
    w_fp8, w_scale = mxfp8_quantize_cuda_3d(
        w,
        block_size=block_size,
        scale_block_n=block_size,
        scale_block_k=scale_block_k,
        scaling_mode=scale_mode.value.lower(),
        blocked_scale_output=True,
    )
    w_fp8_ref, w_scale_ref = mxfp8_quantize_cuda_3d(
        w,
        block_size=block_size,
        scale_block_n=block_size,
        scale_block_k=scale_block_k,
        scaling_mode=scale_mode.value.lower(),
        blocked_scale_output=False,
    )

    grad_out_scale_ref = grad_out_scale.repeat_interleave(block_size, dim=1)
    grad_out_ref = grad_out_fp8.to(torch.bfloat16) * grad_out_scale_ref.to(
        torch.bfloat16
    )
    if scale_block_k == 1:
        w_scale_ref = w_scale_ref.repeat_interleave(block_size, dim=1)
    else:
        w_scale_ref = w_scale_ref.repeat_interleave(
            block_size, dim=1
        ).repeat_interleave(block_size, dim=2)
    w_ref = w_fp8_ref.to(torch.bfloat16) * w_scale_ref.to(torch.bfloat16)
    ref_out = torch._grouped_mm(
        grad_out_ref, w_ref, offs=offs_ref, out_dtype=torch.bfloat16
    )
    A_scales_blocked = mx_block_rearrange_2d_M_groups_cuda(grad_out_scale, offs)
    out = torch._scaled_grouped_mm(
        grad_out_fp8,
        w_fp8,
        A_scales_blocked,
        w_scale,
        offs=offs,
        out_dtype=torch.bfloat16,
    )

    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M", (1024, 4096))
@pytest.mark.parametrize("N", (1024, 4096))
@pytest.mark.parametrize("num_experts", (8, 16))
def test_emulate_mxfp8_grouped_gemm_2d_2d(M, N, num_experts):
    # Simluate 2d-2d grouped gemm grad_weight = grad_output_t @ x
    block_size = 32
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    grad_out_t = grad_out.t().contiguous()
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)
    x_ref, grad_out_t_ref, offs_ref = x.clone(), grad_out_t.clone(), offs.clone()

    # bf16 reference grouped gemm
    ref_out = torch._grouped_mm(
        grad_out_t_ref,
        x_ref,
        offs=offs_ref,
        out_dtype=torch.bfloat16,
    )

    # mxpf8 grouped gemm
    x_scale, x_mx = to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=block_size)
    grad_out_t_mx, grad_out_t_scale = _to_mxfp8_per_group_rowwise(
        grad_out_t,
        offs=offs,
        block_size=block_size,
    )
    x_mx, x_scale = _to_mxfp8_per_group_colwise(
        x,
        offs=offs,
        block_size=block_size,
    )
    out = _emulated_mxfp8_scaled_grouped_mm_2d_2d(
        grad_out_t_mx,
        grad_out_t_scale,
        x_mx.transpose(-2, -1),  # (K, N) -> (N, K)
        x_scale.transpose(-2, -1),  # (K//block_size, N) -> (N, K//block_size)
        offs=offs,
        out_dtype=torch.bfloat16,
        block_size=block_size,
    )

    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M,K,N", [(32768, 5120, 8192), (16640, 7168, 2048)])
@pytest.mark.parametrize("num_experts", (1, 8))
@pytest.mark.parametrize("wgrad_with_hp", (True, False))
@pytest.mark.parametrize("use_compile", (False, True))
@pytest.mark.parametrize(
    "kernel_preference", (KernelPreference.AUTO, KernelPreference.EMULATED)
)
@pytest.mark.parametrize(
    "scale_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_mxfp8_grouped_gemm_with_dq_fwd_bwd(
    M,
    K,
    N,
    num_experts,
    wgrad_with_hp,
    use_compile,
    kernel_preference,
    scale_mode,
):
    # MXFP8 hardware path requires SM100
    if kernel_preference != KernelPreference.EMULATED and not is_sm_version(10, 0):
        pytest.skip(
            f"Skipping MXFP8 hardware mode tests, only supported on compute capability 10.0 and found {torch.cuda.get_device_capability()}"
        )
    if kernel_preference == KernelPreference.EMULATED and use_compile:
        pytest.skip(
            "torch native dynamic per group pad/unpad functions do not work with torch.compile yet: https://github.com/pytorch/pytorch/issues/176770"
        )

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1).requires_grad_(True)

    offs = generate_jagged_offs(num_experts, M, multiple_of=128)
    x_ref, w_t_ref, offs_ref = (
        x.clone().detach().requires_grad_(True),
        w_t.clone().detach().requires_grad_(True),
        offs.clone(),
    )

    # Forward
    mxfp8_gmm = (
        torch.compile(_to_mxfp8_then_scaled_grouped_mm, fullgraph=True)
        if use_compile
        else _to_mxfp8_then_scaled_grouped_mm
    )
    out = mxfp8_gmm(
        x,
        w_t,
        offs=offs,
        kernel_preference=kernel_preference,
        wgrad_with_hp=wgrad_with_hp,
        scale_calculation_mode=scale_mode,
        pad_token_groups_for_grouped_mm=False,
    )
    ref_out = torch._grouped_mm(x_ref, w_t_ref, offs=offs_ref, out_dtype=torch.bfloat16)
    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"Output sqnr {sqnr} is too low, must be >= {min_sqnr}"

    # Backward
    labels = torch.ones_like(ref_out)
    ref_loss = F.mse_loss(ref_out, labels)
    out_loss = F.mse_loss(out, labels)
    ref_loss.backward()
    out_loss.backward()

    # Check input grads
    min_input_grad_sqnr = 25.0
    sqnr = compute_error(x_ref.grad, x.grad)
    assert sqnr >= min_input_grad_sqnr, (
        f"Input grad sqnr {sqnr} is too low, must be >= {min_input_grad_sqnr}"
    )

    # Check weight grads
    min_weight_grad_sqnr = 24.0
    sqnr = compute_error(w_t_ref.grad, w_t.grad)
    assert sqnr >= min_weight_grad_sqnr, (
        f"Weight grad sqnr {sqnr} is too low, must be >= {min_weight_grad_sqnr}"
    )


@skip_if_rocm("ROCm not supported")
def test_mxfp8_grouped_gemm_from_qdata_and_scales_matches_dynamic():
    block_size = 32
    M, K, N, num_experts = 4096, 1024, 2048, 8
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1).requires_grad_(True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=128)

    x_ref = x.detach().clone().requires_grad_(True)
    w_t_ref = w_t.detach().clone().requires_grad_(True)

    x_scale, x_qdata = to_mx(
        x.detach(),
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )
    x_mx = MXTensor.from_qdata_and_scales(
        x_qdata,
        x_scale,
        orig_dtype=x.dtype,
        block_size=block_size,
        is_swizzled_scales=False,
    )
    out = _to_mxfp8_then_scaled_grouped_mm(
        x_mx,
        w_t,
        offs=offs,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    out_ref = _to_mxfp8_then_scaled_grouped_mm(
        x_ref,
        w_t_ref,
        offs=offs,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )

    output_sqnr = compute_error(out_ref, out)
    min_output_sqnr = 60.0
    assert output_sqnr >= min_output_sqnr, (
        f"Output sqnr {output_sqnr} is too low, must be >= {min_output_sqnr}"
    )

    labels = torch.ones_like(out_ref)
    F.mse_loss(out_ref, labels).backward()
    F.mse_loss(out, labels).backward()

    assert x.grad is None, (
        "MXTensor inputs are not connected back to the source HP tensor"
    )

    weight_grad_sqnr = compute_error(w_t_ref.grad, w_t.grad)
    # MXTensor inputs dequantize for the `wgrad_with_hp` path, so the weight
    # gradient is expected to be close to, but not identical to, the HP path.
    min_weight_grad_sqnr = 30.0
    assert weight_grad_sqnr >= min_weight_grad_sqnr, (
        f"Weight grad sqnr {weight_grad_sqnr} is too low, must be >= {min_weight_grad_sqnr}"
    )


@skip_if_rocm("ROCm not supported")
def test_mxfp8_grouped_gemm_from_qdata_and_scales_forward():
    block_size = 32
    M, K, N, num_experts = 4096, 1024, 2048, 8
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1)
    offs = generate_jagged_offs(num_experts, M, multiple_of=128)

    x_scale, x_qdata = to_mx(
        x.detach(),
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )
    x_mx = MXTensor.from_qdata_and_scales(
        x_qdata,
        x_scale,
        orig_dtype=x.dtype,
        block_size=block_size,
        is_swizzled_scales=False,
    )
    out_mx = _to_mxfp8_then_scaled_grouped_mm(
        x_mx,
        w_t,
        offs=offs,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    out_ref = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )

    output_sqnr = compute_error(out_ref, out_mx)
    min_output_sqnr = 60.0
    assert output_sqnr >= min_output_sqnr, (
        f"Output sqnr {output_sqnr} is too low, must be >= {min_output_sqnr}"
    )


@skip_if_rocm("ROCm not supported")
def test_mxfp8_grouped_gemm_mxtensor_requires_wgrad_with_hp():
    block_size = 32
    M, K, N, num_experts = 1024, 1024, 2048, 4
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1)
    offs = generate_jagged_offs(num_experts, M, multiple_of=128)

    x_scale, x_qdata = to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )
    x_mx = MXTensor.from_qdata_and_scales(
        x_qdata,
        x_scale,
        orig_dtype=x.dtype,
        block_size=block_size,
        is_swizzled_scales=False,
    )

    with pytest.raises(AssertionError, match="wgrad_with_hp"):
        _to_mxfp8_then_scaled_grouped_mm(
            x_mx,
            w_t,
            offs=offs,
            out_dtype=torch.bfloat16,
            kernel_preference=KernelPreference.EMULATED,
            wgrad_with_hp=False,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
        )


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize(
    "kernel_preference", (KernelPreference.AUTO, KernelPreference.EMULATED)
)
@pytest.mark.parametrize("M,K,N", [(4096, 1024, 2048)])
@pytest.mark.parametrize("num_experts", (1, 8))
def test_mxfp8_grouped_gemm_bias_forward(kernel_preference, M, K, N, num_experts):
    """Test that bias is correctly added to the forward output."""
    if kernel_preference != KernelPreference.EMULATED and not is_sm_version(10, 0):
        pytest.skip("MXFP8 hardware mode tests require SM100")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    w_t = w.transpose(-2, -1)
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M, multiple_of=128)

    # Output without bias
    out_no_bias = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        bias=None,
        out_dtype=torch.bfloat16,
        kernel_preference=kernel_preference,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
        pad_token_groups_for_grouped_mm=False,
    )

    # Output with bias
    out_with_bias = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        bias=bias,
        out_dtype=torch.bfloat16,
        kernel_preference=kernel_preference,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
        pad_token_groups_for_grouped_mm=False,
    )

    # The difference should be exactly the bias (broadcast along dim0)
    diff = out_with_bias - out_no_bias
    expected_bias = bias.unsqueeze(0).expand_as(diff)
    torch.testing.assert_close(diff, expected_bias, rtol=0, atol=0)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize(
    "kernel_preference", (KernelPreference.AUTO, KernelPreference.EMULATED)
)
@pytest.mark.parametrize("M,K,N", [(4096, 1024, 2048)])
@pytest.mark.parametrize("num_experts", (1, 8))
def test_mxfp8_grouped_gemm_bias_backward(kernel_preference, M, K, N, num_experts):
    """Test that bias gradient is correctly computed via autograd."""
    if kernel_preference != KernelPreference.EMULATED and not is_sm_version(10, 0):
        pytest.skip("MXFP8 hardware mode tests require SM100")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    w_t = w.transpose(-2, -1).requires_grad_(True)
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=128)

    # Reference: bf16 grouped mm + bias
    x_ref = x.clone().detach().requires_grad_(True)
    w_t_ref = w_t.clone().detach().requires_grad_(True)
    bias_ref = bias.clone().detach().requires_grad_(True)
    ref_out = torch._grouped_mm(x_ref, w_t_ref, offs=offs, out_dtype=torch.bfloat16)
    ref_out = ref_out + bias_ref

    # MXFP8 with bias
    out = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        bias=bias,
        out_dtype=torch.bfloat16,
        kernel_preference=kernel_preference,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
        pad_token_groups_for_grouped_mm=False,
    )

    # Backward
    labels = torch.ones_like(ref_out)
    F.mse_loss(ref_out, labels).backward()
    F.mse_loss(out, labels).backward()

    # Bias gradient should exist and be close to reference
    assert bias.grad is not None, "bias.grad should be computed"
    assert bias_ref.grad is not None, "bias_ref.grad should be computed"
    sqnr = compute_error(bias_ref.grad, bias.grad)
    min_sqnr = 25.0
    assert sqnr >= min_sqnr, f"Bias grad sqnr {sqnr} is too low, must be >= {min_sqnr}"

    # Input and weight grads should still be correct
    assert x.grad is not None, "x.grad should be computed"
    sqnr_input = compute_error(x_ref.grad, x.grad)
    assert sqnr_input >= 25.0, f"Input grad sqnr {sqnr_input} is too low"

    assert w_t.grad is not None, "w_t.grad should be computed"
    sqnr_weight = compute_error(w_t_ref.grad, w_t.grad)
    assert sqnr_weight >= 24.0, f"Weight grad sqnr {sqnr_weight} is too low"


@skip_if_rocm("ROCm not supported")
def test_mxfp8_grouped_gemm_bias_none_unchanged():
    """Test that passing bias=None produces the same result as not passing bias."""
    M, K, N, num_experts = 4096, 1024, 2048, 8
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    w_t = w.transpose(-2, -1)
    offs = generate_jagged_offs(num_experts, M, multiple_of=128)

    out_default = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )

    out_none = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        bias=None,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )

    torch.testing.assert_close(out_default, out_none, rtol=0, atol=0)
