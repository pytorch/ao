# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn import functional as F

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.utils import is_sm_version, torch_version_at_least

# We need to skip before doing any imports which would use triton, since
# triton won't be available on CPU builds and torch < 2.5
if not (
    torch_version_at_least("2.7.0")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.float8_utils import compute_error
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
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.quantization.quantize_.common import KernelPreference
from torchao.testing.utils import skip_if_rocm

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8, 16))
def test_emulate_mxfp8_grouped_gemm_2d_3d(M, K, N, num_experts):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)
    x_ref, w_ref, offs_ref = x.clone(), w.clone(), offs.clone()

    # Quantize inputs to mxpf8 for emulated mxfp8 scaled grouped mm
    block_size = 32
    x_scale, x_fp8 = to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

    # To cast B_t per-expert to mxfp8 across dim1, we transpose the experts, cast along dim -1, then untranspose.
    w_scale, w_fp8 = to_mx(
        w,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
    )

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

    block_size = 32
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1).requires_grad_(True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)
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
        block_size=block_size,
        kernel_preference=kernel_preference,
        wgrad_with_hp=wgrad_with_hp,
        scale_calculation_mode=scale_mode,
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
