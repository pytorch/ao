# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if not (torch_version_at_least("2.7.0") and torch.cuda.is_available()):
    pytest.skip("CUDA and PyTorch 2.7.0+ required", allow_module_level=True)

from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


def _build_linear_pair(
    in_features: int,
    out_features: int,
    bias: bool,
    kernel_preference: KernelPreference,
    wgrad_with_hp: bool,
):
    ref = torch.nn.Linear(
        in_features, out_features, bias=bias, device="cuda", dtype=torch.bfloat16
    )
    mxfp8 = MXFP8Linear(
        in_features,
        out_features,
        bias=bias,
        device="cuda",
        dtype=torch.bfloat16,
        kernel_preference=kernel_preference,
        wgrad_with_hp=wgrad_with_hp,
    )
    # Share weights/bias so the only difference is the quantized matmul path.
    with torch.no_grad():
        mxfp8.weight.copy_(ref.weight)
        if bias:
            mxfp8.bias.copy_(ref.bias)
    return ref, mxfp8


@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="Real MXFP8 kernels require SM100+"
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("wgrad_with_hp", [False, True])
@pytest.mark.parametrize(
    "kernel_preference", [KernelPreference.EMULATED, KernelPreference.AUTO]
)
def test_mxfp8_linear_fwd_bwd_sqnr(bias, wgrad_with_hp, kernel_preference):
    if kernel_preference == KernelPreference.AUTO and not is_sm_at_least_100():
        pytest.skip("Real MXFP8 kernels require SM100+")

    M, K, N = 1024, 1024, 2048
    ref, mxfp8 = _build_linear_pair(K, N, bias, kernel_preference, wgrad_with_hp)

    x_ref = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x_mxfp8 = x_ref.clone().detach().requires_grad_(True)

    out_ref = ref(x_ref)
    out_mxfp8 = mxfp8(x_mxfp8)

    # Forward checks
    assert out_mxfp8.shape == out_ref.shape
    assert out_mxfp8.dtype == torch.bfloat16

    sqnr_fwd = compute_error(out_ref, out_mxfp8)
    min_sqnr_fwd = 26.0 if bias else 27.0
    assert sqnr_fwd >= min_sqnr_fwd, f"Forward SQNR {sqnr_fwd} below {min_sqnr_fwd}"

    # Backward checks
    labels = torch.ones_like(out_ref)
    F.mse_loss(out_ref, labels).backward()
    F.mse_loss(out_mxfp8, labels).backward()

    sqnr_input_grad = compute_error(x_ref.grad, x_mxfp8.grad)
    assert sqnr_input_grad >= 25.0, f"Input grad SQNR {sqnr_input_grad} below 25.0"

    sqnr_weight_grad = compute_error(ref.weight.grad, mxfp8.weight.grad)
    # wgrad_with_hp keeps grad_weight in bf16, so SQNR should be much higher.
    min_sqnr_weight_grad = 34.0 if wgrad_with_hp else 24.0
    assert sqnr_weight_grad >= min_sqnr_weight_grad, (
        f"Weight grad SQNR {sqnr_weight_grad} below {min_sqnr_weight_grad}"
    )

    if bias:
        # Bias grad is a simple sum over the batch dim and should match closely.
        sqnr_bias_grad = compute_error(ref.bias.grad, mxfp8.bias.grad)
        assert sqnr_bias_grad >= 40.0, f"Bias grad SQNR {sqnr_bias_grad} below 40.0"


def test_mxfp8_linear_3d_input():
    """MXFP8Linear should accept inputs with leading batch/sequence dims."""
    K, N = 1024, 2048
    ref, mxfp8 = _build_linear_pair(
        K,
        N,
        bias=True,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=False,
    )

    x = torch.randn(4, 128, K, dtype=torch.bfloat16, device="cuda")
    out_ref = ref(x)
    out_mxfp8 = mxfp8(x)

    assert out_mxfp8.shape == out_ref.shape == (4, 128, N)
    sqnr = compute_error(out_ref, out_mxfp8)
    assert sqnr >= 26.0, f"Forward SQNR {sqnr} below 26.0"
