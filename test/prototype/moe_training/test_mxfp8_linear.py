# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_100

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

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


@pytest.mark.parametrize(
    "kernel_preference", [KernelPreference.EMULATED, KernelPreference.AUTO]
)
def test_mxfp8_linear_frozen_weight_skips_wgrad(kernel_preference):
    """A frozen weight (requires_grad=False) gets no grad_weight while
    grad_input is unchanged, and the weight-gradient casts are skipped.

    grad_weight is discarded by autograd for a non-trainable weight, so
    computing it (the wgrad GEMM plus its two dim1 MXFP8 casts) is pure
    overhead for a frozen base in LoRA / frozen-layer finetuning.
    """
    if kernel_preference == KernelPreference.AUTO and not is_sm_at_least_100():
        pytest.skip("Real MXFP8 kernels require SM100+")

    import torchao.prototype.moe_training.mxfp8_linear as mxfp8_mod

    M, K, N = 256, 512, 1024
    mxfp8 = MXFP8Linear(
        K,
        N,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        kernel_preference=kernel_preference,
    )
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    def run(weight_requires_grad):
        mxfp8.weight.requires_grad_(weight_requires_grad)
        mxfp8.weight.grad = None
        x_in = x.clone().requires_grad_(True)

        n_dim1_casts = 0
        real_dim1 = mxfp8_mod._to_mxfp8_dim1_kernel_wrapper

        def counting_dim1(*args, **kwargs):
            nonlocal n_dim1_casts
            n_dim1_casts += 1
            return real_dim1(*args, **kwargs)

        with patch.object(mxfp8_mod, "_to_mxfp8_dim1_kernel_wrapper", counting_dim1):
            mxfp8(x_in).backward(grad_output)
        return x_in.grad, mxfp8.weight.grad, n_dim1_casts

    grad_in_train, grad_w_train, casts_train = run(True)
    grad_in_frozen, grad_w_frozen, casts_frozen = run(False)

    # A trainable weight gets a grad; a frozen weight must not.
    assert grad_w_train is not None
    assert grad_w_frozen is None
    # Skipping wgrad must not perturb the input gradient (dgrad).
    torch.testing.assert_close(grad_in_frozen, grad_in_train, atol=0.0, rtol=0.0)
    # The frozen path skips the weight-gradient dim1 casts.
    assert casts_frozen < casts_train


@pytest.mark.parametrize(
    "kernel_preference", [KernelPreference.EMULATED, KernelPreference.AUTO]
)
def test_mxfp8_linear_frozen_weight_compile(kernel_preference):
    """The frozen-weight backward must stay torch.compile-safe.

    ``ctx.needs_input_grad`` is a tuple of Python bools, so skipping the weight
    gradient is a trace-time-constant branch rather than data-dependent control
    flow. ``fullgraph=True`` therefore traces without a graph break and the
    compiled result must match eager.
    """
    if kernel_preference == KernelPreference.AUTO and not is_sm_at_least_100():
        pytest.skip("Real MXFP8 kernels require SM100+")

    M, K, N = 256, 512, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    def make_frozen():
        m = MXFP8Linear(
            K,
            N,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
            kernel_preference=kernel_preference,
        )
        m.weight.requires_grad_(False)
        return m

    eager = make_frozen()
    x_eager = x.clone().requires_grad_(True)
    eager(x_eager).backward(grad_output)

    compiled_mod = make_frozen()
    with torch.no_grad():
        compiled_mod.weight.copy_(eager.weight)
    x_compiled = x.clone().requires_grad_(True)
    torch.compile(compiled_mod, fullgraph=True)(x_compiled).backward(grad_output)

    # Frozen weight gets no grad in either mode, and grad_input must match.
    assert eager.weight.grad is None
    assert compiled_mod.weight.grad is None
    assert x_compiled.grad is not None
    torch.testing.assert_close(x_compiled.grad, x_eager.grad, atol=1e-2, rtol=1e-2)


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
