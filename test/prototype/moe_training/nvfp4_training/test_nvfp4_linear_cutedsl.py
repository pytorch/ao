# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for the CuteDSL kernel_preference path of NVFP4 training linear.

The CuteDSL preference uses the CuteDSL kernels for the amax and forward RTNE quantize
but falls back to Triton for the stochastic-rounding backward and weight quantize, so
these tests require both the CuteDSL runtime and Triton.
"""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_linear
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
    NVFP4TrainingConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.utils import compute_error

_HARDCODED_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)

# CuteDSL path needs M % 256 == 0, K/N % 128 == 0; backward SR quantize needs Triton.
_skip = pytest.mark.skipif(
    not (cutedsl_nvfp4_kernels_available() and has_triton()),
    reason="requires SM100 + CuteDSL runtime and Triton (cutedsl uses triton for SR backward)",
)


@_skip
@torch.no_grad()
def test_cutedsl_forward_matches_triton():
    """Forward output matches the Triton path: the forward GEMM consumes the rowwise
    (plain-A) quantize, which is bitwise-identical between the two backends."""
    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device="cuda")
    kw = dict(sign_vector=_HARDCODED_SIGN_VECTOR, sr_seed=seed)
    out_t = nvfp4_linear(x, W, None, kernel_preference=KernelPreference.TRITON, **kw)
    out_c = nvfp4_linear(x, W, None, kernel_preference=KernelPreference.CUTEDSL, **kw)
    assert compute_error(out_t.float(), out_c.float()) >= 40.0


@_skip
def test_cutedsl_forward_sqnr_and_finite_grads():
    """Forward reconstructs the bf16 result (SQNR >= 15 dB) and backward grads are finite."""
    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out = nvfp4_linear(
        x, W, None, sign_vector=_HARDCODED_SIGN_VECTOR,
        kernel_preference=KernelPreference.CUTEDSL,
    )
    ref = torch.nn.functional.linear(x.detach(), W.detach())
    assert compute_error(ref.float(), out.float()) >= 15.0
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(W.grad).all()
    assert x.grad.shape == x.shape and W.grad.shape == W.shape


@_skip
def test_cutedsl_config_path_converges_like_triton():
    """quantize_(NVFP4TrainingConfig(CUTEDSL)) trains a learnable task; loss decreases and
    tracks the Triton path closely."""

    M = K = N = 512
    torch.manual_seed(2)
    W_true = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    target = data.float() @ W_true.float().t()

    def train(kernel_preference, steps=40):
        torch.manual_seed(3)
        m = torch.nn.Sequential(torch.nn.Linear(K, N, bias=False)).cuda().bfloat16()
        quantize_(m, NVFP4TrainingConfig(kernel_preference=kernel_preference))
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            loss = (m(data).float() - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    cutedsl = train(KernelPreference.CUTEDSL)
    triton = train(KernelPreference.TRITON)

    # The config swap produced NVFP4Linear modules.
    assert cutedsl[-1] < 0.85 * cutedsl[0], "cutedsl loss did not decrease"
    assert abs(cutedsl[-1] - triton[-1]) / triton[-1] < 0.1, (
        f"cutedsl final loss {cutedsl[-1]:.3f} far from triton {triton[-1]:.3f}"
    )


@_skip
def test_cutedsl_config_swap_sets_preference():
    m = torch.nn.Sequential(torch.nn.Linear(512, 512, bias=False)).cuda().bfloat16()
    quantize_(m, NVFP4TrainingConfig(kernel_preference=KernelPreference.CUTEDSL))
    assert isinstance(m[0], NVFP4Linear)
    assert m[0].kernel_preference == KernelPreference.CUTEDSL


@_skip
def test_cutedsl_tensor_parallel_unsupported():
    """CuteDSL + tensor parallel must raise (the TP path is Triton-only)."""
    layer = (
        NVFP4Linear(
            512, 512, bias=False, kernel_preference=KernelPreference.CUTEDSL,
            process_group=object(),
        )
        .cuda()
        .to(torch.bfloat16)
    )
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError):
        layer(x)


@_skip
def test_nvfp4_linear_rejects_unsupported_kernel_preference():
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        nvfp4_linear(
            x, W, None, sign_vector=_HARDCODED_SIGN_VECTOR,
            kernel_preference=KernelPreference.TORCH,
        )
