# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for the CuteDSL kernel_preference path of NVFP4 training linear.

The CuteDSL preference runs the full path on CuteDSL — the amax, the forward RTNE quantize,
the backward SR (cvt.rs) quantize, and the 2D weight quantize. These tests still require Triton
to cross-check against the Triton backend.
"""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
    cutedsl_prepare_for_cuda_graph,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_linear
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
    NVFP4TrainingConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.utils import torch_version_at_least

_HARDCODED_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)

# CuteDSL path needs M % 256 == 0, K/N % 128 == 0 (and weight out_features % 256 == 0).
_skip = pytest.mark.skipif(
    not (cutedsl_nvfp4_kernels_available() and has_triton()),
    reason="requires SM100 + CuteDSL runtime and Triton (tests cross-check the Triton backend)",
)


@_skip
@torch.no_grad()
def test_cutedsl_forward_close_to_triton():
    """Both backends produce an accurate NVFP4 forward vs the bf16 reference. They do NOT match
    bitwise: the CuteDSL weight quantize uses canonical 1x16 block scales while the Triton 2D
    weight kernel shares one scale per 16x16 block, so CuteDSL is in fact slightly *more* accurate
    (the activation rowwise quantize is identical between backends; only the weight differs)."""
    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device="cuda")
    kw = dict(sign_vector=_HARDCODED_SIGN_VECTOR, sr_seed=seed)
    out_t = nvfp4_linear(x, W, None, kernel_preference=KernelPreference.TRITON, **kw)
    out_c = nvfp4_linear(x, W, None, kernel_preference=KernelPreference.CUTEDSL, **kw)
    ref = torch.nn.functional.linear(x.float(), W.float())
    sqnr_t = compute_error(ref, out_t.float())
    sqnr_c = compute_error(ref, out_c.float())
    assert sqnr_t >= 15.0 and sqnr_c >= 15.0
    assert sqnr_c >= sqnr_t - 0.5, (
        f"CuteDSL forward ({sqnr_c:.1f} dB) materially worse than Triton ({sqnr_t:.1f} dB)"
    )


@_skip
def test_cutedsl_forward_sqnr_and_finite_grads():
    """Forward reconstructs the bf16 result (SQNR >= 15 dB) and backward grads are finite."""
    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out = nvfp4_linear(
        x,
        W,
        None,
        sign_vector=_HARDCODED_SIGN_VECTOR,
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
def test_nvfp4_linear_rejects_unsupported_kernel_preference():
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        nvfp4_linear(
            x,
            W,
            None,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            kernel_preference=KernelPreference.TORCH,
        )


@_skip
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_cutedsl_torch_compile_default_mode():
    """The CuteDSL path runs under torch.compile (default mode) for repeated fwd+bwd: finite
    results, and per-step-varying SR noise in the backward (the default CUDA RNG advances each
    call, so consecutive weight grads differ)."""
    M = K = N = 512
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(K, N, bias=False)).cuda().bfloat16()
    quantize_(m, NVFP4TrainingConfig(kernel_preference=KernelPreference.CUTEDSL))
    mc = torch.compile(m)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(M, N, device="cuda", dtype=torch.bfloat16).float()

    grads = []
    for _ in range(4):
        m.zero_grad()
        out = mc(x)
        (out.float() - target).pow(2).mean().backward()
        assert torch.isfinite(out).all() and torch.isfinite(m[0].weight.grad).all()
        grads.append(m[0].weight.grad.detach().clone())
    # SR noise advances per step -> consecutive backward grads are not identical.
    assert (grads[-1] - grads[-2]).abs().max() > 0


@_skip
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_cutedsl_torch_compile_cuda_graph():
    """The CuteDSL path runs under torch.compile(mode="reduce-overhead") (CUDA graphs) once
    cutedsl_prepare_for_cuda_graph has pre-allocated the per-device CuteDSL state outside the
    cudagraph pool (the same setup the Triton path needs via prepare_for_cuda_graph). Captures +
    replays fwd+bwd with finite results, and SR noise still varies per replay because the persistent
    RNG-seed buffer is rewritten from the per-call (default-RNG) offset, which advances each replay."""
    M = K = N = 512
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(K, N, bias=False)).cuda().bfloat16()
    quantize_(m, NVFP4TrainingConfig(kernel_preference=KernelPreference.CUTEDSL))
    cutedsl_prepare_for_cuda_graph("cuda", sign_vectors=(m[0].rht_sign_vector,))
    mc = torch.compile(m, mode="reduce-overhead")
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(M, N, device="cuda", dtype=torch.bfloat16).float()

    grads = []
    for _ in range(6):  # graph captures ~iter 3, replays after
        m.zero_grad()
        out = mc(x)
        (out.float() - target).pow(2).mean().backward()
        assert torch.isfinite(out).all() and torch.isfinite(m[0].weight.grad).all()
        grads.append(m[0].weight.grad.detach().clone())
    torch.cuda.synchronize()
    assert (grads[-1] - grads[-2]).abs().max() > 0  # SR advances per replay
