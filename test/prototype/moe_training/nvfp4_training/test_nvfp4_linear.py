# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training import (
    nvfp4_linear as nvfp4_linear_mod,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
    cutedsl_prepare_for_cuda_graph,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    prepare_for_cuda_graph,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
    _resolve_use_cutedsl,
    nvfp4_linear,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
    NVFP4TrainingConfig,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_HARDCODED_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)

# Both paths need M, K, out_features % 128 == 0. 512 exercises the CuteDSL 256-row
# supertile; the 128-row supertile (M % 128 but not % 256) is covered by the
# kernel-level tests.
_M = _K = _N = 512

_TRITON_MARKS = [
    pytest.mark.skipif(not has_triton(), reason="unsupported without triton"),
    pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+"),
    pytest.mark.skipif(
        not torch_version_at_least("2.10.0"),
        reason="torch.compile requires PyTorch 2.10+",
    ),
]
_CUTEDSL_MARKS = [
    pytest.mark.skipif(
        not cutedsl_nvfp4_kernels_available(),
        reason="requires SM100 + the CuteDSL runtime",
    ),
]

_KERNEL_PREFS = [
    # AUTO is the default, so it is parametrized alongside the two explicit choices
    # rather than left to be covered transitively by whichever one it resolves to.
    pytest.param(KernelPreference.AUTO, marks=_TRITON_MARKS, id="auto"),
    pytest.param(KernelPreference.TRITON, marks=_TRITON_MARKS, id="triton"),
    pytest.param(KernelPreference.CUTEDSL, marks=_CUTEDSL_MARKS, id="cutedsl"),
]

_requires_both_backends = pytest.mark.skipif(
    not (cutedsl_nvfp4_kernels_available() and has_triton()),
    reason="requires SM100 + CuteDSL runtime and Triton (tests cross-check the Triton backend)",
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _prepare_backend_for_cuda_graph(kernel_preference, sign_vector) -> None:
    """Pre-allocate the selected backend's persistent per-device state outside the cudagraph pool.

    Resolved through ``_resolve_use_cutedsl`` rather than compared against CUTEDSL:
    AUTO also lands on CuteDSL wherever the runtime allows, and preparing the wrong
    backend would allocate into the cudagraph pool at capture time.
    """
    if _resolve_use_cutedsl(kernel_preference):
        cutedsl_prepare_for_cuda_graph("cuda", sign_vectors=(sign_vector,))
    else:
        prepare_for_cuda_graph(torch.device("cuda"), sign_vectors=(sign_vector,))


def test_nvfp4_linear_rht_sign_vector_state_dict_roundtrip():
    torch.manual_seed(123)
    layer = NVFP4Linear(128, 128, bias=False, kernel_preference=KernelPreference.TRITON)
    expected_sign_vector = layer.rht_sign_vector
    assert len(expected_sign_vector) == 16
    assert set(expected_sign_vector) <= {-1, 1}
    state_dict = layer.state_dict()

    torch.manual_seed(456)
    loaded = NVFP4Linear(
        128, 128, bias=False, kernel_preference=KernelPreference.TRITON
    )
    loaded.load_state_dict(state_dict)

    assert loaded.rht_sign_vector == expected_sign_vector
    torch.testing.assert_close(
        loaded._rht_sign_vector.cpu(),
        layer._rht_sign_vector.cpu(),
        atol=0,
        rtol=0,
    )


def test_nvfp4_linear_from_linear_preserves_rht_sign_vector():
    sign_vector = tuple(1 if i % 2 == 0 else -1 for i in range(16))
    layer = NVFP4Linear(
        128,
        128,
        bias=False,
        kernel_preference=KernelPreference.TRITON,
        rht_sign_vector=sign_vector,
    )

    converted = NVFP4Linear.from_linear(layer)

    assert converted.rht_sign_vector == sign_vector
    torch.testing.assert_close(
        converted._rht_sign_vector.cpu(),
        layer._rht_sign_vector.cpu(),
        atol=0,
        rtol=0,
    )


@_requires_cuda
def test_nvfp4_linear_rejects_unsupported_kernel_preference():
    x = torch.randn(_M, _K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(_N, _K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        nvfp4_linear(
            x,
            W,
            None,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            kernel_preference=KernelPreference.TORCH,
        )


@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFS)
def test_forward_sqnr_and_finite_grads(kernel_preference):
    """Forward reconstructs the bf16 result (SQNR >= 15 dB) and backward grads are finite."""
    torch.manual_seed(0)
    M, K, N = _M, _K, _N
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out = nvfp4_linear(
        x,
        W,
        None,
        sign_vector=_HARDCODED_SIGN_VECTOR,
        kernel_preference=kernel_preference,
    )
    ref = torch.nn.functional.linear(x.detach(), W.detach())
    assert compute_error(ref.float(), out.float()) >= 15.0
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(W.grad).all()
    assert x.grad.shape == x.shape and W.grad.shape == W.shape


@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFS)
@torch.no_grad()
def test_fast_math_defaults_on_and_is_observable(kernel_preference):
    """nvfp4_linear defaults to fast math, and turning it off changes the result.

    Without this the default would be untestable from outside: fast and exact differ by
    ~48 dB SQNR, so every reconstruction bar in this file passes either way. Asserting
    the default *equals* the explicit True and *differs* from False pins which one the
    module actually selects.
    """
    torch.manual_seed(0)
    x = torch.randn(_M, _K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(_N, _K, dtype=torch.bfloat16, device="cuda")
    kwargs = dict(
        sign_vector=_HARDCODED_SIGN_VECTOR, kernel_preference=kernel_preference
    )

    default = nvfp4_linear(x, W, None, **kwargs)
    fast = nvfp4_linear(x, W, None, use_fast_math=True, **kwargs)
    exact = nvfp4_linear(x, W, None, use_fast_math=False, **kwargs)

    assert torch.equal(default, fast), "nvfp4_linear no longer defaults to fast math"
    assert not torch.equal(fast, exact), (
        "fast and exact math produced identical output; the flag is not reaching "
        "the quantize kernels"
    )


@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFS)
def test_config_swap_sets_preference(kernel_preference):
    m = torch.nn.Sequential(torch.nn.Linear(_K, _N, bias=False)).cuda().bfloat16()
    quantize_(m, NVFP4TrainingConfig(kernel_preference=kernel_preference))
    assert isinstance(m[0], NVFP4Linear)
    assert m[0].kernel_preference == kernel_preference


@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFS)
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_torch_compile_default_mode(kernel_preference):
    """The selected path runs under torch.compile (default mode) for repeated fwd+bwd: finite
    results, and per-step-varying SR noise in the backward (the default CUDA RNG advances each
    call, so consecutive weight grads differ)."""
    M, K, N = _M, _K, _N
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(K, N, bias=False)).cuda().bfloat16()
    quantize_(m, NVFP4TrainingConfig(kernel_preference=kernel_preference))
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


@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFS)
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_torch_compile_cuda_graph(kernel_preference):
    """nvfp4_linear works under torch.compile(mode="reduce-overhead") (CUDA graphs) once the
    backend's persistent per-device state has been pre-allocated outside the cudagraph pool
    (prepare_for_cuda_graph for Triton, cutedsl_prepare_for_cuda_graph for CuteDSL). Captures and
    replays fwd+bwd with finite, replay-stable forwards, and SR noise still varies per replay
    because the persistent RNG-seed buffer is rewritten from the per-call (default-RNG) offset,
    which advances each replay."""
    torch.manual_seed(0)
    M, K, N = _M, _K, _N
    layer = (
        NVFP4Linear(K, N, bias=False, kernel_preference=kernel_preference)
        .cuda()
        .to(torch.bfloat16)
    )
    _prepare_backend_for_cuda_graph(kernel_preference, layer.rht_sign_vector)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    compiled_layer = torch.compile(layer, mode="reduce-overhead", fullgraph=True)
    compiled_bwd = torch.compile(fullgraph=True, mode="reduce-overhead")

    for _ in range(3):  # graph captures ~iter 3, replays after
        with torch._dynamo.compiled_autograd._enable(compiled_bwd):
            out = compiled_layer(x)
            assert torch.isfinite(out).all()
            out.sum().backward()
        assert torch.isfinite(layer.weight.grad).all()

    r1 = compiled_layer(x)
    r2 = compiled_layer(x)
    torch.testing.assert_close(r1, r2)

    x_hp = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    ref = torch.nn.functional.linear(x_hp, layer.weight)
    nvfp4_out = layer(x_hp)
    sqnr = compute_error(ref, nvfp4_out)
    assert sqnr >= 15.0, f"Forward SQNR {sqnr:.2f} dB < 15 dB"

    # Use a fixed non-constant upstream gradient. With .sum().backward(), grad_output
    # is all ones, whose RHT quantization lands on exact values and can be
    # deterministic even when stochastic rounding is enabled.
    grad_out = torch.randn_like(r1)

    def one_step():
        x.grad = None
        layer.weight.grad = None
        with torch._dynamo.compiled_autograd._enable(compiled_bwd):
            out = compiled_layer(x)
            assert torch.isfinite(out).all()
            out.backward(grad_out)
        assert torch.isfinite(layer.weight.grad).all()
        return layer.weight.grad.detach().clone()

    for _ in range(3):
        one_step()

    g1 = one_step()
    g2 = one_step()
    torch.cuda.synchronize()

    assert not torch.equal(g1, g2), (
        "Backward SR grad_weight must differ across steps because default CUDA RNG "
        "advances each replay"
    )
    assert (g2 - g1).abs().max() > 0  # SR advances per replay


@_requires_both_backends
@torch.no_grad()
def test_cutedsl_forward_close_to_triton():
    """Both backends produce an accurate NVFP4 forward vs the bf16 reference."""
    torch.manual_seed(0)
    M, K, N = _M, _K, _N
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


@_requires_both_backends
def test_cutedsl_config_path_converges_like_triton():
    """quantize_(NVFP4TrainingConfig(CUTEDSL)) trains a learnable task; loss decreases and
    tracks the Triton path closely."""

    M, K, N = _M, _K, _N
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


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
)
def test_resolve_use_cutedsl_falls_back_without_cutedsl(monkeypatch):
    """AUTO degrades to Triton where CuteDSL cannot run; CUTEDSL says so instead.

    The linear twin of test_nvfp4_grouped_mm's fallback test. AUTO is the default, so
    without monkeypatching availability this branch is unreachable on any box that has
    the CuteDSL runtime -- which is every box that can run the rest of this file.
    """
    monkeypatch.setattr(
        nvfp4_linear_mod, "cutedsl_nvfp4_kernels_available", lambda: False
    )

    assert _resolve_use_cutedsl(KernelPreference.AUTO) is False
    assert _resolve_use_cutedsl(KernelPreference.TRITON) is False
    with pytest.raises(RuntimeError, match="CUTEDSL requires"):
        _resolve_use_cutedsl(KernelPreference.CUTEDSL)
    with pytest.raises(ValueError, match="AUTO, TRITON, or CUTEDSL"):
        _resolve_use_cutedsl(KernelPreference.TORCH)


@pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(), reason="requires SM100 + the CuteDSL runtime"
)
def test_resolve_use_cutedsl_prefers_cutedsl_when_available():
    """With the runtime present, AUTO and CUTEDSL agree; TRITON still opts out."""
    assert _resolve_use_cutedsl(KernelPreference.AUTO) is True
    assert _resolve_use_cutedsl(KernelPreference.CUTEDSL) is True
    assert _resolve_use_cutedsl(KernelPreference.TRITON) is False


@_requires_cuda
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
)
def test_nvfp4_linear_auto_runs_on_triton_fallback(monkeypatch):
    """AUTO still computes when CuteDSL is unavailable, matching an explicit TRITON call.

    Exercises the fallback end to end rather than through the resolver alone: the
    fallback must reach nvfp4_matmul with use_cutedsl=False, not merely return it.
    RTNE forward only, so the two are bitwise comparable (the SR streams are not).
    """
    torch.manual_seed(123)
    x = torch.randn(_M, _K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(_N, _K, dtype=torch.bfloat16, device="cuda")
    kw = dict(sign_vector=_HARDCODED_SIGN_VECTOR)

    expected = nvfp4_linear(x, w, None, kernel_preference=KernelPreference.TRITON, **kw)
    monkeypatch.setattr(
        nvfp4_linear_mod, "cutedsl_nvfp4_kernels_available", lambda: False
    )
    got = nvfp4_linear(x, w, None, kernel_preference=KernelPreference.AUTO, **kw)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(), reason="requires SM100 + the CuteDSL runtime"
)
def test_cutedsl_prepare_for_cuda_graph_warms_every_kernel_the_path_uses():
    """After the pre-capture warm-up, no CuteDSL kernel compiles lazily.

    The whole point of cutedsl_prepare_for_cuda_graph is that a lazy cute.compile must
    not fire inside a captured region, and nothing else observes whether it worked: the
    warm-up once compiled a set of lru_cache keys no runtime call could reach (keyword
    vs positional args make distinct keys, and the fast-math axis was missing entirely),
    so it warmed twelve kernels and hit zero of them while every test still passed. This
    asserts the property directly -- new compiles after prepare, which must be zero --
    rather than trusting a smoke test that survives the warm-up being a no-op.

    Both halves of the warm-up are covered. The grouped caches live in a separate module
    with their own key shapes, so a linear-only assertion would leave the grouped MoE
    kernels exposed to exactly the drift this test exists to catch.
    """
    from torchao.prototype.moe_training.nvfp4_training import (
        _cutedsl_group_kernels_impl,
        _cutedsl_kernels_impl,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
        _to_nvfp4_rht_rs_then_scaled_grouped_mm,
    )

    cutedsl_prepare_for_cuda_graph("cuda", sign_vectors=(_HARDCODED_SIGN_VECTOR,))
    caches = {
        "fused": _cutedsl_kernels_impl._compile_fused_kernel,
        "amax": _cutedsl_kernels_impl._compile_amax_tc_kernel,
        "group_fused": _cutedsl_group_kernels_impl._compile_group_fused_kernel,
        "group_amax": _cutedsl_group_kernels_impl._compile_group_amax_kernel,
    }
    before = {k: c.cache_info().misses for k, c in caches.items()}

    x = torch.randn(_M, _K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(_N, _K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    # Both arithmetic modes, and a backward so the stochastic-rounding variant is
    # reached as well -- each is a separate compiled kernel and a separate cache key.
    for use_fast_math in (True, False):
        out = nvfp4_linear(
            x,
            w,
            None,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            kernel_preference=KernelPreference.CUTEDSL,
            use_fast_math=use_fast_math,
        )
        out.sum().backward()

    # Grouped MoE: the forward is RTNE and the backward is SR, so one fwd+bwd per
    # arithmetic mode covers all four grouped fused keys plus the grouped amax. The
    # grouped weight quantize compiles out of the linear `fused` cache (grouped=True).
    num_experts = 2
    gx = torch.randn(_M, _K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    gw = torch.randn(
        num_experts, _N, _K, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    offs = generate_jagged_offs(num_experts, _M, multiple_of=128, dtype=torch.int32)
    sr_seed = torch.tensor([1234], dtype=torch.int64, device="cuda")
    for use_fast_math in (True, False):
        gout = _to_nvfp4_rht_rs_then_scaled_grouped_mm(
            gx,
            gw,
            _HARDCODED_SIGN_VECTOR,
            sr_seed,
            offs=offs,
            pad_token_groups_for_grouped_mm=False,
            kernel_preference=KernelPreference.CUTEDSL,
            use_fast_math=use_fast_math,
        )
        gout.sum().backward()

    new = {k: c.cache_info().misses - before[k] for k, c in caches.items()}
    assert set(new.values()) == {0}, (
        f"{sum(new.values())} CuteDSL kernel(s) compiled after "
        f"cutedsl_prepare_for_cuda_graph ({new}); they would compile inside a "
        "CUDA-graph capture"
    )
