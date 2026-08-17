# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the fused MXFP8 grouped-MLP kernel family (SM100).

Three custom ops, one physical kernel launch each:

* ``torchao::mxfp8_grouped_gemm_swiglu_fwd``  (A) FC1 grouped GEMM + SwiGLU +
  rowwise 1x32 and columnwise 32x1 MXFP8 RCEIL quantization
* ``torchao::mxfp8_grouped_gemm_dswiglu_bwd`` (B) FC2 dgrad grouped GEMM +
  dSwiGLU + dual quantization
* ``torchao::mxfp8_grouped_gemm_wgrad``       (C) grouped MXFP8 wgrad

References are deliberately bridge-free: eager per-expert BF16/FP64 matmuls,
``F.silu`` / the closed-form dSwiGLU, and the pure-torch ``to_mx`` (RCEIL) +
``to_blocked`` quantizers. The CuTe DSL standalone quantizer ops and
``cute_utils`` are never imported.

Numerical strategy:

* GEMM outputs (``z``, ``dh``, ``dw``): SQNR / tolerance vs an FP64 oracle
  (reduction order is free), plus BITWISE equality on exact-integer operand
  configs where every partial sum is exactly representable in FP32.
* Quantized outputs: bitwise vs ``to_mx`` wherever the activation is exact.
  ``silu(g) == g`` exactly for ``g >= 128`` (sigmoid saturates to 1.0f), so a
  saturated gate turns the fused activation into exact products and the
  quantization stage must match ``to_mx`` byte-for-byte, including special
  values. Random-input forward comparisons are ALSO bitwise (measured zero
  mismatches: the kernel's sigmoid composition matches torch's float32 silu
  exactly); backward random inputs are SQNR-gated because the reference dh
  comes from an FP64 oracle whose BF16 rounding can differ at reduction-order
  boundaries, with bitwise coverage provided by the exact-dh configuration.

FakeTensor and validation tests run without a GPU; kernel tests require SM100.
"""

import random

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

from torchao.float8.float8_utils import compute_error  # noqa: E402
from torchao.prototype.moe_training.utils import generate_jagged_offs  # noqa: E402
from torchao.prototype.mx_formats.config import ScaleCalculationMode  # noqa: E402
from torchao.prototype.mx_formats.mx_tensor import to_mx  # noqa: E402
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked  # noqa: E402
from torchao.testing._mxfp8_test_utils import make_mxfp8_semantic_cases  # noqa: E402

# Importing the ops module registers the three custom ops. The public wrapper
# module is preferred once it exists; both expose the same wrapper names.
try:
    from torchao.prototype.moe_training import mxfp8_grouped_mlp as _api
except ImportError:
    from torchao.prototype.moe_training.kernels.mxfp8 import grouped_mlp_ops as _api

_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_BLOCK = 32
_RCEIL = ScaleCalculationMode.RCEIL

_OP_NAMES = (
    "mxfp8_grouped_gemm_swiglu_fwd",
    "mxfp8_grouped_gemm_dswiglu_bwd",
    "mxfp8_grouped_gemm_wgrad",
)


def _is_sm_10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


_gpu = pytest.mark.skipif(not _is_sm_10x(), reason="MXFP8 requires CUDA SM 10.x")

torch._dynamo.config.cache_size_limit = 1000


# ---------------------------------------------------------------------------
# Reference helpers (pure torch)
# ---------------------------------------------------------------------------


def _round_up(x: int, to: int) -> int:
    return ((x + to - 1) // to) * to


def _blocked_numel(rows: int, cols: int) -> int:
    return _round_up(rows, 128) * _round_up(cols, 4)


def _quantize_rowwise_ref(x: torch.Tensor):
    """[M, K] high precision -> (qdata [M, K] row-major, flat blocked scales)."""
    scale, q = to_mx(x, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, to_blocked(scale)


def _quantize_colwise_ref(x: torch.Tensor):
    """[R, N] high precision -> (qdata [R, N] stride (1, R), flat blocked scales
    for the logical [N, R/32] scale matrix). Recipe from the repo bench."""
    scale_t, q_t = to_mx(x.t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q_t.t(), to_blocked(scale_t)


def _dequant_rowwise(q: torch.Tensor, sf_flat: torch.Tensor, dtype=torch.float64):
    """Dequantize a row-major [M, K] E4M3 tensor with flat blocked scales."""
    m, k = q.shape
    scale = from_blocked(sf_flat.view(_E8M0).reshape(-1), m, k // _BLOCK)
    return q.to(dtype) * scale.to(dtype).repeat_interleave(_BLOCK, dim=1)


def _dequant_colwise(q_col: torch.Tensor, sf_flat: torch.Tensor, dtype=torch.float64):
    """Dequantize a [R, N] stride-(1, R) E4M3 tensor (scales logical [N, R/32])."""
    r, n = q_col.shape
    scale = from_blocked(sf_flat.view(_E8M0).reshape(-1), n, r // _BLOCK)
    return (q_col.t().to(dtype) * scale.to(dtype).repeat_interleave(_BLOCK, dim=1)).t()


def _dswiglu_closed_form(dh: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    """CONTRACT section 6.3 normative math, all fp32 in, (dgate, dup) fp32 out."""
    sig = torch.sigmoid(gate)
    silu = gate * sig
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    return dh * up * dsilu, dh * silu


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _mismatch_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    return (a != b).float().mean().item()


# ---------------------------------------------------------------------------
# Input builders. Tests own the offsets, so references never need a D2H sync.
# ---------------------------------------------------------------------------


def _mk_offsets(sizes, device):
    ends = torch.tensor(sizes, dtype=torch.int64).cumsum(0)
    return ends.to(torch.int32).to(device)


def _pack_grouped_weight(q_list, sf_list, device):
    """Per-expert row-major [N, K] qdata -> [G, K, N] stride (K*N, 1, K) + [G, sf]."""
    g = len(q_list)
    n, k = q_list[0].shape
    w = torch.empty_strided((g, k, n), (k * n, 1, k), dtype=_E4M3, device=device)
    for i, q in enumerate(q_list):
        w[i].copy_(q.t())
    sf = torch.stack([s.reshape(-1) for s in sf_list])
    return w, sf


def _random_grouped_weight(g, n, k, device, exact_int=False):
    """Random per-expert [N, K] weights; returns (packed q, packed sf, dequants)."""
    qs, sfs, deqs = [], [], []
    for _ in range(g):
        if exact_int:
            q = torch.randint(-6, 7, (n, k), device=device).to(_E4M3)
            logical = torch.full(
                (n, k // _BLOCK), 127, dtype=torch.uint8, device=device
            )
            sf = to_blocked(logical.view(_E8M0))
        else:
            w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
            q, sf = _quantize_rowwise_ref(w)
        qs.append(q)
        sfs.append(sf)
        deqs.append(_dequant_rowwise(q, sf))
    packed_q, packed_sf = _pack_grouped_weight(qs, sfs, device)
    return packed_q, packed_sf, deqs


def _random_activation(r, k, device, exact_int=False):
    if exact_int:
        q = torch.randint(-6, 7, (r, k), device=device).to(_E4M3)
        logical = torch.full((r, k // _BLOCK), 127, dtype=torch.uint8, device=device)
        sf = to_blocked(logical.view(_E8M0))
    else:
        x = torch.randn(r, k, device=device, dtype=torch.bfloat16)
        q, sf = _quantize_rowwise_ref(x)
    return q, sf, _dequant_rowwise(q, sf)


def _ref_grouped_gemm(x_deq, w_deqs, offsets_sizes):
    """Per-expert fp64 x @ w.T over test-owned split sizes; inactive tail = 0."""
    r = x_deq.shape[0]
    out = torch.zeros(r, w_deqs[0].shape[0], dtype=torch.float64, device=x_deq.device)
    start = 0
    for g, size in enumerate(offsets_sizes):
        if size:
            out[start : start + size] = x_deq[start : start + size] @ w_deqs[g].t()
        start += size
    return out


def _make_a_inputs(r, d, f, sizes, device, exact_int=False):
    x_q, x_sf, x_deq = _random_activation(r, d, device, exact_int)
    w_q, w_sf, w_deqs = _random_grouped_weight(len(sizes), 2 * f, d, device, exact_int)
    offsets = _mk_offsets(sizes, device)
    z_ref = _ref_grouped_gemm(x_deq, w_deqs, sizes)
    return (x_q, x_sf, w_q, w_sf, offsets), z_ref


def _make_b_inputs(r, d, f, sizes, device, exact_int=False, z=None):
    do_q, do_sf, do_deq = _random_activation(r, d, device, exact_int)
    w_q, w_sf, w_deqs = _random_grouped_weight(len(sizes), f, d, device, exact_int)
    offsets = _mk_offsets(sizes, device)
    if z is None:
        z = torch.randn(r, f, 2, device=device, dtype=torch.bfloat16)
        active = sum(sizes)
        if active < r:
            # The inactive tail of z is read-forbidden: poison it so any read
            # shows up as NaN contamination in the outputs.
            z[active:] = float("nan")
    dh_ref = _ref_grouped_gemm(do_deq, w_deqs, sizes)
    return (do_q, do_sf, w_q, w_sf, z, offsets), dh_ref


def _make_c_inputs(r, n, k, sizes, device, exact_int=False):
    def colwise(rows, cols):
        if exact_int:
            q_rm = torch.randint(-6, 7, (cols, rows), device=device).to(_E4M3)
            logical = torch.full(
                (cols, rows // _BLOCK), 127, dtype=torch.uint8, device=device
            )
            sf = to_blocked(logical.view(_E8M0))
            return q_rm.t(), sf
        x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        return _quantize_colwise_ref(x)

    dy_q, dy_sf = colwise(r, n)
    x_q, x_sf = colwise(r, k)
    offsets = _mk_offsets(sizes, device)
    return dy_q, dy_sf, x_q, x_sf, offsets


def _ref_wgrad(dy_q, dy_sf, x_q, x_sf, sizes):
    dy = _dequant_colwise(dy_q, dy_sf)
    x = _dequant_colwise(x_q, x_sf)
    g = len(sizes)
    n, k = dy.shape[1], x.shape[1]
    dw = torch.zeros(g, n, k, dtype=torch.float64, device=dy.device)
    start = 0
    for i, size in enumerate(sizes):
        if size:
            dw[i] = dy[start : start + size].t() @ x[start : start + size]
        start += size
    return dw.to(torch.bfloat16)


def _b_reference_dz(dh_bf16, z):
    """CONTRACT section 6.3: bf16 dh + saved z -> interleaved dz (bf16 [R, 2F])."""
    dgate, dup = _dswiglu_closed_form(
        dh_bf16.float(), z[..., 0].float(), z[..., 1].float()
    )
    dz = torch.stack((dgate.bfloat16(), dup.bfloat16()), dim=-1)
    return dz.reshape(dh_bf16.shape[0], -1)


def _assert_quantized_pair(
    q_row, sf_row, q_col, sf_col, ref_bf16, max_qdata_rate=0.0, max_scale_rate=0.0
):
    """Compare both fused quantized orientations against to_mx of ``ref_bf16``."""
    ref_row_q, ref_row_sf = _quantize_rowwise_ref(ref_bf16)
    ref_col_q, ref_col_sf = _quantize_colwise_ref(ref_bf16)
    checks = (
        ("h_row_q", _bytes(q_row), _bytes(ref_row_q), max_qdata_rate),
        ("h_row_sf", _bytes(sf_row.reshape(-1)), _bytes(ref_row_sf), max_scale_rate),
        # Column-major output: compare bytes of the same logical view without
        # forcing contiguity (the stride IS the ABI).
        ("h_col_q", _bytes(q_col.t()), _bytes(ref_col_q.t()), max_qdata_rate),
        ("h_col_sf", _bytes(sf_col.reshape(-1)), _bytes(ref_col_sf), max_scale_rate),
    )
    for name, got, want, budget in checks:
        rate = _mismatch_rate(got, want)
        assert rate <= budget, f"{name}: byte mismatch rate {rate} > {budget}"


# ---------------------------------------------------------------------------
# 1. Registration and public surface
# ---------------------------------------------------------------------------


def test_ops_registered():
    for name in _OP_NAMES:
        assert hasattr(torch.ops.torchao, name), name
        assert hasattr(_api, name), f"{_api.__name__} must export {name}"


# ---------------------------------------------------------------------------
# 2. Fake / meta output contracts (no GPU required)
# ---------------------------------------------------------------------------


def _fake_a_inputs(r, d, f, g, device="cuda"):
    x_q = torch.empty(r, d, dtype=_E4M3, device=device)
    x_sf = torch.empty(_blocked_numel(r, d // _BLOCK), dtype=_E8M0, device=device)
    w_q = torch.empty_strided(
        (g, d, 2 * f), (d * 2 * f, 1, d), dtype=_E4M3, device=device
    )
    w_sf = torch.empty(
        g, _blocked_numel(2 * f, d // _BLOCK), dtype=_E8M0, device=device
    )
    offs = torch.empty(g, dtype=torch.int32, device=device)
    return x_q, x_sf, w_q, w_sf, offs


def _fake_b_inputs(r, d, f, g, device="cuda"):
    do_q = torch.empty(r, d, dtype=_E4M3, device=device)
    do_sf = torch.empty(_blocked_numel(r, d // _BLOCK), dtype=_E8M0, device=device)
    w_q = torch.empty_strided((g, d, f), (d * f, 1, d), dtype=_E4M3, device=device)
    w_sf = torch.empty(g, _blocked_numel(f, d // _BLOCK), dtype=_E8M0, device=device)
    z = torch.empty_strided(
        (r, f, 2), (2 * f, 2, 1), dtype=torch.bfloat16, device=device
    )
    offs = torch.empty(g, dtype=torch.int32, device=device)
    return do_q, do_sf, w_q, w_sf, z, offs


def _fake_c_inputs(r, n, k, g, device="cuda"):
    dy = torch.empty_strided((r, n), (1, r), dtype=_E4M3, device=device)
    dy_sf = torch.empty(_blocked_numel(n, r // _BLOCK), dtype=_E8M0, device=device)
    x = torch.empty_strided((r, k), (1, r), dtype=_E4M3, device=device)
    x_sf = torch.empty(_blocked_numel(k, r // _BLOCK), dtype=_E8M0, device=device)
    offs = torch.empty(g, dtype=torch.int32, device=device)
    return dy, dy_sf, x, x_sf, offs


def test_fake_swiglu_fwd_contract():
    r, d, f, g = 256, 256, 128, 2
    with FakeTensorMode():
        z, hq, hsf, hcq, hcsf = torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(
            *_fake_a_inputs(r, d, f, g)
        )
    assert z.shape == (r, f, 2) and z.stride() == (2 * f, 2, 1)
    assert z.dtype == torch.bfloat16
    assert hq.shape == (r, f) and hq.stride() == (f, 1) and hq.dtype == _E4M3
    assert hsf.numel() == _blocked_numel(r, f // _BLOCK)
    assert hcq.shape == (r, f) and hcq.stride() == (1, r) and hcq.dtype == _E4M3
    assert hcsf.numel() == _blocked_numel(f, r // _BLOCK)


def test_fake_dswiglu_bwd_contract():
    r, d, f, g = 256, 256, 128, 2
    with FakeTensorMode():
        dzq, dzsf, dzcq, dzcsf = torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
            *_fake_b_inputs(r, d, f, g)
        )
    assert dzq.shape == (r, 2 * f) and dzq.stride() == (2 * f, 1)
    assert dzsf.numel() == _blocked_numel(r, 2 * f // _BLOCK)
    assert dzcq.shape == (r, 2 * f) and dzcq.stride() == (1, r)
    assert dzcsf.numel() == _blocked_numel(2 * f, r // _BLOCK)


def test_fake_wgrad_contract():
    r, n, k, g = 256, 256, 128, 2
    with FakeTensorMode():
        dw = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*_fake_c_inputs(r, n, k, g))
    assert dw.shape == (g, n, k)
    assert dw.stride() == (n * k, k, 1)
    assert dw.dtype == torch.bfloat16


def test_fake_validation_rejects_bad_metadata():
    with FakeTensorMode():
        args = list(_fake_c_inputs(256, 192, 128, 2))  # N = 192, not 128-multiple
        with pytest.raises(ValueError, match="multiple of 128"):
            torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)
        a_args = list(_fake_a_inputs(192, 256, 128, 2))  # R = 192
        with pytest.raises(ValueError, match="multiple of 128"):
            torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(*a_args)


# ---------------------------------------------------------------------------
# 3. Validation negatives (real tensors, small shapes)
# ---------------------------------------------------------------------------


def _valid_c_args(device):
    return list(_make_c_inputs(256, 256, 128, [128, 128], device))


_NEGATIVE_CASES = [
    "dtype",
    "row_major_colwise_operand",
    "scale_numel",
    "offsets_int64",
    "offsets_cpu",
    "offsets_2d",
    "offsets_numel",
    "offsets_noncontig",
    "n_not_128",
    "misaligned_view",
    "z_stride",
]


@_gpu
@pytest.mark.parametrize("case", _NEGATIVE_CASES)
def test_validation_negatives(case):
    device = "cuda"
    args = _valid_c_args(device)
    if case == "dtype":
        args[0] = torch.empty_strided(
            (256, 256), (1, 256), dtype=torch.bfloat16, device=device
        )
        err = "float8_e4m3fn"
    elif case == "row_major_colwise_operand":
        args[0] = torch.empty(256, 256, dtype=_E4M3, device=device)
        err = "stride"
    elif case == "scale_numel":
        args[1] = args[1].reshape(-1)[:-1]
        err = "blocked scale bytes"
    elif case == "offsets_int64":
        args[4] = args[4].to(torch.int64)
        err = "int32"
    elif case == "offsets_cpu":
        args[4] = args[4].cpu()
        err = "CUDA"
    elif case == "offsets_2d":
        args[4] = args[4].reshape(1, -1)
        err = "1D"
    elif case == "offsets_numel":
        args[4] = torch.tensor([128, 128, 256], dtype=torch.int32, device=device)
        # wgrad takes G from offsets, so a numel change alone is legal there;
        # use kernel A where G comes from the weight tensor instead.
        a_args = list(_fake_a_inputs(256, 256, 128, 2, device=device))
        a_args[4] = args[4]
        with pytest.raises(ValueError, match="one entry per"):
            torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(*a_args)
        return
    elif case == "offsets_noncontig":
        base = torch.zeros(4, dtype=torch.int32, device=device)
        args[4] = base.as_strided((2,), (2,))
        err = "contiguous"
    elif case == "n_not_128":
        args = _valid_c_args(device)
        dy = torch.empty_strided((256, 192), (1, 256), dtype=_E4M3, device=device)
        dy_sf = torch.empty(
            _blocked_numel(192, 256 // _BLOCK), dtype=_E8M0, device=device
        )
        args[0], args[1] = dy, dy_sf
        err = "multiple of 128"
    elif case == "misaligned_view":
        base = torch.zeros(256 * 256 + 32, dtype=_E4M3, device=device)
        args[0] = base.as_strided((256, 256), (1, 256), 2)
        err = "aligned"
    elif case == "z_stride":
        b_args = list(_fake_b_inputs(256, 256, 128, 2, device=device))
        # materialize real tensors with a wrong z layout
        b_args = [torch.empty_like(t) if t.is_cuda else t for t in b_args]
        b_args[4] = torch.empty(
            256, 2, 128, dtype=torch.bfloat16, device=device
        ).permute(0, 2, 1)
        with pytest.raises(ValueError, match="stride"):
            torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(*b_args)
        return
    with pytest.raises(ValueError, match=err):
        torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)


@_gpu
def test_validation_rejects_g0():
    device = "cuda"
    args = _valid_c_args(device)
    args[4] = torch.empty(0, dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)


@pytest.mark.skipif(
    not (_is_sm_10x() and torch.cuda.device_count() >= 2),
    reason="needs two CUDA devices",
)
def test_validation_rejects_cross_device():
    args = _valid_c_args("cuda:0")
    args[2] = args[2].to("cuda:1")
    with pytest.raises(ValueError, match="device"):
        torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)


# ---------------------------------------------------------------------------
# 4. R == 0 and zero-token experts
# ---------------------------------------------------------------------------


@_gpu
def test_r0_all_ops():
    device = "cuda"
    d, f, g = 256, 128, 2
    a_args = list(_fake_a_inputs(0, d, f, g, device=device))
    a_args = [torch.empty_like(t) for t in a_args]
    a_args[4] = torch.zeros(g, dtype=torch.int32, device=device)
    outs = torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(*a_args)
    assert all(o.shape[0] == 0 or o.numel() == 0 for o in outs)

    b_args = list(_fake_b_inputs(0, d, f, g, device=device))
    b_args = [torch.empty_like(t) for t in b_args]
    b_args[5] = torch.zeros(g, dtype=torch.int32, device=device)
    outs = torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(*b_args)
    assert all(o.numel() == 0 for o in outs)

    c_args = list(_fake_c_inputs(0, 256, 128, g, device=device))
    c_args = [torch.empty_like(t) for t in c_args]
    c_args[4] = torch.zeros(g, dtype=torch.int32, device=device)
    dw = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*c_args)
    assert dw.shape == (g, 256, 128)
    assert (dw == 0).all()


@_gpu
def test_wgrad_zero_token_expert():
    torch.manual_seed(0)
    device = "cuda"
    sizes = [128, 0, 256]
    args = _make_c_inputs(384, 256, 128, sizes, device)
    dw = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)
    assert (dw[1] == 0).all(), "zero-token expert must produce an all-zero slice"
    ref = _ref_wgrad(args[0], args[1], args[2], args[3], sizes)
    assert compute_error(ref[0].float(), dw[0].float()) >= 24.0
    assert compute_error(ref[2].float(), dw[2].float()) >= 24.0


# ---------------------------------------------------------------------------
# 5. Kernel C numerics
# ---------------------------------------------------------------------------


@_gpu
@pytest.mark.parametrize(
    "r,n,k,sizes",
    [
        (256, 256, 128, [128, 128]),  # FC1-like: N = 2F, K = D (small)
        (1536, 2816, 2048, [512, 0, 640, 384]),  # 16B FC1 wgrad class
        (1536, 2048, 1408, [512, 0, 640, 384]),  # 16B FC2 wgrad class
    ],
    ids=["small", "fc1_16b", "fc2_16b"],
)
def test_wgrad_numerics_random(r, n, k, sizes):
    torch.manual_seed(1)
    args = _make_c_inputs(r, n, k, sizes, "cuda")
    dw = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)
    ref = _ref_wgrad(args[0], args[1], args[2], args[3], sizes)
    assert dw.dtype == torch.bfloat16 and dw.shape == (len(sizes), n, k)
    torch.testing.assert_close(dw.float(), ref.float(), atol=2e-3, rtol=0.01)
    assert compute_error(ref.float(), dw.float()) >= 24.0


@_gpu
def test_wgrad_bitwise_exact_integers():
    torch.manual_seed(2)
    sizes = [256, 128, 384]
    args = _make_c_inputs(768, 256, 256, sizes, "cuda", exact_int=True)
    dw = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)
    ref = _ref_wgrad(args[0], args[1], args[2], args[3], sizes)
    assert torch.equal(_bytes(dw), _bytes(ref)), "exact-integer wgrad must be bitwise"


@_gpu
def test_wgrad_rejects_kgroups_scale_ordering_by_result():
    """Whole-matrix vs per-group K-groups blocked scales differ whenever N > 128;
    no length check can catch it, so prove the kernel is sensitive to it."""
    torch.manual_seed(3)
    device = "cuda"
    sizes = [128, 128]
    r, n, k = 256, 256, 128
    dy = torch.randn(r, n, device=device, dtype=torch.bfloat16)
    x = torch.randn(r, k, device=device, dtype=torch.bfloat16)
    dy_q, dy_sf = _quantize_colwise_ref(dy)
    x_q, x_sf = _quantize_colwise_ref(x)
    offsets = _mk_offsets(sizes, device)

    good = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(dy_q, dy_sf, x_q, x_sf, offsets)
    ref = _ref_wgrad(dy_q, dy_sf, x_q, x_sf, sizes)
    torch.testing.assert_close(good.float(), ref.float(), atol=2e-3, rtol=0.01)

    # Re-encode dy's scales per group (torchao K-groups form) and rerun.
    scale_t, _ = to_mx(dy.t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    per_group = torch.cat(
        [
            to_blocked(scale_t[:, s // _BLOCK : e // _BLOCK]).reshape(-1)
            for s, e in ((0, 128), (128, 256))
        ]
    )
    assert per_group.numel() == dy_sf.numel()
    assert not torch.equal(_bytes(per_group), _bytes(dy_sf.reshape(-1)))
    bad = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
        dy_q, per_group, x_q, x_sf, offsets
    )
    assert not torch.equal(bad, good), (
        "kernel must consume whole-matrix blocked scales; a K-groups buffer of "
        "identical length must change the result"
    )


# ---------------------------------------------------------------------------
# 6. Kernel A numerics
# ---------------------------------------------------------------------------

_A_SHAPES = [
    (256, 256, 128, [128, 128]),
    (1024, 512, 256, [256, 0, 512, 256]),  # zero-token expert + ragged
    (1536, 2048, 1408, [512, 128, 640, 256]),  # 16B class
]


@_gpu
@pytest.mark.parametrize("r,d,f,sizes", _A_SHAPES, ids=["small", "ragged", "16b"])
def test_swiglu_fwd_random(r, d, f, sizes):
    torch.manual_seed(4)
    args, z_ref = _make_a_inputs(r, d, f, sizes, "cuda")
    z, hq, hsf, hcq, hcsf = torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(*args)
    active = sum(sizes)
    assert (
        compute_error(z_ref[:active].float(), z[:active].reshape(active, -1).float())
        >= 27.0
    )
    # Quantized outputs compare against the normative chain applied to the
    # kernel's own z (removes GEMM reduction-order noise from the comparison).
    # Measured bitwise-identical on all of these shapes (the kernel's sigmoid
    # composition matches torch's float32 silu exactly), so no mismatch budget.
    gate = z[..., 0].float()
    up = z[..., 1].float()
    h_ref = (F.silu(gate) * up).bfloat16()
    _assert_quantized_pair(hq, hsf, hcq, hcsf, h_ref)


@_gpu
def test_swiglu_fwd_bitwise_exact_integers():
    torch.manual_seed(5)
    r, d, f, sizes = 512, 256, 128, [256, 256]
    args, z_ref = _make_a_inputs(r, d, f, sizes, "cuda", exact_int=True)
    z, hq, hsf, hcq, hcsf = torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(*args)
    assert torch.equal(
        _bytes(z.reshape(r, -1)), _bytes(z_ref.bfloat16().reshape(r, -1))
    ), "integer-exact z must be bitwise"


@_gpu
def test_swiglu_fwd_saturated_gate_bitwise():
    """gate == 128 makes silu exact in any implementation, so the fused dual
    quantization must match to_mx byte-for-byte, including special values."""
    torch.manual_seed(6)
    device = "cuda"
    r, d, f = 256, 128, 128
    sizes = [256]
    # x = identity blocks: row r selects weight column r % d.
    x_q = torch.zeros(r, d, dtype=torch.uint8, device=device)
    x_q[torch.arange(r), torch.arange(r) % d] = 0x38  # 1.0
    x_q = x_q.view(_E4M3)
    x_logical = torch.full((r, d // _BLOCK), 127, dtype=torch.uint8, device=device)
    x_sf = to_blocked(x_logical.view(_E8M0))

    # Up rows: random E4M3 bytes (finite lanes), then crafted special features.
    # Feature f's specials land in rowwise scale block f // 32 of every row.
    up_bytes = torch.randint(0, 0x7E, (f, d), dtype=torch.uint8, device=device)
    up_scales = torch.randint(
        120, 134, (f, d // _BLOCK), dtype=torch.uint8, device=device
    )
    up_bytes[0, :] = 0x7F  # NaN up -> h NaN in block 0
    up_bytes[1, :] = 0x00  # zero up
    up_bytes[40, :] = 0x7E  # 448 * 2^119: z_up ~ 2.98e38 (finite bf16);
    up_scales[40, :] = 127 + 119  # h = 128 * z_up overflows f32 -> +Inf, block 1
    w_q2f = torch.zeros(2 * f, d, dtype=torch.uint8, device=device)
    w_scale2f = torch.zeros(2 * f, d // _BLOCK, dtype=torch.uint8, device=device)
    w_q2f[0::2] = 0x38
    w_scale2f[0::2] = 127 + 7
    w_q2f[1::2] = up_bytes
    w_scale2f[1::2] = up_scales
    w_q = w_q2f.view(_E4M3)
    w_sf = to_blocked(w_scale2f.view(_E8M0))
    w_packed, w_sf_packed = _pack_grouped_weight([w_q], [w_sf], device)
    offsets = _mk_offsets(sizes, device)

    z, hq, hsf, hcq, hcsf = torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(
        x_q, x_sf, w_packed, w_sf_packed, offsets
    )
    gate = z[..., 0].float()
    up = z[..., 1].float()
    assert torch.equal(gate, torch.full_like(gate, 128.0)), "gate must be exactly 128"
    h_ref = (128.0 * up).bfloat16()  # silu(128) == 128 exactly
    _assert_quantized_pair(hq, hsf, hcq, hcsf, h_ref)  # zero mismatch budget
    # Special-value spot checks straight from the RCEIL table:
    hq_bytes = _bytes(hq).reshape(r, f)
    hsf_logical = _bytes(
        from_blocked(hsf.view(_E8M0).reshape(-1), r, f // _BLOCK)
    ).reshape(r, f // _BLOCK)
    nan_blocks = torch.isnan(h_ref).view(r, f // _BLOCK, _BLOCK).any(-1)
    inf_blocks = torch.isinf(h_ref).view(r, f // _BLOCK, _BLOCK).any(-1)
    assert nan_blocks.any() and inf_blocks.any(), "crafted specials must appear"
    assert (hsf_logical[nan_blocks | inf_blocks] == 0xFF).all()
    nonfinite_cols = (nan_blocks | inf_blocks).repeat_interleave(_BLOCK, dim=1)
    assert (hq_bytes[nonfinite_cols] == 0x7F).all()


@_gpu
def test_swiglu_fwd_tail_and_poison():
    torch.manual_seed(7)
    device = "cuda"
    r, d, f, sizes = 512, 256, 128, [128, 256]  # active 384, tail 128
    args, _ = _make_a_inputs(r, d, f, sizes, device)
    z, hq, hsf, hcq, hcsf = torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(*args)
    active = sum(sizes)
    assert (_bytes(z.reshape(r, -1))[active:] == 0).all(), "z tail must be zero bytes"
    assert (_bytes(hq)[active:] == 0).all()
    assert (_bytes(hcq.t())[:, active:] == 0).all()
    hsf_logical = _bytes(from_blocked(hsf.view(_E8M0).reshape(-1), r, f // _BLOCK))
    assert (hsf_logical.reshape(r, -1)[active:] == 0).all()
    hcsf_logical = _bytes(from_blocked(hcsf.view(_E8M0).reshape(-1), f, r // _BLOCK))
    assert (hcsf_logical.reshape(f, -1)[:, active // _BLOCK :] == 0).all()


# ---------------------------------------------------------------------------
# 7. Kernel B numerics
# ---------------------------------------------------------------------------


@_gpu
@pytest.mark.parametrize(
    "r,d,f,sizes",
    [
        (256, 256, 128, [128, 128]),
        (1024, 512, 256, [256, 0, 512, 128]),  # zero-token + strict tail
        (1536, 2048, 1408, [512, 128, 640, 256]),  # 16B class
    ],
    ids=["small", "ragged_tail", "16b"],
)
def test_dswiglu_bwd_random(r, d, f, sizes):
    torch.manual_seed(8)
    args, dh_ref = _make_b_inputs(r, d, f, sizes, "cuda")
    dzq, dzsf, dzcq, dzcsf = torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(*args)
    active = sum(sizes)
    z = args[4]
    dz_ref = _b_reference_dz(dh_ref.bfloat16(), z)
    got = _dequant_rowwise(dzq, dzsf, dtype=torch.float32)
    assert compute_error(dz_ref[:active].float(), got[:active]) >= 25.0
    got_col = _dequant_colwise(dzcq, dzcsf, dtype=torch.float32)
    assert compute_error(dz_ref[:active].float(), got_col[:active]) >= 25.0
    if active < r:
        assert (_bytes(dzq)[active:] == 0).all(), "dz tail must be zero bytes"
        assert (_bytes(dzcq.t())[:, active:] == 0).all()


@_gpu
def test_dswiglu_bwd_bitwise_exact():
    """Exact dh (integer GEMM) + saturated/zero gates: dz is exact products, so
    all four outputs must match to_mx byte-for-byte; interleave order checked."""
    torch.manual_seed(9)
    device = "cuda"
    r, d, f, sizes = 256, 128, 128, [256]
    # dh == 1.0 exactly: do = identity rows, w2 = all ones.
    do_q = torch.zeros(r, d, dtype=torch.uint8, device=device)
    do_q[torch.arange(r), torch.arange(r) % d] = 0x38
    do_q = do_q.view(_E4M3)
    do_sf = to_blocked(
        torch.full((r, d // _BLOCK), 127, dtype=torch.uint8, device=device).view(_E8M0)
    )
    w_q = torch.full((f, d), 0x38, dtype=torch.uint8, device=device).view(_E4M3)
    w_sf = to_blocked(
        torch.full((f, d // _BLOCK), 127, dtype=torch.uint8, device=device).view(_E8M0)
    )
    w_packed, w_sf_packed = _pack_grouped_weight([w_q], [w_sf], device)

    # Saturated gates make dsilu == 1 and silu == gate exactly; gate rows of z
    # also cover 0 (silu(0) == 0, dsilu(0) == 0.5 -- both exact).
    z = torch.zeros(r, f, 2, device=device, dtype=torch.bfloat16)
    z[..., 0] = 128.0
    z[: r // 2, :, 1] = torch.randn(r // 2, f, device=device).bfloat16()
    z[r // 2 :, :, 0] = 0.0  # gate 0 rows: dgate = 0.5 * up, dup = 0
    z[r // 2 :, :, 1] = (
        torch.randn(r - r // 2, f, device=device).bfloat16().float() * 2.0
    ).bfloat16()
    z[0, 0:4, 0] = float("nan")  # NaN gate+up block
    z[0, 0:4, 1] = float("nan")
    z[1, 0:16, 1] = 448.0  # uniform 448 dgate lanes
    offsets = _mk_offsets(sizes, device)

    dzq, dzsf, dzcq, dzcsf = torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
        do_q, do_sf, w_packed, w_sf_packed, z, offsets
    )
    dh = torch.ones(r, f, device=device, dtype=torch.bfloat16)
    dz_ref = _b_reference_dz(dh, z)
    # Interleave check on exact lanes (bitwise: NaN blocks are present).
    dgate_ref, dup_ref = _dswiglu_closed_form(
        dh.float(), z[..., 0].float(), z[..., 1].float()
    )
    assert torch.equal(_bytes(dz_ref[:, 0::2]), _bytes(dgate_ref.bfloat16()))
    assert torch.equal(_bytes(dz_ref[:, 1::2]), _bytes(dup_ref.bfloat16()))
    _assert_quantized_pair(dzq, dzsf, dzcq, dzcsf, dz_ref)  # zero budget


@_gpu
def test_dswiglu_bwd_semantic_blocks():
    """Drive the shared MXFP8 semantic contract through the fused backward.

    Uniform cases with |value| >= 128 are constructed exactly (gate = up =
    value under a saturated gate gives a uniform dz block); the rest are
    covered transitively: kernel bytes must equal to_mx bytes on blocks
    containing the case values, and to_mx itself is asserted against the
    shared table in test_to_mx_matches_semantic_table."""
    device = "cuda"
    cases = make_mxfp8_semantic_cases(torch.bfloat16, _RCEIL, device=device)
    n_cases = len(cases.names)
    r, d, f = 128, 128, max(128, _round_up(n_cases * 16, 128))
    do_q = torch.zeros(r, d, dtype=torch.uint8, device=device)
    do_q[torch.arange(r), torch.arange(r) % d] = 0x38
    do_q = do_q.view(_E4M3)
    do_sf = to_blocked(
        torch.full((r, d // _BLOCK), 127, dtype=torch.uint8, device=device).view(_E8M0)
    )
    w_q = torch.full((f, d), 0x38, dtype=torch.uint8, device=device).view(_E4M3)
    w_sf = to_blocked(
        torch.full((f, d // _BLOCK), 127, dtype=torch.uint8, device=device).view(_E8M0)
    )
    w_packed, w_sf_packed = _pack_grouped_weight([w_q], [w_sf], device)

    # Row 0: each case occupies 16 features -> one 32-wide dz block.
    z = torch.zeros(r, f, 2, device=device, dtype=torch.bfloat16)
    z[..., 0] = 128.0
    direct_table = []
    for idx in range(n_cases):
        vals = cases.inputs[idx].to(device)
        f0 = idx * 16
        uniform = bool((vals == vals[0]).all()) and not torch.isnan(vals).any()
        if uniform and abs(float(vals[0])) >= 128.0:
            # gate = up = v: dgate = v, dup = gate = |v|-signed... both = v
            # requires gate == v which needs v >= 128; negatives via up lane.
            v = float(vals[0])
            if v >= 128.0:
                z[0, f0 : f0 + 16, 0] = v
                z[0, f0 : f0 + 16, 1] = v
                direct_table.append((idx, None))
            else:
                z[0, f0 : f0 + 16, 0] = -v
                z[0, f0 : f0 + 16, 1] = v
                direct_table.append((idx, "even_only"))
        else:
            # Transitive: dgate lanes carry the case's even values, dup lanes
            # its odd values scaled through the saturated gate where possible;
            # fall back to plain interleave of the case into up lanes.
            z[0, f0 : f0 + 16, 1] = vals[0::2]
            z[0, f0 : f0 + 16, 0] = torch.where(
                torch.isfinite(vals[1::2]) & (vals[1::2].abs() >= 128),
                vals[1::2],
                torch.full_like(vals[1::2], 128.0),
            )
    offsets = _mk_offsets([r], device)
    dzq, dzsf, dzcq, dzcsf = torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
        do_q, do_sf, w_packed, w_sf_packed, z, offsets
    )
    dh = torch.ones(r, f, device=device, dtype=torch.bfloat16)
    dz_ref = _b_reference_dz(dh, z)
    _assert_quantized_pair(dzq, dzsf, dzcq, dzcsf, dz_ref)  # bitwise transitivity
    # Direct table assertions where the block is exactly the case input.
    row_sf_logical = _bytes(
        from_blocked(dzsf.view(_E8M0).reshape(-1), r, 2 * f // _BLOCK)
    ).reshape(r, -1)
    dz_bytes = _bytes(dzq).reshape(r, -1)
    for idx, mode in direct_table:
        blk = slice(idx * _BLOCK, (idx + 1) * _BLOCK)
        want_scale = int(cases.expected_scales[idx])
        want_data = cases.expected_data[idx].to(device)
        assert row_sf_logical[0, idx] == want_scale, cases.names[idx]
        got = dz_bytes[0, blk]
        if mode is None:
            assert torch.equal(got, want_data.to(got.device)), cases.names[idx]
        else:
            assert torch.equal(got[0::2], want_data.to(got.device)[0::2]), cases.names[
                idx
            ]


def _semantic_table_dtype_check(device):
    cases = make_mxfp8_semantic_cases(torch.bfloat16, _RCEIL, device=device)
    scale, q = to_mx(cases.inputs, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    assert torch.equal(
        _bytes(q).cpu().reshape(len(cases.names), 32), cases.expected_data
    )
    assert torch.equal(
        _bytes(scale.reshape(-1)).cpu(), cases.expected_scales.reshape(-1)
    )


@_gpu
def test_to_mx_matches_semantic_table():
    """Anchors the transitive comparisons above: to_mx == the shared table."""
    _semantic_table_dtype_check("cuda")


# ---------------------------------------------------------------------------
# 8. torch.compile
# ---------------------------------------------------------------------------


@_gpu
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_compile_matches_eager(op_name):
    torch.manual_seed(10)
    device = "cuda"
    if op_name == "mxfp8_grouped_gemm_wgrad":
        args = _make_c_inputs(256, 256, 128, [128, 128], device)
    elif op_name == "mxfp8_grouped_gemm_swiglu_fwd":
        args, _ = _make_a_inputs(256, 256, 128, [128, 128], device)
    else:
        args, _ = _make_b_inputs(256, 256, 128, [128, 128], device)
    op = getattr(torch.ops.torchao, op_name)
    eager = op(*args)
    compiled_fn = torch.compile(lambda *a: op(*a), fullgraph=True)
    compiled = compiled_fn(*args)
    eager = eager if isinstance(eager, tuple) else (eager,)
    compiled = compiled if isinstance(compiled, tuple) else (compiled,)
    for e, c in zip(eager, compiled):
        assert e.stride() == c.stride()
        assert torch.equal(_bytes(e.reshape(-1)), _bytes(c.reshape(-1)))


# ---------------------------------------------------------------------------
# 9. One physical launch per op (profiler evidence)
# ---------------------------------------------------------------------------


def _device_kernel_names(fn):
    fn()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for evt in prof.key_averages():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if "Memset" in evt.key or "Memcpy" in evt.key:
            continue
        names.extend([evt.key] * evt.count)
    return names


@_gpu
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_one_physical_launch(op_name):
    torch.manual_seed(11)
    device = "cuda"
    if op_name == "mxfp8_grouped_gemm_wgrad":
        args = _make_c_inputs(256, 256, 128, [128, 128], device)
    elif op_name == "mxfp8_grouped_gemm_swiglu_fwd":
        args, _ = _make_a_inputs(256, 256, 128, [128, 128], device)
    else:
        args, _ = _make_b_inputs(256, 256, 128, [128, 128], device)
    op = getattr(torch.ops.torchao, op_name)
    names = _device_kernel_names(lambda: op(*args))
    assert len(names) == 1, f"{op_name} must be ONE kernel launch, saw: {names}"
    banned = ("quantize", "silu", "elementwise", "vectorized")
    assert not any(b in names[0].lower() for b in banned), names


# ---------------------------------------------------------------------------
# 10. Large DSv3-class shapes (memory gated)
# ---------------------------------------------------------------------------


def _enough_memory(bytes_needed: int) -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory >= bytes_needed


@_gpu
@pytest.mark.skipif(not _enough_memory(16 << 30), reason="needs >= 16 GiB")
def test_wgrad_dsv3_671b_class():
    torch.manual_seed(12)
    sizes = [256, 0, 512, 256]
    args = _make_c_inputs(1024, 4096, 7168, sizes, "cuda")
    dw = torch.ops.torchao.mxfp8_grouped_gemm_wgrad(*args)
    ref = _ref_wgrad(args[0], args[1], args[2], args[3], sizes)
    assert compute_error(ref.float(), dw.float()) >= 24.0


@_gpu
@pytest.mark.skipif(not _enough_memory(16 << 30), reason="needs >= 16 GiB")
def test_swiglu_fwd_dsv3_671b_class():
    torch.manual_seed(13)
    args, z_ref = _make_a_inputs(1024, 7168, 2048, [256, 0, 512, 256], "cuda")
    z, hq, hsf, hcq, hcsf = torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(*args)
    active = 1024
    assert (
        compute_error(z_ref[:active].float(), z[:active].reshape(active, -1).float())
        >= 27.0
    )


# ---------------------------------------------------------------------------
# 11. Availability / unsupported environments
# ---------------------------------------------------------------------------


def test_wrapper_unavailable_raises_cleanly(monkeypatch):
    mod = pytest.importorskip("torchao.prototype.moe_training.mxfp8_grouped_mlp")
    flag = "_mxfp8_grouped_mlp_kernels_available"
    if not hasattr(mod, flag):
        pytest.skip("availability flag not exposed")
    monkeypatch.setattr(mod, flag, False)
    with pytest.raises(NotImplementedError):
        with FakeTensorMode():
            mod.mxfp8_grouped_gemm_wgrad(*_fake_c_inputs(256, 256, 128, 2))


def test_jagged_offs_generator_contract():
    """The test/bench offsets helper must produce 128-multiples when asked."""
    random.seed(0)
    if torch.cuda.is_available():
        offs = generate_jagged_offs(4, 1024, multiple_of=128)
    else:
        offs = generate_jagged_offs(4, 1024, multiple_of=128, device="cpu")
    sizes = torch.diff(offs.cpu(), prepend=torch.tensor([0], dtype=offs.dtype))
    assert (sizes % 128 == 0).all()
    assert offs[-1].item() == 1024
