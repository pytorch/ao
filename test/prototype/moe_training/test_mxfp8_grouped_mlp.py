# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MXFP8 fused grouped-MLP custom ops.

The four ops wrap the cudnn-frontend package's CuTe DSL grouped-GEMM kernels
(``cudnn.grouped_gemm_{glu,quant,dglu,wgrad}_wrapper_sm100``).

Every op is validated against a standalone plain-PyTorch reference (the
``_ref_*`` functions: dequantization, per-expert FP32 ``torch`` matmuls, and
``F.silu``/``torch.sigmoid`` only) under fixed ``min_sqnr`` gates; the gate
rationale and measured margins live next to the gate constants below.

The scale-byte-flip negative control uses a DERIVED gate instead: the
variability between two legitimate FP32 reduction orders of the same
reference (whole-K vs chunked-K) minus a 12 dB margin, capped at 60 dB --
measured 63-75 dB on GB200 at the debug shapes, so a single corrupted scale
byte must drag the op below a bar that legitimate scheduling never crosses.

Layout vocabulary (probe-derived): columnwise operands are accepted in ANY
major -- "rowmajor" here means un-transposed ``[R, N]`` row-major bytes (also
the layout the fwd/bwd ops emit for their columnwise outputs) and "native"
means the transposed-memory layout torchao's dim1 quantizers produce.
Columnwise scale buffers are PER-GROUP blocked (each expert's block
``to_blocked``-ed independently, concatenated); whole-matrix blocking has the
same byte count and is silently wrong -- a dedicated negative control asserts
the gap.
"""

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from torchao.utils import is_sm_version

# Exactly SM 10.0, matching the ops module's availability gate: the wrapped
# cudnn kernels are *_sm100-specific.
if not (torch.cuda.is_available() and is_sm_version(10, 0)):
    pytest.skip(
        "MXFP8 fused grouped MLP requires CUDA SM100",
        allow_module_level=True,
    )

try:
    from torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp import (
        _mxfp8_grouped_mlp_kernels_available,
        _mxfp8_grouped_mlp_unavailable_reason,
        is_supported,
        validate_group_offsets,
    )
except ImportError:
    pytest.skip(
        "installed torchao does not provide the cudnn_grouped_mlp module",
        allow_module_level=True,
    )

if not _mxfp8_grouped_mlp_kernels_available:
    pytest.skip(
        f"cudnn-frontend grouped-GEMM wrappers unavailable: "
        f"{_mxfp8_grouped_mlp_unavailable_reason}",
        allow_module_level=True,
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.kernels.mxfp8 import torch_to_blocked_per_group_3d
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import get_fp_scale, to_mx
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_BLOCK = 32
_RCEIL = ScaleCalculationMode.RCEIL

_OPS = torch.ops.torchao


# ---------------------------------------------------------------------------
# Pure-torch quantization / dequantization helpers.
# ---------------------------------------------------------------------------


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _quant_rowwise(x: torch.Tensor):
    """[M, K] -> (qdata [M, K] e4m3 row-major, flat blocked scales)."""
    s, q = to_mx(x, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, to_blocked(s.view(_E8M0)).view(_E8M0)


def _quant_colwise(x: torch.Tensor):
    """[M, K] quantized along M in 32-blocks, un-transposed row-major bytes.

    Scales: flat blocked of the transposed [K, M/32] scale matrix (one group).
    """
    M, K = x.shape
    if M == 0:
        return (
            torch.empty(0, K, dtype=_E4M3, device=x.device),
            torch.empty(0, dtype=_E8M0, device=x.device),
        )
    s_t, q_t = to_mx(x.t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q_t.t().contiguous(), to_blocked(s_t.view(_E8M0)).view(_E8M0)


def _cat8(ts, dim=0):
    dt = ts[0].dtype
    return torch.cat([t.view(torch.uint8) for t in ts], dim).view(dt)


def _quant_colwise_grouped(x: torch.Tensor, sizes, native: bool):
    """Ragged [R, K]: per-group colwise quantization, per-group blocked scales."""
    qs, sfs = [], []
    off = 0
    for m in sizes:
        q, sf = _quant_colwise(x[off : off + m])
        qs.append(q)
        sfs.append(sf.reshape(-1))
        off += m
    q = _cat8(qs, 0)
    if native:
        q = q.t().contiguous().t()  # values identical; (1, R) strides
    return q, _cat8(sfs)


def _quant_weight_rowwise(w: torch.Tensor):
    """[G, N, K] quantized along K -> (contiguous stack, per-group blocked)."""
    s, q = to_mx(w, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, torch_to_blocked_per_group_3d(s.view(_E8M0)).reshape(-1)


def _quant_weight_colwise(w: torch.Tensor, native: bool = False):
    """[G, N, K] quantized along N.

    native=False: contiguous row-major [G, N, K] bytes.
    native=True: the dim1-quantizer memory-transposed major -- [G, N, K]
    logical with per-group (1, N) strides (values identical).
    """
    s_t, q_t = to_mx(
        w.transpose(-2, -1).contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL
    )
    q = q_t.transpose(-2, -1)
    if not native:
        q = q.contiguous()
    return q, torch_to_blocked_per_group_3d(s_t.view(_E8M0)).reshape(-1)


def _dequant_rowwise(q: torch.Tensor, sf_flat: torch.Tensor):
    M, K = q.shape
    s = get_fp_scale(from_blocked(sf_flat.view(_E8M0), M, K // _BLOCK)).double()
    return (q.to(torch.float64) * s.repeat_interleave(_BLOCK, dim=1)).to(torch.float32)


def _dequant_colwise_grouped(q: torch.Tensor, sf_flat: torch.Tensor, sizes, K: int):
    """q [R, K] logical (any major), per-group blocked scales -> f32 [R, K]."""
    R = q.shape[0]
    out = torch.empty(R, K, dtype=torch.float32, device=q.device)
    off, soff = 0, 0
    for m in sizes:
        if m == 0:
            continue
        n = K * (m // _BLOCK)
        s_t = get_fp_scale(
            from_blocked(sf_flat[soff : soff + n].view(_E8M0), K, m // _BLOCK)
        ).double()
        block = q[off : off + m].to(torch.float64)
        out[off : off + m] = (block * s_t.t().repeat_interleave(_BLOCK, dim=0)).to(
            torch.float32
        )
        off += m
        soff += n
    return out


def _mk_offsets(sizes, device):
    return torch.cumsum(
        torch.tensor(sizes, dtype=torch.int64, device=device), dim=0
    ).to(torch.int32)


def _zsplit(z: torch.Tensor, hidden: int):
    """32-block interleaved [R, 2F] -> (gate [R, F], up [R, F])."""
    v = z.view(z.shape[0], hidden // _BLOCK, 2, _BLOCK)
    return (
        v[:, :, 0, :].reshape(z.shape[0], hidden),
        v[:, :, 1, :].reshape(z.shape[0], hidden),
    )


def _to_32block(w13_elem: torch.Tensor) -> torch.Tensor:
    """Element-interleaved [G, F, 2, D] -> 32-block GLU order [G, 2F, D]."""
    G, hidden, _, D = w13_elem.shape
    return (
        w13_elem.view(G, hidden // _BLOCK, _BLOCK, 2, D)
        .permute(0, 1, 3, 2, 4)
        .reshape(G, 2 * hidden, D)
        .contiguous()
    )


def _grouped_matmul(a_f32, b_f32_per_group, sizes, transpose_b: bool, chunks: int = 1):
    """Per-group f32 matmul with an optional chunked-K reduction order."""
    R = a_f32.shape[0]
    N = b_f32_per_group[0].shape[0 if transpose_b else 1]
    out = torch.zeros(R, N, dtype=torch.float32, device=a_f32.device)
    off = 0
    for g, m in enumerate(sizes):
        b = b_f32_per_group[g]
        bt = b.t() if transpose_b else b
        if chunks == 1:
            out[off : off + m] = a_f32[off : off + m] @ bt
        else:
            K = a_f32.shape[1]
            step = K // chunks
            acc = torch.zeros(m, N, dtype=torch.float32, device=a_f32.device)
            for c in range(chunks):
                lo, hi = c * step, K if c == chunks - 1 else (c + 1) * step
                acc += a_f32[off : off + m, lo:hi] @ bt[lo:hi]
            out[off : off + m] = acc
        off += m
    return out


def _refA_ref(a_f32, b_f32_per_group, sizes, transpose_b):
    """Reference GEMM + its derived exactness gate (reduction-order band - 12 dB)."""
    ref = _grouped_matmul(a_f32, b_f32_per_group, sizes, transpose_b)
    ref2 = _grouped_matmul(a_f32, b_f32_per_group, sizes, transpose_b, chunks=4)
    band = compute_error(ref.bfloat16(), ref2.bfloat16()).item()
    return ref, min(band - 12.0, 60.0)


def _dswiglu(dh, gate, up):
    s = torch.sigmoid(gate)
    return dh * up * (s * (1 + gate * (1 - s))), dh * F.silu(gate)


# ---------------------------------------------------------------------------
# Case construction: quantize everything once per case, with references.
# ---------------------------------------------------------------------------

_CASES = {
    # name: (D, F, sizes)
    "dbg_zero_token": (256, 256, [256, 0, 512, 256]),
    "dnef": (256, 384, [512, 256]),
    "g1": (256, 256, [512]),
    "16b": (2048, 1408, [256] * 8),
}


def _build_case(D, hidden, sizes, device="cuda", seed=0):
    torch.manual_seed(seed)
    G = len(sizes)
    R = sum(sizes)
    c = {}
    c["sizes"], c["G"], c["R"], c["D"], c["F"] = sizes, G, R, D, hidden
    c["offsets"] = _mk_offsets(sizes, device)
    c["x"] = torch.randn(R, D, dtype=torch.bfloat16, device=device) * 0.5
    w13_elem = torch.randn(G, hidden, 2, D, dtype=torch.bfloat16, device=device) * 0.02
    c["w13"] = _to_32block(w13_elem)
    c["w2"] = torch.randn(G, D, hidden, dtype=torch.bfloat16, device=device) * 0.02
    c["dy"] = torch.randn(R, D, dtype=torch.bfloat16, device=device) * 0.5

    c["x_q"], c["x_sf"] = _quant_rowwise(c["x"])
    c["w13_q"], c["w13_sf"] = _quant_weight_rowwise(c["w13"])
    c["w2_q"], c["w2_sf"] = _quant_weight_rowwise(c["w2"])
    c["w13c_q"], c["w13c_sf"] = _quant_weight_colwise(c["w13"])
    c["w2c_q"], c["w2c_sf"] = _quant_weight_colwise(c["w2"])
    c["dy_q"], c["dy_sf"] = _quant_rowwise(c["dy"])
    c["x_colq"], c["x_col_sf"] = _quant_colwise_grouped(c["x"], sizes, native=True)
    c["dy_colq"], c["dy_col_sf"] = _quant_colwise_grouped(c["dy"], sizes, native=True)

    c["x_deq"] = _dequant_rowwise(c["x_q"], c["x_sf"])
    c["w13_deq"] = [
        _dequant_rowwise(c["w13_q"][g], c["w13_sf"].view(G, -1)[g]) for g in range(G)
    ]
    c["z_ref"], c["z_refA_gate"] = _refA_ref(
        c["x_deq"], c["w13_deq"], sizes, transpose_b=True
    )
    return c


@pytest.fixture(scope="module")
def dbg():
    return _build_case(*_CASES["dbg_zero_token"])


# ---------------------------------------------------------------------------
# Registration / availability / fakes (no GPU launch).
# ---------------------------------------------------------------------------


def test_ops_registered():
    for name in (
        "mxfp8_grouped_gemm_swiglu_fwd_cudnn",
        "mxfp8_grouped_gemm_cudnn",
        "mxfp8_grouped_gemm_dswiglu_bwd_cudnn",
        "mxfp8_grouped_gemm_wgrad_cudnn",
    ):
        assert hasattr(_OPS, name), f"torchao::{name} is not registered"


def test_is_supported():
    assert is_supported(2048, 1408)
    assert is_supported(256, 256)
    assert not is_supported(192, 256)
    assert not is_supported(256, 64)
    assert not is_supported(0, 256)


def _fake_chain_shapes(R=512, D=256, hidden=256, G=2):
    N1 = 2 * hidden
    dev = "cuda"
    x_q = torch.empty(R, D, dtype=_E4M3, device=dev)
    x_sf = torch.empty(R * D // _BLOCK, dtype=_E8M0, device=dev)
    w13_q = torch.empty(G, N1, D, dtype=_E4M3, device=dev)
    w13_sf = torch.empty(G * N1 * D // _BLOCK, dtype=_E8M0, device=dev)
    offsets = torch.empty(G, dtype=torch.int32, device=dev)
    outs = {}
    outs["fwd"] = _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(
        x_q, x_sf, w13_q, w13_sf, offsets
    )
    w2_q = torch.empty(G, D, hidden, dtype=_E4M3, device=dev)
    w2_sf = torch.empty(G * D * hidden // _BLOCK, dtype=_E8M0, device=dev)
    outs["mm"] = _OPS.mxfp8_grouped_gemm_cudnn(
        outs["fwd"][1], outs["fwd"][2], w2_q, w2_sf, offsets
    )
    dy_q = torch.empty(R, D, dtype=_E4M3, device=dev)
    dy_sf = torch.empty(R * D // _BLOCK, dtype=_E8M0, device=dev)
    w2c_q = torch.empty(G, D, hidden, dtype=_E4M3, device=dev)
    w2c_sf = torch.empty(G * D * hidden // _BLOCK, dtype=_E8M0, device=dev)
    outs["bwd"] = _OPS.mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
        dy_q, dy_sf, w2c_q, w2c_sf, outs["fwd"][0], offsets
    )
    outs["wgrad"] = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
        torch.empty(R, D, dtype=_E4M3, device=dev),
        torch.empty(D * R // _BLOCK, dtype=_E8M0, device=dev),
        torch.empty(R, hidden, dtype=_E4M3, device=dev),
        torch.empty(
            ((hidden + 127) // 128 * 128) * R // _BLOCK, dtype=_E8M0, device=dev
        ),
        offsets,
    )
    return outs


def test_fake_contracts_match_specs():
    """All four fakes produce the documented shapes/dtypes/contiguity."""
    with FakeTensorMode():
        outs = _fake_chain_shapes()
    R, D, hidden, N1 = 512, 256, 256, 512
    z, hq, hsf, hcq, hcsf = outs["fwd"]
    assert tuple(z.shape) == (R, N1) and z.dtype == torch.bfloat16
    assert tuple(hq.shape) == (R, hidden) and hq.dtype == _E4M3
    assert hsf.numel() == R * hidden // _BLOCK and hsf.dtype == _E8M0
    assert tuple(hcq.shape) == (R, hidden) and hcq.dtype == _E4M3
    assert hcsf.numel() == hidden * R // _BLOCK
    assert all(t.is_contiguous() for t in outs["fwd"])
    y = outs["mm"]
    assert tuple(y.shape) == (R, D) and y.dtype == torch.bfloat16 and y.is_contiguous()
    dz_q, dz_sf, dzc_q, dzc_sf = outs["bwd"]
    assert tuple(dz_q.shape) == (R, N1) and dz_sf.numel() == R * N1 // _BLOCK
    assert tuple(dzc_q.shape) == (R, N1) and dzc_sf.numel() == N1 * R // _BLOCK
    dw = outs["wgrad"]
    assert tuple(dw.shape) == (2, D, hidden) and dw.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Per-op plain-PyTorch references: each op is validated in isolation against
# a reference built from nothing but dequantization, per-expert FP32 torch
# matmuls, and F.silu / torch.sigmoid.
# ---------------------------------------------------------------------------

# The op output is an FP32-accumulated GEMM over the same dequantized operands
# as the reference, so agreement is GEMM-exactness: measured 85-160 dB at
# these shapes, while layout/scale bugs land at 2-25 dB.
_MIN_SQNR_GEMM_EXACT = 50.0
# Requantized op outputs carry one extra MXFP8 quantization that the FP32
# reference does not; the requantization band alone measures ~30-40 dB here.
_MIN_SQNR_REQUANT = 27.0


def _assert_sqnr(ref, actual, min_db, label):
    sqnr = compute_error(ref, actual).item()
    assert sqnr >= min_db, f"{label} sqnr {sqnr} is too low, must be >= {min_db}"


def _sizes_from_offsets(offsets: torch.Tensor):
    ends = offsets.tolist()
    return [end - start for start, end in zip([0] + ends[:-1], ends)]


def _ref_grouped_gemm_swiglu_fwd(x_q, x_sf, w13_q, w13_sf, offsets):
    """Plain-PyTorch reference for mxfp8_grouped_gemm_swiglu_fwd_cudnn:
    dequantize both operands, per-expert FP32 matmul, split the 32-block
    interleaved gate/up halves, SwiGLU. Returns FP32 (z_ref, h_ref)."""
    sizes = _sizes_from_offsets(offsets)
    G, two_hidden, _ = w13_q.shape
    x_deq = _dequant_rowwise(x_q, x_sf)
    w13_deq = [_dequant_rowwise(w13_q[g], w13_sf.view(G, -1)[g]) for g in range(G)]
    z_ref = _grouped_matmul(x_deq, w13_deq, sizes, transpose_b=True)
    gate, up = _zsplit(z_ref, two_hidden // 2)
    return z_ref, F.silu(gate) * up


def _ref_grouped_gemm(a_q, a_sf, b_q, b_sf, offsets):
    """Plain-PyTorch reference for mxfp8_grouped_gemm_cudnn:
    out[r] = dequant(a[r]) @ dequant(b[g(r)]).T in FP32."""
    sizes = _sizes_from_offsets(offsets)
    G = b_q.shape[0]
    a_deq = _dequant_rowwise(a_q, a_sf)
    b_deq = [_dequant_rowwise(b_q[g], b_sf.view(G, -1)[g]) for g in range(G)]
    return _grouped_matmul(a_deq, b_deq, sizes, transpose_b=True)


def _ref_grouped_gemm_dswiglu_bwd(dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets):
    """Plain-PyTorch reference for mxfp8_grouped_gemm_dswiglu_bwd_cudnn:
    dh = dequant(dy) @ dequant(w2_col) per expert, then the closed-form
    dSwiGLU against z's gate/up halves, re-interleaved into the 32-block
    order. Returns FP32 dz_ref [R, 2F]."""
    sizes = _sizes_from_offsets(offsets)
    G, model_dim, hidden = w2_col_q.shape
    R = dy_q.shape[0]
    dy_deq = _dequant_rowwise(dy_q, dy_sf)
    w2_deq = [
        _dequant_colwise_grouped(
            w2_col_q[g], w2_col_sf.view(G, -1)[g], [model_dim], hidden
        )
        for g in range(G)
    ]
    dh = _grouped_matmul(dy_deq, w2_deq, sizes, transpose_b=False)
    gate, up = _zsplit(z_bf16.float(), hidden)
    dgate, dup = _dswiglu(dh, gate, up)
    dz_ref = torch.empty(R, 2 * hidden, dtype=torch.float32, device=dy_q.device)
    v = dz_ref.view(R, hidden // _BLOCK, 2, _BLOCK)
    v[:, :, 0, :] = dgate.view(R, hidden // _BLOCK, _BLOCK)
    v[:, :, 1, :] = dup.view(R, hidden // _BLOCK, _BLOCK)
    return dz_ref


def _ref_grouped_gemm_wgrad(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    """Plain-PyTorch reference for mxfp8_grouped_gemm_wgrad_cudnn:
    dw[g] = dequant(dy_g).T @ dequant(x_g) per expert in FP32."""
    sizes = _sizes_from_offsets(offsets)
    out_features, in_features = dy_col_q.shape[1], x_col_q.shape[1]
    dy_deq = _dequant_colwise_grouped(dy_col_q, dy_col_sf, sizes, out_features)
    x_deq = _dequant_colwise_grouped(x_col_q, x_col_sf, sizes, in_features)
    dw_ref = torch.zeros(
        len(sizes),
        out_features,
        in_features,
        dtype=torch.float32,
        device=dy_col_q.device,
    )
    off = 0
    for g, m in enumerate(sizes):
        dw_ref[g] = dy_deq[off : off + m].t() @ x_deq[off : off + m]
        off += m
    return dw_ref


@pytest.mark.parametrize("case", list(_CASES))
def test_mxfp8_grouped_gemm_swiglu_fwd_cudnn_matches_reference(case):
    c = _build_case(*_CASES[case])
    args = (c["x_q"], c["x_sf"], c["w13_q"], c["w13_sf"], c["offsets"])
    z, h_q, h_sf, h_colq, h_col_sf = _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(*args)
    z_ref, h_ref = _ref_grouped_gemm_swiglu_fwd(*args)
    _assert_sqnr(z_ref.bfloat16(), z, _MIN_SQNR_GEMM_EXACT, "z")
    for name, deq in (
        ("h_row", _dequant_rowwise(h_q, h_sf)),
        ("h_col", _dequant_colwise_grouped(h_colq, h_col_sf, c["sizes"], c["F"])),
    ):
        _assert_sqnr(h_ref, deq, _MIN_SQNR_REQUANT, name)


@pytest.mark.parametrize("case", list(_CASES))
def test_mxfp8_grouped_gemm_cudnn_matches_reference(case):
    c = _build_case(*_CASES[case])
    torch.manual_seed(11)
    a = torch.randn(c["R"], c["F"], dtype=torch.bfloat16, device="cuda") * 0.5
    a_q, a_sf = _quant_rowwise(a)
    out = _OPS.mxfp8_grouped_gemm_cudnn(a_q, a_sf, c["w2_q"], c["w2_sf"], c["offsets"])
    ref = _ref_grouped_gemm(a_q, a_sf, c["w2_q"], c["w2_sf"], c["offsets"])
    _assert_sqnr(ref.bfloat16(), out, _MIN_SQNR_GEMM_EXACT, "y")


@pytest.mark.parametrize("case", list(_CASES))
def test_mxfp8_grouped_gemm_dswiglu_bwd_cudnn_matches_reference(case):
    c = _build_case(*_CASES[case])
    torch.manual_seed(12)
    # Any BF16 [R, 2F] tensor is a valid z for isolated numerics: the kernel
    # computes dSwiGLU from whatever z it is given.
    z = torch.randn(c["R"], 2 * c["F"], dtype=torch.bfloat16, device="cuda")
    args = (c["dy_q"], c["dy_sf"], c["w2c_q"], c["w2c_sf"], z, c["offsets"])
    dz_q, dz_sf, dz_colq, dz_col_sf = _OPS.mxfp8_grouped_gemm_dswiglu_bwd_cudnn(*args)
    dz_ref = _ref_grouped_gemm_dswiglu_bwd(*args)
    for name, deq in (
        ("dz_row", _dequant_rowwise(dz_q, dz_sf)),
        (
            "dz_col",
            _dequant_colwise_grouped(dz_colq, dz_col_sf, c["sizes"], 2 * c["F"]),
        ),
    ):
        _assert_sqnr(dz_ref, deq, _MIN_SQNR_REQUANT, name)


@pytest.mark.parametrize("case", list(_CASES))
def test_mxfp8_grouped_gemm_wgrad_cudnn_matches_reference(case):
    c = _build_case(*_CASES[case])
    args = (c["dy_colq"], c["dy_col_sf"], c["x_colq"], c["x_col_sf"], c["offsets"])
    dw = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(*args)
    dw_ref = _ref_grouped_gemm_wgrad(*args)
    _assert_sqnr(dw_ref.bfloat16(), dw, _MIN_SQNR_GEMM_EXACT, "dw")


# ---------------------------------------------------------------------------
# The production 6-op chain runner, shared by the determinism, layout, and
# scale-mode tests.
# ---------------------------------------------------------------------------


def _run_chain(c):
    """fwd -> FC2 mm -> bwd -> FC1-dgrad mm -> wgrad x2, production layouts."""
    r = {}
    r["z"], r["h_q"], r["h_sf"], r["h_colq"], r["h_col_sf"] = (
        _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(
            c["x_q"], c["x_sf"], c["w13_q"], c["w13_sf"], c["offsets"]
        )
    )
    r["y"] = _OPS.mxfp8_grouped_gemm_cudnn(
        r["h_q"], r["h_sf"], c["w2_q"], c["w2_sf"], c["offsets"]
    )
    r["dz_q"], r["dz_sf"], r["dz_colq"], r["dz_col_sf"] = (
        _OPS.mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
            c["dy_q"], c["dy_sf"], c["w2c_q"], c["w2c_sf"], r["z"], c["offsets"]
        )
    )
    # FC1 dgrad: colwise weight cast enters the mm op TRANSPOSED into
    # [G, N=D, K=2F].
    r["dx"] = _OPS.mxfp8_grouped_gemm_cudnn(
        r["dz_q"],
        r["dz_sf"],
        c["w13c_q"].transpose(-2, -1),
        c["w13c_sf"],
        c["offsets"],
    )
    # Production wgrad layout mixes: native dy x kernel h; kernel dz x native x.
    r["dw2"] = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
        c["dy_colq"], c["dy_col_sf"], r["h_colq"], r["h_col_sf"], c["offsets"]
    )
    r["dw13"] = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
        r["dz_colq"], r["dz_col_sf"], c["x_colq"], c["x_col_sf"], c["offsets"]
    )
    return r


# ---------------------------------------------------------------------------
# Wgrad stride matrix: both operands in each major, all four combinations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_native", [False, True], ids=["aRM", "aNat"])
@pytest.mark.parametrize("b_native", [False, True], ids=["bRM", "bNat"])
def test_wgrad_stride_matrix(dbg, a_native, b_native):
    c = dbg
    sizes = c["sizes"]
    dy_q, dy_sf = _quant_colwise_grouped(c["dy"], sizes, native=a_native)
    x_q, x_sf = _quant_colwise_grouped(c["x"], sizes, native=b_native)
    dw = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(dy_q, dy_sf, x_q, x_sf, c["offsets"])
    ref = _ref_grouped_gemm_wgrad(dy_q, dy_sf, x_q, x_sf, c["offsets"])
    db = compute_error(ref.bfloat16(), dw).item()
    assert db >= 50.0, f"wgrad[{a_native=} {b_native=}] {db:.1f} dB < 50"


def test_native_weight_major_mm_bwd(dbg):
    """Ops 2 and 3 accept the production dim1-native (memory-transposed)
    colwise weight major. Both majors carry identical logical values, so each
    native arm must agree with the rowmajor arm far above any
    reduction-order band."""
    c = dbg
    r = _run_chain(c)
    # Op 3: dim1-native w2 colwise major.
    w2c_nat, w2c_nat_sf = _quant_weight_colwise(c["w2"], native=True)
    assert not w2c_nat.is_contiguous()
    assert torch.equal(_bytes(w2c_nat), _bytes(c["w2c_q"]))
    dz_q, dz_sf, _, _ = _OPS.mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
        c["dy_q"], c["dy_sf"], w2c_nat, w2c_nat_sf, r["z"], c["offsets"]
    )
    db = compute_error(
        _dequant_rowwise(r["dz_q"], r["dz_sf"]), _dequant_rowwise(dz_q, dz_sf)
    ).item()
    assert db >= 50.0, f"native-major w2 dz vs rowmajor arm: {db:.1f} dB < 50"
    # Op 2 (FC1 dgrad): dim1-native w13 colwise major, transposed into
    # [G, N=D, K=2F] exactly like the production call.
    w13c_nat, w13c_nat_sf = _quant_weight_colwise(c["w13"], native=True)
    dx_nat = _OPS.mxfp8_grouped_gemm_cudnn(
        r["dz_q"], r["dz_sf"], w13c_nat.transpose(-2, -1), w13c_nat_sf, c["offsets"]
    )
    db = compute_error(r["dx"].float(), dx_nat.float()).item()
    assert db >= 50.0, f"native-major w13 dx vs rowmajor arm: {db:.1f} dB < 50"


# ---------------------------------------------------------------------------
# Bitwise invariance: poisoned A < R tails and zero-token-group removal.
# ---------------------------------------------------------------------------


def _quant_and_run_chain(x, dy, w13, w2, sizes, r_alloc):
    """Quantize raw [A, D] activations and run the production 6-op chain with
    the activation buffers allocated at ``r_alloc`` rows. When ``r_alloc``
    exceeds the routed total the extra rows are POISONED: NaN/Inf activations
    (so the rowwise qdata tail is garbage), NaN-byte colwise qdata, and
    NaN-byte (0xFF) scale padding out to the R-sized colwise buffers the ops
    require -- none of which the kernels may read."""
    dev = x.device
    a_rows = sum(sizes)
    offsets = _mk_offsets(sizes, dev)
    x_colq, x_col_sf = _quant_colwise_grouped(x, sizes, native=True)
    dy_colq, dy_col_sf = _quant_colwise_grouped(dy, sizes, native=True)
    if r_alloc > a_rows:
        nan_rows = torch.full(
            (r_alloc - a_rows, x.shape[1]), float("nan"), dtype=x.dtype, device=dev
        )
        dy_tail = nan_rows.clone()
        dy_tail[::2] = float("inf")
        x = torch.cat([x, nan_rows])
        dy = torch.cat([dy, dy_tail])

        def _poison_cols(q):
            tail = torch.full(
                (r_alloc - a_rows, q.shape[1]), 0x7F, dtype=torch.uint8, device=dev
            ).view(_E4M3)
            return _cat8([q.contiguous(), tail], 0)

        def _sf_pad(sf, feats):
            pad = torch.full(
                (feats * (r_alloc - a_rows) // _BLOCK,),
                0xFF,
                dtype=torch.uint8,
                device=dev,
            ).view(_E8M0)
            return _cat8([sf, pad])

        x_colq, dy_colq = _poison_cols(x_colq), _poison_cols(dy_colq)
        x_col_sf = _sf_pad(x_col_sf, x.shape[1])
        dy_col_sf = _sf_pad(dy_col_sf, dy.shape[1])
    x_q, x_sf = _quant_rowwise(x)
    dy_q, dy_sf = _quant_rowwise(dy)
    w13_q, w13_sf = _quant_weight_rowwise(w13)
    w2_q, w2_sf = _quant_weight_rowwise(w2)
    w13c_q, w13c_sf = _quant_weight_colwise(w13)
    w2c_q, w2c_sf = _quant_weight_colwise(w2)
    r = {}
    r["z"], r["h_q"], r["h_sf"], r["h_colq"], r["h_col_sf"] = (
        _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(x_q, x_sf, w13_q, w13_sf, offsets)
    )
    r["y"] = _OPS.mxfp8_grouped_gemm_cudnn(r["h_q"], r["h_sf"], w2_q, w2_sf, offsets)
    r["dz_q"], r["dz_sf"], r["dz_colq"], r["dz_col_sf"] = (
        _OPS.mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
            dy_q, dy_sf, w2c_q, w2c_sf, r["z"], offsets
        )
    )
    r["dx"] = _OPS.mxfp8_grouped_gemm_cudnn(
        r["dz_q"], r["dz_sf"], w13c_q.transpose(-2, -1), w13c_sf, offsets
    )
    r["dw2"] = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
        dy_colq, dy_col_sf, r["h_colq"], r["h_col_sf"], offsets
    )
    r["dw13"] = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
        r["dz_colq"], r["dz_col_sf"], x_colq, x_col_sf, offsets
    )
    return r


def test_tail_a_lt_r_poisoned():
    """A < R allocation with a poisoned tail: every active-region output of
    the padded chain must be BITWISE identical to the same chain run at exact
    size. NaN/Inf activation rows, NaN-byte colwise qdata, and NaN-byte scale
    padding are planted past ``offsets[-1]``, so ANY tail read shows up as a
    byte difference against the finite exact-size run."""
    D = hidden = 256
    sizes = [256, 0, 512, 256]
    A, R = sum(sizes), 1280
    torch.manual_seed(3)
    dev = "cuda"
    x = torch.randn(A, D, dtype=torch.bfloat16, device=dev) * 0.5
    dy = torch.randn(A, D, dtype=torch.bfloat16, device=dev) * 0.5
    w13 = _to_32block(
        torch.randn(4, hidden, 2, D, dtype=torch.bfloat16, device=dev) * 0.02
    )
    w2 = torch.randn(4, D, hidden, dtype=torch.bfloat16, device=dev) * 0.02
    rp = _quant_and_run_chain(x, dy, w13, w2, sizes, R)
    re = _quant_and_run_chain(x, dy, w13, w2, sizes, A)
    for key in ("y", "dw2", "dw13"):
        assert re[key].float().isfinite().all(), f"exact-size {key} is not finite"
    for key in ("z", "h_q", "h_colq", "y", "dz_q", "dz_colq", "dx"):
        assert torch.equal(_bytes(rp[key][:A]), _bytes(re[key])), (
            f"{key}: active rows differ between the padded and exact-size runs"
        )
    for key, feat in (("h_sf", hidden), ("dz_sf", 2 * hidden)):
        padded = from_blocked(rp[key].view(_E8M0), R, feat // _BLOCK)[:A]
        exact = from_blocked(re[key].view(_E8M0), A, feat // _BLOCK)
        assert torch.equal(_bytes(padded), _bytes(exact)), f"{key} active rows differ"
    for key in ("h_col_sf", "dz_col_sf"):
        n = re[key].numel()
        assert torch.equal(_bytes(rp[key][:n]), _bytes(re[key])), f"{key} differs"
    for key in ("dw2", "dw13"):
        assert torch.equal(_bytes(rp[key]), _bytes(re[key])), f"{key} differs"
    # A routed-A-sized colwise scale buffer must be rejected up front: cudnn
    # validates the R-derived scale shape only on cold plan-building calls, so
    # accepting it would make the op's behavior depend on call history.
    with pytest.raises(ValueError, match="allocated row count"):
        _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
            torch.zeros(R, D, dtype=_E4M3, device=dev),
            torch.zeros(D * A // _BLOCK, dtype=_E8M0, device=dev),
            torch.zeros(R, hidden, dtype=_E4M3, device=dev),
            torch.zeros(hidden * R // _BLOCK, dtype=_E8M0, device=dev),
            _mk_offsets(sizes, dev),
        )


def test_zero_token_group_bitwise_inert():
    """A zero-token expert group is bitwise inert: dropping it from offsets
    and the weight stacks entirely ([256, 0, 512, 256] -> [256, 512, 256])
    leaves every output byte of every op unchanged for the surviving groups,
    and the dropped group's own weight gradients are written as exact
    zeros."""
    D = hidden = 256
    torch.manual_seed(5)
    dev = "cuda"
    A = 1024
    x = torch.randn(A, D, dtype=torch.bfloat16, device=dev) * 0.5
    dy = torch.randn(A, D, dtype=torch.bfloat16, device=dev) * 0.5
    w13 = _to_32block(
        torch.randn(4, hidden, 2, D, dtype=torch.bfloat16, device=dev) * 0.02
    )
    w2 = torch.randn(4, D, hidden, dtype=torch.bfloat16, device=dev) * 0.02
    keep = [0, 2, 3]
    r4 = _quant_and_run_chain(x, dy, w13, w2, [256, 0, 512, 256], A)
    r3 = _quant_and_run_chain(
        x, dy, w13[keep].contiguous(), w2[keep].contiguous(), [256, 512, 256], A
    )
    for key in r4:
        a = r4[key][keep] if key in ("dw2", "dw13") else r4[key]
        assert torch.equal(_bytes(a), _bytes(r3[key])), (
            f"{key} changed when the zero-token group was removed"
        )
    for key in ("dw2", "dw13"):
        assert (r4[key][1] == 0).all(), (
            f"zero-token expert {key} slice must be written as exact zeros"
        )


# ---------------------------------------------------------------------------
# Determinism and R == 0.
# ---------------------------------------------------------------------------


def test_determinism_all_ops_bitwise(dbg):
    c = dbg
    r1 = _run_chain(c)
    r2 = _run_chain(c)
    for key in r1:
        assert torch.equal(_bytes(r1[key]), _bytes(r2[key])), (
            f"{key} is not bitwise deterministic across identical launches"
        )


def test_r0_all_ops():
    dev = "cuda"
    D = hidden = 256
    offsets = torch.zeros(2, dtype=torch.int32, device=dev)
    args = _valid_fwd_args(R=0)
    args["offsets"] = offsets  # the builder's row offsets assume R=512
    z, h_q, h_sf, h_colq, h_col_sf = _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(**args)
    assert z.shape == (0, 2 * hidden) and h_q.shape == (0, hidden)
    assert h_sf.numel() == 0 and h_col_sf.numel() == 0
    a0 = torch.empty(0, D, dtype=_E4M3, device=dev)
    h0 = torch.empty(0, hidden, dtype=_E4M3, device=dev)
    sf0 = torch.empty(0, dtype=_E8M0, device=dev)
    w2_q = torch.zeros(2, D, hidden, dtype=_E4M3, device=dev)
    w2_sf = torch.empty(2 * D * hidden // _BLOCK, dtype=_E8M0, device=dev)
    y = _OPS.mxfp8_grouped_gemm_cudnn(h0, sf0, w2_q, w2_sf, offsets)
    assert y.shape == (0, D) and y.dtype == torch.bfloat16
    z0 = torch.empty(0, 2 * hidden, dtype=torch.bfloat16, device=dev)
    dz_q, dz_sf, dz_colq, dz_col_sf = _OPS.mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
        a0, sf0, w2_q, w2_sf, z0, offsets
    )
    assert dz_q.shape == (0, 2 * hidden) and dz_sf.numel() == 0
    assert dz_colq.shape == (0, 2 * hidden) and dz_col_sf.numel() == 0
    dw = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(a0, sf0, h0, sf0, offsets)
    assert dw.shape == (2, D, hidden) and (dw == 0).all()


# ---------------------------------------------------------------------------
# Negative controls: each sabotage must fail the numerics gates decisively.
# ---------------------------------------------------------------------------


def test_negative_control_whole_matrix_colwise_scales(dbg):
    """Whole-matrix to_blocked colwise scales: same bytes, silently wrong order."""
    c = dbg
    sizes = c["sizes"]
    dy_q, dy_sf_pg = _quant_colwise_grouped(c["dy"], sizes, native=False)
    x_q, x_sf_pg = _quant_colwise_grouped(c["x"], sizes, native=False)
    # Rebuild the SAME logical scales in whole-matrix blocked order.
    s_t, _ = to_mx(c["dy"].t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    dy_sf_wm = to_blocked(s_t.view(_E8M0)).view(_E8M0)
    assert dy_sf_wm.numel() == dy_sf_pg.numel()
    good = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
        dy_q, dy_sf_pg, x_q, x_sf_pg, c["offsets"]
    )
    bad = _OPS.mxfp8_grouped_gemm_wgrad_cudnn(
        dy_q, dy_sf_wm, x_q, x_sf_pg, c["offsets"]
    )
    ref = _ref_grouped_gemm_wgrad(dy_q, dy_sf_pg, x_q, x_sf_pg, c["offsets"])
    good_db = compute_error(ref.bfloat16(), good).item()
    bad_db = compute_error(ref.bfloat16(), bad).item()
    assert good_db >= 50.0
    assert bad_db < 25.0, (
        f"whole-matrix colwise scales scored {bad_db:.1f} dB -- the negative "
        f"control lost its teeth (good arm: {good_db:.1f})"
    )


def test_negative_control_gate_up_swap(dbg):
    """Swapping the gate/up 32-blocks must collapse h against the correct ref."""
    c = dbg
    hidden, G, D = c["F"], c["G"], c["D"]
    w13_sw = (
        c["w13"]
        .view(G, hidden // _BLOCK, 2, _BLOCK, D)
        .flip(2)
        .reshape(G, 2 * hidden, D)
        .contiguous()
    )
    w13_sw_q, w13_sw_sf = _quant_weight_rowwise(w13_sw)
    _, h_q, h_sf, _, _ = _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(
        c["x_q"], c["x_sf"], w13_sw_q, w13_sw_sf, c["offsets"]
    )
    gate_f, up_f = _zsplit(c["z_ref"], hidden)
    h_ref = F.silu(gate_f) * up_f
    _, h_q_good, h_sf_good, _, _ = _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(
        c["x_q"], c["x_sf"], c["w13_q"], c["w13_sf"], c["offsets"]
    )
    good_db = compute_error(h_ref, _dequant_rowwise(h_q_good, h_sf_good)).item()
    bad_db = compute_error(h_ref, _dequant_rowwise(h_q, h_sf)).item()
    assert bad_db < good_db - 10.0, (
        f"gate/up swap only moved h from {good_db:.1f} to {bad_db:.1f} dB -- "
        "the 32-block order convention is not actually being exercised"
    )


def test_negative_control_scale_byte_flip(dbg):
    """One +2-code E8M0 flip (x4) in the weight scales must break refA."""
    c = dbg
    sf_bad = c["w13_sf"].view(torch.uint8).clone()
    sf_bad[sf_bad.numel() // 2] += 2
    z_bad = _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(
        c["x_q"], c["x_sf"], c["w13_q"], sf_bad, c["offsets"]
    )[0]
    gate = c["z_refA_gate"]
    bad_db = compute_error(c["z_ref"].bfloat16(), z_bad).item()
    assert bad_db < gate, (
        f"single scale-byte flip still passes refA ({bad_db:.1f} >= {gate:.1f} dB)"
    )


def test_kernel_scale_mode_is_rceil(dbg):
    """The fwd op's h scale bytes must match RCEIL, and not FLOOR, quantization."""
    c = dbg
    r = _run_chain(c)
    gate_f, up_f = _zsplit(r["z"].float(), c["F"])
    h_ref = (F.silu(gate_f) * up_f).bfloat16()
    s_rceil, _ = to_mx(h_ref, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    s_floor, _ = to_mx(h_ref, _E4M3, _BLOCK, scaling_mode=ScaleCalculationMode.FLOOR)
    got = from_blocked(r["h_sf"].view(_E8M0), c["R"], c["F"] // _BLOCK).view(
        torch.uint8
    )
    rceil_frac = (got == s_rceil.view(torch.uint8)).float().mean().item()
    floor_frac = (got == s_floor.view(torch.uint8)).float().mean().item()
    assert rceil_frac > 0.98, f"h scales match RCEIL on only {rceil_frac:.3f}"
    assert rceil_frac > floor_frac + 0.1, (
        f"RCEIL ({rceil_frac:.3f}) does not dominate FLOOR ({floor_frac:.3f})"
    )


def test_zero_amax_requant_block():
    """An h 32-block that is EXACTLY zero (its up-half weight block zeroed)
    must requantize like ``to_mx(RCEIL)``: the kernel's scale byte matches the
    reference zero-block scale and the qdata dequantizes to exactly zero. The
    kernel preserves the sign of ``silu(gate) * (+-0)``, so the block's bytes
    are {0x00, 0x80} rather than all-zero. A NaN or junk scale here would
    poison the wgrad that consumes these scales."""
    D = hidden = 256
    sizes = [512, 512]
    A = sum(sizes)
    torch.manual_seed(7)
    dev = "cuda"
    x = torch.randn(A, D, dtype=torch.bfloat16, device=dev) * 0.5
    w13 = _to_32block(
        torch.randn(2, hidden, 2, D, dtype=torch.bfloat16, device=dev) * 0.02
    )
    blk = 1  # zero expert 0's up-half 32-block: rows [blk*64+32, blk*64+64)
    w13[0, blk * 64 + 32 : blk * 64 + 64] = 0
    x_q, x_sf = _quant_rowwise(x)
    w13_q, w13_sf = _quant_weight_rowwise(w13)
    _, h_q, h_sf, h_colq, h_col_sf = _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(
        x_q, x_sf, w13_q, w13_sf, _mk_offsets(sizes, dev)
    )
    rows, cols = slice(0, sizes[0]), slice(32 * blk, 32 * blk + 32)
    s_ref, _ = to_mx(
        torch.zeros(1, 32, dtype=torch.bfloat16, device=dev),
        _E4M3,
        _BLOCK,
        scaling_mode=_RCEIL,
    )
    ref_byte = s_ref.view(torch.uint8).item()
    for name, q in (("h_row", h_q[rows, cols]), ("h_col", h_colq[rows, cols])):
        codes = q.contiguous().view(torch.uint8).unique().tolist()
        assert set(codes) <= {0x00, 0x80}, f"{name} zero-block codes {codes}"
    row_scales = from_blocked(h_sf.view(_E8M0), A, hidden // _BLOCK)
    got = row_scales.view(torch.uint8)[rows, blk].unique().tolist()
    assert got == [ref_byte], f"rowwise zero-block scale {got} != RCEIL {ref_byte}"
    g0 = from_blocked(
        h_col_sf[: hidden * sizes[0] // _BLOCK].view(_E8M0), hidden, sizes[0] // _BLOCK
    )
    got = g0.view(torch.uint8)[cols].unique().tolist()
    assert got == [ref_byte], f"colwise zero-block scale {got} != RCEIL {ref_byte}"


# ---------------------------------------------------------------------------
# Validation: rejection matrix and the opt-in offsets path.
# ---------------------------------------------------------------------------


def _valid_fwd_args(device="cuda", R=512, D=256, hidden=256, G=2):
    N1 = 2 * hidden
    return dict(
        x_q=torch.zeros(R, D, dtype=_E4M3, device=device),
        x_sf=torch.zeros(R * D // _BLOCK, dtype=_E8M0, device=device),
        w13_q=torch.zeros(G, N1, D, dtype=_E4M3, device=device),
        w13_sf=torch.zeros(G * N1 * D // _BLOCK, dtype=_E8M0, device=device),
        offsets=torch.tensor([256, 512], dtype=torch.int32, device=device),
    )


_NEGATIVES = [
    ("x_q_dtype", lambda a: a.update(x_q=a["x_q"].view(torch.int8)), "float8_e4m3fn"),
    ("x_q_cpu", lambda a: a.update(x_q=a["x_q"].cpu()), "CUDA"),
    (
        "r_not_256",
        lambda a: a.update(
            x_q=torch.zeros(384, 256, dtype=_E4M3, device="cuda"),
            x_sf=torch.zeros(384 * 8, dtype=_E8M0, device="cuda"),
        ),
        "multiple of 256",
    ),
    (
        "d_192",
        lambda a: a.update(
            x_q=torch.zeros(512, 192, dtype=_E4M3, device="cuda"),
            x_sf=torch.zeros(512 * 6, dtype=_E8M0, device="cuda"),
            w13_q=torch.zeros(2, 512, 192, dtype=_E4M3, device="cuda"),
            w13_sf=torch.zeros(2 * 512 * 6, dtype=_E8M0, device="cuda"),
        ),
        "multiple of 128",
    ),
    (
        "offsets_i64",
        lambda a: a.update(offsets=a["offsets"].to(torch.int64)),
        "int32",
    ),
    (
        "offsets_wrong_len",
        lambda a: a.update(offsets=a["offsets"][:1]),
        "one entry per local expert",
    ),
    (
        "g0",
        lambda a: a.update(
            w13_q=torch.zeros(0, 512, 256, dtype=_E4M3, device="cuda"),
            w13_sf=torch.zeros(0, dtype=_E8M0, device="cuda"),
            offsets=torch.zeros(0, dtype=torch.int32, device="cuda"),
        ),
        "at least one expert group",
    ),
    (
        "x_sf_short",
        lambda a: a.update(x_sf=a["x_sf"][:-8].clone()),
        "blocked scale bytes",
    ),
    (
        "w13_stride",
        lambda a: a.update(
            w13_q=a["w13_q"].transpose(-2, -1).contiguous().transpose(-2, -1)
        ),
        "stride",
    ),
]


@pytest.mark.parametrize("case", _NEGATIVES, ids=[c[0] for c in _NEGATIVES])
def test_validation_negatives(case):
    _name, mutate, needle = case
    args = _valid_fwd_args()
    mutate(args)
    with pytest.raises(ValueError) as exc_info:
        _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(**args)
    assert needle.lower() in str(exc_info.value).lower(), (
        f"rejection message {str(exc_info.value)!r} does not name the defect "
        f"({needle!r})"
    )


def test_optin_offsets_validation(monkeypatch):
    """Offset VALUES are checked only under TORCHAO_MXFP8_VALIDATE_OFFSETS=1.

    The default-build non-rejection of a 128-row group is asserted on the
    validator directly: launching a kernel with misaligned offsets is the
    exact out-of-contract config the module documents as corrupting silently
    and nondeterministically, so this test must never perform that launch.
    The opt-in rejections DO go through the ops, which raise before any
    launch.
    """
    args = _valid_fwd_args()
    bad_offsets = torch.tensor([128, 512], dtype=torch.int32, device="cuda")

    # Default build: metadata-only, misaligned VALUES are not (and cannot be)
    # caught without a D2H sync.
    monkeypatch.delenv("TORCHAO_MXFP8_VALIDATE_OFFSETS", raising=False)
    validate_group_offsets(
        bad_offsets, num_groups=2, allocated_rows=512, device=bad_offsets.device
    )

    monkeypatch.setenv("TORCHAO_MXFP8_VALIDATE_OFFSETS", "1")
    bad = dict(args, offsets=bad_offsets)
    with pytest.raises(ValueError, match="FIX_PAD_SIZE"):
        _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(**bad)

    dec = dict(args, offsets=torch.tensor([512, 256], dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="nondecreasing"):
        _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(**dec)

    over = dict(
        args, offsets=torch.tensor([256, 768], dtype=torch.int32, device="cuda")
    )
    with pytest.raises(ValueError, match="exceeds the allocated row count"):
        _OPS.mxfp8_grouped_gemm_swiglu_fwd_cudnn(**over)

    # The opt-in check must not break fake tracing (no values to read).
    with FakeTensorMode():
        _fake_chain_shapes()
