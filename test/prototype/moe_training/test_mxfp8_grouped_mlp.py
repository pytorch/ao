# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MXFP8 fused grouped-MLP custom ops.

The four ops wrap the cudnn-frontend package's CuTe DSL grouped-GEMM kernels
(``cudnn.grouped_gemm_{glu,quant,dglu,wgrad}_wrapper_sm100``).

Every numerics gate is DERIVED at test time, never hard-coded:

* ``refA`` (GEMM-exactness) gates come from the variability between two
  legitimate evaluations of the same dequantized-operand reference that differ
  only in FP32 reduction order (whole-K vs chunked-K), minus a 12 dB margin,
  capped at 60 dB. Measured bands on GB200: 63-75 dB at the debug shapes
  (gates land at the 51-60 dB cap region); op outputs measure 85-160 dB.
* ``refB`` (independent-chain) gates come from the SQNR of a quantized-unfused
  evaluation against the exact FP32 chain computed from the ORIGINAL BF16
  inputs with no quantization at all, minus a 6 dB margin. This reference
  shares NO layout helper with the op inputs, so a self-consistent layout bug
  (wrong scale blocking built and decoded the same wrong way) cannot pass it.
  Measured band: ~30-40 dB at these shapes (pure MXFP8 requantization error).

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

if not (torch.cuda.is_available() and is_sm_version(10, 0)):
    pytest.skip(
        "MXFP8 fused grouped MLP requires CUDA SM100",
        allow_module_level=True,
    )

from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_ops import (
    _mxfp8_grouped_mlp_kernels_available,
    _mxfp8_grouped_mlp_unavailable_reason,
    is_supported,
)

if not _mxfp8_grouped_mlp_kernels_available:
    pytest.skip(
        f"cudnn-frontend grouped-GEMM wrappers unavailable: "
        f"{_mxfp8_grouped_mlp_unavailable_reason}",
        allow_module_level=True,
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx
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


def _e8m0_to_f64(s: torch.Tensor) -> torch.Tensor:
    u = s.view(torch.uint8).to(torch.int32)
    out = torch.exp2((u - 127).to(torch.float64))
    return torch.where(u == 255, torch.full_like(out, float("nan")), out)


def _quant_rowwise(x: torch.Tensor):
    """[M, K] -> (qdata [M, K] e4m3 row-major, flat blocked scales)."""
    s, q = to_mx(x, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, to_blocked(s.view(_E8M0)).view(_E8M0)


def _quant_colwise(x: torch.Tensor, native: bool):
    """[M, K] quantized along M in 32-blocks.

    native=False: un-transposed row-major [M, K] bytes ("rowmajor").
    native=True: the dim1-quantizer layout, [M, K] logical with (1, M) strides.
    Scales: flat blocked of the transposed [K, M/32] scale matrix (one group).
    """
    M, K = x.shape
    if M == 0:
        return (
            torch.empty(0, K, dtype=_E4M3, device=x.device),
            torch.empty(0, dtype=_E8M0, device=x.device),
        )
    s_t, q_t = to_mx(x.t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    q = q_t.t() if native else q_t.t().contiguous()
    return q, to_blocked(s_t.view(_E8M0)).view(_E8M0)


def _cat8(ts, dim=0):
    dt = ts[0].dtype
    return torch.cat([t.view(torch.uint8) for t in ts], dim).view(dt)


def _quant_colwise_grouped(x: torch.Tensor, sizes, native: bool):
    """Ragged [R, K]: per-group colwise quantization, per-group blocked scales."""
    qs, sfs = [], []
    off = 0
    for m in sizes:
        q, sf = _quant_colwise(x[off : off + m], native=False)
        qs.append(q)
        sfs.append(sf.reshape(-1))
        off += m
    q = _cat8(qs, 0)
    if native:
        q = q.t().contiguous().t()  # values identical; (1, R) strides
    return q, _cat8(sfs)


def _quant_weight_rowwise(w: torch.Tensor):
    """[G, N, K] quantized along K -> (contiguous stack, per-group blocked)."""
    qs, sfs = [], []
    for g in range(w.shape[0]):
        q, sf = _quant_rowwise(w[g])
        qs.append(q.view(torch.uint8))
        sfs.append(sf.reshape(-1))
    return torch.stack(qs).view(_E4M3), _cat8(sfs)


def _quant_weight_colwise(w: torch.Tensor):
    """[G, N, K] quantized along N (dim1-native strides per group)."""
    qs, sfs = [], []
    for g in range(w.shape[0]):
        q, sf = _quant_colwise(w[g], native=False)
        qs.append(q.view(torch.uint8))
        sfs.append(sf.reshape(-1))
    return torch.stack(qs).view(_E4M3), _cat8(sfs)


def _dequant_rowwise(q: torch.Tensor, sf_flat: torch.Tensor):
    M, K = q.shape
    s = _e8m0_to_f64(from_blocked(sf_flat.view(_E8M0), M, K // _BLOCK))
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
        s_t = _e8m0_to_f64(
            from_blocked(sf_flat[soff : soff + n].view(_E8M0), K, m // _BLOCK)
        )
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


def _refA_gate(ref_whole: torch.Tensor, ref_chunked: torch.Tensor) -> float:
    """GEMM-exactness gate from the reduction-order variability band - 12 dB."""
    band = compute_error(ref_whole.bfloat16(), ref_chunked.bfloat16()).item()
    return min(band - 12.0, 60.0)


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
    c["w2_deq"] = [
        _dequant_rowwise(c["w2_q"][g], c["w2_sf"].view(G, -1)[g]) for g in range(G)
    ]
    return c


@pytest.fixture(scope="module")
def dbg():
    return _build_case(*_CASES["dbg_zero_token"])


# ---------------------------------------------------------------------------
# Registration / availability / fakes (no GPU launch).
# ---------------------------------------------------------------------------


def test_ops_registered():
    for name in (
        "mxfp8_grouped_gemm_swiglu_fwd",
        "mxfp8_grouped_gemm",
        "mxfp8_grouped_gemm_dswiglu_bwd",
        "mxfp8_grouped_gemm_wgrad",
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
    outs["fwd"] = _OPS.mxfp8_grouped_gemm_swiglu_fwd(x_q, x_sf, w13_q, w13_sf, offsets)
    w2_q = torch.empty(G, D, hidden, dtype=_E4M3, device=dev)
    w2_sf = torch.empty(G * D * hidden // _BLOCK, dtype=_E8M0, device=dev)
    outs["mm"] = _OPS.mxfp8_grouped_gemm(
        outs["fwd"][1], outs["fwd"][2], w2_q, w2_sf, offsets
    )
    dy_q = torch.empty(R, D, dtype=_E4M3, device=dev)
    dy_sf = torch.empty(R * D // _BLOCK, dtype=_E8M0, device=dev)
    w2c_q = torch.empty(G, D, hidden, dtype=_E4M3, device=dev)
    w2c_sf = torch.empty(G * D * hidden // _BLOCK, dtype=_E8M0, device=dev)
    outs["bwd"] = _OPS.mxfp8_grouped_gemm_dswiglu_bwd(
        dy_q, dy_sf, w2c_q, w2c_sf, outs["fwd"][0], offsets
    )
    outs["wgrad"] = _OPS.mxfp8_grouped_gemm_wgrad(
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
# Full-chain numerics: two references per stage, derived gates.
# ---------------------------------------------------------------------------


def _run_chain(c):
    """fwd -> FC2 mm -> bwd -> FC1-dgrad mm -> wgrad x2, production layouts."""
    r = {}
    r["z"], r["h_q"], r["h_sf"], r["h_colq"], r["h_col_sf"] = (
        _OPS.mxfp8_grouped_gemm_swiglu_fwd(
            c["x_q"], c["x_sf"], c["w13_q"], c["w13_sf"], c["offsets"]
        )
    )
    r["y"] = _OPS.mxfp8_grouped_gemm(
        r["h_q"], r["h_sf"], c["w2_q"], c["w2_sf"], c["offsets"]
    )
    r["dz_q"], r["dz_sf"], r["dz_colq"], r["dz_col_sf"] = (
        _OPS.mxfp8_grouped_gemm_dswiglu_bwd(
            c["dy_q"], c["dy_sf"], c["w2c_q"], c["w2c_sf"], r["z"], c["offsets"]
        )
    )
    # FC1 dgrad: colwise weight cast enters the mm op TRANSPOSED into
    # [G, N=D, K=2F].
    r["dx"] = _OPS.mxfp8_grouped_gemm(
        r["dz_q"],
        r["dz_sf"],
        c["w13c_q"].transpose(-2, -1),
        c["w13c_sf"],
        c["offsets"],
    )
    # Production wgrad layout mixes: native dy x kernel h; kernel dz x native x.
    r["dw2"] = _OPS.mxfp8_grouped_gemm_wgrad(
        c["dy_colq"], c["dy_col_sf"], r["h_colq"], r["h_col_sf"], c["offsets"]
    )
    r["dw13"] = _OPS.mxfp8_grouped_gemm_wgrad(
        r["dz_colq"], r["dz_col_sf"], c["x_colq"], c["x_col_sf"], c["offsets"]
    )
    return r


@pytest.mark.parametrize("case", list(_CASES))
def test_chain_numerics(case):
    D, hidden, sizes = _CASES[case]
    c = _build_case(D, hidden, sizes)
    G, R = c["G"], c["R"]
    r = _run_chain(c)

    # --- z: refA (dequantized operands, two reduction orders)
    z_ref = _grouped_matmul(c["x_deq"], c["w13_deq"], sizes, transpose_b=True)
    z_ref2 = _grouped_matmul(
        c["x_deq"], c["w13_deq"], sizes, transpose_b=True, chunks=4
    )
    gate_a = _refA_gate(z_ref, z_ref2)
    z_db = compute_error(z_ref.bfloat16(), r["z"]).item()
    assert z_db >= gate_a, f"z {z_db:.1f} dB < derived refA gate {gate_a:.1f}"

    # --- z: refB (exact chain from ORIGINAL bf16 tensors; no quant helpers)
    z_exact = _grouped_matmul(
        c["x"].float(), [w.float() for w in c["w13"]], sizes, transpose_b=True
    )
    band_b = compute_error(z_exact, z_ref).item()  # quantization band
    z_db_b = compute_error(z_exact.bfloat16(), r["z"]).item()
    assert z_db_b >= band_b - 6.0, (
        f"z vs independent exact chain {z_db_b:.1f} dB < band {band_b:.1f} - 6"
    )

    # --- h (both quantized orientations) vs silu ref from the KERNEL's z
    gate_f, up_f = _zsplit(r["z"].float(), hidden)
    h_ref = F.silu(gate_f) * up_f
    band_h = compute_error(
        h_ref, _dequant_rowwise(*_quant_rowwise(h_ref.bfloat16()))
    ).item()
    h_deq = _dequant_rowwise(r["h_q"], r["h_sf"])
    h_db = compute_error(h_ref, h_deq).item()
    assert h_db >= band_h - 6.0, f"h {h_db:.1f} dB < requant band {band_h:.1f} - 6"
    h_col_deq = _dequant_colwise_grouped(r["h_colq"], r["h_col_sf"], sizes, hidden)
    h_col_db = compute_error(h_ref, h_col_deq).item()
    assert h_col_db >= band_h - 6.0, (
        f"h_col {h_col_db:.1f} dB < requant band {band_h:.1f} - 6 "
        "(a whole-matrix-vs-per-group scale layout bug lands at 2-5 dB)"
    )

    # --- y: refA from the op's own quantized h + refB independent chain
    w2_deq = c["w2_deq"]
    y_ref = _grouped_matmul(h_deq, w2_deq, sizes, transpose_b=True)
    y_ref2 = _grouped_matmul(h_deq, w2_deq, sizes, transpose_b=True, chunks=4)
    y_gate = _refA_gate(y_ref, y_ref2)
    y_db = compute_error(y_ref.bfloat16(), r["y"]).item()
    assert y_db >= y_gate, f"y {y_db:.1f} dB < derived refA gate {y_gate:.1f}"
    # refB for y: the whole forward computed from ORIGINAL bf16 tensors only.
    gate_x, up_x = _zsplit(z_exact, hidden)
    y_exact = _grouped_matmul(
        F.silu(gate_x) * up_x, [w.float() for w in c["w2"]], sizes, transpose_b=True
    )
    y_band_b = compute_error(y_exact, y_ref).item()
    y_db_b = compute_error(y_exact.bfloat16(), r["y"]).item()
    assert y_db_b >= y_band_b - 6.0, (
        f"y vs independent chain {y_db_b:.1f} dB < band {y_band_b:.1f} - 6"
    )

    # --- dz vs closed-form dSwiGLU from the kernel's z
    dy_deq = _dequant_rowwise(c["dy_q"], c["dy_sf"])
    w2c_deq = [
        _dequant_colwise_grouped(c["w2c_q"][g], c["w2c_sf"].view(G, -1)[g], [D], hidden)
        for g in range(G)
    ]
    dh_ref = _grouped_matmul(dy_deq, w2c_deq, sizes, transpose_b=False)
    dgate, dup = _dswiglu(dh_ref, gate_f, up_f)
    dz_ref = torch.empty(R, 2 * hidden, dtype=torch.float32, device="cuda")
    v = dz_ref.view(R, hidden // _BLOCK, 2, _BLOCK)
    v[:, :, 0, :] = dgate.view(R, hidden // _BLOCK, _BLOCK)
    v[:, :, 1, :] = dup.view(R, hidden // _BLOCK, _BLOCK)
    band_dz = compute_error(
        dz_ref, _dequant_rowwise(*_quant_rowwise(dz_ref.bfloat16()))
    ).item()
    dz_deq = _dequant_rowwise(r["dz_q"], r["dz_sf"])
    dz_db = compute_error(dz_ref, dz_deq).item()
    assert dz_db >= band_dz - 6.0, f"dz {dz_db:.1f} dB < band {band_dz:.1f} - 6"

    # --- dx refA
    w13c_deq = [
        _dequant_colwise_grouped(
            c["w13c_q"][g], c["w13c_sf"].view(G, -1)[g], [2 * hidden], D
        )
        for g in range(G)
    ]
    dx_ref = _grouped_matmul(dz_deq, w13c_deq, sizes, transpose_b=False)
    dx_ref2 = _grouped_matmul(dz_deq, w13c_deq, sizes, transpose_b=False, chunks=4)
    dx_gate = _refA_gate(dx_ref, dx_ref2)
    dx_db = compute_error(dx_ref.bfloat16(), r["dx"]).item()
    assert dx_db >= dx_gate, f"dx {dx_db:.1f} dB < derived refA gate {dx_gate:.1f}"

    # --- wgrads refA (production layout mixes)
    dy_col_deq = _dequant_colwise_grouped(c["dy_colq"], c["dy_col_sf"], sizes, D)
    dz_col_deq = _dequant_colwise_grouped(
        r["dz_colq"], r["dz_col_sf"], sizes, 2 * hidden
    )
    x_col_deq = _dequant_colwise_grouped(c["x_colq"], c["x_col_sf"], sizes, D)
    off = 0
    dw2_ref = torch.zeros(G, D, hidden, dtype=torch.float32, device="cuda")
    dw13_ref = torch.zeros(G, 2 * hidden, D, dtype=torch.float32, device="cuda")
    for g, m in enumerate(sizes):
        dw2_ref[g] = dy_col_deq[off : off + m].t() @ h_col_deq[off : off + m]
        dw13_ref[g] = dz_col_deq[off : off + m].t() @ x_col_deq[off : off + m]
        off += m
    dw2_db = compute_error(dw2_ref.bfloat16(), r["dw2"]).item()
    dw13_db = compute_error(dw13_ref.bfloat16(), r["dw13"]).item()
    assert dw2_db >= 50.0, f"dw2 {dw2_db:.1f} dB < 50 (probe level: 98-155)"
    assert dw13_db >= 50.0, f"dw13 {dw13_db:.1f} dB < 50 (probe level: 91-160)"

    # zero-token experts must come back written as exact zeros
    for g, m in enumerate(sizes):
        if m == 0:
            assert (r["dw2"][g] == 0).all() and (r["dw13"][g] == 0).all(), (
                f"zero-token expert {g} weight gradients must be exactly zero"
            )


# ---------------------------------------------------------------------------
# Wgrad stride matrix: both operands in each major, all four combinations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_native", [False, True], ids=["aRM", "aNat"])
@pytest.mark.parametrize("b_native", [False, True], ids=["bRM", "bNat"])
def test_wgrad_stride_matrix(dbg, a_native, b_native):
    c = dbg
    sizes, D = c["sizes"], c["D"]
    dy_q, dy_sf = _quant_colwise_grouped(c["dy"], sizes, native=a_native)
    x_q, x_sf = _quant_colwise_grouped(c["x"], sizes, native=b_native)
    dw = _OPS.mxfp8_grouped_gemm_wgrad(dy_q, dy_sf, x_q, x_sf, c["offsets"])
    dy_deq = _dequant_colwise_grouped(dy_q, dy_sf, sizes, D)
    x_deq = _dequant_colwise_grouped(x_q, x_sf, sizes, D)
    ref = torch.zeros(c["G"], D, D, dtype=torch.float32, device="cuda")
    off = 0
    for g, m in enumerate(sizes):
        ref[g] = dy_deq[off : off + m].t() @ x_deq[off : off + m]
        off += m
    db = compute_error(ref.bfloat16(), dw).item()
    assert db >= 50.0, f"wgrad[{a_native=} {b_native=}] {db:.1f} dB < 50"


# ---------------------------------------------------------------------------
# A < R strict tail with planted garbage.
# ---------------------------------------------------------------------------


def test_tail_a_lt_r_poisoned():
    D = hidden = 256
    sizes = [256, 0, 512, 256]
    A, R = sum(sizes), 1280
    torch.manual_seed(3)
    dev = "cuda"
    offsets = _mk_offsets(sizes, dev)
    x = torch.randn(R, D, dtype=torch.bfloat16, device=dev) * 0.5
    dy = torch.randn(R, D, dtype=torch.bfloat16, device=dev) * 0.5
    x[A:] = float("nan")
    dy[A::2] = float("inf")
    dy[A + 1 :: 2] = float("nan")
    w13 = _to_32block(
        torch.randn(4, hidden, 2, D, dtype=torch.bfloat16, device=dev) * 0.02
    )
    w2 = torch.randn(4, D, hidden, dtype=torch.bfloat16, device=dev) * 0.02

    x_q, x_sf = _quant_rowwise(x)
    dy_q, dy_sf = _quant_rowwise(dy)
    w13_q, w13_sf = _quant_weight_rowwise(w13)
    w2_q, w2_sf = _quant_weight_rowwise(w2)
    w2c_q, w2c_sf = _quant_weight_colwise(w2)

    z, h_q, h_sf, h_colq, h_col_sf = _OPS.mxfp8_grouped_gemm_swiglu_fwd(
        x_q, x_sf, w13_q, w13_sf, offsets
    )
    assert not z[:A].isnan().any(), "active z rows contaminated by the poisoned tail"
    y = _OPS.mxfp8_grouped_gemm(h_q, h_sf, w2_q, w2_sf, offsets)
    assert not y[:A].isnan().any(), "active y rows contaminated"
    dz_q, dz_sf, dz_colq, dz_col_sf = _OPS.mxfp8_grouped_gemm_dswiglu_bwd(
        dy_q, dy_sf, w2c_q, w2c_sf, z, offsets
    )
    w13c_q, w13c_sf = _quant_weight_colwise(w13)
    dx = _OPS.mxfp8_grouped_gemm(
        dz_q, dz_sf, w13c_q.transpose(-2, -1), w13c_sf, offsets
    )
    assert not dx[:A].isnan().any(), "active dx rows contaminated"

    # wgrad: colwise scales cover only the routed A rows; the qdata tail is
    # additionally poisoned with NaN bytes and must never be read.
    dy_colq, dy_col_sf = _quant_colwise_grouped(dy[:A], sizes, native=True)
    dy_colq_full = _cat8(
        [
            dy_colq.contiguous(),
            torch.full((R - A, D), 0x7F, dtype=torch.uint8, device=dev).view(_E4M3),
        ],
        0,
    )
    dw2 = _OPS.mxfp8_grouped_gemm_wgrad(
        dy_colq_full, dy_col_sf, h_colq, h_col_sf, offsets
    )
    assert not dw2.isnan().any(), "wgrad read the NaN-poisoned inactive tail"
    dy_col_deq = _dequant_colwise_grouped(dy_colq, dy_col_sf, sizes, D)
    h_col_deq = _dequant_colwise_grouped(h_colq[:A], h_col_sf, sizes, hidden)
    ref = torch.zeros(4, D, hidden, dtype=torch.float32, device=dev)
    off = 0
    for g, m in enumerate(sizes):
        ref[g] = dy_col_deq[off : off + m].t() @ h_col_deq[off : off + m]
        off += m
    db = compute_error(ref.bfloat16(), dw2).item()
    assert db >= 50.0, f"tail-poisoned dw2 {db:.1f} dB < 50"


# ---------------------------------------------------------------------------
# Determinism, compile, R == 0.
# ---------------------------------------------------------------------------


def test_determinism_all_ops_bitwise(dbg):
    c = dbg
    r1 = _run_chain(c)
    r2 = _run_chain(c)
    for key in r1:
        assert torch.equal(_bytes(r1[key]), _bytes(r2[key])), (
            f"{key} is not bitwise deterministic across identical launches"
        )


def test_compile_fullgraph_bitwise(dbg):
    c = dbg

    def fwd_then_mm(x_q, x_sf, w13_q, w13_sf, w2_q, w2_sf, offsets):
        z, h_q, h_sf, h_colq, h_col_sf = _OPS.mxfp8_grouped_gemm_swiglu_fwd(
            x_q, x_sf, w13_q, w13_sf, offsets
        )
        y = _OPS.mxfp8_grouped_gemm(h_q, h_sf, w2_q, w2_sf, offsets)
        return z, h_q, y

    eager = fwd_then_mm(
        c["x_q"],
        c["x_sf"],
        c["w13_q"],
        c["w13_sf"],
        c["w2_q"],
        c["w2_sf"],
        c["offsets"],
    )
    compiled = torch.compile(fwd_then_mm, fullgraph=True)(
        c["x_q"],
        c["x_sf"],
        c["w13_q"],
        c["w13_sf"],
        c["w2_q"],
        c["w2_sf"],
        c["offsets"],
    )
    for e, co, name in zip(eager, compiled, ("z", "h_q", "y")):
        assert torch.equal(_bytes(e), _bytes(co)), f"compiled {name} != eager"


def test_r0_all_ops():
    dev = "cuda"
    D = hidden = 256
    offsets = torch.zeros(2, dtype=torch.int32, device=dev)
    z, h_q, h_sf, h_colq, h_col_sf = _OPS.mxfp8_grouped_gemm_swiglu_fwd(
        torch.empty(0, D, dtype=_E4M3, device=dev),
        torch.empty(0, dtype=_E8M0, device=dev),
        torch.zeros(2, 2 * hidden, D, dtype=torch.uint8, device=dev).view(_E4M3),
        torch.empty(2 * 2 * hidden * D // _BLOCK, dtype=_E8M0, device=dev),
        offsets,
    )
    assert z.shape == (0, 2 * hidden) and h_q.shape == (0, hidden)
    assert h_sf.numel() == 0 and h_col_sf.numel() == 0
    dw = _OPS.mxfp8_grouped_gemm_wgrad(
        torch.empty(0, D, dtype=_E4M3, device=dev),
        torch.empty(0, dtype=_E8M0, device=dev),
        torch.empty(0, hidden, dtype=_E4M3, device=dev),
        torch.empty(0, dtype=_E8M0, device=dev),
        offsets,
    )
    assert dw.shape == (2, D, hidden) and (dw == 0).all()


# ---------------------------------------------------------------------------
# Negative controls: each sabotage must fail the numerics gates decisively.
# ---------------------------------------------------------------------------


def test_negative_control_whole_matrix_colwise_scales(dbg):
    """Whole-matrix to_blocked colwise scales: same bytes, silently wrong order."""
    c = dbg
    sizes, D, G = c["sizes"], c["D"], c["G"]
    dy_q, dy_sf_pg = _quant_colwise_grouped(c["dy"], sizes, native=False)
    x_q, x_sf_pg = _quant_colwise_grouped(c["x"], sizes, native=False)
    # Rebuild the SAME logical scales in whole-matrix blocked order.
    s_t, _ = to_mx(c["dy"].t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    dy_sf_wm = to_blocked(s_t.view(_E8M0)).view(_E8M0)
    assert dy_sf_wm.numel() == dy_sf_pg.numel()
    good = _OPS.mxfp8_grouped_gemm_wgrad(dy_q, dy_sf_pg, x_q, x_sf_pg, c["offsets"])
    bad = _OPS.mxfp8_grouped_gemm_wgrad(dy_q, dy_sf_wm, x_q, x_sf_pg, c["offsets"])
    dy_deq = _dequant_colwise_grouped(dy_q, dy_sf_pg, sizes, D)
    x_deq = _dequant_colwise_grouped(x_q, x_sf_pg, sizes, D)
    ref = torch.zeros(G, D, D, dtype=torch.float32, device="cuda")
    off = 0
    for g, m in enumerate(sizes):
        ref[g] = dy_deq[off : off + m].t() @ x_deq[off : off + m]
        off += m
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
    _, h_q, h_sf, _, _ = _OPS.mxfp8_grouped_gemm_swiglu_fwd(
        c["x_q"], c["x_sf"], w13_sw_q, w13_sw_sf, c["offsets"]
    )
    z_ref = _grouped_matmul(c["x_deq"], c["w13_deq"], c["sizes"], transpose_b=True)
    gate_f, up_f = _zsplit(z_ref, hidden)
    h_ref = F.silu(gate_f) * up_f
    good_db = compute_error(
        h_ref,
        _dequant_rowwise(
            *(
                _OPS.mxfp8_grouped_gemm_swiglu_fwd(
                    c["x_q"], c["x_sf"], c["w13_q"], c["w13_sf"], c["offsets"]
                )[1:3]
            )
        ),
    ).item()
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
    z_bad = _OPS.mxfp8_grouped_gemm_swiglu_fwd(
        c["x_q"], c["x_sf"], c["w13_q"], sf_bad, c["offsets"]
    )[0]
    z_ref = _grouped_matmul(c["x_deq"], c["w13_deq"], c["sizes"], transpose_b=True)
    z_ref2 = _grouped_matmul(
        c["x_deq"], c["w13_deq"], c["sizes"], transpose_b=True, chunks=4
    )
    gate = _refA_gate(z_ref, z_ref2)
    bad_db = compute_error(z_ref.bfloat16(), z_bad).item()
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
        _OPS.mxfp8_grouped_gemm_swiglu_fwd(**args)
    assert needle.lower() in str(exc_info.value).lower(), (
        f"rejection message {str(exc_info.value)!r} does not name the defect "
        f"({needle!r})"
    )


def test_optin_offsets_validation(monkeypatch):
    args = _valid_fwd_args()
    bad = dict(args)
    bad["offsets"] = torch.tensor([128, 512], dtype=torch.int32, device="cuda")

    # Default build: metadata-only, misaligned VALUES are not (and cannot be)
    # caught without a D2H sync.
    monkeypatch.delenv("TORCHAO_MXFP8_VALIDATE_OFFSETS", raising=False)
    _OPS.mxfp8_grouped_gemm_swiglu_fwd(**bad)

    monkeypatch.setenv("TORCHAO_MXFP8_VALIDATE_OFFSETS", "1")
    with pytest.raises(ValueError, match="FIX_PAD_SIZE"):
        _OPS.mxfp8_grouped_gemm_swiglu_fwd(**bad)

    over = dict(args)
    over["offsets"] = torch.tensor([256, 768], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="exceeds the allocated row count"):
        _OPS.mxfp8_grouped_gemm_swiglu_fwd(**over)

    # The opt-in check must not break fake tracing (no values to read).
    with FakeTensorMode():
        _fake_chain_shapes()
