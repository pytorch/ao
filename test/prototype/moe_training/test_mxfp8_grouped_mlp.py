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

Columnwise operands are accepted in ANY major (probe-derived).
Columnwise scale buffers are PER-GROUP blocked (each expert's block
``to_blocked``-ed independently, concatenated); whole-matrix blocking has the
same byte count and is silently wrong.
"""

import pytest
import torch
import torch.nn.functional as F

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


def _quant_colwise_grouped(x: torch.Tensor, sizes):
    """Ragged [R, K]: per-group colwise quantization, per-group blocked scales."""
    qs, sfs = [], []
    off = 0
    for m in sizes:
        q, sf = _quant_colwise(x[off : off + m])
        qs.append(q)
        sfs.append(sf.reshape(-1))
        off += m
    q = _cat8(qs, 0)
    q = q.t().contiguous().t()  # values identical; (1, R) strides
    return q, _cat8(sfs)


def _quant_weight_rowwise(w: torch.Tensor):
    """[G, N, K] quantized along K -> (contiguous stack, per-group blocked)."""
    s, q = to_mx(w, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, torch_to_blocked_per_group_3d(s.view(_E8M0)).reshape(-1)


def _quant_weight_colwise(w: torch.Tensor):
    """[G, N, K] quantized along N."""
    s_t, q_t = to_mx(
        w.transpose(-2, -1).contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL
    )
    q = q_t.transpose(-2, -1)
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


def _grouped_matmul(a_f32, b_f32_per_group, sizes, transpose_b: bool):
    """Per-group f32 matmul."""
    R = a_f32.shape[0]
    N = b_f32_per_group[0].shape[0 if transpose_b else 1]
    out = torch.zeros(R, N, dtype=torch.float32, device=a_f32.device)
    off = 0
    for g, m in enumerate(sizes):
        b = b_f32_per_group[g]
        bt = b.t() if transpose_b else b
        out[off : off + m] = a_f32[off : off + m] @ bt
        off += m
    return out


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
    c["sizes"], c["R"], c["F"] = sizes, R, hidden
    c["offsets"] = _mk_offsets(sizes, device)
    c["x"] = torch.randn(R, D, dtype=torch.bfloat16, device=device) * 0.5
    w13_elem = torch.randn(G, hidden, 2, D, dtype=torch.bfloat16, device=device) * 0.02
    c["w13"] = _to_32block(w13_elem)
    c["w2"] = torch.randn(G, D, hidden, dtype=torch.bfloat16, device=device) * 0.02
    c["dy"] = torch.randn(R, D, dtype=torch.bfloat16, device=device) * 0.5

    c["x_q"], c["x_sf"] = _quant_rowwise(c["x"])
    c["w13_q"], c["w13_sf"] = _quant_weight_rowwise(c["w13"])
    c["w2_q"], c["w2_sf"] = _quant_weight_rowwise(c["w2"])
    c["w2c_q"], c["w2c_sf"] = _quant_weight_colwise(c["w2"])
    c["dy_q"], c["dy_sf"] = _quant_rowwise(c["dy"])
    c["x_colq"], c["x_col_sf"] = _quant_colwise_grouped(c["x"], sizes)
    c["dy_colq"], c["dy_col_sf"] = _quant_colwise_grouped(c["dy"], sizes)
    return c


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
