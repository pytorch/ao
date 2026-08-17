# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark the fused MXFP8 grouped-MLP kernel family against the existing
decomposed torchao path and TransformerEngine, on identical inputs, offsets,
shapes and RCEIL scale mode.

Lanes (``--lane``):

* ``torchao`` -- the existing decomposed SM100 path: triton/CUDA quantizers +
  ``torch._scaled_grouped_mm`` + eager SwiGLU/dSwiGLU, staged to match each
  fused kernel's covered work. Uses no CuTe DSL code.
* ``ours``    -- the three fused ops ``torchao::mxfp8_grouped_gemm_swiglu_fwd``
  / ``_dswiglu_bwd`` / ``_wgrad`` (one kernel launch each).
* ``te``      -- TransformerEngine: the fused CuTe-DSL lane
  (``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1``; per-kernel times recovered from a
  profiler pass by kernel-name fragment) plus the modular lane's single-kernel
  ``tex.swiglu`` / ``tex.dswiglu`` gated-activation+dual-quantize points.
* ``all``     -- ``torchao`` + ``ours``.

The TE lane must run in a separate process from ``ours``: our kernels need the
public ``nvidia-cutlass-dsl`` 4.7.0 wheel on the user site, while TE's fused
lane uses the container-native cuDNN/cutlass stack, which that wheel shadows.
Invocation on the GB200 dev host::

    # ours / torchao lanes
    bash ./run_te.sh env PYTHONUSERBASE=/.local PYTHONPATH=/ao \\
        python /ao/benchmarks/prototype/moe_training/mxfp8/bench_grouped_mlp.py --lane all
    # TE lane (no PYTHONUSERBASE)
    bash ./run_te.sh env PYTHONPATH=/ao \\
        python /ao/benchmarks/prototype/moe_training/mxfp8/bench_grouped_mlp.py --lane te

Caveats stated up front so the numbers are read honestly:

* The decomposed wgrad stage includes its own dim1 (columnwise) quantization of
  both operands, because that is what the existing path pays; the fused wgrad
  consumes columnwise operands produced by kernels A/B. The A/B stages of both
  lanes start from identically prequantized GEMM inputs.
* The TE fused forward also applies per-token router probs in-kernel; we pass
  probs = 1 so the work matches.
* Eager launch timing (``do_bench`` median) only. No CUDA-graph replay column.
* Absolute microseconds from a clock-capped host (this dev box pins app clocks
  at 1200 MHz) are not publishable; the startup banner prints the clocks.

``MXFP8_BENCH_VALIDATE=1`` cross-checks the fused outputs against pure-torch
``to_mx``/``to_blocked`` references (SQNR gates; the checked-in test suite owns
the bitwise contracts).
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    grouped_mlp_ops,  # noqa: F401  (registers the three fused ops)
    mx_block_rearrange_2d_M_groups_cuda,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_to_mxfp8_dim0,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import (
    _to_mxfp8_dim1_kernel_wrapper,
    from_blocked,
    to_blocked,
)
from torchao.quantization.quantize_.common import KernelPreference

device = torch.device("cuda")
VALIDATE = os.environ.get("MXFP8_BENCH_VALIDATE", "0") == "1"
BLOCK = 32
RCEIL = ScaleCalculationMode.RCEIL


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    rows: int  # R: padded token rows (sum of per-expert rows)
    model_dim: int  # D
    hidden_dim: int  # F
    num_groups: int  # G (local experts)
    distribution: str  # "balanced" | "skewed"


@dataclass(frozen=True)
class ExperimentResult:
    # Per fused-kernel-equivalent stage, microseconds (median).
    a_us: float
    b_us: float
    c_fc1_us: float
    c_fc2_us: float
    seq_us: float
    # Derived TFLOP/s for the GEMM in each stage.
    a_tflops: float
    b_tflops: float
    c_fc1_tflops: float
    c_fc2_tflops: float


@dataclass(frozen=True)
class Experiment:
    lane: str
    config: ExperimentConfig
    result: ExperimentResult


def get_configs(args) -> List[ExperimentConfig]:
    if args.shape is not None:
        r, d, f, g = (int(v) for v in args.shape.split(","))
        return [ExperimentConfig(r, d, f, g, dist) for dist in args.dists]
    shapes = [
        # smoke
        (512, 256, 256, 2),
        # DeepSeekV3 16B class (D=2048, F=1408, G=8 local experts)
        (2048, 2048, 1408, 8),
        (8192, 2048, 1408, 8),
        (16384, 2048, 1408, 8),
        # DeepSeekV3 671B class (D=7168, F=2048, G=4 local experts)
        (2048, 7168, 2048, 4),
        (8192, 7168, 2048, 4),
        (16384, 7168, 2048, 4),
    ]
    return [
        ExperimentConfig(r, d, f, g, dist)
        for (r, d, f, g) in shapes
        for dist in args.dists
    ]


def make_offsets(cfg: ExperimentConfig) -> torch.Tensor:
    """Exclusive per-expert end offsets, every group a multiple of 128."""
    r, g = cfg.rows, cfg.num_groups
    if cfg.distribution == "balanced":
        per = r // g
        if per % 128 != 0 or per * g != r:
            raise ValueError(
                f"balanced distribution needs R/G to be a 128 multiple, got {r}/{g}"
            )
        return torch.arange(1, g + 1, device=device, dtype=torch.int32) * per
    return generate_jagged_offs(g, r, multiple_of=128, device=device)


# --------------------------------------------------------------------------
# Pure-torch quantization recipes (input prep + validation only, never timed)
# --------------------------------------------------------------------------


def ref_quantize_rowwise_1x32(x: torch.Tensor):
    """[M, K] -> (E4M3 [M, K] row-major, flat blocked E8M0 for [M, K/32])."""
    scale, q = to_mx(x, torch.float8_e4m3fn, BLOCK, scaling_mode=RCEIL)
    return q, to_blocked(scale)


def ref_quantize_colwise_32x1(x: torch.Tensor):
    """[R, N] -> (E4M3 [R, N] stride (1, R), flat blocked E8M0 for [N, R/32])."""
    scale_t, q_t = to_mx(
        x.t().contiguous(), torch.float8_e4m3fn, BLOCK, scaling_mode=RCEIL
    )
    return q_t.t(), to_blocked(scale_t)


def ref_dequant_colwise(q_col: torch.Tensor, sf_blocked: torch.Tensor):
    """FP32 dequant of a columnwise operand (for validation oracles)."""
    rows, cols = q_col.shape
    logical = from_blocked(sf_blocked, cols, rows // BLOCK)  # [N, R/32]
    scales = logical.t().to(torch.float32).repeat_interleave(BLOCK, dim=0)
    return q_col.to(torch.float32) * scales


def ref_dequant_rowwise(q_row: torch.Tensor, sf_blocked: torch.Tensor):
    """FP32 dequant of a rowwise 1x32-quantized operand (for validation oracles)."""
    rows, cols = q_row.shape
    scales = from_blocked(sf_blocked, rows, cols // BLOCK).to(torch.float32)
    return q_row.to(torch.float32) * scales.repeat_interleave(BLOCK, dim=1)


def sqnr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    err = (ref - actual).float().pow(2).mean()
    if err == 0:
        return float("inf")
    return (10 * torch.log10(ref.float().pow(2).mean() / err)).item()


# --------------------------------------------------------------------------
# Shared input bundle
# --------------------------------------------------------------------------


class Inputs:
    """All operands both non-TE lanes consume, prepared once per config.

    GEMM operands are prequantized identically for both lanes (kernels A and B
    take prequantized inputs by contract, and the decomposed path accepts
    prequantized MX operands at the same seam).
    """

    def __init__(self, cfg: ExperimentConfig):
        r, d, f, g = cfg.rows, cfg.model_dim, cfg.hidden_dim, cfg.num_groups
        torch.manual_seed(0)
        self.offsets = make_offsets(cfg)
        self.offsets_host = self.offsets.tolist()

        self.x = torch.randn(r, d, device=device, dtype=torch.bfloat16) / d**0.5
        self.do = torch.randn(r, d, device=device, dtype=torch.bfloat16) / d**0.5
        # Element-interleaved gate/up FC1 weight [G, 2F, D] and FC2-dgrad
        # weight-view source [G, F, D]; both quantized along D (the GEMM
        # contraction), then freely transposed into the K-major ABI layouts.
        self.w13i = (
            torch.randn(g, 2 * f, d, device=device, dtype=torch.bfloat16) / d**0.5
        )
        self.w2d = torch.randn(g, f, d, device=device, dtype=torch.bfloat16) / d**0.5

        self.x_q, self.x_sf = ref_quantize_rowwise_1x32(self.x)
        self.do_q, self.do_sf = ref_quantize_rowwise_1x32(self.do)

        w13_q, w13_sf = zip(
            *(ref_quantize_rowwise_1x32(self.w13i[i]) for i in range(g))
        )
        self.w13_t_q = torch.stack(list(w13_q)).transpose(-2, -1)  # [G, D, 2F]
        self.w13_t_sf = torch.stack(list(w13_sf))
        w2_q, w2_sf = zip(*(ref_quantize_rowwise_1x32(self.w2d[i]) for i in range(g)))
        self.w2_t_q = torch.stack(list(w2_q)).transpose(-2, -1)  # [G, D, F]
        self.w2_t_sf = torch.stack(list(w2_sf))

        # Reference forward intermediates (bf16), computed per expert from the
        # DEQUANTIZED operands — the values the fused kernels consume by
        # contract — so the validation SQNR isolates each kernel's own work
        # instead of stacking input-quantization error on top of it. The same
        # z then feeds kernel B and the decomposed dSwiGLU identically.
        x_f32 = ref_dequant_rowwise(self.x_q, self.x_sf)
        do_f32 = ref_dequant_rowwise(self.do_q, self.do_sf)
        z = torch.zeros(r, 2 * f, device=device, dtype=torch.bfloat16)
        prev = 0
        for i in range(g):
            end = self.offsets_host[i]
            if end > prev:
                w13_f32 = ref_dequant_rowwise(w13_q[i], w13_sf[i])
                z[prev:end] = (x_f32[prev:end] @ w13_f32.t()).to(torch.bfloat16)
            prev = end
        self.z_flat = z
        self.z_bf16 = z.view(r, f, 2)
        gate = self.z_bf16[..., 0].float()
        up = self.z_bf16[..., 1].float()
        self.h = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        sig = torch.sigmoid(gate)
        dh = torch.zeros(r, f, device=device, dtype=torch.bfloat16)
        prev = 0
        for i in range(g):
            end = self.offsets_host[i]
            if end > prev:
                # do [m, D] contracts with the [D, F] dgrad weight view; the
                # previous `do @ w2d[i]` only type-checked when D == F and
                # computed the transpose of the intended dgrad.
                w2_f32 = ref_dequant_rowwise(w2_q[i], w2_sf[i])
                dh[prev:end] = (do_f32[prev:end] @ w2_f32.t()).to(torch.bfloat16)
            prev = end
        dhf = dh.float()
        dgate = (dhf * up * (sig * (1.0 + gate * (1.0 - sig)))).to(torch.bfloat16)
        dup = (dhf * (gate * sig)).to(torch.bfloat16)
        self.dz_flat = torch.stack((dgate, dup), dim=-1).view(r, 2 * f)

        # Columnwise operands for the two wgrad calls (produced by A/B in the
        # fused regime, by standalone quantizers in the decomposed one).
        self.dz_col_q, self.dz_col_sf = ref_quantize_colwise_32x1(self.dz_flat)
        self.x_col_q, self.x_col_sf = ref_quantize_colwise_32x1(self.x)
        self.do_col_q, self.do_col_sf = ref_quantize_colwise_32x1(self.do)
        self.h_col_q, self.h_col_sf = ref_quantize_colwise_32x1(self.h)


def stage_flops(cfg: ExperimentConfig) -> Dict[str, float]:
    r, d, f = cfg.rows, cfg.model_dim, cfg.hidden_dim
    return {
        "a": 2.0 * r * d * 2 * f,
        "b": 2.0 * r * d * f,
        "c_fc1": 2.0 * r * 2 * f * d,
        "c_fc2": 2.0 * r * d * f,
    }


# --------------------------------------------------------------------------
# Lane: ours (the three fused ops)
# --------------------------------------------------------------------------


def lane_ours(cfg: ExperimentConfig, inp: Inputs) -> Dict[str, Callable]:
    ops = torch.ops.torchao

    def a():
        return ops.mxfp8_grouped_gemm_swiglu_fwd(
            inp.x_q, inp.x_sf, inp.w13_t_q, inp.w13_t_sf, inp.offsets
        )

    def b():
        return ops.mxfp8_grouped_gemm_dswiglu_bwd(
            inp.do_q, inp.do_sf, inp.w2_t_q, inp.w2_t_sf, inp.z_bf16, inp.offsets
        )

    def c_fc1():
        return ops.mxfp8_grouped_gemm_wgrad(
            inp.dz_col_q, inp.dz_col_sf, inp.x_col_q, inp.x_col_sf, inp.offsets
        )

    def c_fc2():
        return ops.mxfp8_grouped_gemm_wgrad(
            inp.do_col_q, inp.do_col_sf, inp.h_col_q, inp.h_col_sf, inp.offsets
        )

    def seq():
        _, _, _, h_col_q, h_col_sf = a()
        _, _, dz_col_q, dz_col_sf = b()
        ops.mxfp8_grouped_gemm_wgrad(
            dz_col_q, dz_col_sf, inp.x_col_q, inp.x_col_sf, inp.offsets
        )
        ops.mxfp8_grouped_gemm_wgrad(
            inp.do_col_q, inp.do_col_sf, h_col_q, h_col_sf, inp.offsets
        )

    return {"a": a, "b": b, "c_fc1": c_fc1, "c_fc2": c_fc2, "seq": seq}


def validate_ours(cfg: ExperimentConfig, inp: Inputs, stages) -> None:
    """SQNR cross-checks against pure-torch references. Bitwise contracts are
    owned by test_mxfp8_grouped_mlp.py; this is a sanity gate for benching."""
    z_k, h_row_q, h_row_sf, h_col_q, h_col_sf = stages["a"]()
    assert z_k.shape == inp.z_bf16.shape and z_k.stride() == inp.z_bf16.stride()
    s = sqnr(inp.z_flat.float(), z_k.reshape(cfg.rows, -1).float())
    assert s >= 27.0, f"A z SQNR {s:.1f} < 27"
    h_deq = ref_dequant_colwise(h_col_q, h_col_sf)
    s = sqnr(inp.h.float(), h_deq)
    assert s >= 27.0, f"A h (dequant colwise) SQNR {s:.1f} < 27"
    row_deq = h_row_q.float() * (
        from_blocked(h_row_sf, cfg.rows, cfg.hidden_dim // BLOCK)
        .to(torch.float32)
        .repeat_interleave(BLOCK, dim=1)
    )
    s = sqnr(inp.h.float(), row_deq)
    assert s >= 27.0, f"A h (dequant rowwise) SQNR {s:.1f} < 27"

    dz_row_q, dz_row_sf, dz_col_q, dz_col_sf = stages["b"]()
    dz_deq = ref_dequant_colwise(dz_col_q, dz_col_sf)
    s = sqnr(inp.dz_flat.float(), dz_deq)
    assert s >= 25.0, f"B dz SQNR {s:.1f} < 25"

    dw = stages["c_fc1"]()
    dy_f32 = ref_dequant_colwise(inp.dz_col_q, inp.dz_col_sf)
    x_f32 = ref_dequant_colwise(inp.x_col_q, inp.x_col_sf)
    prev = 0
    ref = torch.zeros_like(dw, dtype=torch.float32)
    for i in range(cfg.num_groups):
        end = inp.offsets_host[i]
        if end > prev:
            ref[i] = dy_f32[prev:end].t() @ x_f32[prev:end]
        prev = end
    s = sqnr(ref, dw.float())
    assert s >= 24.0, f"C dw SQNR {s:.1f} < 24"
    print("  validate(ours): OK (A z/h, B dz, C dw)")


# --------------------------------------------------------------------------
# Lane: torchao decomposed (existing path; no CuTe DSL)
# --------------------------------------------------------------------------


def lane_torchao(cfg: ExperimentConfig, inp: Inputs) -> Dict[str, Callable]:
    r, f = cfg.rows, cfg.hidden_dim
    offs = inp.offsets

    def dual_quantize(t: torch.Tensor):
        # Rowwise via the triton dim0 quantizer (the CUDA kernel is
        # colwise-only today), colwise via the CUDA quantizer, plus the two
        # scale rearranges the existing SM100 path performs.
        out_row, s_row = triton_to_mxfp8_dim0(t, BLOCK, "rceil")
        s_row_blocked = mx_block_rearrange_2d_M_groups_cuda(
            s_row.view(torch.uint8), offs
        )
        _, out_col, _, s_col = mxfp8_quantize_cuda(
            t, rowwise=False, colwise=True, scaling_mode="rceil"
        )
        s_col_blocked = triton_mx_block_rearrange_2d_K_groups(
            s_col.view(torch.uint8), offs // BLOCK
        )
        return out_row, s_row_blocked, out_col, s_col_blocked

    def a():
        # FC1 grouped GEMM (prequantized inputs) -> eager SwiGLU -> dual quant.
        z = torch._scaled_grouped_mm(
            inp.x_q,
            inp.w13_t_q,
            inp.x_sf.view(r, -1),
            inp.w13_t_sf.view(cfg.num_groups, -1),
            offs=offs,
            out_dtype=torch.bfloat16,
        )
        zv = z.view(r, f, 2)
        h = (torch.nn.functional.silu(zv[..., 0].float()) * zv[..., 1].float()).to(
            torch.bfloat16
        )
        return dual_quantize(h)

    def b():
        dh = torch._scaled_grouped_mm(
            inp.do_q,
            inp.w2_t_q,
            inp.do_sf.view(r, -1),
            inp.w2_t_sf.view(cfg.num_groups, -1),
            offs=offs,
            out_dtype=torch.bfloat16,
        )
        gate = inp.z_bf16[..., 0].float()
        up = inp.z_bf16[..., 1].float()
        sig = torch.sigmoid(gate)
        dhf = dh.float()
        dgate = (dhf * up * (sig * (1.0 + gate * (1.0 - sig)))).to(torch.bfloat16)
        dup = (dhf * (gate * sig)).to(torch.bfloat16)
        dz = torch.stack((dgate, dup), dim=-1).view(r, 2 * f)
        return dual_quantize(dz)

    def make_wgrad(dy: torch.Tensor, x: torch.Tensor):
        # Verbatim shape of the existing wgrad stage: CUDA dim1 quantization of
        # both operands + K-groups scale rearranges + scaled grouped GEMM.
        # (The fused kernel C instead consumes columnwise operands produced by
        # kernels A/B, so this stage's quantization cost is the decomposed
        # path's own.)
        def run():
            dy_t_mx = _to_mxfp8_dim1_kernel_wrapper(
                dy,
                BLOCK,
                elem_dtype=torch.float8_e4m3fn,
                hp_dtype=dy.dtype,
                kernel_preference=KernelPreference.AUTO,
                cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
                scale_calculation_mode=RCEIL,
            )
            x_t_mx = _to_mxfp8_dim1_kernel_wrapper(
                x,
                BLOCK,
                elem_dtype=torch.float8_e4m3fn,
                hp_dtype=x.dtype,
                kernel_preference=KernelPreference.AUTO,
                cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
                scale_calculation_mode=RCEIL,
            )
            scale_offs = offs // BLOCK
            dy_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
                dy_t_mx.scale, scale_offs
            )
            x_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
                x_t_mx.scale, scale_offs
            )
            return torch._scaled_grouped_mm(
                dy_t_mx.qdata,
                x_t_mx.qdata.transpose(-2, -1),
                dy_scales_blocked,
                x_scales_blocked,
                offs=offs,
                out_dtype=torch.bfloat16,
            )

        return run

    c_fc1 = make_wgrad(inp.dz_flat, inp.x)
    c_fc2 = make_wgrad(inp.do, inp.h)

    def seq():
        a()
        b()
        c_fc1()
        c_fc2()

    return {"a": a, "b": b, "c_fc1": c_fc1, "c_fc2": c_fc2, "seq": seq}


# --------------------------------------------------------------------------
# Lane: TransformerEngine
# --------------------------------------------------------------------------


def run_te_lane(cfg: ExperimentConfig) -> None:
    """TE fused lane (one CuTe kernel per stage) + modular-lane single-kernel
    gated-activation points. Prints its own tables; per-kernel CUDA times come
    from a profiler pass filtered by kernel-name fragment."""
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    r, d, f, g = cfg.rows, cfg.model_dim, cfg.hidden_dim, cfg.num_groups
    torch.manual_seed(0)
    offsets = make_offsets(cfg)
    sizes = torch.diff(offsets, prepend=torch.zeros(1, device=device).int())
    sizes = sizes.to(torch.int32)
    if (sizes % 128 != 0).any():
        raise ValueError(
            "TE fused lane hard-crashes the CUDA context on per-expert sizes "
            f"that are not multiples of 128, got {sizes.tolist()}"
        )
    recipe = MXFP8BlockScaling()

    # Fused lane: one Sequential MLP, per-kernel attribution by name fragment.
    fc1 = te.ops.GroupedLinear(
        g, d, 2 * f, bias=False, device="cuda", dtype=torch.bfloat16
    )
    act = te.ops.ScaledSwiGLU(glu_interleave_size=32)
    fc2 = te.ops.GroupedLinear(g, f, d, bias=False, device="cuda", dtype=torch.bfloat16)
    mlp = te.ops.Sequential(fc1, act, fc2)
    x = torch.randn(r, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.ones(r, device=device, dtype=torch.bfloat16)

    def fwd():
        with te.autocast(enabled=True, recipe=recipe):
            return mlp(x, sizes, probs, sizes)

    y = fwd()
    dy = torch.randn_like(y)

    def fwd_bwd():
        out = fwd()
        out.backward(dy)

    fwd_bwd()  # warmup / lazy init
    torch.cuda.synchronize()
    fwd_us = benchmark_cuda_function_in_microseconds(fwd)
    fwd_bwd_us = benchmark_cuda_function_in_microseconds(fwd_bwd)

    fragments = {
        "A analog (GroupedGemmGlu)": "GroupedGemmGlu",
        "B analog (GroupedGemmDglu)": "GroupedGemmDglu",
        "C analog (GroupedGemmWgrad)": "GroupedGemmWgrad",
        "plain GEMM (GroupedGemmQuant)": "GroupedGemmQuant",
    }
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(5):
            fwd_bwd()
    torch.cuda.synchronize()
    sums = dict.fromkeys(fragments, 0.0)
    counts = dict.fromkeys(fragments, 0)
    for evt in prof.key_averages():
        for label, frag in fragments.items():
            if frag in evt.key and "helper" not in evt.key:
                sums[label] += evt.self_device_time_total
                counts[label] += evt.count
    rows = [
        [label, counts[label] / 5.0, sums[label] / 5.0]
        for label in fragments
        if counts[label]
    ]
    print(f"\nTE fused lane (NVTE_CUTEDSL_FUSED_GROUPED_MLP=1)  {cfg}")
    print(f"  fwd wall: {fwd_us:.1f} us   fwd+bwd wall: {fwd_bwd_us:.1f} us")
    print(
        tabulate(
            rows,
            headers=["main kernel", "launches/iter", "device us/iter"],
            floatfmt=".1f",
        )
    )
    print(
        "  (fwd+bwd wall also covers FC2-fwd/FC1-dgrad GEMMs, input/dy "
        "quantize and offsets prep, matching a full MLP step)"
    )

    # Modular lane: the single fused gated-act+dual-quant kernels.
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    z = torch.randn(r, 2 * f, device=device, dtype=torch.bfloat16)
    dh = torch.randn(r, f, device=device, dtype=torch.bfloat16)
    tex.swiglu(z, q)
    tex.dswiglu(dh, z, q)
    swiglu_us = benchmark_cuda_function_in_microseconds(tex.swiglu, z, q)
    dswiglu_us = benchmark_cuda_function_in_microseconds(tex.dswiglu, dh, z, q)
    print(
        f"TE modular lane single kernels: tex.swiglu {swiglu_us:.1f} us, "
        f"tex.dswiglu {dswiglu_us:.1f} us (gated activation + dual MXFP8 "
        "quantize only; GEMMs are per-expert cuBLASLt in this lane)"
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_experiment(
    lane: str, cfg: ExperimentConfig, inp: Inputs
) -> Optional[ExperimentResult]:
    stages = (lane_ours if lane == "ours" else lane_torchao)(cfg, inp)
    if VALIDATE and lane == "ours":
        validate_ours(cfg, inp, stages)
    times: Dict[str, float] = {}
    for name, fn in stages.items():
        fn()  # warmup + lazy compile
        torch.cuda.synchronize()
        times[name] = benchmark_cuda_function_in_microseconds(fn)
    flops = stage_flops(cfg)
    return ExperimentResult(
        a_us=times["a"],
        b_us=times["b"],
        c_fc1_us=times["c_fc1"],
        c_fc2_us=times["c_fc2"],
        seq_us=times["seq"],
        a_tflops=flops["a"] / times["a"] / 1e6,
        b_tflops=flops["b"] / times["b"] / 1e6,
        c_fc1_tflops=flops["c_fc1"] / times["c_fc1"] / 1e6,
        c_fc2_tflops=flops["c_fc2"] / times["c_fc2"] / 1e6,
    )


def print_banner() -> None:
    props = torch.cuda.get_device_properties(device)
    clocks = os.popen(
        "nvidia-smi --query-gpu=clocks.applications.graphics,clocks.max.graphics "
        "--format=csv,noheader 2>/dev/null"
    ).read()
    try:
        import cutlass

        dsl = cutlass.__version__
    except Exception:
        dsl = "n/a"
    print(
        f"device: {props.name} (cc {props.major}.{props.minor}, index "
        f"{torch.cuda.current_device()}), torch {torch.__version__}, CUDA "
        f"{torch.version.cuda}, nvidia-cutlass-dsl {dsl}"
    )
    print(f"app clocks / max (per GPU):\n{clocks.strip()}")
    print(
        "NOTE: if app clocks are capped below max (e.g. 1200 MHz on the GB200 "
        "dev hosts), absolute microseconds are NOT publishable; use ratios."
    )


def print_results(experiments: List[Experiment]) -> None:
    headers = [
        "lane",
        "R",
        "D",
        "F",
        "G",
        "dist",
        "A us",
        "B us",
        "C_fc1 us",
        "C_fc2 us",
        "seq us",
        "A TF/s",
        "B TF/s",
        "C1 TF/s",
        "C2 TF/s",
    ]
    rows = []
    for e in experiments:
        c, r = e.config, e.result
        rows.append(
            [
                e.lane,
                c.rows,
                c.model_dim,
                c.hidden_dim,
                c.num_groups,
                c.distribution,
                f"{r.a_us:.1f}",
                f"{r.b_us:.1f}",
                f"{r.c_fc1_us:.1f}",
                f"{r.c_fc2_us:.1f}",
                f"{r.seq_us:.1f}",
                f"{r.a_tflops:.1f}",
                f"{r.b_tflops:.1f}",
                f"{r.c_fc1_tflops:.1f}",
                f"{r.c_fc2_tflops:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))
    print(
        "stage coverage: A = FC1 grouped GEMM + SwiGLU + dual MXFP8 quantize; "
        "B = FC2 dgrad + dSwiGLU + dual quantize; C_* = grouped wgrad "
        "(decomposed lane's C includes its own dim1 operand quantization); "
        "seq = A;B;C_fc1;C_fc2."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        choices=["torchao", "ours", "te", "all"],
        default="all",
        help="'all' = torchao + ours; 'te' must run in its own process "
        "(container-native stack, no PYTHONUSERBASE)",
    )
    parser.add_argument(
        "--shape",
        default=None,
        help="single shape as 'R,D,F,G' instead of the built-in sweep",
    )
    parser.add_argument(
        "--dist",
        choices=["balanced", "skewed", "both"],
        default="balanced",
        dest="dist",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="export a chrome trace of the fused-op sequence per config",
    )
    args = parser.parse_args()
    args.dists = ["balanced", "skewed"] if args.dist == "both" else [args.dist]

    print_banner()
    configs = get_configs(args)

    if args.lane == "te":
        for cfg in configs:
            run_te_lane(cfg)
        return

    lanes = ["torchao", "ours"] if args.lane == "all" else [args.lane]
    experiments: List[Experiment] = []
    for cfg in tqdm(configs):
        inp = Inputs(cfg)
        for lane in lanes:
            result = run_experiment(lane, cfg, inp)
            experiments.append(Experiment(lane, cfg, result))
        if args.profile and "ours" in lanes:
            from benchmarks.utils import profile_fn

            stages = lane_ours(cfg, inp)
            profile_fn(
                stages["seq"],
                profile_name=f"grouped_mlp_seq_R{cfg.rows}_D{cfg.model_dim}"
                f"_F{cfg.hidden_dim}_G{cfg.num_groups}",
            )
        del inp
        torch.cuda.empty_cache()
    print_results(experiments)


if __name__ == "__main__":
    main()
