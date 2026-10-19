# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark int1-int8 weight-only quantization on MPS with a real model.

Compares the new simdgroup-tiled GEMM kernel (intNgemm.metal) against the
upstream per-bitwidth GEMM kernels (intNmm_opt.metal) for prefill (M > 1),
and the new qmv kernel for decode (M == 1).  The kernel selection is
controlled by the TORCHAO_MPS_DISABLE_SIMDGROUP_GEMM environment variable:

  - Not set (default): use the new simdgroup-tiled GEMM for prefill
  - "1": use the upstream per-bitwidth GEMM kernels (no simdgroup MMA)

Both paths use the same qmv kernel for decode (M == 1).

The benchmark also demonstrates 8-bit support, which the upstream path
did not provide (upstream supported bitwidths 1-7 only).

Metrics: decode/prefill latency and top-1 next-token agreement with the
bf16 baseline. Results are checkpointed to /tmp for incremental runs.
"""

import argparse
import gc
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.experimental.quant_api import (
    UIntxChooseQParamsAlgorithm,
    UIntxWeightOnlyConfig,
)
from torchao.quantization import Int8WeightOnlyConfig, quantize_

MODEL_ID = "Qwen/Qwen3.5-4B"
BITWIDTHS = [2, 4, 6, 8]
GROUP_SIZES = [64, 128]
PREFILL_M_VALUES = [2048]
ALGORITHMS = ["min_max", "hqq", "torch", "wo"]

# Native PyTorch MPS int4 group sizes (macOS 15+)
NATIVE_INT4_GROUP_SIZES = [64, 128]

SEQ_LEN = 2048           # tokens per prefill window
RUNS = 10                # timed runs per measurement
WARMUP = 5               # warmup runs before timing
NUM_WINDOWS = RUNS + WARMUP  # one window per run (warmup + timed)
CHECKPOINT_FILE = "/tmp/benchmark_mps_checkpoint.json"
RESULTS_MD_FILE = "/tmp/results.md"


class Latency(NamedTuple):
    """A latency measurement: mean and 95% CI half-width in ms."""
    mean: float
    ci: float


class Result(NamedTuple):
    """Benchmark result for one config."""
    nbit: int
    gs: int
    dc: Latency
    pf: dict[int, Latency]  # M -> Latency
    top1: Latency           # top-1 next-token agreement with bf16 baseline (%)
    algo: UIntxChooseQParamsAlgorithm


class Baseline(NamedTuple):
    """bf16 baseline measurements."""
    dc: Latency
    pf: dict[int, Latency]  # M -> Latency
    top1_argmax: list[torch.Tensor]  # per-window argmax of baseline logits (CPU)


@dataclass
class Args:
    """Parsed CLI arguments."""
    model: str
    group_sizes: list[int]
    prefill_m: list[int]
    algorithms: list[str]
    bits: list[int]
    clear_cache: bool = False


# ---------------------------------------------------------------------------
# Checkpoint (save/load results for incremental runs)
# ---------------------------------------------------------------------------

def _result_to_dict(r: Result, category, nbit, gs, algorithm):
    """Serialize a Result to a JSON-safe dict."""
    if r is None:
        return None
    return {
        "category": category,
        "nbit": nbit,
        "gs": gs,
        "algorithm": algorithm.value if hasattr(algorithm, "value") else str(algorithm),
        "dc_mean": r.dc.mean,
        "dc_ci": r.dc.ci,
        "pf_results": {str(m): [v.mean, v.ci] for m, v in r.pf.items()},
        "top1_mean": r.top1.mean,
        "top1_ci": r.top1.ci,
    }


def _dict_to_result(d) -> Result:
    """Deserialize a checkpoint dict back to a Result."""
    algo = UIntxChooseQParamsAlgorithm(d["algorithm"])
    pf = {int(m): Latency(v[0], v[1]) for m, v in d["pf_results"].items()}
    # Backward compat: old checkpoints stored top1 as a scalar
    if isinstance(d.get("top1"), (int, float)):
        top1 = Latency(d["top1"], 0.0)
    else:
        top1 = Latency(d["top1_mean"], d["top1_ci"])
    return Result(
        nbit=d["nbit"],
        gs=d["gs"],
        dc=Latency(d["dc_mean"], d["dc_ci"]),
        pf=pf,
        top1=top1,
        algo=algo,
    )


def _load_checkpoint():
    """Load checkpointed results. Returns dict of saved config results."""
    if not os.path.exists(CHECKPOINT_FILE):
        return {}
    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)
    return data.get("results", {})


def _save_checkpoint(results_dict):
    """Save checkpointed results to disk."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"results": results_dict}, f, indent=2)


def _checkpoint_key(category, nbit, gs, algorithm):
    """Build a unique key for a config."""
    return f"{category}_{nbit}_{gs}_{algorithm}"


def _run_with_checkpoint(category, key, nbit, gs, algorithm, run_fn, ckpt_results):
    """Run a config, or return checkpointed result if available.

    Saves each result to disk immediately after completion.
    """
    if key in ckpt_results:
        print(
            f"  [checkpoint] {category} nbit={nbit} gs={gs} "
            f"algo={algorithm.value if hasattr(algorithm, 'value') else algorithm}",
            flush=True,
        )
        return _dict_to_result(ckpt_results[key])
    r = run_fn()
    if r is not None:
        ckpt_results[key] = _result_to_dict(r, category, nbit, gs, algorithm)
        _save_checkpoint(ckpt_results)
    return r


def _run_specs(category, specs, run_fn_factory, ckpt_results):
    """Run a list of ``(nbit, gs, algorithm)`` specs under one checkpoint category.

    ``run_fn_factory(nbit, gs, algorithm)`` returns the callable that executes
    the config. Results are returned in spec order; ``None`` indicates failure.
    """
    results = []
    for nbit, gs, algorithm in specs:
        key = _checkpoint_key(category, nbit, gs, algorithm.value)
        run_fn = run_fn_factory(nbit, gs, algorithm)
        results.append(
            _run_with_checkpoint(category, key, nbit, gs, algorithm, run_fn,
                                 ckpt_results)
        )
    return results


# ---------------------------------------------------------------------------
# Native PyTorch MPS int4 quantization
# ---------------------------------------------------------------------------

def _quantize_int4_native(weight, group_size):
    """Quantize a weight tensor for native PyTorch MPS int4 _weight_int4pack_mm.

    Returns (packed_weight, qScaleAndZeros) on MPS.
    Packing: pre-packed uint8 (2 int4 per byte, hi-nibble first).
    Dequant: w = (q - 8) * scale + zero, where zero = w_min + 8*scale.
    qScaleAndZeros: [K//gs, N, 2] with (scale, zero) in model dtype.

    The fp32 working copy (w.float()) is kept on CPU to avoid doubling MPS
    peak memory.  Only the small packed result is moved to MPS.
    _convert_weight_to_int4pack is MPS-only, so w_packed is moved to MPS
    just for that call.
    """
    model_dtype = weight.dtype
    N, K = weight.shape
    w = weight.cpu().float()
    qmin, qmax = 0, 15
    w_g = w.reshape(N, K // group_size, group_size)
    w_min = w_g.amin(dim=2, keepdim=True)
    w_max = w_g.amax(dim=2, keepdim=True)
    scales = (w_max - w_min) / (qmax - qmin)
    zeros = qmin - torch.round(w_min / scales)
    # Quantize in the grouped shape to avoid materializing full-shaped
    # scales_exp / zeros_exp (each ~4x the bf16 weight size in float32).
    w_q_g = torch.clamp(torch.round(w_g / scales + zeros), qmin, qmax).to(
        torch.uint8
    )
    w_q = w_q_g.reshape(N, K)
    del w_g, w, w_q_g
    # Keep small grouped tensors for qScaleAndZeros; free the rest.
    scale_flat = scales.reshape(N, K // group_size)
    w_min_flat = w_min.reshape(N, K // group_size)
    del w_min, w_max, scales, zeros

    # Pack: hi-nibble first
    w_packed = ((w_q[:, 0::2] << 4) | w_q[:, 1::2]).to(torch.uint8)
    del w_q
    # _convert_weight_to_int4pack is MPS-only — move just this small tensor.
    packed = torch.ops.aten._convert_weight_to_int4pack(w_packed.to("mps"), 8)

    zero_native = w_min_flat + 8 * scale_flat
    qScaleAndZeros = (
        torch.stack([scale_flat.T, zero_native.T], dim=2)
        .to("mps")
        .to(model_dtype)
    )
    return packed, qScaleAndZeros


class _NativeQuantizedLinear(torch.nn.Module):
    """Base for native PyTorch MPS int4/int8 Linear wrappers.

    Subclasses set ``in_features``/``out_features``, register their packed
    parameters, call ``_setup_bias``, and implement ``_mm``.
    """

    def _mm(self, x_2d):
        raise NotImplementedError

    def _setup_bias(self, orig_linear):
        self.has_bias = orig_linear.bias is not None
        if self.has_bias:
            self.bias = torch.nn.Parameter(
                orig_linear.bias.data.clone(), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        y = self._mm(x_2d)
        if self.has_bias:
            y = y + self.bias
        return y.reshape(*orig_shape[:-1], self.out_features)


class _NativeInt4Linear(_NativeQuantizedLinear):
    """nn.Linear wrapper using native PyTorch MPS int4 _weight_int4pack_mm."""

    def __init__(self, orig_linear, group_size):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.group_size = group_size
        packed, qSZ = _quantize_int4_native(orig_linear.weight.data, group_size)
        self.packed_weight = torch.nn.Parameter(packed, requires_grad=False)
        self.qScaleAndZeros = torch.nn.Parameter(qSZ, requires_grad=False)
        self._setup_bias(orig_linear)

    def _mm(self, x_2d):
        return torch.ops.aten._weight_int4pack_mm(
            x_2d, self.packed_weight, self.group_size, self.qScaleAndZeros
        )


def _replace_linears(model, wrapper_cls, *args):
    """Recursively replace all nn.Linear modules with ``wrapper_cls(orig, *args)``."""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, wrapper_cls(module, *args))
        else:
            _replace_linears(module, wrapper_cls, *args)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

_UL = "\033[4m"
_UL_R = "\033[0m"


def _ul(s: str, width: int = 0) -> str:
    """Underline ``s``, right-padded to ``width``. Only the text is underlined,
    not the padding."""
    underlined = f"{_UL}{s}{_UL_R}"
    if width > 0:
        return " " * max(0, width - len(s)) + underlined
    return underlined


def _mean_ci(times):
    """Return Latency(mean, 95% CI half-width) in ms."""
    n = len(times)
    mean = sum(times) / n
    if n < 2:
        return Latency(mean, 0.0)
    var = sum((t - mean) ** 2 for t in times) / (n - 1)
    se = math.sqrt(var / n)
    # t critical values for 95% CI, common sample sizes
    t_vals = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145,
        20: 2.093, 30: 2.045,
    }
    t = t_vals.get(n, 1.96)
    return Latency(mean, t * se)


def _measure_latency(fn, runs=RUNS, warmup=WARMUP):
    """Run ``fn`` under torch.no_grad with warmup + timed loop. Returns Latency."""
    for _ in range(warmup):
        with torch.no_grad():
            out = fn()
        del out
    torch.mps.synchronize()
    times = []
    for _ in range(runs):
        torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = fn()
        torch.mps.synchronize()
        times.append((time.time() - t0) * 1000)
        del out
    return _mean_ci(times)


def measure_decode_latency(model, input_ids):
    """Measure single-token decode latency on MPS. Returns Latency in ms."""
    x = input_ids[:, :1]
    return _measure_latency(lambda: model(x))


def measure_prefill_latency_multi(model, windows_mps, prefill_m_values):
    """Measure prefill latency across multiple windows. Returns dict[M -> Latency].

    Each window is a separate [1, SEQ_LEN] input_ids tensor. We run
    WARMUP + RUNS windows (one per run), timing the prefill.
    """
    pf = {}
    for m in prefill_m_values:
        lats = []
        for i in range(WARMUP + RUNS):
            w = windows_mps[i % len(windows_mps)][:, :m]
            torch.mps.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = model(w)
            torch.mps.synchronize()
            if i >= WARMUP:
                lats.append((time.time() - t0) * 1000)
            del out
        pf[m] = _mean_ci(lats)
    return pf


def measure_top1_multi(model, windows_mps, baseline_argmax_list):
    """Measure top-1 agreement across all windows. Returns Latency.

    Runs one forward pass per window (all NUM_WINDOWS), captures argmax
    and compares to baseline. Top-1 is deterministic per window, so all
    windows contribute (no warmup discard). No logits are retained.
    """
    top1_scores = []
    for i in range(len(windows_mps)):
        w = windows_mps[i]
        with torch.no_grad():
            logits = model(w).logits
        pred = logits[:, :-1, :].argmax(dim=-1)
        base = baseline_argmax_list[i]
        agreement = (pred == base.to(pred.device)).float().mean().item()
        top1_scores.append(agreement * 100)
        del logits, pred
    return _mean_ci(top1_scores)


def measure_prefill_with_top1(model, windows_mps, baseline_argmax_list, prefill_m_values):
    """Measure prefill latency and top-1 agreement across multiple windows.

    Convenience wrapper: runs prefill latency, then top-1 (which does its
    own forward passes). Returns (pf_latencies, top1_latency).
    """
    pf = measure_prefill_latency_multi(model, windows_mps, prefill_m_values)
    top1 = measure_top1_multi(model, windows_mps, baseline_argmax_list)
    return pf, top1


def cleanup():
    """Force garbage collection and MPS cache clearing."""
    gc.collect()
    torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Quantization functions
# ---------------------------------------------------------------------------

def _quantize_uintx(nbit, gs, algorithm):
    """Return a quantize_fn(model) using the UIntxWeightOnlyConfig path."""
    def fn(model):
        quantize_(
            model,
            UIntxWeightOnlyConfig(
                bitwidth=nbit, group_size=gs,
                uintx_choose_qparams_algorithm=algorithm,
            ),
        )
    return fn


def _quantize_native(nbit, gs):
    """Return a quantize_fn(model) using native PyTorch MPS int4 ops."""
    def fn(model):
        if nbit == 4:
            _replace_linears(model, _NativeInt4Linear, gs)
        else:
            raise ValueError(f"Native path only supports 4 bits, got {nbit}")
    return fn


def _quantize_int8wo():
    """Return a quantize_fn(model) using stable Int8WeightOnlyConfig.

    This produces Int8Tensor subclasses that do NOT dispatch to
    _weight_int8pack_mm — they dequantize to float and use torch.mm.
    Included to benchmark the gap vs the native int8 kernel.
    """
    def fn(model):
        quantize_(model, Int8WeightOnlyConfig())
    return fn


# ---------------------------------------------------------------------------
# Config execution
# ---------------------------------------------------------------------------

def _run_config(
    nbit, gs, windows_mps, baseline_argmax_list, *,
    model_id, quantize_fn, algorithm, desc, prefill_m_values,
) -> Result | None:
    """Load, quantize, and benchmark a single config. Returns None on failure."""
    print(f"  Loading {model_id} ({desc})...", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model.eval()
        quantize_fn(model)
        model.to("mps")

        dc = measure_decode_latency(model, windows_mps[0])
        pf, top1 = measure_prefill_with_top1(
            model, windows_mps, baseline_argmax_list, prefill_m_values,
        )

        del model
        cleanup()
        return Result(nbit, gs, dc, pf, top1, algorithm)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        cleanup()
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(tokenizer):
    """Load wikitext-103 test split and return NUM_WINDOWS [1, SEQ_LEN] windows.

    Windows are sampled uniformly from the full tokenized test split with
    a fixed seed (42) for reproducibility, giving content diversity across
    windows for top-1 measurement.
    """
    import random

    from datasets import Dataset
    arrow_path = os.path.expanduser(
        "~/.cache/huggingface/datasets/Salesforce___wikitext/"
        "wikitext-103-raw-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow"
    )
    if os.path.exists(arrow_path):
        ds = Dataset.from_file(arrow_path)
        text = "\n\n".join([t for t in ds["text"] if t.strip()])
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join([t for t in ds["text"] if t.strip()])
    # Tokenize the full test split (no truncation) so we can sample from it.
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total_tokens = input_ids.shape[0]
    if total_tokens < NUM_WINDOWS * SEQ_LEN:
        raise ValueError(
            f"Need at least {NUM_WINDOWS * SEQ_LEN} tokens for {NUM_WINDOWS} "
            f"non-overlapping windows of {SEQ_LEN}, got {total_tokens}"
        )
    # Pick NUM_WINDOWS random non-overlapping start positions with seed 42.
    rng = random.Random(42)
    max_start = total_tokens - SEQ_LEN
    starts = sorted(rng.sample(range(0, max_start + 1, SEQ_LEN), NUM_WINDOWS))
    windows = [input_ids[s:s + SEQ_LEN].unsqueeze(0) for s in starts]
    return windows


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def measure_baseline(args, windows_mps) -> Baseline:
    """Load bf16 model to MPS, measure speed, capture argmax per window for top-1."""
    print(f"Loading {args.model} (bf16 baseline)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("mps")
    model.eval()
    dc = measure_decode_latency(model, windows_mps[0])
    pf = measure_prefill_latency_multi(model, windows_mps, args.prefill_m)

    # Capture baseline argmax for each window (tiny: [1, SEQ_LEN-1] int64 each).
    baseline_argmax_list = []
    for w in windows_mps:
        with torch.no_grad():
            logits = model(w).logits
        baseline_argmax_list.append(logits[:, :-1, :].argmax(dim=-1).cpu())
        del logits

    del model
    cleanup()
    return Baseline(dc, pf, baseline_argmax_list)


# ---------------------------------------------------------------------------
# Run all configs
# ---------------------------------------------------------------------------

def run_all_configs(args, windows_mps, baseline, ckpt_results):
    """Run all quantized configs (old/new/native). Returns dict keyed by category."""
    prefill_m_values = args.prefill_m
    bitwidths = args.bits
    group_sizes = args.group_sizes
    algorithms = args.algorithms
    model_id = args.model
    baseline_argmax_list = baseline.top1_argmax

    def uintx_factory(nbit, gs, algorithm):
        algo_name = "hqq" if algorithm == UIntxChooseQParamsAlgorithm.HQQ else "minmax"
        desc = f"{nbit}-bit gs={gs} {algo_name}"
        return lambda: _run_config(
            nbit, gs, windows_mps, baseline_argmax_list,
            model_id=model_id,
            quantize_fn=_quantize_uintx(nbit, gs, algorithm),
            algorithm=algorithm, desc=desc,
            prefill_m_values=prefill_m_values,
        )

    def native_factory(nbit, gs, algorithm):
        desc = f"native {nbit}-bit gs={gs}"
        return lambda: _run_config(
            nbit, gs, windows_mps, baseline_argmax_list,
            model_id=model_id,
            quantize_fn=_quantize_native(nbit, gs),
            algorithm=UIntxChooseQParamsAlgorithm.MIN_MAX, desc=desc,
            prefill_m_values=prefill_m_values,
        )

    def int8wo_factory(nbit, gs, algorithm):
        desc = f"Int8WeightOnlyConfig {nbit}-bit per-channel"
        return lambda: _run_config(
            nbit, gs, windows_mps, baseline_argmax_list,
            model_id=model_id,
            quantize_fn=_quantize_int8wo(),
            algorithm=UIntxChooseQParamsAlgorithm.MIN_MAX, desc=desc,
            prefill_m_values=prefill_m_values,
        )

    results = {}

    # Old kernels: force fallback to upstream per-bitwidth GEMM (no simdgroup MMA)
    print("\n=== Old prefill kernels (upstream, no simdgroup MMA) ===", flush=True)
    os.environ["TORCHAO_MPS_DISABLE_SIMDGROUP_GEMM"] = "1"
    old_specs = []
    if "min_max" in algorithms:
        for nbit in [b for b in bitwidths if b != 8]:
            for gs in group_sizes:
                old_specs.append((nbit, gs, UIntxChooseQParamsAlgorithm.MIN_MAX))
    if "hqq" in algorithms and 4 in bitwidths and 64 in group_sizes:
        old_specs.append((4, 64, UIntxChooseQParamsAlgorithm.HQQ))
    if "min_max" in algorithms and 8 in bitwidths:
        for gs in group_sizes:
            old_specs.append((8, gs, UIntxChooseQParamsAlgorithm.MIN_MAX))
    results["old"] = _run_specs("old", old_specs, uintx_factory, ckpt_results)

    # New kernels: use simdgroup-tiled GEMM (or native delegation for int8 per-ch)
    print("\n=== New prefill kernels (simdgroup MMA) ===", flush=True)
    os.environ["TORCHAO_MPS_DISABLE_SIMDGROUP_GEMM"] = "0"
    new_specs = []
    if "min_max" in algorithms:
        for nbit in bitwidths:
            for gs in group_sizes:
                new_specs.append((nbit, gs, UIntxChooseQParamsAlgorithm.MIN_MAX))
    # int8 per-channel: IntxMPSExperimentalTensor delegates to native _weight_int8pack_mm
    if "min_max" in algorithms and 8 in bitwidths:
        new_specs.append((8, -1, UIntxChooseQParamsAlgorithm.MIN_MAX))
    if "hqq" in algorithms and 4 in bitwidths and 64 in group_sizes:
        new_specs.append((4, 64, UIntxChooseQParamsAlgorithm.HQQ))
    results["new"] = _run_specs("new", new_specs, uintx_factory, ckpt_results)

    # Native PyTorch MPS int4
    print("\n=== Native PyTorch MPS int4 ===", flush=True)
    native_specs = []
    if "torch" in algorithms:
        if 4 in bitwidths:
            for gs in [g for g in NATIVE_INT4_GROUP_SIZES if g in group_sizes]:
                native_specs.append((4, gs, UIntxChooseQParamsAlgorithm.MIN_MAX))
    results["torch"] = _run_specs("torch", native_specs, native_factory, ckpt_results)

    # Stable Int8WeightOnlyConfig (Int8Tensor — does NOT use _weight_int8pack_mm)
    print("\n=== Stable Int8WeightOnlyConfig (Int8Tensor) ===", flush=True)
    os.environ["TORCHAO_MPS_DISABLE_SIMDGROUP_GEMM"] = "0"
    int8wo_specs = []
    if "wo" in algorithms and 8 in bitwidths:
        int8wo_specs.append((8, -1, UIntxChooseQParamsAlgorithm.MIN_MAX))
    results["wo"] = _run_specs("wo", int8wo_specs, int8wo_factory, ckpt_results)

    cleanup()
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _make_label(nbit, suffix, algorithm):
    if algorithm == UIntxChooseQParamsAlgorithm.HQQ:
        return f"{nbit}{suffix}hqq"
    return f"{nbit}{suffix}"


def _prepare_report_data(results_by_category, prefill_m_values):
    """Collect, sort rows and compute significant bests.

    Shared by ``print_report`` (stdout) and ``write_markdown_report`` (file).
    Returns ``(rows, rows_by_nbit, sig_best, sorted_m)``.
    """
    sorted_m = sorted(prefill_m_values)

    # Collect results
    category_priority = {"old": 0, "new": 1, "torch": 2, "wo": 3}
    tagged = (
        [(r, "old") for r in results_by_category.get("old", [])]
        + [(r, "new") for r in results_by_category.get("new", [])]
        + [(r, "torch") for r in results_by_category.get("torch", [])]
        + [(r, "wo") for r in results_by_category.get("wo", [])]
    )
    rows = []
    for r, suffix in tagged:
        if r is None:
            continue
        label = _make_label(r.nbit, suffix, r.algo)
        # int8 per-channel via the new class delegates to native
        # _weight_int8pack_mm; label it as "8torch" to reflect this.
        if r.nbit == 8 and suffix == "new" and r.gs == -1:
            label = "8torch"
        priority = category_priority[suffix]
        is_hqq = 1 if "hqq" in label else 0
        sort_key = (r.nbit, r.gs, priority, is_hqq)
        rows.append((sort_key, r, label))
    rows.sort(key=lambda x: x[0])

    # Group rows by bitwidth to find best-within-bitwidth
    rows_by_nbit = {}
    for _, r, label in rows:
        rows_by_nbit.setdefault(r.nbit, []).append((r, label))

    # For each bitwidth group, find the best value per metric.
    # "Best" = lowest mean for latency, highest for top1.
    # Only underline if the best is significantly better than the
    # second-best (CIs don't overlap).
    def _is_significant_best(vals_cis, lower_is_better):
        """Return set of indices that are significant bests.

        Finds all entries tied for best (CIs overlap with the best) that
        collectively beat the rest. If the best doesn't significantly beat
        the second-best, returns empty (no underline).
        """
        if len(vals_cis) < 2:
            return set()
        sorted_idx = sorted(
            range(len(vals_cis)),
            key=lambda i: vals_cis[i][0] if lower_is_better else -vals_cis[i][0],
        )
        # Find the "best group": entries whose CIs overlap with the best.
        # These are tied — we can't distinguish them.
        best_mean, best_ci = vals_cis[sorted_idx[0]]
        if lower_is_better:
            best_upper = best_mean + best_ci
        else:
            best_lower = best_mean - best_ci

        best_group = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            mean, ci = vals_cis[idx]
            if lower_is_better:
                # Tied if this entry's lower bound overlaps best's upper bound
                if mean - ci <= best_upper:
                    best_group.append(idx)
                else:
                    break
            else:
                # Tied if this entry's upper bound overlaps best's lower bound
                if mean + ci >= best_lower:
                    best_group.append(idx)
                else:
                    break

        # Only underline if the best group significantly beats the rest.
        # "The rest" = first entry not in the best group.
        # If all entries are tied (best group covers everything), no underline.
        if len(best_group) >= len(vals_cis):
            return set()
        rest_idx = sorted_idx[len(best_group)]
        rest_mean, rest_ci = vals_cis[rest_idx]
        if lower_is_better:
            # Significant if best group's upper bound < rest's lower bound
            if best_upper < rest_mean - rest_ci:
                return set(best_group)
        else:
            # Significant if best group's lower bound > rest's upper bound
            if best_lower > rest_mean + rest_ci:
                return set(best_group)
        return set()

    # Compute significant bests per bitwidth group
    # Keys: (nbit, metric_key) -> set of row indices that are tied for best
    sig_best = {}
    for nbit, group in rows_by_nbit.items():
        # DCmean
        dc_vals = [(r.dc.mean, r.dc.ci) for r, _ in group]
        sig_best[(nbit, "dc")] = _is_significant_best(dc_vals, lower_is_better=True)
        # Prefill M
        for m in sorted_m:
            pf_vals = [(r.pf[m].mean, r.pf[m].ci) for r, _ in group]
            sig_best[(nbit, ("pf", m))] = _is_significant_best(pf_vals, lower_is_better=True)
        # top1 — highest wins, must be significant
        top1_vals = [(r.top1.mean, r.top1.ci) for r, _ in group]
        sig_best[(nbit, "top1")] = _is_significant_best(top1_vals, lower_is_better=False)

    return rows, rows_by_nbit, sig_best, sorted_m


def print_report(device_name, model_id, baseline, results_by_category, prefill_m_values):
    """Print the full report: speed table + top-1."""
    print()
    print(f"Device: {device_name}")
    print(f"Model: {model_id}")
    print("All latencies in ms; speedups vs bf16 eager baseline.")
    print()
    print("  DCmean     = decode latency (M=1)")
    print("  DCsp       = decode speedup vs bf16")
    print("  M=N        = prefill latency for N tokens")
    print("  Msp        = prefill speedup vs bf16")
    print("  top1       = next-token agreement with bf16 baseline (%)")
    print()
    print("  old        = upstream per-bitwidth GEMM (no simdgroup MMA)")
    print("  new        = simdgroup-tiled GEMM (intNgemm.metal)")
    print("  torch      = native PyTorch MPS (_weight_int4pack_mm / _weight_int8pack_mm)")
    print("  wo         = stable Int8WeightOnlyConfig (Int8Tensor, no _weight_int8pack_mm)")
    print("  hqq        = HQQ qparams algorithm (same _linear_fp_act kernel;")
    print("               included to show the new GEMM speeds up HQQ too)")
    print()
    print("  Underlined = significantly best within bitwidth")
    print("               (CIs don't overlap aside from ties)")
    print()
    print(f"  Samples: {RUNS} timed runs per latency measurement "
          f"(after {WARMUP} warmup), {NUM_WINDOWS} windows for top-1.")
    print("  CIs: 95% t-distribution (two-sided, Bessel-corrected).")
    print()
    print(f"  Data: wikitext-103 test split, {NUM_WINDOWS} non-overlapping")
    print(f"         {SEQ_LEN}-token windows sampled uniformly with a fixed")
    print("         seed (42). Latency runs cycle through these windows (one")
    print("         per run); top-1 uses all windows (no warmup discard,")
    print("         since top-1 is deterministic per window).")
    print()
    print("  Quality metric: top-1 next-token agreement with bf16 baseline.")
    print("         KL divergence is not reported because it requires retaining")
    print("         the full baseline logits tensor for every window (~970 MB")
    print("         per window, ~14.5 GB for 15 windows) to compare against each")
    print("         quantized config. Top-1 only needs the argmax (a [1, SEQ_LEN]")
    print("         int64 tensor per window, ~16 KB), which is negligible.")
    print()
    print()

    rows, rows_by_nbit, sig_best, sorted_m = _prepare_report_data(
        results_by_category, prefill_m_values
    )

    # --- Results table ---
    header = f"{'Bits':>8} {'gs':>6} {'DCmean':>7} {'DCsp':>6}"
    for m in sorted_m:
        header += f" {'M=' + str(m):>7} {'Msp':>5}"
    header += f" {'top1':>6}"
    print(header)
    print("-" * len(header))

    # Baseline row
    base_parts = [
        f"{'bf16':>8} {'-':>6} {baseline.dc.mean:>7.2f} {'1.00x':>6}"
    ]
    for m in sorted_m:
        lat = baseline.pf[m]
        base_parts.append(f" {lat.mean:>7.1f} {'1.00x':>5}")
    base_parts.append(f" {'100.0%':>6}")
    print("".join(base_parts))
    # Baseline CI row
    ci_parts = [f"{'':>8} {'':>6} {'±' + f'{baseline.dc.ci:.1f}':>7} {'':>6}"]
    for m in sorted_m:
        lat = baseline.pf[m]
        ci_parts.append(f" {'±' + f'{lat.ci:.1f}':>7} {'':>5}")
    ci_parts.append(f" {'':>6}")
    print("".join(ci_parts))
    print("-" * len(header))

    prev_nbit = None
    for _, r, label in rows:
        if prev_nbit is not None and r.nbit != prev_nbit:
            print("-" * len(header))
        prev_nbit = r.nbit

        # Find this row's index within its bitwidth group
        group = rows_by_nbit[r.nbit]
        row_idx = next(i for i, (gr, _) in enumerate(group) if gr is r)

        dc_sp = baseline.dc.mean / r.dc.mean if r.dc.mean > 0 else 0
        gs_str = "per-ch" if r.gs == -1 else str(r.gs)

        # Underline significant bests (latency + speedup + top1)
        dc_best = row_idx in sig_best.get((r.nbit, "dc"), set())
        dc_str = _ul(f"{r.dc.mean:.2f}", 7) if dc_best else f"{r.dc.mean:>7.2f}"
        dc_sp_str = _ul(f"{dc_sp:.2f}x", 6) if dc_best else f"{f'{dc_sp:.2f}x':>6}"
        parts = [f"{label:>8} {gs_str:>6} {dc_str} {dc_sp_str}"]
        for m in sorted_m:
            lat = r.pf.get(m, Latency(0, 0))
            base = baseline.pf.get(m, Latency(0, 0))
            sp = base.mean / lat.mean if lat.mean > 0 else 0
            pf_best = row_idx in sig_best.get((r.nbit, ("pf", m)), set())
            lat_str = _ul(f"{lat.mean:.1f}", 7) if pf_best else f"{lat.mean:>7.1f}"
            sp_str = _ul(f"{sp:.2f}x", 5) if pf_best else f"{f'{sp:.2f}x':>5}"
            parts.append(f" {lat_str} {sp_str}")
        top1_best = row_idx in sig_best.get((r.nbit, "top1"), set())
        top1_str = _ul(f"{r.top1.mean:.1f}%", 6) if top1_best else f"{r.top1.mean:>5.1f}%"
        parts.append(f" {top1_str}")
        print("".join(parts))
        # CI row
        ci_parts = [f"{'':>8} {'':>6} {'±' + f'{r.dc.ci:.1f}':>7} {'':>6}"]
        for m in sorted_m:
            lat = r.pf.get(m, Latency(0, 0))
            ci_parts.append(f" {'±' + f'{lat.ci:.1f}':>7} {'':>5}")
        ci_parts.append(f" {'±' + f'{r.top1.ci:.1f}':>6}")
        print("".join(ci_parts))


def write_markdown_report(
    path, device_name, model_id, baseline, results_by_category, prefill_m_values
):
    """Write the results table as GitHub-compatible markdown to ``path``.

    Differences from ``print_report`` (stdout):
    - Significant bests are **bold** instead of ANSI-underlined.
    - CI half-widths are on a new line within each cell (``<br>±X.X``)
      instead of on a separate row.
    - The table uses GitHub markdown table syntax.
    """
    rows, rows_by_nbit, sig_best, sorted_m = _prepare_report_data(
        results_by_category, prefill_m_values
    )

    lines = []
    lines.append(f"**Device:** {device_name}")
    lines.append("")
    lines.append(f"**Model:** {model_id}")
    lines.append("")
    lines.append("All latencies in ms; speedups vs bf16 eager baseline.")
    lines.append("")

    # Legend as a small table
    lines.append("| Key | Description |")
    lines.append("|-----|-------------|")
    lines.append("| `DCmean` | decode latency (M=1) |")
    lines.append("| `DCsp` | decode speedup vs bf16 |")
    lines.append("| `M=N` | prefill latency for N tokens |")
    lines.append("| `Msp` | prefill speedup vs bf16 |")
    lines.append("| `top1` | next-token agreement with bf16 baseline (%) |")
    lines.append("| `old` | upstream per-bitwidth GEMM (no simdgroup MMA) |")
    lines.append("| `new` | simdgroup-tiled GEMM (intNgemm.metal) |")
    lines.append("| `torch` | native PyTorch MPS (_weight_int4pack_mm / _weight_int8pack_mm) |")
    lines.append("| `wo` | stable Int8WeightOnlyConfig (Int8Tensor, no _weight_int8pack_mm) |")
    lines.append("| `hqq` | HQQ qparams algorithm (same _linear_fp_act kernel) |")
    lines.append("")
    lines.append(
        "**Bold** = significantly best within bitwidth "
        "(CIs don't overlap aside from ties)"
    )
    lines.append("")
    lines.append(
        f"**Samples:** {RUNS} timed runs per latency measurement "
        f"(after {WARMUP} warmup), {NUM_WINDOWS} windows for top-1. "
        f"**CIs:** 95% t-distribution (two-sided, Bessel-corrected)."
    )
    lines.append("")
    lines.append(
        f"**Data:** wikitext-103 test split, {NUM_WINDOWS} non-overlapping "
        f"{SEQ_LEN}-token windows sampled uniformly with a fixed seed (42). "
        f"Latency runs cycle through these windows (one per run); top-1 uses "
        f"all windows (no warmup discard, since top-1 is deterministic per "
        f"window)."
    )
    lines.append("")
    lines.append(
        "**Quality metric:** top-1 next-token agreement with bf16 baseline. "
        "KL divergence is not reported because it requires retaining the "
        "full baseline logits tensor for every window (~970 MB per "
        "window, ~14.5 GB for 15 windows) to compare against each quantized "
        "config. Top-1 only needs the argmax (a [1, SEQ_LEN] int64 tensor "
        "per window, ~16 KB), which is negligible."
    )
    lines.append("")

    # --- Results table ---
    header_cells = ["Bits", "gs", "DCmean", "DCsp"]
    for m in sorted_m:
        header_cells.append(f"M={m}")
        header_cells.append("Msp")
    header_cells.append("top1")

    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

    # Baseline row
    base_cells = ["bf16", "-"]
    base_cells.append(f"{baseline.dc.mean:.2f}<br>±{baseline.dc.ci:.1f}")
    base_cells.append("1.00x")
    for m in sorted_m:
        lat = baseline.pf[m]
        base_cells.append(f"{lat.mean:.1f}<br>±{lat.ci:.1f}")
        base_cells.append("1.00x")
    base_cells.append("100.0%<br>±0.0")
    lines.append("| " + " | ".join(base_cells) + " |")

    # Data rows
    for _, r, label in rows:
        group = rows_by_nbit[r.nbit]
        row_idx = next(i for i, (gr, _) in enumerate(group) if gr is r)

        dc_sp = baseline.dc.mean / r.dc.mean if r.dc.mean > 0 else 0
        gs_str = "per-ch" if r.gs == -1 else str(r.gs)

        dc_best = row_idx in sig_best.get((r.nbit, "dc"), set())
        dc_val = f"{r.dc.mean:.2f}"
        dc_cell = (
            f"**{dc_val}**<br>±{r.dc.ci:.1f}" if dc_best
            else f"{dc_val}<br>±{r.dc.ci:.1f}"
        )
        dc_sp_val = f"{dc_sp:.2f}x"
        dc_sp_cell = f"**{dc_sp_val}**" if dc_best else dc_sp_val

        cells = [label, gs_str, dc_cell, dc_sp_cell]
        for m in sorted_m:
            lat = r.pf.get(m, Latency(0, 0))
            base = baseline.pf.get(m, Latency(0, 0))
            sp = base.mean / lat.mean if lat.mean > 0 else 0
            pf_best = row_idx in sig_best.get((r.nbit, ("pf", m)), set())
            lat_val = f"{lat.mean:.1f}"
            lat_cell = (
                f"**{lat_val}**<br>±{lat.ci:.1f}" if pf_best
                else f"{lat_val}<br>±{lat.ci:.1f}"
            )
            sp_val = f"{sp:.2f}x"
            sp_cell = f"**{sp_val}**" if pf_best else sp_val
            cells.append(lat_cell)
            cells.append(sp_cell)

        top1_best = row_idx in sig_best.get((r.nbit, "top1"), set())
        top1_val = f"{r.top1.mean:.1f}%"
        top1_cell = (
            f"**{top1_val}**<br>±{r.top1.ci:.1f}" if top1_best
            else f"{top1_val}<br>±{r.top1.ci:.1f}"
        )
        cells.append(top1_cell)

        lines.append("| " + " | ".join(cells) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults table written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> Args:
    """Parse CLI args; fall back to module-level defaults when unset."""
    parser = argparse.ArgumentParser(description="Benchmark MPS weight-only quantization.")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("-g", "--group-sizes", type=int, nargs="*", default=None,
                        help="Override group sizes (default: 64 128)")
    parser.add_argument("-m", "--prefill-m", type=int, nargs="*", default=None,
                        help="Override prefill M values (default: 2048)")
    parser.add_argument("-a", "--algorithms", nargs="*", default=None,
                        help="Override algorithms (default: min_max hqq torch wo)")
    parser.add_argument("-b", "--bits", type=int, nargs="*", default=None,
                        help="Override bitwidths (default: 4 6 8)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear checkpoint cache before running")
    ns = parser.parse_args()
    return Args(
        model=ns.model,
        group_sizes=ns.group_sizes if ns.group_sizes is not None else list(GROUP_SIZES),
        prefill_m=ns.prefill_m if ns.prefill_m is not None else list(PREFILL_M_VALUES),
        algorithms=ns.algorithms if ns.algorithms is not None else list(ALGORITHMS),
        bits=ns.bits if ns.bits is not None else list(BITWIDTHS),
        clear_cache=ns.clear_cache,
    )


def main():
    """Entry point: parse args, load data, measure, run, report."""
    args = parse_args()

    if args.clear_cache and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Cleared checkpoint cache: {CHECKPOINT_FILE}")

    # Compile the HQQ proximal solver for faster quantization.
    if "hqq" in args.algorithms:
        import torchao.quantization.quant_primitives as _qp
        torch._dynamo.config.capture_scalar_outputs = True
        _compiled_optimize = torch.compile(
            _qp.optimize_weights_proximal_legacy, mode="reduce-overhead"
        )
        _fn = _qp._choose_qparams_and_quantize_affine_hqq
        _fn.__defaults__ = (*_fn.__defaults__[:-1], _compiled_optimize)
        print("HQQ quantization optimizer compiled (torch.compile, reduce-overhead)")

    # Load checkpoint (if any) from previous runs
    ckpt_results = _load_checkpoint()
    if ckpt_results:
        print(
            f"Loaded {len(ckpt_results)} checkpointed results from {CHECKPOINT_FILE}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    windows = load_dataset(tokenizer)
    windows_mps = [w.to("mps") for w in windows]

    baseline = measure_baseline(args, windows_mps)

    device_name = (
        subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        or "Apple Silicon"
    )

    results = run_all_configs(args, windows_mps, baseline, ckpt_results)
    cleanup()
    print_report(device_name, args.model, baseline, results, args.prefill_m)
    write_markdown_report(
        RESULTS_MD_FILE, device_name, args.model, baseline, results, args.prefill_m
    )


if __name__ == "__main__" and torch.backends.mps.is_available():
    main()
