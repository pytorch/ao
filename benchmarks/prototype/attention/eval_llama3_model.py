# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for evaluating attention backends on LLaMA 3.

Two-phase benchmark:
  Phase 1 (Perplexity): Evaluates quality on WikiText-2 test set using lm_eval,
      comparing baseline and test backends.
  Phase 2 (Runtime): Measures forward-pass latency across sequence lengths
      from 1K to 128K tokens using random token IDs.

Available backends:
    fa2      - Flash Attention 2 (default SDPA)
    fa3      - Flash Attention 3
    fa3_fp8  - Flash Attention 3 with FP8 quantization (fused RoPE + FP8 SDPA)

Usage:
    # Default: FA3 vs FA3 FP8
    python eval_llama3_model.py

    # FA2 vs FA3
    python eval_llama3_model.py --baseline fa2 --test fa3

    # With torch.compile (applies to non-FP8 backends)
    python eval_llama3_model.py --compile
"""

import argparse
import gc

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from torch._inductor.compile_fx import compile_fx
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.prototype.attention import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
    apply_low_precision_attention,
)
from torchao.prototype.attention.shared_utils.fusion_utils import (
    _is_sdpa_node,
    _sdpa_is_fusible,
    _strip_causal_mask,
    detect_causal_mask,
)

# =============================================================================
# Backend Configuration
# =============================================================================

BACKENDS = {
    "fa2": {
        "flash_impl": None,
        "fp8": False,
        "label": "FA2 BF16",
    },
    "fa3": {
        "flash_impl": "FA3",
        "fp8": False,
        "label": "FA3 BF16",
    },
    "fa3_fp8": {
        "flash_impl": "FA3",
        "fp8": True,
        "fp8_backend": AttentionBackend.FP8_FA3,
        "label": "FA3 FP8",
    },
}

RANDOM_SEED = 42

SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]  # , 65536, 131072]


def cleanup_gpu():
    """Free GPU memory between benchmark phases."""
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()


def _make_strip_causal_mask_pass(strip_causal_mask: bool):
    """Create a pre-grad pass that strips HF's materialized causal masks from SDPA nodes.

    During torch.compile, HuggingFace materializes causal masks as 4D bool
    tensors and passes them to SDPA with is_causal=False.  This forces the
    math backend instead of flash attention.  This pass detects those masks,
    removes them, and sets is_causal=True so flash attention can be used.

    Args:
        strip_causal_mask: Result of ``detect_causal_mask`` pre-flight check.
    """

    def _strip_causal_mask_pass(graph):
        for node in graph.nodes:
            if _is_sdpa_node(node):
                _, needs_strip = _sdpa_is_fusible(
                    node, strip_causal_mask=strip_causal_mask
                )
                if needs_strip:
                    _strip_causal_mask(node)
        graph.eliminate_dead_code()

    return _strip_causal_mask_pass


def _compile_with_mask_strip(model, flash_impl_name=None):
    """Compile a model with the causal mask stripping pass."""
    strip_causal_mask = detect_causal_mask(model, flash_impl_name=flash_impl_name)
    pass_fn = _make_strip_causal_mask_pass(strip_causal_mask)

    def mask_strip_backend(gm, example_inputs):
        old_pass = inductor_config.pre_grad_custom_pass
        inductor_config.pre_grad_custom_pass = pass_fn
        try:
            return compile_fx(gm, example_inputs)
        finally:
            inductor_config.pre_grad_custom_pass = old_pass

    torch._dynamo.reset()
    return torch.compile(model, backend=mask_strip_backend)


def setup_backend(orig_model, backend_name, compile_flag, fuse_rope=False):
    """Set up a backend for a benchmark phase.

    All backends use the HuggingFace SDPA attention path (the model must
    be loaded with ``attn_implementation="sdpa"``).  The specific flash
    attention kernel (FA2, FA3, FA4) is selected at runtime via
    ``activate_flash_attention_impl`` around each forward call.

    For FP8 backends: applies low-precision attention which handles
    compilation and the fusion pass internally.

    For other backends: optionally compiles if compile_flag is set.
    When compiling, a custom pre-grad pass strips HF's materialized causal
    masks so SDPA dispatches to flash attention instead of the math backend.

    Also manages ``config.use_cache``: FP8 backends need it disabled
    (DynamicCache.update() inserts torch.cat that blocks RoPE fusion),
    and non-FP8 backends also disable it when compiling (the causal mask
    stripping pass assumes no KV cache).

    Args:
        orig_model: The original (uncompiled, unwrapped) model.
        backend_name: Name of the backend.
        compile_flag: Whether --compile was passed.
        fuse_rope: Whether to fuse RoPE into the FP8 kernel (FP8 backends only).

    Returns:
        (model, flash_impl) where model is the model to use for this phase
        and flash_impl is the flash attention implementation to activate
        around forward calls.
    """
    cfg = BACKENDS[backend_name]

    if cfg["fp8"]:
        print(f"  Applying low-precision FP8 attention ({backend_name})...")
        # Disable KV cache: DynamicCache.update() torch.cat nodes block
        # the RoPE + SDPA fusion pass required by FP8 compilation.
        orig_model.config.use_cache = False
        fp8_config = LowPrecisionAttentionConfig(
            backend=cfg["fp8_backend"],
            fuse_rope=fuse_rope,
        )
        model = apply_low_precision_attention(orig_model, fp8_config)
        if compile_flag:
            print(f"  Compiling model with torch.compile ({backend_name})...")
            model = torch.compile(model)
        return model, cfg["flash_impl"]
    else:
        if compile_flag:
            print(f"  Compiling model with torch.compile ({backend_name})...")
            # Disable KV cache so the causal mask stripping pass works
            # (DynamicCache generates masks that block flash attention).
            orig_model.config.use_cache = False
            model = _compile_with_mask_strip(
                orig_model, flash_impl_name=cfg["flash_impl"]
            )
            return model, cfg["flash_impl"]
        # Restore use_cache in case a prior setup disabled it.
        orig_model.config.use_cache = True
        return orig_model, cfg["flash_impl"]


# =============================================================================
# Helpers
# =============================================================================


def evaluate_perplexity(model, tokenizer, flash_impl) -> float:
    """Evaluate perplexity on WikiText-2 using lm_eval.

    For FP8 backends, the model wrapper handles flash activation internally.
    For non-FP8 backends, flash activation is set for the duration of the
    evaluation so that lm_eval's internal forward calls use the correct kernel.
    """
    if flash_impl:
        activate_flash_attention_impl(flash_impl)
    try:
        results = evaluator.simple_evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer),
            tasks=["wikitext"],
            batch_size=1,
        )
    finally:
        if flash_impl:
            restore_flash_attention_impl()

    ppl = results["results"]["wikitext"]["word_perplexity,none"]
    return ppl


def benchmark_runtime(
    model,
    seq_len,
    vocab_size,
    device,
    flash_impl,
    num_warmup,
    num_iters,
) -> float:
    """Benchmark forward-pass latency at a given sequence length. Returns median ms."""
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    if flash_impl:
        activate_flash_attention_impl(flash_impl)
    try:
        # Warmup
        for _ in range(num_warmup):
            model(input_ids)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            start_events[i].record()
            model(input_ids)
            end_events[i].record()
        torch.cuda.synchronize()
    finally:
        if flash_impl:
            restore_flash_attention_impl()

    del input_ids

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[num_iters // 2]  # median


# =============================================================================
# Benchmark
# =============================================================================


@torch.inference_mode()
def run_benchmark(
    model_id: str,
    baseline_backend: str = "fa3",
    test_backend: str = "fa3_fp8",
    num_runtime_iters: int = 10,
    num_warmup: int = 3,
    compile: bool = False,
    fuse_rope: bool = False,
):
    baseline_label = BACKENDS[baseline_backend]["label"]
    test_label = BACKENDS[test_backend]["label"]
    compile_str = " + torch.compile" if compile else ""

    print("=" * 80)
    print("Attention Backend Benchmark for LLaMA 3")
    print(f"  Baseline: {baseline_label}  |  Test: {test_label}{compile_str}")
    print(f"  Model: {model_id}")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print("=" * 80)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    device = "cuda"

    # ----- Load model -----
    print(f"\nLoading model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    orig_model = model
    vocab_size = model.config.vocab_size

    # =====================================================================
    # Phase 1: Perplexity
    # =====================================================================
    print("\n" + "=" * 80)
    print("Phase 1: Perplexity (WikiText-2 via lm_eval)")
    print("=" * 80)

    # --- Baseline perplexity ---
    print(f"\n  Computing perplexity with {baseline_label}...")
    baseline_model, baseline_flash = setup_backend(
        orig_model,
        baseline_backend,
        compile,
        fuse_rope=fuse_rope,
    )
    baseline_ppl = evaluate_perplexity(baseline_model, tokenizer, baseline_flash)
    print(f"  {baseline_label} perplexity: {baseline_ppl:.2f}")

    # --- Test perplexity ---
    print(f"\n  Computing perplexity with {test_label}...")
    test_model, test_flash = setup_backend(
        orig_model,
        test_backend,
        compile,
        fuse_rope=fuse_rope,
    )
    test_ppl = evaluate_perplexity(test_model, tokenizer, test_flash)
    print(f"  {test_label} perplexity: {test_ppl:.2f}")

    print(f"\n  Delta: {test_ppl - baseline_ppl:+.2f}")

    del baseline_model, test_model
    cleanup_gpu()

    # =====================================================================
    # Phase 2: Runtime
    # =====================================================================
    print("\n" + "=" * 80)
    print(
        f"Phase 2: Runtime ({num_runtime_iters} iters, {num_warmup} warmup per seq_len)"
    )
    print("=" * 80)

    # --- Baseline runtime (all sequence lengths) ---
    print(f"\n  Running baseline ({baseline_label})...")
    baseline_model, baseline_flash = setup_backend(
        orig_model,
        baseline_backend,
        compile,
        fuse_rope=fuse_rope,
    )
    baseline_runtimes = {}
    for S in SEQ_LENGTHS:
        try:
            ms = benchmark_runtime(
                baseline_model,
                S,
                vocab_size,
                device,
                baseline_flash,
                num_warmup,
                num_runtime_iters,
            )
            baseline_runtimes[S] = ms
            print(f"    seq_len={S:>6}: {ms:.1f} ms")
        except torch.cuda.OutOfMemoryError:
            print(f"    seq_len={S:>6}: OOM")
            torch.cuda.empty_cache()
            break
        torch.cuda.empty_cache()

    del baseline_model
    cleanup_gpu()

    # --- Test runtime (all sequence lengths) ---
    print(f"\n  Running test ({test_label})...")
    test_model, test_flash = setup_backend(
        orig_model,
        test_backend,
        compile,
        fuse_rope=fuse_rope,
    )
    test_runtimes = {}
    for S in baseline_runtimes:
        try:
            ms = benchmark_runtime(
                test_model,
                S,
                vocab_size,
                device,
                test_flash,
                num_warmup,
                num_runtime_iters,
            )
            test_runtimes[S] = ms
            print(f"    seq_len={S:>6}: {ms:.1f} ms")
        except torch.cuda.OutOfMemoryError:
            print(f"    seq_len={S:>6}: OOM")
            torch.cuda.empty_cache()
            break
        torch.cuda.empty_cache()

    # --- Print comparison table ---
    col_baseline = f"{baseline_label} (ms)"
    col_test = f"{test_label} (ms)"
    col_w = max(len(col_baseline), len(col_test), 12)

    header = (
        f"{'SeqLen':>8} | "
        f"{col_baseline:>{col_w}} | "
        f"{col_test:>{col_w}} | "
        f"{'Speedup':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    runtime_results = []

    for S in baseline_runtimes:
        baseline_ms = baseline_runtimes[S]
        test_ms = test_runtimes.get(S)

        if test_ms is not None:
            speedup = baseline_ms / test_ms
            print(
                f"{S:>8} | "
                f"{baseline_ms:>{col_w}.1f} | "
                f"{test_ms:>{col_w}.1f} | "
                f"{speedup:>7.2f}x"
            )
            runtime_results.append(
                {
                    "seq_len": S,
                    "baseline_ms": baseline_ms,
                    "test_ms": test_ms,
                    "speedup": speedup,
                }
            )
        else:
            print(f"{S:>8} | {baseline_ms:>{col_w}.1f} | {'OOM':>{col_w}} |")

    print("-" * len(header))

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Baseline ({baseline_label}) perplexity: {baseline_ppl:.2f}")
    print(f"  Test     ({test_label}) perplexity: {test_ppl:.2f}")
    print(f"  Perplexity delta: {test_ppl - baseline_ppl:+.2f}")
    if runtime_results:
        avg_speedup = sum(r["speedup"] for r in runtime_results) / len(runtime_results)
        print(f"  Avg runtime speedup: {avg_speedup:.2f}x")
    print("=" * 80)

    return {
        "baseline_ppl": baseline_ppl,
        "test_ppl": test_ppl,
        "runtime_results": runtime_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention backends on LLaMA 3"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="fa3",
        choices=list(BACKENDS.keys()),
        help="Baseline attention backend (default: fa3)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="fa3_fp8",
        choices=list(BACKENDS.keys()),
        help="Test attention backend (default: fa3_fp8)",
    )
    parser.add_argument(
        "--num_runtime_iters",
        type=int,
        default=10,
        help="Number of timed forward passes per sequence length",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=3,
        help="Number of warmup iterations per sequence length",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile (applies to non-FP8 backends)",
    )
    parser.add_argument(
        "--fuse_rope",
        action="store_true",
        help="Fuse RoPE into the FP8 kernel (compile path, off by default)",
    )
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model_id,
        baseline_backend=args.baseline,
        test_backend=args.test,
        num_runtime_iters=args.num_runtime_iters,
        num_warmup=args.num_warmup,
        compile=args.compile,
        fuse_rope=args.fuse_rope,
    )


if __name__ == "__main__":
    main()
