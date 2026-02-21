# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for evaluating attention backends on LLaMA 3.

Two-phase benchmark:
  Phase 1 (Perplexity): Evaluates quality on WikiText-2 test set at a fixed
      sequence length, comparing baseline and test backends.
  Phase 2 (Runtime): Measures forward-pass latency across sequence lengths
      from 1K to 128K tokens using random token IDs.

Available backends:
    fa2      - Flash Attention 2 (default SDPA)
    fa3      - Flash Attention 3
    fa3_fp8  - Flash Attention 3 with FP8 quantization
    fa4      - Flash Attention 4

Usage:
    # Default: FA3 vs FA3 FP8
    python eval_llama3_model.py

    # FA2 vs FA3
    python eval_llama3_model.py --baseline fa2 --test fa3

    # FA3 vs FA4
    python eval_llama3_model.py --baseline fa3 --test fa4

    # With torch.compile
    python eval_llama3_model.py --compile
"""

import argparse
import math
import warnings
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa

# =============================================================================
# Backend Configuration
# =============================================================================

_original_sdpa = F.scaled_dot_product_attention

BACKENDS = {
    "fa2": {
        "hf_attn_impl": "flash_attention_2",
        "flash_impl": None,
        "fp8": False,
        "label": "FA2 BF16",
    },
    "fa3": {
        "hf_attn_impl": "flash_attention_3",
        "flash_impl": None,
        "fp8": False,
        "label": "FA3 BF16",
    },
    "fa3_fp8": {
        "hf_attn_impl": "sdpa",
        "flash_impl": "FA3",
        "fp8": True,
        "label": "FA3 FP8",
    },
    "fa4": {
        "hf_attn_impl": "sdpa",
        "flash_impl": "FA4",
        "fp8": False,
        "label": "FA4 BF16",
    },
}

RANDOM_SEED = 42
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B"

SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


@contextmanager
def backend_context(backend_name: str, model):
    """Context manager that activates the appropriate attention backend.

    For fa2/fa3, sets HuggingFace's attn_implementation so HuggingFace
    calls the flash_attn packages directly (bypassing SDPA and avoiding
    the causal-mask incompatibility with flash attention).

    For fa3_fp8/fa4, uses the SDPA path with activate_flash_attention_impl
    since HuggingFace doesn't have native support for these.
    """
    cfg = BACKENDS[backend_name]
    old_attn_impl = model.config._attn_implementation

    model.config._attn_implementation = cfg["hf_attn_impl"]

    if cfg["flash_impl"]:
        activate_flash_attention_impl(cfg["flash_impl"])
    if cfg["fp8"]:
        def _fp8_sdpa_wrapper(
            query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False,
        ):
            if dropout_p != 0.0 or attn_mask is not None:
                warnings.warn(
                    "Dropout and attention mask not supported for FP8 FA3. "
                    "Falling back to regular SDPA."
                )
                return _original_sdpa(
                    query, key, value, attn_mask=attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal,
                    scale=scale, enable_gqa=enable_gqa,
                )
            return fp8_fa3_sdpa(
                query, key, value, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, enable_gqa=enable_gqa,
            )
        F.scaled_dot_product_attention = _fp8_sdpa_wrapper
    try:
        yield
    finally:
        if cfg["fp8"]:
            F.scaled_dot_product_attention = _original_sdpa
        if cfg["flash_impl"]:
            restore_flash_attention_impl()
        model.config._attn_implementation = old_attn_impl


# =============================================================================
# Helpers
# =============================================================================


def load_wikitext2_tokens(tokenizer, seq_len: int):
    """Load WikiText-2 test set, tokenize, and chunk into fixed-length segments."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    # Chunk into non-overlapping segments of (seq_len + 1) so we have
    # seq_len input tokens and 1 target token for each chunk.
    n_chunks = tokens.size(0) // (seq_len + 1)
    tokens = tokens[: n_chunks * (seq_len + 1)]
    chunks = tokens.view(n_chunks, seq_len + 1)

    print(
        f"  WikiText-2 test: {tokens.numel()} tokens, "
        f"{n_chunks} chunks of {seq_len} tokens"
    )
    return chunks


def compute_perplexity(model, chunks, device: str, backend_name: str) -> float:
    """Compute perplexity over chunked token sequences using a given backend."""
    total_loss = 0.0
    n_chunks = chunks.size(0)

    with backend_context(backend_name, model):
        for i in range(n_chunks):
            chunk = chunks[i].unsqueeze(0).to(device)
            input_ids = chunk[:, :-1]
            labels = chunk[:, 1:]

            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.reshape(-1)
            )
            total_loss += loss.item()

    avg_loss = total_loss / n_chunks
    return math.exp(avg_loss)


def benchmark_runtime(
    model, seq_len: int, vocab_size: int, device: str, backend_name: str,
    num_warmup: int, num_iters: int,
) -> float:
    """Benchmark forward-pass latency at a given sequence length. Returns median ms."""
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    with backend_context(backend_name, model):
        # Warmup
        for _ in range(num_warmup):
            model(input_ids)
        torch.cuda.synchronize()

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_iters)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_iters)
        ]

        for i in range(num_iters):
            start_events[i].record()
            model(input_ids)
            end_events[i].record()
        torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[num_iters // 2]  # median


# =============================================================================
# Benchmark
# =============================================================================


@torch.inference_mode()
def run_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    baseline_backend: str = "fa3",
    test_backend: str = "fa3_fp8",
    num_runtime_iters: int = 10,
    num_warmup: int = 3,
    perplexity_seq_len: int = 2048,
    compile: bool = False,
):
    baseline_label = BACKENDS[baseline_backend]["label"]
    test_label = BACKENDS[test_backend]["label"]
    compile_str = " + torch.compile" if compile else ""

    print("=" * 80)
    print("Attention Backend Benchmark for LLaMA 3")
    print(
        f"  Baseline: {baseline_label}  |  Test: {test_label}{compile_str}"
    )
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
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    if compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    vocab_size = model.config.vocab_size

    # =====================================================================
    # Phase 1: Perplexity
    # =====================================================================
    print("\n" + "=" * 80)
    print(
        f"Phase 1: Perplexity (WikiText-2 test, seq_len={perplexity_seq_len})"
    )
    print("=" * 80)

    chunks = load_wikitext2_tokens(tokenizer, perplexity_seq_len)

    print(f"\n  Computing perplexity with {baseline_label}...")
    baseline_ppl = compute_perplexity(
        model, chunks, device, baseline_backend,
    )
    print(f"  {baseline_label} perplexity: {baseline_ppl:.2f}")

    print(f"\n  Computing perplexity with {test_label}...")
    test_ppl = compute_perplexity(model, chunks, device, test_backend)
    print(f"  {test_label} perplexity: {test_ppl:.2f}")

    print(f"\n  Delta: {test_ppl - baseline_ppl:+.2f}")

    # =====================================================================
    # Phase 2: Runtime
    # =====================================================================
    print("\n" + "=" * 80)
    print(
        f"Phase 2: Runtime "
        f"({num_runtime_iters} iters, {num_warmup} warmup per seq_len)"
    )
    print("=" * 80)

    col_baseline = f"{baseline_label} (ms)"
    col_test = f"{test_label} (ms)"
    col_w = max(len(col_baseline), len(col_test), 12)

    header = (
        f"{'SeqLen':>8} | "
        f"{col_baseline:>{col_w}} | "
        f"{col_test:>{col_w}} | "
        f"{'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    runtime_results = []

    for S in SEQ_LENGTHS:
        try:
            baseline_ms = benchmark_runtime(
                model, S, vocab_size, device, baseline_backend,
                num_warmup, num_runtime_iters,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{S:>8} | {'OOM':>{col_w}} |")
            torch.cuda.empty_cache()
            break

        try:
            test_ms = benchmark_runtime(
                model, S, vocab_size, device, test_backend,
                num_warmup, num_runtime_iters,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"{S:>8} | "
                f"{baseline_ms:>{col_w}.1f} | "
                f"{'OOM':>{col_w}} |"
            )
            torch.cuda.empty_cache()
            break

        speedup = baseline_ms / test_ms

        print(
            f"{S:>8} | "
            f"{baseline_ms:>{col_w}.1f} | "
            f"{test_ms:>{col_w}.1f} | "
            f"{speedup:>7.2f}x"
        )

        runtime_results.append({
            "seq_len": S,
            "baseline_ms": baseline_ms,
            "test_ms": test_ms,
            "speedup": speedup,
        })

        torch.cuda.empty_cache()

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
        avg_speedup = (
            sum(r["speedup"] for r in runtime_results) / len(runtime_results)
        )
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
        default=DEFAULT_MODEL_ID,
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
        "--perplexity_seq_len",
        type=int,
        default=2048,
        help="Sequence length for perplexity evaluation",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile",
    )
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model_id,
        baseline_backend=args.baseline,
        test_backend=args.test,
        num_runtime_iters=args.num_runtime_iters,
        num_warmup=args.num_warmup,
        perplexity_seq_len=args.perplexity_seq_len,
        compile=args.compile,
    )


if __name__ == "__main__":
    main()
