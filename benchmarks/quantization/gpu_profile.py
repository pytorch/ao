# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark peak memory: BF16 vs W8A8-INT vs INT8+compile on Qwen3-8B.

Result on RTX 5090:
  Metric              BF16    W8A8-INT  INT8+compile
  Allocated (MiB)  15643.9    11396.1       8430.6
  Reserved  (MiB)  15658.0    21334.0      15312.0
  Fragmentation %      0.1       46.6          44.9
"""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

MODEL_ID = "Qwen/Qwen3-8B"
# TODO: Consider long (> 32k) tokens for long-context task validation
# This can be extended to `long_bench_e` dataset using lm_eval library
PROMPT = "Reduce memory footprint with torchao and torch.compile"
MAX_NEW_TOKENS = 64
mb = lambda n: n / 1024**2


def measure(model, tokenizer, label):
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()

    s = torch.cuda.memory_stats()
    peak_active = s.get("active_bytes.all.peak", 0)
    peak_reserved = s.get("reserved_bytes.all.peak", 0)
    allocated = torch.cuda.max_memory_allocated()
    return {
        "label": label,
        "alloc": allocated,
        "reserved": torch.cuda.max_memory_reserved(),
        "frag_pct": (peak_reserved - peak_active) / peak_reserved * 100
        if peak_reserved
        else 0,
    }


def print_results(results):
    labels, W = [r["label"] for r in results], 13
    print(f"\n{'Metric':<24}" + "".join(f"{l:>{W}}" for l in labels))
    print("-" * (24 + W * len(results)))
    for key, lbl, scale in [
        ("alloc", "Allocated (MiB)", 1024**2),
        ("reserved", "Reserved  (MiB)", 1024**2),
        ("frag_pct", "Fragmentation %", 1),
    ]:
        print(f"{lbl:<24}" + "".join(f"{r[key] / scale:>{W}.1f}" for r in results))
    print()


def run(label, tokenizer, quantize_fn=None):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    if quantize_fn:
        quantize_fn(model)
    result = measure(model, tokenizer, label)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_ID} | max_new_tokens: {MAX_NEW_TOKENS}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def int8_compile(m):
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        m.forward = torch.compile(m.forward, mode="max-autotune")

    print_results(
        [
            run("BF16", tok),
            run(
                "W8A8-INT",
                tok,
                lambda m: quantize_(m, Int8DynamicActivationInt8WeightConfig()),
            ),
            run("INT8+compile", tok, int8_compile),
        ]
    )
