# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for evaluating attention backends on FLUX.1-schnell.

Compares two selectable attention backends using LPIPS (perceptual similarity).
Uses DrawBench dataset for standardized prompt evaluation.

Available backends:
    fa2      - Flash Attention 2 (default SDPA)
    fa3      - Flash Attention 3
    fa3_fp8  - Flash Attention 3 with FP8 quantization (fused RoPE + FP8 SDPA)
    fa4      - Flash Attention 4
    fa4_fp8  - Flash Attention 4 with FP8 quantization (fused RoPE + FP8 SDPA)

Usage:
    # Compare FA3 vs FA3 FP8 (default)
    python eval_flux_model.py --debug_prompt "A red car"

    # Compare FA2 vs FA3
    python eval_flux_model.py --baseline fa2 --test fa3

    # Compare FA3 vs FA4
    python eval_flux_model.py --baseline fa3 --test fa4

    # Full benchmark with 200 prompts
    python eval_flux_model.py --num_prompts 200

    # With torch.compile
    python eval_flux_model.py --compile
"""

import argparse
import gc
import random
from typing import Optional

import lpips
import numpy as np
import torch
import torch._dynamo
from datasets import load_dataset
from diffusers import FluxPipeline
from PIL import Image
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.attention import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
    apply_low_precision_attention,
)

# =============================================================================
# Backend Configuration
# =============================================================================

BACKENDS = {
    "fa2": {"flash_impl": None, "fp8": False},
    "fa3": {"flash_impl": "FA3", "fp8": False},
    "fa3_fp8": {
        "flash_impl": "FA3",
        "fp8": True,
        "fp8_backend": AttentionBackend.FP8_FA3,
    },
    "fa4": {"flash_impl": "FA4", "fp8": False},
    "fa4_fp8": {
        "flash_impl": "FA4",
        "fp8": True,
        "fp8_backend": AttentionBackend.FP8_FA4,
    },
}

IMAGE_SIZE = (512, 512)  # (width, height) - resize for consistent LPIPS
RANDOM_SEED = 42
MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def cleanup_gpu():
    """Free GPU memory between benchmark phases."""
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()


def setup_backend(pipe, backend_name, compile_flag, orig_transformer, fuse_rope=False):
    """Set up a backend for a benchmark phase.

    For FP8 backends (fa3_fp8): applies low-precision attention which
    handles compilation and the fusion pass internally.

    For other backends: optionally compiles if compile_flag is set.

    Args:
        pipe: The diffusion pipeline.
        backend_name: Name of the backend.
        compile_flag: Whether --compile was passed.
        orig_transformer: The original (uncompiled) transformer.
        fuse_rope: Whether to fuse RoPE into the FP8 kernel (FP8 backends only).

    Returns:
        flash_impl to pass to generate_image (None for FP8 backends
        since FA3 is managed internally by the wrapper).
    """
    cfg = BACKENDS[backend_name]
    pipe.transformer = orig_transformer

    if cfg["fp8"]:
        print(f"Applying low-precision FP8 attention ({backend_name})...")
        fp8_config = LowPrecisionAttentionConfig(
            backend=cfg["fp8_backend"],
            fuse_rope=fuse_rope,
        )
        pipe.transformer = apply_low_precision_attention(pipe.transformer, fp8_config)
        if compile_flag:
            print(f"Compiling transformer with torch.compile ({backend_name})...")
            pipe.transformer = torch.compile(pipe.transformer)
        # Return the flash impl (e.g. "FA3") so generate_image activates it
        # for the entire pipe() call.  This keeps non-transformer SDPA calls
        # (VAE, text encoder) consistent with the baseline FA3 backend.
        return cfg["flash_impl"]
    else:
        if compile_flag:
            print(f"Compiling transformer with torch.compile ({backend_name})...")
            pipe.transformer = torch.compile(pipe.transformer)
        return cfg["flash_impl"]


# =============================================================================
# Helpers
# =============================================================================


def pil_to_lpips_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """Convert a PIL Image to a tensor suitable for LPIPS computation."""
    t = (
        torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                .view(img.size[1], img.size[0], 3)
                .numpy()
            )
        ).float()
        / 255.0
    )
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = t * 2.0 - 1.0
    return t.to(device)


def generate_image(
    pipe,
    prompt: str,
    seed: int,
    device: str,
    num_inference_steps: int,
    flash_impl: Optional[str] = None,
) -> Image.Image:
    """Generate an image from a prompt with deterministic seed."""
    generator = torch.Generator(device=device).manual_seed(seed)

    if flash_impl:
        activate_flash_attention_impl(flash_impl)
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            generator=generator,
        ).images[0]
    finally:
        if flash_impl:
            restore_flash_attention_impl()

    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    return image


# =============================================================================
# Benchmark
# =============================================================================


@torch.inference_mode()
def run_benchmark(
    baseline_backend: str = "fa3",
    test_backend: str = "fa3_fp8",
    num_prompts: int = 50,
    num_inference_steps: int = 20,
    debug_prompt: Optional[str] = None,
    warmup_iters: int = 2,
    compile: bool = False,
    fuse_rope: bool = False,
):
    """
    Run the attention backend benchmark on FLUX.1-schnell.

    Args:
        baseline_backend: Baseline attention backend name.
        test_backend: Test attention backend name.
        num_prompts: Number of prompts to use (50 or 200 recommended).
        num_inference_steps: Number of diffusion steps per image.
        debug_prompt: If specified, use only this prompt (for debugging).
        warmup_iters: Number of warmup iterations before benchmarking.
        compile: If True, wrap the model with torch.compile.
        fuse_rope: If True (default), fuse RoPE into the FP8 kernel.
            If False, only replace SDPA with FP8 (skip RoPE fusion).
    """
    compile_str = " + torch.compile" if compile else ""
    print("=" * 80)
    print("Attention Backend Benchmark for FLUX.1-schnell")
    print(f"Baseline: {baseline_backend}  |  Test: {test_backend}{compile_str}")
    print("=" * 80)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cuda"

    # ----- Load prompts -----
    if debug_prompt is not None:
        prompts = [debug_prompt]
        print(f"Using debug prompt: {debug_prompt}")
    else:
        print("Loading DrawBench dataset...")
        dataset = load_dataset("sayakpaul/drawbench", split="train")
        all_prompts = [item["Prompts"] for item in dataset]
        prompts = all_prompts[:num_prompts]
        print(
            f"Using {len(prompts)} prompts from DrawBench "
            f"(total available: {len(all_prompts)})"
        )

    # ----- Load model and LPIPS -----
    print(f"\nLoading FLUX.1-schnell from {MODEL_ID}...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print("Loading LPIPS model (VGG)...")
    loss_fn = lpips.LPIPS(net="vgg").to(device)

    orig_transformer = pipe.transformer

    if compile:
        pipe.vae.decode = torch.compile(pipe.vae.decode)

    # ----- Phase 1: Baseline backend -----
    print("\n" + "-" * 80)
    print(f"Phase 1: Generating images ({baseline_backend})")
    print("-" * 80)

    baseline_flash_impl = setup_backend(
        pipe,
        baseline_backend,
        compile,
        orig_transformer,
        fuse_rope=fuse_rope,
    )

    print(f"Warming up {baseline_backend} with {warmup_iters} iterations...")
    warmup_prompt = prompts[0]
    for i in range(warmup_iters):
        _ = generate_image(
            pipe,
            warmup_prompt,
            RANDOM_SEED,
            device,
            num_inference_steps,
            flash_impl=baseline_flash_impl,
        )
        print(f"  Warmup {i + 1}/{warmup_iters} complete")

    baseline_data = []
    baseline_times_ms = []

    for idx, prompt in enumerate(prompts):
        print(f"[{idx + 1}/{len(prompts)}] {baseline_backend}: {prompt[:50]}...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        baseline_img = generate_image(
            pipe,
            prompt,
            RANDOM_SEED,
            device,
            num_inference_steps,
            flash_impl=baseline_flash_impl,
        )
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)

        baseline_tensor = pil_to_lpips_tensor(baseline_img, device)
        # Store tensors on CPU to free GPU memory for the test phase.
        baseline_data.append((prompt, baseline_tensor.cpu()))
        baseline_times_ms.append(elapsed_ms)

    avg_baseline_ms = sum(baseline_times_ms) / len(baseline_times_ms)
    print(
        f"\n{baseline_backend} complete. Avg time per image: {avg_baseline_ms:.1f} ms"
    )

    # ----- Cleanup before test phase -----
    cleanup_gpu()

    # ----- Phase 2: Test backend -----
    print("\n" + "-" * 80)
    print(f"Phase 2: Generating images ({test_backend})")
    print("-" * 80)

    test_flash_impl = setup_backend(
        pipe,
        test_backend,
        compile,
        orig_transformer,
        fuse_rope=fuse_rope,
    )

    print(f"Warming up {test_backend} with {warmup_iters} iterations...")
    for i in range(warmup_iters):
        _ = generate_image(
            pipe,
            warmup_prompt,
            RANDOM_SEED,
            device,
            num_inference_steps,
            flash_impl=test_flash_impl,
        )
        print(f"  Warmup {i + 1}/{warmup_iters} complete")

    lpips_values = []
    test_times_ms = []

    for idx, (prompt, baseline_tensor_cpu) in enumerate(baseline_data):
        print(f"[{idx + 1}/{len(prompts)}] {test_backend}: {prompt[:50]}...")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        test_img = generate_image(
            pipe,
            prompt,
            RANDOM_SEED,
            device,
            num_inference_steps,
            flash_impl=test_flash_impl,
        )
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        test_times_ms.append(elapsed_ms)

        test_tensor = pil_to_lpips_tensor(test_img, device)
        lpips_value = loss_fn(baseline_tensor_cpu.to(device), test_tensor).item()
        lpips_values.append(lpips_value)

        print(f"    LPIPS: {lpips_value:.4f}, Time: {elapsed_ms:.1f} ms")

    avg_test_ms = sum(test_times_ms) / len(test_times_ms)

    # ----- Results -----
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    avg_lpips = sum(lpips_values) / len(lpips_values)
    max_lpips = max(lpips_values)
    min_lpips = min(lpips_values)
    std_lpips = np.std(lpips_values)

    print("\nLPIPS Statistics (lower is better, 0 = identical):")
    print(f"  Average LPIPS: {avg_lpips:.4f}")
    print(f"  Std Dev:       {std_lpips:.4f}")
    print(f"  Min LPIPS:     {min_lpips:.4f}")
    print(f"  Max LPIPS:     {max_lpips:.4f}")

    print("\nTiming Statistics:")
    print(f"  Avg {baseline_backend} time:  {avg_baseline_ms:.1f} ms per image")
    print(f"  Avg {test_backend} time: {avg_test_ms:.1f} ms per image")
    print(f"  Speedup:                {avg_baseline_ms / avg_test_ms:.2f}x")

    print("\nBenchmark Configuration:")
    print(f"  Baseline backend:  {baseline_backend}")
    print(f"  Test backend:      {test_backend}")
    print(f"  torch.compile:     {compile}")
    print(f"  Model:             {MODEL_ID}")
    print(f"  Prompts tested:    {len(prompts)}")
    print(f"  Inference steps:   {num_inference_steps}")
    print(f"  Image size:        {IMAGE_SIZE}")
    print(f"  Random seed:       {RANDOM_SEED}")
    print("=" * 80)

    return {
        "avg_lpips": avg_lpips,
        "std_lpips": std_lpips,
        "min_lpips": min_lpips,
        "max_lpips": max_lpips,
        "speedup": avg_baseline_ms / avg_test_ms,
        "lpips_values": lpips_values,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention backends on FLUX.1-schnell"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="fa3",
        choices=list(BACKENDS.keys()),
        help="Baseline attention backend",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="fa3_fp8",
        choices=list(BACKENDS.keys()),
        help="Test attention backend",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=200,
        help="Number of prompts to use (50 for quick, 200 for full benchmark)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Number of diffusion inference steps",
    )
    parser.add_argument(
        "--debug_prompt",
        type=str,
        default=None,
        help="Use a single debug prompt instead of DrawBench",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile for both backends",
    )
    parser.add_argument(
        "--fuse_rope",
        action="store_true",
        help="Fuse RoPE into the FP8 kernel (compile path, off by default)",
    )

    args = parser.parse_args()

    run_benchmark(
        baseline_backend=args.baseline,
        test_backend=args.test,
        num_prompts=args.num_prompts,
        num_inference_steps=args.num_inference_steps,
        debug_prompt=args.debug_prompt,
        warmup_iters=args.warmup_iters,
        compile=args.compile,
        fuse_rope=args.fuse_rope,
    )


if __name__ == "__main__":
    main()
