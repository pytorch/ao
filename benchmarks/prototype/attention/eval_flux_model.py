# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for evaluating FP8 attention accuracy on FLUX.1-schnell.

Compares regular inference with low-precision FP8 attention using LPIPS
(perceptual similarity). Uses DrawBench dataset for standardized prompt
evaluation.

Usage:
    # Default (auto-selected backend), 50 prompts
    python eval_flux_sdpa.py --num_prompts 50

    # With torch.compile
    python eval_flux_sdpa.py --compile

    # Full benchmark with 200 prompts
    python eval_flux_sdpa.py --num_prompts 200

    # Debug with single prompt
    python eval_flux_sdpa.py --debug_prompt "A photo of an astronaut riding a horse"
"""

import argparse
import random
import time
from typing import Optional

import lpips
import numpy as np
import torch
from datasets import load_dataset
from diffusers import FluxPipeline
from PIL import Image

from torchao.prototype.attention import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
    apply_low_precision_attention,
)

# =============================================================================
# Configuration
# =============================================================================

# Modify this config to change the low-precision attention behavior.
# Default: None (auto-selects the best available backend).
# Example: LowPrecisionAttentionConfig(backend=AttentionBackend.FP8_FA3)
ATTENTION_CONFIG = LowPrecisionAttentionConfig(backend=AttentionBackend.FP8_FA3)

IMAGE_SIZE = (512, 512)  # (width, height) - resize for consistent LPIPS
RANDOM_SEED = 42
MODEL_ID = "black-forest-labs/FLUX.1-schnell"


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
) -> Image.Image:
    """Generate an image from a prompt with deterministic seed."""
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    return image


# =============================================================================
# Benchmark
# =============================================================================


@torch.inference_mode()
def run_benchmark(
    num_prompts: int = 50,
    num_inference_steps: int = 20,
    debug_prompt: Optional[str] = None,
    warmup_iters: int = 2,
    compile: bool = False,
):
    """
    Run the FP8 attention accuracy benchmark on FLUX.1-schnell.

    Args:
        num_prompts: Number of prompts to use (50 or 200 recommended)
        num_inference_steps: Number of diffusion steps per image
        debug_prompt: If specified, use only this prompt (for debugging)
        warmup_iters: Number of warmup iterations before benchmarking
        compile: If True, wrap the model with torch.compile
    """
    config_str = str(ATTENTION_CONFIG) if ATTENTION_CONFIG is not None else "auto"
    compile_str = " + torch.compile" if compile else ""
    print("=" * 80)
    print("FP8 Attention Benchmark for FLUX.1-schnell")
    print(f"Config: {config_str}{compile_str}")
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

    # ----- Optionally compile for baseline -----
    if compile:
        print("\nCompiling transformer and vae.decode with torch.compile...")
        pipe.transformer = torch.compile(orig_transformer)
        pipe.vae.decode = torch.compile(pipe.vae.decode)

    # ----- Warmup -----
    print(f"\nWarming up with {warmup_iters} iterations...")
    warmup_prompt = prompts[0]
    for i in range(warmup_iters):
        _ = generate_image(
            pipe, warmup_prompt, RANDOM_SEED, device, num_inference_steps
        )
        print(f"  Warmup {i + 1}/{warmup_iters} complete")

    # ----- Generate baseline images -----
    print("\n" + "-" * 80)
    print("Phase 1: Generating baseline images (regular SDPA)")
    print("-" * 80)

    baseline_data = []
    baseline_times = []

    for idx, prompt in enumerate(prompts):
        print(f"[{idx + 1}/{len(prompts)}] Generating baseline: {prompt[:50]}...")
        t0 = time.time()
        baseline_img = generate_image(
            pipe, prompt, RANDOM_SEED, device, num_inference_steps
        )
        t1 = time.time()

        baseline_tensor = pil_to_lpips_tensor(baseline_img, device)
        baseline_data.append((prompt, baseline_tensor))
        baseline_times.append(t1 - t0)

    avg_baseline_time = sum(baseline_times) / len(baseline_times)
    print(
        f"\nBaseline generation complete. Avg time per image: {avg_baseline_time:.2f}s"
    )

    # ----- Apply low-precision attention -----
    print("\n" + "-" * 80)
    print("Phase 2: Generating FP8 attention images")
    print("-" * 80)

    if compile:
        print("Restoring original transformer before applying FP8 attention...")
        pipe.transformer = orig_transformer

    print(f"Applying low-precision attention (config: {config_str})...")
    apply_low_precision_attention(pipe.transformer, ATTENTION_CONFIG)

    if compile:
        print("\nCompiling FP8 attention transformer with torch.compile...")
        pipe.transformer = torch.compile(pipe.transformer)

    # Warmup FP8 path
    print(f"Warming up FP8 attention with {warmup_iters} iterations...")
    for i in range(warmup_iters):
        _ = generate_image(
            pipe, warmup_prompt, RANDOM_SEED, device, num_inference_steps
        )
        print(f"  FP8 warmup {i + 1}/{warmup_iters} complete")

    # Generate FP8 images and compute LPIPS
    lpips_values = []
    fp8_times = []

    for idx, (prompt, baseline_tensor) in enumerate(baseline_data):
        print(f"[{idx + 1}/{len(prompts)}] Generating FP8 attention: {prompt[:50]}...")

        t0 = time.time()
        fp8_img = generate_image(pipe, prompt, RANDOM_SEED, device, num_inference_steps)
        t1 = time.time()
        fp8_times.append(t1 - t0)

        fp8_tensor = pil_to_lpips_tensor(fp8_img, device)
        lpips_value = loss_fn(baseline_tensor, fp8_tensor).item()
        lpips_values.append(lpips_value)

        print(f"    LPIPS: {lpips_value:.4f}, Time: {t1 - t0:.2f}s")

    avg_fp8_time = sum(fp8_times) / len(fp8_times)

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
    print(f"  Avg baseline time:      {avg_baseline_time:.2f}s per image")
    print(f"  Avg FP8 attention time: {avg_fp8_time:.2f}s per image")
    print(f"  Speedup:                {avg_baseline_time / avg_fp8_time:.2f}x")

    print("\nBenchmark Configuration:")
    print(f"  Attention config:  {config_str}")
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
        "speedup": avg_baseline_time / avg_fp8_time,
        "lpips_values": lpips_values,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP8 attention accuracy on FLUX.1-schnell"
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
        help="Wrap the model with torch.compile for both baseline and FP8 modes",
    )

    args = parser.parse_args()

    run_benchmark(
        num_prompts=args.num_prompts,
        num_inference_steps=args.num_inference_steps,
        debug_prompt=args.debug_prompt,
        warmup_iters=args.warmup_iters,
        compile=args.compile,
    )


if __name__ == "__main__":
    main()
