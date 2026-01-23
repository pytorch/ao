# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for evaluating FP8 quantization accuracy on Stable Diffusion 3.5.

Compares regular inference with FP8 variants using LPIPS (perceptual similarity).
Uses DrawBench dataset for standardized prompt evaluation.

Modes:
    - fp8_sdpa: Replace SDPA with FP8 SDPA (quantize Q/K/V before attention)
    - fp8_linear: Replace linear layers with FP8 dynamic activation + FP8 weight

Usage:
    # FP8 SDPA with 50 prompts
    python eval_sd35_fp8_sdpa.py --mode fp8_sdpa --num_prompts 50

    # FP8 Linear with 50 prompts
    python eval_sd35_fp8_sdpa.py --mode fp8_linear --num_prompts 50

    # Full benchmark with 200 prompts
    python eval_sd35_fp8_sdpa.py --mode fp8_sdpa --num_prompts 200

    # Debug with single prompt
    python eval_sd35_fp8_sdpa.py --debug_prompt "A photo of an astronaut riding a horse"
"""

import argparse
import random
import time
from typing import Optional

import lpips
import numpy as np
import torch
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from torchao.prototype.fp8_sdpa_inference import wrap_module_with_fp8_sdpa
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    quantize_,
)

# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = (512, 512)  # (width, height) - resize for consistent LPIPS
RANDOM_SEED = 42
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"


def pil_to_lpips_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """
    Convert a PIL Image to a tensor suitable for LPIPS computation.

    Args:
        img: PIL Image to convert
        device: Device to place the tensor on ('cuda' or 'cpu')

    Returns:
        Tensor in shape (1, 3, H, W) normalized to [-1, 1]
    """
    t = (
        torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                .view(img.size[1], img.size[0], 3)
                .numpy()
            )
        ).float()
        / 255.0
    )  # [0, 1]
    # reshape to (1, 3, H, W) and scale to [-1, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
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
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    # Resize to fixed size for consistent LPIPS comparison
    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    return image


@torch.inference_mode()
def run_benchmark(
    mode: str = "fp8_sdpa",
    num_prompts: int = 50,
    num_inference_steps: int = 20,
    debug_prompt: Optional[str] = None,
    warmup_iters: int = 2,
):
    """
    Run the FP8 accuracy benchmark on Stable Diffusion 3.5.

    Args:
        mode: Quantization mode - "fp8_sdpa" or "fp8_linear"
        num_prompts: Number of prompts to use (50 or 200 recommended)
        num_inference_steps: Number of diffusion steps per image
        debug_prompt: If specified, use only this prompt (for debugging)
        warmup_iters: Number of warmup iterations before benchmarking
    """
    mode_display = "FP8 SDPA" if mode == "fp8_sdpa" else "FP8 Linear"
    print("=" * 80)
    print(f"{mode_display} Accuracy Benchmark for Stable Diffusion 3.5")
    print("=" * 80)

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cuda"

    # -----------------------------
    # 1. Load prompts
    # -----------------------------
    if debug_prompt is not None:
        prompts = [debug_prompt]
        print(f"Using debug prompt: {debug_prompt}")
    else:
        print("Loading DrawBench dataset...")
        dataset = load_dataset("sayakpaul/drawbench", split="train")
        all_prompts = [item["Prompts"] for item in dataset]
        prompts = all_prompts[:num_prompts]
        print(
            f"Using {len(prompts)} prompts from DrawBench (total available: {len(all_prompts)})"
        )

    # -----------------------------
    # 2. Load model and LPIPS
    # -----------------------------
    print(f"\nLoading Stable Diffusion 3.5 from {MODEL_ID}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print("Loading LPIPS model (VGG)...")
    loss_fn = lpips.LPIPS(net="vgg").to(device)

    # -----------------------------
    # 3. Warmup
    # -----------------------------
    print(f"\nWarming up with {warmup_iters} iterations...")
    warmup_prompt = prompts[0]
    for i in range(warmup_iters):
        _ = generate_image(
            pipe, warmup_prompt, RANDOM_SEED, device, num_inference_steps
        )
        print(f"  Warmup {i + 1}/{warmup_iters} complete")

    # -----------------------------
    # 4. Generate baseline images (regular SDPA)
    # -----------------------------
    print("\n" + "-" * 80)
    print("Phase 1: Generating baseline images (regular SDPA)")
    print("-" * 80)

    baseline_data = []  # List of (prompt, baseline_tensor)
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

    # -----------------------------
    # 5. Apply FP8 quantization based on mode
    # -----------------------------
    print("\n" + "-" * 80)
    print(f"Phase 2: Generating {mode_display} images")
    print("-" * 80)

    if mode == "fp8_sdpa":
        # Wrap the transformer to use FP8 SDPA
        print("Wrapping transformer with FP8 SDPA...")
        pipe.transformer = wrap_module_with_fp8_sdpa(pipe.transformer)
    elif mode == "fp8_linear":
        # Quantize linear layers with FP8 dynamic activation + FP8 weight
        print("Quantizing transformer linear layers with FP8...")
        quantize_(
            pipe.transformer,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Warmup FP8 path
    print(f"Warming up {mode_display} with {warmup_iters} iterations...")
    for i in range(warmup_iters):
        _ = generate_image(
            pipe, warmup_prompt, RANDOM_SEED, device, num_inference_steps
        )
        print(f"  {mode_display} warmup {i + 1}/{warmup_iters} complete")

    # Generate FP8 images and compute LPIPS
    lpips_values = []
    fp8_times = []

    for idx, (prompt, baseline_tensor) in enumerate(baseline_data):
        print(f"[{idx + 1}/{len(prompts)}] Generating {mode_display}: {prompt[:50]}...")

        t0 = time.time()
        fp8_img = generate_image(pipe, prompt, RANDOM_SEED, device, num_inference_steps)
        t1 = time.time()
        fp8_times.append(t1 - t0)

        # Compute LPIPS
        fp8_tensor = pil_to_lpips_tensor(fp8_img, device)
        lpips_value = loss_fn(baseline_tensor, fp8_tensor).item()
        lpips_values.append(lpips_value)

        print(f"    LPIPS: {lpips_value:.4f}, Time: {t1 - t0:.2f}s")

    avg_fp8_time = sum(fp8_times) / len(fp8_times)

    # -----------------------------
    # 6. Summary and results
    # -----------------------------
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
    print(f"  Avg baseline time:   {avg_baseline_time:.2f}s per image")
    print(f"  Avg {mode_display} time: {avg_fp8_time:.2f}s per image")
    print(f"  Speedup:             {avg_baseline_time / avg_fp8_time:.2f}x")

    print("\nBenchmark Configuration:")
    print(f"  Mode:              {mode_display}")
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
        description="Benchmark FP8 quantization accuracy on Stable Diffusion 3.5"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fp8_sdpa",
        choices=["fp8_sdpa", "fp8_linear"],
        help="Quantization mode: fp8_sdpa (FP8 SDPA) or fp8_linear (FP8 linear layers)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=50,
        help="Number of prompts to use (50 for quick, 200 for full benchmark)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
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

    args = parser.parse_args()

    run_benchmark(
        mode=args.mode,
        num_prompts=args.num_prompts,
        num_inference_steps=args.num_inference_steps,
        debug_prompt=args.debug_prompt,
        warmup_iters=args.warmup_iters,
    )


if __name__ == "__main__":
    main()
