# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import random
import time

import fire
import lpips
import numpy as np
import torch
from datasets import load_dataset
from diffusers import FluxPipeline
from PIL import Image, ImageDraw, ImageFont
from utils import string_to_config

from torchao.quantization import (
    FqnToConfig,
    quantize_,
)

# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = (512, 512)  # (width, height)
OUTPUT_DIR = "benchmarks/data/flux_eval"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_pipeline_architecture(pipe):
    """
    Print the PyTorch model architecture for each component of a diffusion pipeline.

    Args:
        pipe: The diffusion pipeline to inspect
    """
    print("\n" + "=" * 80)
    print("DIFFUSION PIPELINE COMPONENTS")
    print("=" * 80)

    # Iterate through components specified in the model config
    total_params = 0
    components = ["vae", "transformer", "text_encoder", "text_encoder_2"]
    for idx, component_name in enumerate(components, 1):
        component = getattr(pipe, component_name)
        print("\n" + "-" * 80)
        print(f"{idx}. {component_name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(component)
        param_count = sum(p.numel() for p in component.parameters())
        print(f"\n{component_name} Parameter Count: {param_count:,}")
        total_params += param_count

    print("\n" + "-" * 80)
    print("Other Components (Non-Neural)")
    print("-" * 80)
    print(f"Tokenizer: {type(pipe.tokenizer).__name__}")
    print(f"Scheduler: {type(pipe.scheduler).__name__}")

    print("\n" + "=" * 80)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("=" * 80 + "\n")


def generate_image(
    pipe, prompt: str, seed: int, device: str, num_inference_steps: int
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,  # can tweak for speed vs quality
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    # Resize (if needed) to a fixed size so LPIPS sees consistent shapes
    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    return image


def create_comparison_image(
    baseline_img: Image.Image,
    modified_img: Image.Image,
    lpips_score: float,
    prompt: str = None,
    margin_top: int = 80,
) -> Image.Image:
    """
    Create a comparison image by stacking two images horizontally with a top margin
    and overlaying the prompt text and LPIPS score.

    Args:
        baseline_img: The baseline image
        modified_img: The modified/quantized image
        lpips_score: The LPIPS score between the two images
        prompt: Optional prompt text to display at the top
        margin_top: Height of the top margin for text (default 80 to fit prompt + LPIPS)
    """
    # Get dimensions
    width1, height1 = baseline_img.size
    width2, height2 = modified_img.size

    # Create new image with top margin
    total_width = width1 + width2
    total_height = max(height1, height2) + margin_top

    # Create composite image with dark gray background for margin
    composite = Image.new("RGB", (total_width, total_height), color=(50, 50, 50))

    # Paste the two images side by side, offset by margin_top
    composite.paste(baseline_img, (0, margin_top))
    composite.paste(modified_img, (width1, margin_top))

    # Add text overlay with prompt and LPIPS score
    draw = ImageDraw.Draw(composite)

    # Try to use reasonable font sizes, fallback to default if truetype fails
    try:
        prompt_font = ImageFont.truetype("arial.ttf", 20)
        lpips_font = ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        prompt_font = ImageFont.load_default()
        lpips_font = ImageFont.load_default()

    # Draw prompt text at the top if provided
    y_offset = 5
    if prompt:
        # Wrap prompt text if it's too long
        max_width = total_width - 20  # 10px padding on each side
        prompt_lines = []
        words = prompt.split()
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=prompt_font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    prompt_lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            prompt_lines.append(" ".join(current_line))

        # Draw each line of the prompt
        for line in prompt_lines:
            bbox = draw.textbbox((0, 0), line, font=prompt_font)
            text_width = bbox[2] - bbox[0]
            text_x = (total_width - text_width) // 2
            draw.text((text_x, y_offset), line, fill=(200, 200, 200), font=prompt_font)
            y_offset += (bbox[3] - bbox[1]) + 2  # line height + small gap

    # Format the LPIPS text
    lpips_text = f"LPIPS: {lpips_score:.4f}"

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), lpips_text, font=lpips_font)
    text_width = bbox[2] - bbox[0]

    # Center the LPIPS text horizontally, place it below the prompt
    text_x = (total_width - text_width) // 2
    text_y = y_offset + 5  # small gap after prompt

    # Draw LPIPS text in white
    draw.text((text_x, text_y), lpips_text, fill=(255, 255, 255), font=lpips_font)

    return composite


def create_combined_comparison_image(
    comparison_images: list[Image.Image],
) -> Image.Image:
    """
    Stack multiple comparison images vertically into a single combined image.

    Args:
        comparison_images: List of comparison images to stack vertically

    Returns:
        Combined image with all comparisons stacked vertically
    """
    if not comparison_images:
        raise ValueError("comparison_images list cannot be empty")

    # Calculate dimensions
    total_height = sum(img.size[1] for img in comparison_images)
    max_width = max(img.size[0] for img in comparison_images)

    # Create combined image
    combined_img = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for comp_img in comparison_images:
        combined_img.paste(comp_img, (0, y_offset))
        y_offset += comp_img.size[1]

    return combined_img


def pil_to_lpips_tensor(img: Image.Image, device: str):
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


@torch.inference_mode()
def run(
    num_prompts: int = None,
    num_inference_steps: int = 20,
    quant_config_str: str = "float8_rowwise",
    use_compile: bool = False,
    torch_compile_mode: str = "default",
    debug_prompt: str | None = None,
    print_model: bool = False,
):
    """
    A performance and accuracy eval script for quantizing flux-1.dev:

      1. load flux-1.dev model
      2. run it on a prompts dataset and save the images
      3. quantize the model, run it on the same dataset and save the images
      4. report performance difference between 2 and 3
      5. report accuracy difference (using LPIPS) between 2 and 3

    Args:
        num_prompts: Optional limit on number of prompts to use (for debugging)
        num_inference_steps: Number of passes through the transformer,
          default 20 for flux-1.dev. Can set to 1 for speeding up debugging.
        quant_config_str: Quantization config to use ('float8_rowwise'). Default: 'float8_rowwise'
        use_compile: if true, uses torch.compile
        torch_compile_mode: mode to use torch.compile with
        debug_prompt: if specified, use this prompt instead of the drawbench dataset
        print_model: if True, prints model architecture
    """
    # TODO(future): maybe support other models and datasets
    model = "black-forest-labs/FLUX.1-dev"
    prompts_dataset = "sayakpaul/drawbench"
    if debug_prompt is not None:
        prompts_dataset = "debug"

    print(f"Model: {model}")
    print(f"Quant config: {quant_config_str}")
    print(f"num_inference_steps: {num_inference_steps}")
    print(f"prompts_dataset: {prompts_dataset}")
    print(f"use_compile: {use_compile}")
    print(f"torch_compile_mode: {torch_compile_mode}")

    # Create model-specific output directory
    output_dir = os.path.join(OUTPUT_DIR, model)
    os.makedirs(output_dir, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load model
    device = "cuda"
    pipe = FluxPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    )
    print("Moving model to device")
    pipe = pipe.to(device)

    loss_fn = lpips.LPIPS(net="vgg").to(device)

    # Store original for restoration later, since we will quantize it
    # and compile the quantized version again
    orig_transformer = pipe.transformer

    if use_compile:
        pipe.transformer = torch.compile(orig_transformer, mode=torch_compile_mode)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=torch_compile_mode)

    # -----------------------------
    # 2. Baseline images (for all prompts)
    # -----------------------------
    # Load prompts from file or HuggingFace dataset
    if debug_prompt is None:
        dataset = load_dataset(prompts_dataset, split="train")
        all_prompts = [item["Prompts"] for item in dataset]
    else:
        all_prompts = [debug_prompt]

    # Limit prompts for debugging if requested
    prompts_to_use = all_prompts if num_prompts is None else all_prompts[:num_prompts]
    print(f"Generating baseline images for {len(prompts_to_use)} prompts")
    baseline_data = []  # List of (prompt_idx, prompt, baseline_img, baseline_t)
    baseline_times = []
    for idx, prompt in enumerate(prompts_to_use):
        prompt_idx = f"prompt_{idx}"
        print(f"Generating baseline for {prompt_idx}: {prompt}")
        t0 = time.time()
        baseline_img = generate_image(
            pipe, prompt, RANDOM_SEED, device, num_inference_steps
        )
        t1 = time.time()
        baseline_t = pil_to_lpips_tensor(baseline_img, device)
        baseline_data.append((prompt_idx, prompt, baseline_img, baseline_t))
        baseline_times.append(t1 - t0)

    if use_compile:
        print("Restoring original (uncompiled) transformer before quantization")
        pipe.transformer = orig_transformer

    # Inspect Linear layers in main component
    component_linear_fqns_and_weight_shapes = []
    for fqn, module in orig_transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_shape = module.weight.shape
            if print_model:
                print(f"  {fqn}: {weight_shape}")
            component_linear_fqns_and_weight_shapes.append([fqn, weight_shape])

    config_obj = string_to_config(quant_config_str)

    # Create FqnToConfig mapping
    fqn_to_config_dict = {}
    for fqn, weight_shape in component_linear_fqns_and_weight_shapes:
        # Hand-crafted heuristic: don't quantize embedding layers, the last two
        # layers, and layers with small weights
        if "embed" in fqn:
            continue
        elif fqn == "norm_out.linear":
            continue
        elif fqn == "proj_out":
            continue
        elif weight_shape[0] < 1024 or weight_shape[1] < 1024:
            continue
        fqn_to_config_dict[fqn] = config_obj
    fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)

    # Quantize the main component using this config
    quantize_(pipe.transformer, fqn_to_config, filter_fn=None)
    if use_compile:
        pipe.transformer = torch.compile(pipe.transformer, mode=torch_compile_mode)
    if print_model:
        print_pipeline_architecture(pipe)

    print("Generating images with quantized model for all prompts")
    lpips_values = []
    comparison_images = []
    times = []
    for prompt_idx, prompt, baseline_img, baseline_t in baseline_data:
        print(f"Generating image for {prompt_idx}")
        t0 = time.time()
        modified_img = generate_image(
            pipe, prompt, RANDOM_SEED, device, num_inference_steps
        )
        t1 = time.time()
        times.append(t1 - t0)

        # Compute LPIPS for fully quantized model
        modified_t = pil_to_lpips_tensor(modified_img, device)
        with torch.no_grad():
            lpips_value = loss_fn(baseline_t, modified_t).item()

        lpips_values.append(lpips_value)
        print(f"LPIPS distance (full quantization, {prompt_idx}): {lpips_value:.4f}")

        # Create and save comparison image
        print("Creating comparison image")
        comparison_img = create_comparison_image(
            baseline_img, modified_img, lpips_value, prompt=prompt
        )
        comparison_images.append(comparison_img)
        comparison_path = os.path.join(
            output_dir,
            f"comparison_prompt_mode_full_quant_config_str_{quant_config_str}_{prompt_idx}.png",
        )
        comparison_img.save(comparison_path)
        print(f"Saved comparison image to: {comparison_path}")

    # Create combined image with all comparisons stacked vertically
    combined_img = create_combined_comparison_image(comparison_images)
    combined_path = os.path.join(
        output_dir,
        f"comparison_prompt_mode_full_quant_config_str_{quant_config_str}_combined.png",
    )
    combined_img.save(combined_path)
    print(f"Saved combined comparison image to: {combined_path}")

    # Print summary
    print("=" * 80)
    print("Test Mode Summary:")
    print(f"  Total Linear layers quantized: {len(fqn_to_config_dict)}")
    print(f"  Prompts tested: {len(baseline_data)}")
    print("")
    print("LPIPS Results:")
    avg_lpips = sum(lpips_values) / len(lpips_values)
    max_lpips = max(lpips_values)
    min_lpips = min(lpips_values)
    print(f"  Average LPIPS: {avg_lpips:.4f}")
    print(f"  Max LPIPS: {max_lpips:.4f}")
    print(f"  Min LPIPS: {min_lpips:.4f}")
    print(f"  All values: {[f'{v:.4f}' for v in lpips_values]}")
    print("=" * 80)
    print("baseline_times", baseline_times)
    print("times", times)
    speedups = [x / y for (x, y) in zip(baseline_times, times)]
    print("speedups", speedups)
    if len(speedups) > 1:
        # ignore first value as it includes torch.compile compilation time
        avg_speedup = sum(speedups[1:]) / len(speedups[1:])
        print("avg speedup ignoring first value", avg_speedup)
    else:
        avg_speedup = sum(speedups) / len(speedups)
        print("avg speedup", avg_speedup)

    # Save summary stats to CSV
    summary_csv_path = os.path.join(
        output_dir,
        f"summary_stats_prompt_mode_full_quant_config_str_{quant_config_str}.csv",
    )
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mode", "full_quant"])
        writer.writerow(["total_linear_layers_quantized", len(fqn_to_config_dict)])
        writer.writerow(["prompts_tested", len(baseline_data)])
        writer.writerow(["average_lpips", f"{avg_lpips:.4f}"])
        writer.writerow(["max_lpips", f"{max_lpips:.4f}"])
        writer.writerow(["min_lpips", f"{min_lpips:.4f}"])
        writer.writerow(["average_speedup", avg_speedup])
        # Write individual LPIPS values
        for idx, val in enumerate(lpips_values):
            writer.writerow([f"lpips_prompt_{idx}", f"{val:.4f}"])
    print(f"Summary stats saved to {summary_csv_path}")


if __name__ == "__main__":
    fire.Fire(run)
