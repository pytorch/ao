# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test FP8 SDPA with Stable Diffusion model from diffusers.
"""

import torch
from diffusers import StableDiffusionPipeline
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_utils import (
    wrap_module_with_fp8_sdpa,
)


def test_stable_diffusion_regular_sdpa():
    from torch.profiler import ProfilerActivity, profile

    """
    Load Stable Diffusion and run regular SDPA.
    """

    # Load Stable Diffusion pipeline
    # Using SD 2.1 as an example - uses SDPA internally
    model_id = "Manojb/stable-diffusion-2-1-base"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    # Run inference
    prompt = "A photo of an astronaut riding a unicorn on Mars"

    # Warmup
    for _ in range(10):
        _ = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]

    # Add the pytorch profiler here too
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]
    # Log to output file instead of printing
    with open(
        "./torchao/prototype/fp8_sdpa_inference/outputs/regular_sdpa_stable_diffusion_profiler.txt",
        "w",
    ) as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    # Save the result
    image.save(
        "./torchao/prototype/fp8_sdpa_inference/outputs/regular_sdpa_stable_diffusion_output.png"
    )
    print(
        "Image saved to ./torchao/prototype/fp8_sdpa_inference/outputs/regular_sdpa_stable_diffusion_output.png"
    )

    return image


def test_stable_diffusion_fp8_sdpa():
    from torch.profiler import ProfilerActivity, profile

    """
    Load Stable Diffusion and convert attention to FP8 SDPA.
    """

    # Load Stable Diffusion pipeline
    # Using SD 2.1 as an example - uses SDPA internally
    model_id = "Manojb/stable-diffusion-2-1-base"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    # Wrap UNet to use FP8 SDPA only during its forward pass
    # VAE and text encoder will use regular SDPA
    pipe.unet = wrap_module_with_fp8_sdpa(pipe.unet)

    # Run inference
    prompt = "A photo of an astronaut riding a unicorn on Mars"

    # Warmup
    for _ in range(10):
        _ = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]

    # Add the pytorch profiler here too
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]
    # Log to output file instead of printing
    with open(
        "./torchao/prototype/fp8_sdpa_inference/outputs/fp8_sdpa_stable_diffusion_profiler.txt",
        "w",
    ) as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    # Save the result
    image.save(
        "./torchao/prototype/fp8_sdpa_inference/outputs/fp8_sdpa_stable_diffusion_output.png"
    )
    print(
        "Image saved to ./torchao/prototype/fp8_sdpa_inference/outputs/fp8_sdpa_stable_diffusion_output.png"
    )

    return image


def test_fp8_sdpa_numerical_accuracy():
    """
    Compare FP8 SDPA output with regular SDPA for numerical accuracy.
    """
    import torch.nn.functional as F

    from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_attention import (
        fp8_sdpa_parallel,
    )

    # Create test inputs matching typical SD attention shapes
    # UNet attention: (B, H, S, D) where S can be large (e.g., 4096 for 64x64 latents)
    B, H, S, D = 2, 8, 1024, 64

    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        # Regular SDPA
        out_regular = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # FP8 SDPA
        activate_flash_attention_impl("FA3")
        out_fp8 = fp8_sdpa_parallel(q, k, v, is_causal=False)
        restore_flash_attention_impl()

    # Test SQNR
    sqnr = 10 * torch.log10(
        torch.mean(out_regular.pow(2)) / torch.mean((out_regular - out_fp8).pow(2))
    )
    print(f"SQNR: {sqnr.item():.2f} dB")

    assert sqnr > 25, "FP8 SDPA is not accurate enough"
    print("SQNR test passed")


def test_fp8_sdpa_benchmark():
    """
    Benchmark FP8 SDPA vs regular SDPA.
    """
    import time

    import torch.nn.functional as F

    from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_attention import (
        fp8_sdpa_parallel,
    )

    # Test various sizes
    configs = [
        (1, 8, 1024, 64),  # Small
        (1, 8, 4096, 64),  # Medium (typical SD)
        (1, 16, 4096, 128),  # Large
        (2, 16, 4096, 128),  # Batched
    ]

    warmup_iters = 10
    bench_iters = 100

    print("\nBenchmark Results:")
    print("-" * 70)
    print(
        f"{'Config (B,H,S,D)':<25} {'Regular (ms)':<15} {'FP8 (ms)':<15} {'Speedup':<10}"
    )
    print("-" * 70)

    for B, H, S, D in configs:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        # Warmup regular
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        # Benchmark regular
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        regular_time = (time.perf_counter() - start) / bench_iters * 1000

        # Warmup FP8
        with torch.no_grad():
            for _ in range(warmup_iters):
                activate_flash_attention_impl("FA3")
                _ = fp8_sdpa_parallel(q, k, v)
                restore_flash_attention_impl()
        torch.cuda.synchronize()

        # Benchmark FP8
        activate_flash_attention_impl("FA3")
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                _ = fp8_sdpa_parallel(q, k, v)
        torch.cuda.synchronize()
        fp8_time = (time.perf_counter() - start) / bench_iters * 1000
        restore_flash_attention_impl()

        speedup = regular_time / fp8_time
        config_str = f"({B}, {H}, {S}, {D})"
        print(
            f"{config_str:<25} {regular_time:<15.3f} {fp8_time:<15.3f} {speedup:<10.2f}x"
        )

    print("-" * 70)


def test_fp8_sdpa_profiler():
    """
    Profiler FP8 SDPA vs regular SDPA.
    """
    from torch.nn.functional import scaled_dot_product_attention
    from torch.profiler import ProfilerActivity, profile

    from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_attention import (
        fp8_sdpa_parallel,
    )

    q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
    activate_flash_attention_impl("FA3")

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = fp8_sdpa_parallel(q, k, v)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            _ = fp8_sdpa_parallel(q, k, v)
    # Log to output file
    with open(
        "./torchao/prototype/fp8_sdpa_inference/outputs/fp8_sdpa_profiler.txt", "w"
    ) as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    restore_flash_attention_impl()

    # Now do the same for regular SDPA

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = scaled_dot_product_attention(q, k, v)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            _ = scaled_dot_product_attention(q, k, v)
    # Log to output file
    with open(
        "./torchao/prototype/fp8_sdpa_inference/outputs/regular_sdpa_profiler.txt", "w"
    ) as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    print("=" * 70)
    print("FP8 SDPA Inference Tests")
    print("=" * 70)
    print()

    # # Run numerical accuracy test first
    # test_fp8_sdpa_numerical_accuracy()
    # print()

    # Run benchmark
    test_fp8_sdpa_benchmark()
    print()

    # Run profiler
    test_fp8_sdpa_profiler()

    # # Run Stable Diffusion test
    # print("Testing with Stable Diffusion...")
    # test_stable_diffusion_fp8_sdpa()

    # print("Testing with Regular SDPA...")
    # test_stable_diffusion_regular_sdpa()
