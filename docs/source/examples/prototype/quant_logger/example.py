"""
Simple example demonstrating quant_logger parameter and activation logging.

Usage:
    python example.py
"""

import torch
from diffusers import DiffusionPipeline

from torchao.prototype.quant_logger.quant_logger import (
    add_activation_loggers,
    log_parameter_info,
    reset_counter,
)

# Load model
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Log parameter statistics
print("=" * 70)
print("Parameter statistics:")
print("=" * 70)
log_parameter_info(pipe.transformer)

# Reset logging counter before logging activations
reset_counter()

# Add activation loggers
add_activation_loggers(pipe.transformer)

# Generate one image
print("=" * 70)
print("Activation statistics during inference:")
print("=" * 70)
result = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    # `num_inference_steps` is usually 4 for FLUX.1-schnell, but set to 1
    # for the purposes of this demo
    num_inference_steps=1,
    generator=torch.manual_seed(0),
)
