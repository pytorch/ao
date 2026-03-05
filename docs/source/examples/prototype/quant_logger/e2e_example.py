"""
Simple example demonstrating quant_logger parameter and activation logging
on the `FLUX-1.schnell` model.

By default, the logging information is printed to stdout. We provide
two convenience functions:
* `enable_log_stats_to_file` to log to a user specified file instead of stdout
* `enable_log_tensor_save_tensors_to_disk` to log entire tensor contents to a directory

Usage:
    python example.py
"""

import torch
from diffusers import DiffusionPipeline

from torchao.prototype.quant_logger import (
    add_activation_loggers,
    log_parameter_info,
    reset_counter,
)

# Load the model
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Log parameter statistics
print("=" * 70)
print("Parameter statistics:")
print("=" * 70)
log_parameter_info(pipe.transformer)
# example output line:
#
#   t=param, c=0, fqn='time_text_embed.timestep_embedder.linear_1.weight', op='', max=0.29, avg=-0.00, std=0.01
#

# example full output: https://gist.github.com/vkuzo/be3ba5e85bb76e547204badca0bf69b2

# Reset logging counter before logging activations
reset_counter()

# Add activation loggers
add_activation_loggers(pipe.transformer)
# example output line after data is fed through the model:
#
#
#   t=act, c=0, fqn='x_embedder.weight', op='linear', extra='MKN=4096|64|3072', max=3.33, avg=0.00, std=1.00
#
# example full output: https://gist.github.com/vkuzo/be3ba5e85bb76e547204badca0bf69b2

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
