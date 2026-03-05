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
    enable_log_stats_to_file,
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
#   t=param, c=0, fqn='time_text_embed.timestep_embedder.linear_1.weight', op='', max_abs=0.29, avg=-0.00, std=0.01
#

# example full output: https://gist.github.com/vkuzo/1fffca0974d1f59099f3c0d16a3a1834

# Reset logging counter before logging activations
reset_counter()

# Add activation loggers
add_activation_loggers(pipe.transformer)
# example output line after data is fed through the model:
#
#
#   t=act, c=0, fqn='x_embedder.weight', op='linear', extra='MKN=4096|64|3072', max_abs=3.33, avg=0.00, std=1.00
#
# example full output: https://gist.github.com/vkuzo/1fffca0974d1f59099f3c0d16a3a1834

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

#
# Now, override logging to save stats to a CSV file for further analysis
#
import csv
import re
import tempfile

from tabulate import tabulate

stats_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
# Override the logger to write to a file instead of stdout
enable_log_stats_to_file(stats_file.name)
reset_counter()

# Generate another image so activation stats are written to the CSV
result2 = pipe(
    prompt="A dog playing fetch in a park",
    height=1024,
    width=1024,
    num_inference_steps=1,
    generator=torch.manual_seed(1),
)

with open(stats_file.name, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


def parse_mkn(extra):
    match = re.match(r"MKN=(\d+)\|(\d+)\|(\d+)", extra)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


# Print the top 20 fqns by largest max_abs value
sorted_rows = sorted(rows, key=lambda r: float(r["max_abs"]), reverse=True)
top_rows = [[row["fqn"], f"{float(row['max_abs']):.2f}"] for row in sorted_rows[:20]]
print("Top 20 fqns by max_abs:")
print(tabulate(top_rows, headers=["fqn", "max_abs"], tablefmt="simple"))
print()
# example output:
#
#   fqn                                                   max_abs
#   --------------------------------------------------  ---------
#   transformer_blocks.18.ff.net.2.weight                  189
#   ...
#
# example full output: https://gist.github.com/vkuzo/1fffca0974d1f59099f3c0d16a3a1834

# Print layers with small activation shapes (any of M, K, N < 1024)
small_rows = []
for row in rows:
    mkn = parse_mkn(row["extra"])
    if mkn is not None:
        M, K, N = mkn
        if min(M, K, N) < 1024:
            small_rows.append([row["fqn"], M, K, N])
print("Small activation shapes (any of M, K, N < 1024):")
print(tabulate(small_rows, headers=["fqn", "M", "K", "N"], tablefmt="simple"))
# example output:
#
#   fqn                                                    M      K      N
#   --------------------------------------------------  ----  -----  -----
#   x_embedder.weight                                   4096     64   3072
#   ...
#
# example full output: https://gist.github.com/vkuzo/1fffca0974d1f59099f3c0d16a3a1834
