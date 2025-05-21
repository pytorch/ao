# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision import models

from torchao.utils import benchmark_model, profiler_runner

torch.set_float32_matmul_precision("high")
# Load Vision Transformer model
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Set the model to evaluation mode
model.eval().cuda().to(torch.bfloat16)

# Input tensor (batch_size, channels, height, width)
inputs = (torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device="cuda"),)

model = torch.compile(model, mode="max-autotune")

# Must run with no_grad when optimizing for inference
with torch.no_grad():
    # warmup
    benchmark_model(model, 5, inputs)
    # benchmark
    print("elapsed_time: ", benchmark_model(model, 100, inputs), " milliseconds")
    # Create a trace
    profiler_runner("bfloat16.json.gz", benchmark_model, model, 5, inputs)
