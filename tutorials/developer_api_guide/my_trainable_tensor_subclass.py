# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
This tutorial demonstrates quantizing a trained model for inference.

Shows:
  • Training a model normally in FP32
  • Applying post-training int8 quantization
  • Evaluating accuracy impact
"""

import torch
import torch.nn as nn

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerRow


class ToyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256, bias=False)
        self.linear2 = nn.Linear(256, 128, bias=False)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def main():
    # Step 1: Train model (simplified - normally you'd train on real data)
    print("Step 1: Training model...")
    model = ToyLinearModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(10):
        x = torch.randn(32, 512)
        y = torch.randn(32, 128)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Step 2: Evaluate baseline
    model.eval()
    test_input = torch.randn(16, 512)
    baseline_output = model(test_input)
    print(f"Baseline output mean: {baseline_output.mean():.4f}")

    # Step 3: Quantize with different configs
    print("\nStep 2: Applying quantization...")

    # Weight-only quantization
    import copy

    model_wo = copy.deepcopy(model)
    quantize_(model_wo, Int8WeightOnlyConfig(granularity=PerRow()))
    wo_output = model_wo(test_input)
    wo_error = (baseline_output - wo_output).abs().mean()
    print(f"Weight-Only - Error: {wo_error:.6f}")

    # Dynamic activation + weight quantization
    model_dyn = copy.deepcopy(model)
    quantize_(model_dyn, Int8DynamicActivationInt8WeightConfig(granularity=PerRow()))
    dyn_output = model_dyn(test_input)
    dyn_error = (baseline_output - dyn_output).abs().mean()
    print(f"Dynamic Act+Weight - Error: {dyn_error:.6f}")

    # Step 3: Compile for speedup
    print("\nStep 3: Compiling quantized model...")
    model_compiled = torch.compile(model_dyn, mode="max-autotune")
    _ = model_compiled(test_input)  # Warmup compilation
    print("Quantized model compiled! Ready for fast inference.")

    print("\nFor benchmarks and more info, see:")
    print("  • benchmarks/quantization/")
    print("  • torchao/quantization/README.md")


if __name__ == "__main__":
    main()
