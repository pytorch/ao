# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Tutorial showing how integer 8-bit quantization works with torchao.

Demonstrates Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig,
and Int8StaticActivationInt8WeightConfig with minimal examples.
"""

import torch

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_primitives import MappingType


class ToyLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 256)

    def forward(self, x):
        return self.linear(x)


def example_weight_only():
    """Weight-only quantization"""
    print("\n1. Weight-Only Quantization")
    model = ToyLinearModel()

    # Per-row granularity (better accuracy)
    quantize_(model, Int8WeightOnlyConfig(granularity=PerRow()))
    print("   Quantized with PerRow granularity")

    # Or per-tensor (simpler)
    model2 = ToyLinearModel()
    quantize_(model2, Int8WeightOnlyConfig(granularity=PerTensor()))
    print("   Quantized with PerTensor granularity")


def example_dynamic():
    """Dynamic activation + weight quantization"""
    print("\n2. Dynamic Activation + Weight Quantization")
    model = ToyLinearModel()

    quantize_(
        model,
        Int8DynamicActivationInt8WeightConfig(
            granularity=PerRow(), act_mapping_type=MappingType.SYMMETRIC
        ),
    )
    print("   Quantized weights statically, activations dynamically")


def example_static():
    """Static activation + weight quantization (requires calibration)"""
    print("\n3. Static Activation + Weight Quantization")
    model = ToyLinearModel()

    # Step 1: Calibrate to get activation scale
    calibration_data = torch.randn(100, 512)
    observer = AffineQuantizedMinMaxObserver(
        mapping_type=MappingType.SYMMETRIC,
        target_dtype=torch.int8,
        granularity=PerRow(),
        eps=torch.finfo(torch.float32).eps,
        scale_dtype=torch.float32,
        zero_point_dtype=torch.int32,
    )

    with torch.no_grad():
        activations = model.linear(calibration_data)
        observer(activations)

    scale, _ = observer.calculate_qparams()  # Returns (scale, zero_point)

    # Step 2: Quantize with calibrated scale
    quantize_(
        model,
        Int8StaticActivationInt8WeightConfig(
            scale=scale, granularity=PerRow(), act_mapping_type=MappingType.SYMMETRIC
        ),
    )
    print("   Quantized with calibrated activation scale")


def main():
    print("=" * 50)
    print("TorchAO Int8 Quantization Tutorial")
    print("=" * 50)

    example_weight_only()
    example_dynamic()
    example_static()

    print("\n" + "=" * 50)
    print("Completed! Use torch.compile for best performance.")
    print("\nFor more information:")
    print("  • Benchmarks: benchmarks/quantization/")
    print("  • README: torchao/quantization/README.md")
    print("=" * 50)


if __name__ == "__main__":
    main()
