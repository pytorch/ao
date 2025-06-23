#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple usage example for AWQ with QDQLayout and ExecuTorch support.

This example demonstrates the complete workflow for using AWQ quantization
with QDQLayout support and 8-bit dynamic activation quantization.
"""

import torch
import torch.nn as nn

from torchao.prototype.awq import (
    insert_awq_observer_qdq_,
    AWQQDQConfig,
)
from torchao.prototype.awq.executorch_awq import _is_awq_observed_linear_qdq
from torchao.quantization import quantize_


def main():
    print("AWQ + QDQLayout + ExecuTorch Example")
    print("=" * 40)

    # 1. Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
    )

    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Insert AWQ observers with QDQLayout support
    print("\n1. Inserting AWQ observers...")
    insert_awq_observer_qdq_(
        model,
        n_validation_examples=5,
        validation_sequence_len=64,
        quant_dtype=torch.uint4,
        group_size=128,
        use_dynamic_activation_quant=True,  # Enable 8-bit dynamic activation quantization
    )

    print("   Observers inserted successfully!")

    # 3. Calibrate the model
    print("\n2. Calibrating model...")
    model.eval()
    with torch.no_grad():
        for i in range(5):
            # Generate random calibration data
            example_input = torch.randn(2, 64, 512)
            model(example_input)
            print(f"   Calibration step {i + 1}/5 completed")

    print("   Calibration completed!")

    # 4. Apply AWQ quantization with QDQLayout
    print("\n3. Applying AWQ quantization with QDQLayout...")
    config = AWQQDQConfig(
        quant_dtype=torch.uint4,
        group_size=128,
        use_dynamic_activation_quant=True,
    )

    # Use the custom filter to target AWQObservedLinearQDQ modules
    quantize_(model, config, filter_fn=_is_awq_observed_linear_qdq)

    print("   Quantization applied successfully!")

    # 5. Test the quantized model
    print("\n4. Testing quantized model...")
    test_input = torch.randn(1, 64, 512)

    with torch.no_grad():
        output = model(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")

    # 6. Verify QDQLayout usage
    print("\n5. Verifying QDQLayout tensors...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            if hasattr(weight, "__tensor_flatten__"):
                print(f"   ✓ {name}: Uses quantized tensor (QDQLayout)")
                # Check for QDQLayout specific attributes
                if hasattr(weight, "int_data"):
                    print(f"     - int_data shape: {weight.int_data.shape}")
                    print(f"     - scale shape: {weight.scale.shape}")
            else:
                print(f"   ✗ {name}: Uses regular tensor")

    print("\n" + "=" * 40)
    print("AWQ + QDQLayout quantization completed successfully!")
    print("The model is now ready for ExecuTorch export.")

    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run the example
    quantized_model = main()

    print(f"\nFinal model type: {type(quantized_model)}")
    print("Example completed successfully!")
