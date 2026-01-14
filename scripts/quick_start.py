# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch

# ================
# | Set up model |
# ================


@torch.compile(mode="max-autotune", fullgraph=True)
class ToyLinearModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dtype,
        device,
        has_bias=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.linear1 = torch.nn.Linear(
            input_dim, hidden_dim, bias=has_bias, dtype=dtype, device=device
        )
        self.linear2 = torch.nn.Linear(
            hidden_dim, output_dim, bias=has_bias, dtype=dtype, device=device
        )

    def example_inputs(self, batch_size=1):
        return (
            torch.randn(
                batch_size,
                self.linear1.in_features,
                dtype=self.dtype,
                device=self.device,
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


model_w16a16 = ToyLinearModel(
    1024, 1024, 1024, device="cuda", dtype=torch.bfloat16
).eval()
model_w8a8 = copy.deepcopy(model_w16a16)  # We will quantize in next chapter!

# ========================
# | torchao quantization |
# ========================

from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

quantize_(model_w8a8, Int8DynamicActivationInt8WeightConfig())
print(type(model_w8a8.linear1.weight).__name__)

# ========================
# | Model Size Comparison |
# ========================

import os

# Save models
torch.save(model_w16a16.state_dict(), "model_w16a16.pth")
torch.save(model_w8a8.state_dict(), "model_w8a8.pth")

# Compare file sizes
original_size = os.path.getsize("model_w16a16.pth") / 1024**2
quantized_size = os.path.getsize("model_w8a8.pth") / 1024**2
print(
    f"Size reduction: {original_size / quantized_size:.2f}x ({original_size:.2f}MB -> {quantized_size:.2f}MB)"
)

# ========================
# | Throughput (Speedup) |
# ========================

import time

# Get example inputs
example_inputs = model_w8a8.example_inputs(batch_size=128)

# Warmup
for _ in range(10):
    _ = model_w8a8(*example_inputs)
    _ = model_w16a16(*example_inputs)
torch.cuda.synchronize()

# Throughput: original model
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = model_w16a16(*example_inputs)
torch.cuda.synchronize()
original_time = time.time() - start

# Throughput: Quantized (W8A8-INT) model
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = model_w8a8(*example_inputs)
torch.cuda.synchronize()
quantized_time = time.time() - start

print(f"Speedup: {original_time / quantized_time:.2f}x")
