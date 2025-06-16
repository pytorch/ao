# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch

from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    benchmark_model,
    unwrap_tensor_subclass,
)

# ================
# | Set up model |
# ================


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


model = ToyLinearModel(1024, 1024, 1024).eval().to(torch.bfloat16).to("cuda")

# Optional: compile model for faster inference and generation
model = torch.compile(model, mode="max-autotune", fullgraph=True)
model_bf16 = copy.deepcopy(model)


# ========================
# | torchao quantization |
# ========================

# torch 2.4+ only
quantize_(model, Int4WeightOnlyConfig(group_size=32))


# =============
# | Benchmark |
# =============

# Temporary workaround for tensor subclass + torch.compile
# Only needed for torch version < 2.5
if not TORCH_VERSION_AT_LEAST_2_5:
    unwrap_tensor_subclass(model)

num_runs = 100
torch._dynamo.reset()
example_inputs = (torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda"),)
bf16_time = benchmark_model(model_bf16, num_runs, example_inputs)
int4_time = benchmark_model(model, num_runs, example_inputs)

print("bf16 mean time: %0.3f ms" % bf16_time)
print("int4 mean time: %0.3f ms" % int4_time)
print("speedup: %0.1fx" % (bf16_time / int4_time))
