# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch

from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.testing.model_architectures import ToyMultiLinearModel
from torchao.utils import benchmark_model

# ================
# | Set up model |
# ================

model = ToyMultiLinearModel(1024, 1024, 1024).eval().to(torch.bfloat16).to("cuda")

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

num_runs = 100
torch._dynamo.reset()
example_inputs = (torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda"),)
bf16_time = benchmark_model(model_bf16, num_runs, example_inputs)
int4_time = benchmark_model(model, num_runs, example_inputs)

print("bf16 mean time: %0.3f ms" % bf16_time)
print("int4 mean time: %0.3f ms" % int4_time)
print("speedup: %0.1fx" % (bf16_time / int4_time))
