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

## Quantization code - start
# int8 dynamic quantization act, int8 weight, see ao/torchao/quantization/README.md
# for APIs for earlier torch version and other quantization techniques

# for torch 2.4+
from torchao.quantization.quant_api import (
    int8_dynamic_activation_int8_weight,
    quantize_,
)

quantize_(model, int8_dynamic_activation_int8_weight())
## Quantization code - end

## compilation configs
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
## compilation configs end

# temporary workaround for the API to work with torch.compile
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, unwrap_tensor_subclass

if not TORCH_VERSION_AT_LEAST_2_5:
    unwrap_tensor_subclass(model)

# temporary workaround to recover the perf with quantized model under torch.compile
torch.backends.mha.set_fastpath_enabled(False)

model = torch.compile(model, mode="max-autotune")

# Must run with no_grad when optimizing for inference
with torch.no_grad():
    # warmup
    benchmark_model(model, 20, inputs)
    # benchmark
    print("elapsed_time: ", benchmark_model(model, 1000, inputs), " milliseconds")
    # Create a trace
    profiler_runner("quant.json.gz", benchmark_model, model, 5, inputs)
