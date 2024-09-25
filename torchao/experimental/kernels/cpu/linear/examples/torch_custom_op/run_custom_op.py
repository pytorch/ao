# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import os

import sys

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)
from quant_api import Int8DynActIntxWeightQuantizer

libs = glob.glob("/tmp/cmake-out/torchao/liblowbit_op_aten.*")
libs = list(filter(lambda l: (l.endswith("so") or l.endswith("dylib")), libs))
torch.ops.load_library(libs[0])

group_size = 256
m = 1
n = 4096
k = 4096
nbit = 4
has_weight_zeros = False
n_layers = 5

print("Creating random model")
layers = [torch.nn.Linear(k, n, bias=False) for _ in range(n_layers)]
model = torch.nn.Sequential(*layers)
model = model.eval()

print("Quantizing random model")
quantized_model = copy.deepcopy(model)
quantizer = Int8DynActIntxWeightQuantizer(
    device="cpu",
    precision=torch.float32,
    bitwidth=nbit,
    groupsize=group_size,
    has_weight_zeros=has_weight_zeros,
)
quantized_model = quantizer.quantize(quantized_model)
quantized_model = quantized_model.eval()

print("Creating random activations")
activations = torch.randn(m, k, dtype=torch.float32)

print("Exporting quantized model")
exported = torch.export.export(quantized_model, (activations,))

print("Using torch.compile on quantized model")
quantized_model_compiled = torch.compile(quantized_model)
with torch.no_grad():
    quantized_model_compiled(activations)

print("Compiling quantized model with AOTI")
torch._export.aot_compile(
    quantized_model,
    (activations,),
    options={"aot_inductor.output_path": "/tmp/torch_custom_op_example_model.so"},
)

print("Running AOTI")
fn = torch._export.aot_load("/tmp/torch_custom_op_example_model.so", "cpu")
fn(activations)
