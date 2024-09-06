# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch_custom_op import quantize, replace_linear_with_quantized_linear
import torch
import copy

group_size = 16
m = 1
n = 4096
k = 4096
nbit = 4
n_layers = 10

print("Creating random model")
layers = [torch.nn.Linear(k, n, bias=False) for _ in range(n_layers)]
model = torch.nn.Sequential(*layers)
model = model.eval()

print("Quantizing random model")
quantized_model = copy.deepcopy(model)
quantized_model =  quantized_model.eval()
replace_linear_with_quantized_linear(quantized_model, kwargs={"group_size": group_size, "nbit": nbit})

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


print("Checking correctness on layer 0")

rtol=1e-05

# default is 1e-8, but PyTorch and C++ (and ARM neon) have different rounding
# conventions for ties (PyTorch rounds half to even and C++ rounds half to odd)
# TODO(T200109708): address this
atol=1e-05 

linear = model[0]
quantized_linear = quantized_model[0]
weight_qvals, weight_scales = quantize(linear.weight, group_size, quantized_linear.nbit, scale_only=True)

activation_qvals, activations_scales, activations_zeros = quantize(activations, k, 8, False)
activations_dequantized = activations_scales * (activation_qvals - activations_zeros)
weights_dequantized = (weight_qvals.reshape(-1, group_size) * weight_scales.reshape(-1, 1)).reshape(n, k)

with torch.no_grad():
    result = quantized_linear(activations)
    expected_result = torch.matmul(activations_dequantized, weights_dequantized.transpose(1, 0))
    non_quantized_result = linear(activations)

if not (torch.allclose(result, expected_result, rtol=rtol, atol=atol)):
    rand_idxs = torch.randint(0, result.shape[1], (5,))
    print("rand_idxs: ", rand_idxs)
    print("kernel_result[rand_idxs]: ", result[0][rand_idxs])
    print("expected_result[rand_idxs]: ", expected_result[0][rand_idxs])
    assert False
else:
    print("Correctness check passed")

print("kernel_result[0:5]: ", result[0][0:5])
print("non_quantized_result[0:5]: ", non_quantized_result[0][0:5])
