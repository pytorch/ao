# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from torch_custom_op import (
    linear_a8sz_w_lowbit_reference_impl,
    replace_linear_with_quantized_linear,
)

group_size = 256
m = 1
n = 4096
k = 4096
nbit = 5
has_weight_zeros = True
n_layers = 5

print("Creating random model")
layers = [torch.nn.Linear(k, n, bias=False) for _ in range(n_layers)]
model = torch.nn.Sequential(*layers)
model = model.eval()

print("Quantizing random model")
quantized_model = copy.deepcopy(model)
quantized_model = quantized_model.eval()
replace_linear_with_quantized_linear(
    quantized_model,
    kwargs={
        "group_size": group_size,
        "nbit": nbit,
        "has_weight_zeros": has_weight_zeros,
    },
)

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


print("\nChecking correctness on layer 0")
linear = model[0]
quantized_linear = quantized_model[0]

with torch.no_grad():
    result = quantized_linear(activations)
    expected_result = linear_a8sz_w_lowbit_reference_impl(
        linear.weight, activations, group_size, nbit, has_weight_zeros
    )
    non_quantized_result = linear(activations)


# Check that entries in result match entries in expected_result
num_mismatch_at_low_tol = 0
num_total = result.reshape(-1).shape[0]
for i in range(num_total):
    actual_val = result.reshape(-1)[i]
    expected_val = expected_result.reshape(-1)[i]
    if not torch.allclose(actual_val, expected_val):
        num_mismatch_at_low_tol += 1

        # If results are not close at a relaxed tolerance, exit with failure
        if not torch.allclose(actual_val, expected_val, atol=1e-6):
            assert False, "Correctness check failed"

# Assert at most 5% of entries are not close at a low tolerance
assert num_mismatch_at_low_tol / num_total <= 0.05, "Correctness check failed"
print(
    "Correctness check passed.  All results are close, and ",
    (num_total - num_mismatch_at_low_tol),
    "/",
    num_total,
    " entries are close at a low tolerance.",
)
print("Quantization errors:")
print("\tL1 error: ", torch.mean(torch.abs(result - non_quantized_result)).item())
print("\tL2 error: ", torch.mean((result - non_quantized_result) ** 2).item())
print("\tquantized_result[0:5]: ", result[0][0:5])
print("\tnon_quantized_result[0:5]: ", non_quantized_result[0][0:5])
