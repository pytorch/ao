import torch
import torchao
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from torchao.quantization.quant_api import quantize_, float8_dynamic_activation_float8_weight
import copy
from utils import (
    get_name_to_shapes_iter,
)
import tqdm

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False).to(torch.bfloat16)
        self.linear2 = torch.nn.Linear(n, k, bias=False).to(torch.bfloat16)

    def example_inputs(self, batch_size=1, dtype=torch.float, device="cuda"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Function to benchmark model evaluation with profiling
def benchmark_model_with_profiling(model, input_data, dtype):
    print('Model before quantization: ', model)
    quantize_(model, float8_dynamic_activation_float8_weight())
    print('Model quantized: ', model)
    model.eval()  # Set the model to evaluation mode
    # input_data = torch.randn(input_size, device=device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(*input_data)

    # Return the profiler output
    return prof

name_to_shapes = get_name_to_shapes_iter("square", None, None, None)



# Set the data types
float8_dtype = torch.float8_e4m3fn  # Replace with the actual float8 dtype from TorchAO
bf16_dtype = torch.bfloat16

# Dictionary to store performance data
performance_data = {
    'Input Size': [],
    'float8 Kernel Times (ms)': [],
    'bf16 Kernel Times (ms)': []
}

# Run benchmarks for each input size
for idx, (name, (m, k, n)) in enumerate(tqdm.tqdm(name_to_shapes)):
    print(f"Profiling model with input size: {m, k, n}")
    
    # Initialize the model with the specified dimensions
    model = ToyLinearModel().eval().to(device)
    example_inputs = model.example_inputs()
    model_bf16 = copy.deepcopy(model).to(device)  # Copy the model to bf
    model_ref =  copy.deepcopy(model).to(device)  # Copy the model for quantization

    
    print('Model created: ', model)
    print('Example inputs: ', len(example_inputs), example_inputs[0].size())

    # Profile float8 model evaluation
    prof_float8 = benchmark_model_with_profiling(model_ref, example_inputs, float8_dtype)
    prof_float8.export_chrome_trace(f"float8_model_{example_inputs[0].size()[0]}.json")  # Save profiling details

    # Profile bf16 model evaluation
    prof_bf16 = benchmark_model_with_profiling(model_bf16, example_inputs, bf16_dtype)
    prof_bf16.export_chrome_trace(f"bf16_model_{example_inputs[0].size()[0]}.json")  # Save profiling details

    print('Profiling keys: ', prof_float8.key_averages())
    # Calculate and store total GPU kernel times
    float8_kernel_time = sum([event.device_time for event in prof_float8.key_averages()])
    bf16_kernel_time = sum([event.device_time for event in prof_bf16.key_averages()])
    
    performance_data['Input Size'].append(f"{example_inputs[0].size()[0]}")
    performance_data['float8 Kernel Times (ms)'].append(float8_kernel_time / 1000)  # Convert from microseconds to milliseconds
    performance_data['bf16 Kernel Times (ms)'].append(bf16_kernel_time / 1000)  # Convert from microseconds to milliseconds

    print('Performance data: ', performance_data)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(performance_data['Input Size'], performance_data['float8 Kernel Times (ms)'], marker='o', label='float8')
plt.plot(performance_data['Input Size'], performance_data['bf16 Kernel Times (ms)'], marker='s', label='bf16')
plt.xlabel('Batch Size')
plt.ylabel('Kernel Time (ms)')
plt.title('Model Evaluation GPU Kernel Performance: float8 vs bf16')
plt.legend()
plt.grid(True)
plt.show()
