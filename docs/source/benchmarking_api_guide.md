# Benchmarking API Guide

This tutorial will guide you through using the TorchAO benchmarking framework. The tutorial contains integrating new APIs with the framework and dashboard.

1. [Add an API to benchmarking recipes](#add-an-api-to-benchmarking-recipes)
2. [Add a model architecture for benchmarking recipes](#add-a-model-to-benchmarking-recipes)
3. [Add an HF model to benchmarking recipes](#add-an-hf-model-to-benchmarking-recipes)
4. [Add an API to micro-benchmarking CI dashboard](#add-an-api-to-benchmarking-ci-dashboard)

## Add an API to Benchmarking Recipes

The framework currently supports quantization and sparsity recipes, which can be run using the quantize_() or sparsity_() functions:

To add a new recipe, add the corresponding string configuration to the function `string_to_config()` in `benchmarks/microbenchmarks/utils.py`.

```python
def string_to_config(
  quantization: Optional[str], sparsity: Optional[str], **kwargs
) -> AOBaseConfig:

# ... existing code ...

elif quantization == "my_new_quantization":
  # If additional information needs to be passed as kwargs, process it here
  return MyNewQuantizationConfig(**kwargs)
elif sparsity == "my_new_sparsity":
  return MyNewSparsityConfig(**kwargs)

# ... rest of existing code ...
```

Now we can use this recipe throughout the benchmarking framework.

**Note:** If the `AOBaseConfig` uses input parameters, like bit-width, group-size etc, you can pass them appended to the string config in input. For example, for `GemliteUIntXWeightOnlyConfig` we can pass bit-width and group-size as `gemlitewo-<bit_width>-<group_size>`

## Add a Model to Benchmarking Recipes

To add a new model architecture to the benchmarking system, you need to modify `torchao/testing/model_architectures.py`.

1. To add a new model type, define your model class in `torchao/testing/model_architectures.py`:

```python
class MyCustomModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dtype=torch.bfloat16):
        super().__init__()
        # Define your model architecture
        self.layer1 = torch.nn.Linear(input_dim, 512, bias=False).to(dtype)
        self.activation = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(512, output_dim, bias=False).to(dtype)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

2. Update the `create_model_and_input_data` function to handle your new model type:

```python
def create_model_and_input_data(
    model_type: str,
    m: int,
    k: int,
    n: int,
    high_precision_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    activation: str = "relu",
):
    # ... existing code ...

    elif model_type == "my_custom_model":
        model = MyCustomModel(k, n, high_precision_dtype).to(device)
        input_data = torch.randn(m, k, device=device, dtype=high_precision_dtype)

    # ... rest of existing code ...
```

### Model Design Considerations

When adding new models:

- **Input/Output Dimensions**: Ensure your model handles the (m, k, n) dimension convention where:
  - `m`: Batch size or sequence length
  - `k`: Input feature dimension
  - `n`: Output feature dimension

- **Data Types**: Support the `high_precision_dtype` parameter (typically `torch.bfloat16`)

- **Device Compatibility**: Ensure your model works on CUDA, CPU, and other target devices

- **Quantization Compatibility**: Design your model to work with TorchAO quantization methods

## Add an HF model to benchmarking recipes
(Coming soon!!!)

## Add an API to Benchmarking CI Dashboard

To integrate your API with the CI [dashboard](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fao&benchmarkName=micro-benchmark+api):

### 1. Modify Existing CI Configuration

Add your quantization method to the existing CI configuration file at `benchmarks/dashboard/microbenchmark_quantization_config.yml`:

```yaml
# benchmarks/dashboard/microbenchmark_quantization_config.yml
benchmark_mode: "inference"
quantization_config_recipe_names:
  - "int8wo"
  - "int8dq"
  - "float8dq-tensor"
  - "float8dq-row"
  - "float8wo"
  - "my_new_quantization"  # Add your method here

output_dir: "benchmarks/microbenchmarks/results"

model_params:
  - name: "small_bf16_linear"
    matrix_shapes:
      - name: "small_sweep"
        min_power: 10
        max_power: 15
    high_precision_dtype: "torch.bfloat16"
    torch_compile_mode: "max-autotune"
    device: "cuda"
    model_type: "linear"
```

### 2. Run CI Benchmarks

Use the CI runner to generate results in PyTorch OSS benchmark database format:

```bash
python benchmarks/dashboard/ci_microbenchmark_runner.py \
    --config benchmarks/dashboard/microbenchmark_quantization_config.yml \
    --output benchmark_results.json
```

### 3. CI Output Format

The CI runner outputs results in a specific JSON format required by the PyTorch OSS benchmark database:

```json
[
  {
    "benchmark": {
      "name": "micro-benchmark api",
      "mode": "inference",
      "dtype": "int8wo",
      "extra_info": {
        "device": "cuda",
        "arch": "NVIDIA A100-SXM4-80GB"
      }
    },
    "model": {
      "name": "1024-1024-1024",
      "type": "micro-benchmark custom layer",
      "origins": ["torchao"]
    },
    "metric": {
      "name": "speedup(wrt bf16)",
      "benchmark_values": [1.25],
      "target_value": 0.0
    },
    "runners": [],
    "dependencies": {}
  }
]
```

### 4. Integration with CI Pipeline

To integrate with your CI pipeline, add the benchmark step to your workflow:

```yaml
# Example GitHub Actions step
- name: Run Microbenchmarks
  run: |
    python benchmarks/dashboard/ci_microbenchmark_runner.py \
      --config benchmarks/dashboard/microbenchmark_quantization_config.yml \
      --output benchmark_results.json

- name: Upload Results
  # Upload benchmark_results.json to your dashboard system
```

## Troubleshooting

### Running Tests

To verify your setup and run the test suite:

```bash
python -m unittest discover benchmarks/microbenchmarks/test
```

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or matrix dimensions
2. **Missing Quantization Methods**: Ensure TorchAO is properly installed
3. **Device Not Available**: Check device availability and drivers

### Best Practices

1. Use `small_sweep` for basic testing, `custom shapes` for comprehensive or model specific analysis
2. Enable profiling only when needed (adds overhead)
3. Test on multiple devices when possible
4. Use consistent naming conventions for reproducibility

For information on different use-cases for benchmarking, refer to [Benchmarking User Guide](benchmarking_user_guide.md)

For more detailed information about the framework components, see the README files in the `benchmarks/microbenchmarks/` directory.
