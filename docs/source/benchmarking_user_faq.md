# Benchmarking Use-Case FAQs

This guide is intended to provide instructions for the most fequent benchmarking use-case. If you have any use-case that is not answered here, please create an issue here: [TorchAO Issues](https://github.com/pytorch/ao/issues)

## Table of Contents
- [Run the performance benchmarking on your PR](#run-the-performance-benchmarking-on-your-pr)
- [Benchmark Your API Locally](#benchmark-your-api-locally)
- [Generate evaluation metrics for your quantized model](#generate-evaluation-metrics-for-your-quantized-model)
- [Advanced Usage](#advanced-usage)

## Run the performance benchmarking on your PR

### 1. Add label to your PR
To trigger the benchmarking CI workflow on your pull request, you need to add a specific label to your PR. Follow these steps:

1. Go to your pull request on GitHub.
2. On the right sidebar, find the "Labels" section.
3. Click on the "Labels" dropdown and select "ciflow/benchmark" from the list of available labels.

Adding this label will automatically trigger the benchmarking CI workflow for your pull request.

### 2. Manually trigger benchmarking workflow on your github branch
To manually trigger the benchmarking workflow for your branch, follow these steps:

1. Navigate to the "Actions" tab in your GitHub repository.
2. Select the benchmarking workflow from the list of available workflows. For microbenchmarks, it's `Microbenchmarks-Perf-Nightly`.
3. Click on the "Run workflow" button.
4. In the dropdown menu, select the branch.
5. Click the "Run workflow" button to start the benchmarking process.

This will execute the benchmarking workflow on the specified branch, allowing you to evaluate the performance of your changes.

## Benchmark Your API Locally

For local development and testing:

### 1. Quick Start

Create a minimal configuration for local testing:

```yaml
# local_test.yml
benchmark_mode: "inference"
quantization_config_recipe_names:
  - "baseline"
  - "int8wo"
  # Add your recipe here

output_dir: "local_results" # Add your output directory here

model_params:
  # Add your model configurations here
  - name: "quick_test"
    matrix_shapes:
      # Define a custom shape, or use one of the predefined shape generators
      - name: "custom"
        shapes: [[1024, 1024, 1024]]
      - name: "small_sweep"
    high_precision_dtype: "torch.bfloat16"
    use_torch_compile: true
    torch_compile_mode: "max-autotune"
    device: "cuda"
    model_type: "linear"
    enable_profiler: true  # Enable profiling for this model
    enable_memory_profiler: true  # Enable memory profiling for this model
```

> **Note:**
> - For a list of latest supported config recipes for quantization or sparsity, please refer to `benchmarks/microbenchmarks/README.md`.
> - For a list of all model types, please refer to `torchao/testing/model_architectures.py`.

### 2. Run Local Benchmark

```bash
python -m benchmarks.microbenchmarks.benchmark_runner --config local_test.yml
```

### 3. Analysing the Output

The output generated after running the benchmarking script, is the form of a csv. It'll contain some of the following:
 - time for inference for running baseline model and quantized model
 - speedup in inference time in quantized model
 - compile or eager mode
 - if enabled, memory snapshot and gpu chrome trace


## Generate evaluation metrics for your quantized model
(Coming soon!!!)

## Advanced Usage

### Multiple Model Configurations

You can benchmark multiple model configurations in a single run:

```yaml
model_params:
  - name: "small_models"
    matrix_shapes:
      - name: "pow2"
        min_power: 10
        max_power: 12
    model_type: "linear"
    device: "cuda"

  - name: "transformer_models"
    matrix_shapes:
      - name: "llama"
    model_type: "transformer_block"
    device: "cuda"

  - name: "cpu_models"
    matrix_shapes:
      - name: "custom"
        shapes: [[512, 512, 512]]
    model_type: "linear"
    device: "cpu"
```

### Interpreting Results

The benchmark results include:

- **Speedup**: Performance improvement compared to baseline (bfloat16)
- **Memory Usage**: Peak memory consumption during inference
- **Latency**: Time taken for inference operations
- **Profiling Data**: Detailed performance traces (when enabled)

Results are saved in CSV format with columns for:

- Model configuration
- Quantization method
- Shape dimensions (M, K, N)
- Performance metrics
- Memory metrics
- Device information

### Best Practices

1. Use `small_sweep` for initial testing, `sweep` for comprehensive analysis
2. Enable profiling only when needed (adds overhead)
3. Test on multiple devices when possible
