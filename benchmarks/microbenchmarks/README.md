# Microbenchmarks

This directory contains microbenchmarking tools for measuring inference performance across different quantization methods and model architectures.

## Overview

The microbenchmarking system works as follows:

![Microbenchmarks Process Flow](../../docs/static/microbenchmarking_process_diagram.png)

## Components

![Microbenchmarks Flow](../../docs/static/microbenchmarks_code_flow_diagram.png)

- **benchmark_runner.py**: Main entry point that orchestrates the benchmarking process
- **benchmark_inference.py**: Handles model creation and inference benchmarking
- **utils.py**: Contains utility functions and configuration classes
- **test\/**: Test files and sample configurations

## Usage

1. Create a configuration YAML file (see example below)
2. Run the benchmark using:

```bash
python -m benchmarks.microbenchmarks.benchmark_runner --config path/to/config.yml
```

### Example Configuration

```yaml
# Sample configuration for inference benchmarks
benchmark_mode: "inference"
quantization_config_recipe_names:
  - "baseline"
  - "int8wo"
  - "float8wo"
  - "float8dq-tensor"

output_dir: "benchmarks/microbenchmarks/results"

model_params:
  matrix_shapes:
    - name: "custom"
      shapes: [
        [1024, 1024, 1024],  # [m, k, n]
        [2048, 4096, 1024],
        [4096, 4096, 1024]
      ]
  high_precision_dtype: "torch.bfloat16"
  compile: "max-autotune" # Options: "default", "max-autotune", "false"
  device: "cuda"  # Options: "cuda", "mps", "xpu", "cpu"
  model_type: "linear"  # Options: "linear", "ln_linear_sigmoid"
  enable_profiler: true  # Enable standard profiling
  enable_memory_profiler: true  # Enable CUDA memory profiling
```

## Configuration Options

### Profiling Options
- `enable_profiler`: Enable standard PyTorch profiling (default: false)
- `enable_memory_profiler`: Enable CUDA memory profiling (default: false)
  - Only works when device is set to "cuda"
  - Generates memory snapshots before and after inference
  - Creates visualizations of memory usage
  - Outputs are saved in the memory_profiler subdirectory

### Quantization Methods
Currently, quantization string is in same format as the one being passed in llama/generate.py.
- `baseline`: No quantization
- `int8wo`: 8-bit weight-only quantization
- `int4wo-{group_size}`: 4-bit weight-only quantization with specified group size
- `int4wo-{group_size}-hqq`: 4-bit weight-only quantization with HQQ
- `gemlitewo-{bit_width}-{group_size}`: 4 or 8 bit integer quantization and utilizes the gemlite triton kernel

### Model Types
- `linear`: Simple linear layer
- `ln_linear_<activation>`: LayerNorm + Linear + Activation, where activation can be:
  - `ln_linear_sigmoid`: LayerNorm + Linear + Sigmoid
  - `ln_linear_relu`: LayerNorm + Linear + ReLU
  - `ln_linear_leakyrelu`: LayerNorm + Linear + LeakyReLU
  - `ln_linear_relu6`: LayerNorm + Linear + ReLU6
  - `ln_linear_gelu`: LayerNorm + Linear + GELU
  - `ln_linear_silu`: LayerNorm + Linear + SiLU
  - `ln_linear_hardswish`: LayerNorm + Linear + Hardswish
- `transformer_block`: Transformer block with self-attention and MLP

### Device Options
- `cuda`: NVIDIA GPU
- `xpu`: Intel GPU
- `mps`: Apple Silicon GPU
- `cpu`: CPU fallback

### Shape Generation Options
- `custom`: Manually specify shapes as a list of [m, k, n] dimensions
  ```yaml
  matrix_shapes:
    - name: "custom"
      shapes: [
        [1024, 1024, 1024],  # [m, k, n]
        [2048, 4096, 1024]
      ]
  ```

- `llama`: Use LLaMa 2 70B single-node weight shapes (assumes fused attn.wqkv and ffn.w13)
  - Generates shapes for: "attn.wqkv", "attn.w0", "ffn.w13", "ffn.w2"
  ```yaml
  matrix_shapes:
    - name: "llama"
  ```

- `pow2`: Generate shapes with dimensions that are powers of 2
  - Parameters:
    - `min_power`: Minimum power of 2 (default: 10, which is 1024)
    - `max_power`: Maximum power of 2 (default: 14, which is 16,384)
  ```yaml
  matrix_shapes:
    - name: "pow2"
      min_power: 10  # 2^10 = 1024
      max_power: 12  # 2^12 = 4096
  ```

- `pow2_extended`: Generate shapes with dimensions that are powers of 2 and powers of 2 + half
  - Parameters:
    - `min_power`: Minimum power of 2 (default: 10, which is 1024)
    - `max_power`: Maximum power of 2 (default: 14, which is 16,384)
  ```yaml
  matrix_shapes:
    - name: "pow2_extended"
      min_power: 10  # Generates: 1024, 1536, 2048, 3072, etc.
      max_power: 11
  ```

- `small_sweep`: Generate a small sweep of shapes with increasing powers of 2 for M, K, N dimensions
  - Parameters:
    - `min_power`: Minimum power of 2 (default: 10, which is 1024)
    - `max_power`: Maximum power of 2 (default: 14, which is 16,384)
  - Note: This generates shapes where M <= K <= N (ensuring increasing order), which produces fewer combinations than the full sweep, and could be good to use for plots like heatmap
  ```yaml
  matrix_shapes:
    - name: "small_sweep"
      min_power: 10  # 2^10 = 1024
      max_power: 15  # 2^15 = 32,768
  ```

- `sweep`: Generate a sweep of shapes with different powers of 2 for M, K, N dimensions
  - Parameters:
    - `min_power`: Minimum power of 2 (default: 8, which is 256)
    - `max_power`: Maximum power of 2 (default: 15, which is 32,768)
  - Note: This generates all combinations of M, K, N dimensions, which can be a large number of shapes
  ```yaml
  matrix_shapes:
    - name: "sweep"
      min_power: 8  # 2^8 = 256
      max_power: 9  # 2^9 = 512
  ```



## Output

Results are saved to a CSV file in the specified output directory

## Running Tests

To run the test suite:

```bash
python -m unittest discover benchmarks/microbenchmarks/test
```
