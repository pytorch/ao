# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

torchao is an Architecture Optimization library that accelerates PyTorch models through quantization and sparsification techniques. It provides optimization for weights, gradients, activations, and more for both inference and training with minimal code changes.

## Prerequisites

### Required Dependencies
```bash
# Install PyTorch (required before torchao installation)
pip install torch torchvision torchaudio

# For development, you may need specific PyTorch versions
# Check requirements.txt or setup.py for version constraints
```

## Development Commands

### Installation & Build
```bash
# Development install (Python-only mode, fastest for development)
USE_CPP=0 python setup.py develop

# Full build with C++/CUDA extensions
python setup.py develop

# Install specific version of ruff for linting
pip install ruff==0.11.6
```

### Testing
```bash
# Run specific test files
pytest test/float8/test_base.py
pytest test/quantization/test_quant_api.py
pytest test/dtypes/test_affine_quantized.py

# Run comprehensive float8 tests
./test/float8/test_everything.sh

# Run all tutorials
./tutorials/run_all.sh
```

### Linting & Formatting
```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Run pre-commit on staged files only
pre-commit run
```

## Architecture Overview

### Workflow-Based Structure (2025H1 Refresh)

TorchAO is transitioning from an AQT-centered structure to a **workflow-based organization** that embraces vertical workflows for optimal performance and maintainability.

### Current Structure

**torchao/quantization/** - User-facing APIs
- `quantize_()` - Main quantization function with workflow-specific configs
- `autoquant.py` - Automatic quantization selection
- Configuration classes for different workflows

**torchao/sparsity/** - User-facing sparsity APIs
- `sparsify_()` - Main sparsification function
- Sparsity configuration classes

**Vertical Workflows** (in transition):
- **int8_weight_only** - INT8 weight-only quantization workflows
- **float8** - High-performance float8 training (1.5x speedup vs FP16)
- **nf4** - NF4 (4-bit normal float) for QLoRA
- **pt2e** - PT2E graph mode quantization (migrating from PyTorch Core)
- **executorch** - ExecutorTorch workflows (moving from experimental)

**torchao/csrc/** - Custom kernels
- CUTLASS-based implementations for maximum performance
- ROCm support for AMD GPUs
- CPU kernels with AVX512 optimizations

**torchao/experimental/** - Experimental features
- MPS acceleration for Apple Silicon
- Low-bit quantization research (1-7 bit weights)
- Prototype workflows before graduation

### Design Philosophy

**Vertical Workflows Over Horizontal Abstractions:**
- Self-contained workflows that can move fast on SOTA performance/accuracy
- Workflows choose abstractions that fit their needs rather than forced repo-wide patterns
- Well-fitting abstractions > no abstractions > poorly fitting abstractions
- Duplicated easy-to-understand code preferred over highly abstracted hard-to-understand code


## Build Configuration

The build system uses environment variables for configuration:

**Core Controls:**
- `USE_CPP=0|1` - Skip C++/CUDA extensions (default: 1, set to 0 for fastest dev setup)
- `USE_CPU_KERNELS=0|1` - Enable optimized CPU kernels (Linux only, default: 0)
- `DEBUG=0|1` - Debug build mode

**Experimental Features:**
- `BUILD_TORCHAO_EXPERIMENTAL=1` - Enable experimental cmake builds
- `TORCHAO_BUILD_CPU_AARCH64=1` - ARM64 CPU kernels (auto-enabled on Apple Silicon)
- `TORCHAO_BUILD_KLEIDIAI=1` - Kleidi AI library integration (experimental, accuracy issues)
- `TORCHAO_BUILD_EXPERIMENTAL_MPS=1` - MPS acceleration (macOS only, disabled by default)
- `USE_AVX512=1` - Enable AVX512 optimizations for x86 CPUs (default on Linux)

## Experimental Features (Alpha)

### Stability Levels
**Alpha Features**: Early development stage requiring further refinement. These are prototypes due to:
- Ongoing hardware support development
- Non-compelling memory benchmarks
- Need for compiler/kernel investment

### Current Experimental Features

**MX Training and Inference:**
- **Status**: Prototype (hardware support not yet available)
- **Description**: Tensors based on OCP MX specification for group-wise scaled float8/float6/float4/int8
- **Usage**: Group-wise quantization with MX data types

**Int8 Quantized Training:**
- **Status**: Prototype (memory benchmarks not yet compelling)
- **Description**: Full int8 training support
- **Usage**: `quantize_(model, int8_weight_only_quantized_training())`

**IntX (Low-bit integers):**
- **Status**: Prototype (needs compiler/kernel investment)
- **Description**: Various integer types through bitpacking in pure PyTorch
- **Note**: Int4 remains more compelling than smaller data types currently

**Bitnet:**
- **Status**: Experimental (dependent on better hardware/kernel support)
- **Description**: Bitnet quantization technique
- **Limitation**: Usefulness highly dependent on hardware improvements

### Hardware-Specific Experimental Features

**MPS Kernels (Apple Silicon):**
- **Status**: Experimental (disabled by default)
- **Requirements**: macOS with ARM64 architecture and MPS available
- **Build**: `export TORCHAO_BUILD_EXPERIMENTAL_MPS=1`
- **Features**: Metal shaders for int1mm, int2mm, int3mm, int4mm, int5mm, int6mm, int7mm

**ARM64/AArch64 CPU Kernels:**
- **Status**: Auto-enabled on ARM64 Macs, manual enable elsewhere
- **Build**: `export TORCHAO_BUILD_CPU_AARCH64=1`
- **Features**:
  - Quantized matrix operations with NEON intrinsics
  - Bit-packing operations for low-bit quantization
  - Lookup table (LUT) operations for weight compression
  - Kleidi AI integration (experimental, accuracy issues in CI)

**Kleidi AI Integration:**
- **Status**: Experimental (disabled by default)
- **Build**: `export TORCHAO_BUILD_KLEIDIAI=1`
- **Requirements**: ARM64 architecture
- **Note**: Increases build time, has shown BF16 accuracy issues in CI tests


## Development Patterns and Workflows

### Common Development Patterns

**One-line optimizations:** Use `quantize_(model, config)` and `sparsify_(model, config)` for quick model optimization
- `quantize_(m, Int4WeightOnlyConfig())` applies 4-bit weight-only quantization
- `sparsify_(model, BlockSparseWeightConfig())` applies block-sparse weight configuration

**Model-specific optimizations:** Specialized patterns for different model types
- **SAM2**: Use "Fast Mode" with `torch.compile` and "Furious Mode" with FP16 precision
- **LLMs**: Common patterns include KV cache quantization and Sparse-Marlin integration

**Composability focus:** Design optimizations to work with `torch.compile()` and FSDP2 without graph breaks

### Workflow Development Approach

**Workflow-First Development:**
- Focus on vertical workflows rather than horizontal tensor subclass abstractions
- Each workflow is self-contained and optimized for its specific use case
- Workflows can choose their own abstractions and implementation patterns

**Workflow Implementation Patterns:**
- **Config-based**: Each workflow provides configuration classes for `quantize_()`
- **Kernel integration**: Workflows integrate with `torchao/csrc/` kernels as needed
- **Composability**: Workflows maintain compatibility with `torch.compile` and FSDP2
- **Independence**: Workflows avoid dependencies on repo-wide abstractions unless beneficial

**Abstraction Selection:**
- Workflows choose abstractions that make their implementation cleaner and more maintainable
- No enforcement of repo-wide abstractions without clear benefits
- Many-to-many mapping between abstractions and workflows is acceptable

### Development Tasks

**Adding a new workflow:**
1. Create a new workflow directory in the appropriate location
2. Implement a configuration class for the workflow
3. Add the config to `torchao/quantization/__init__.py` for `quantize_()` integration
4. Implement the workflow using patterns that fit your use case (tensor subclass, module swap, etc.)
5. Add any required kernels to `torchao/csrc/` or `torchao/kernel/`
6. Choose helpful abstractions from common utilities as needed

**Current workflow examples:**
- `int8_weight_only` - Uses AQT patterns where beneficial
- `float8` - Uses `Float8Tensor` and specialized training patterns
- `nf4` - Uses NF4-specific tensor subclass for QLoRA
- `pt2e` - Uses graph mode quantization patterns

**Performance optimization:**
1. Custom kernels go in `torchao/csrc/` with architecture-specific builds (SM90a, SM100a)
2. Use `opcheck()` in tests to ensure `torch.compile` compatibility
3. Implement fallback paths for unsupported configurations

**Testing:**
1. Follow patterns in `test/` directory, use `pytest` for individual tests
2. Use `TorchAOBasicTestCase` and `TorchAOCompileTestCase` for tensor subclass tests
3. Include SQNR assertions for quantization accuracy verification

**Experimental workflows:**
1. Develop in `torchao/experimental/` or `torchao/prototype/` as appropriate
2. Use `_check_torchao_ops_loaded()` to verify experimental kernels are loaded
3. Follow Alpha feature guidelines for prototype development
4. Graduate to main workflow structure when ready for production use
5. Focus on vertical workflow patterns rather than forcing horizontal abstractions

## Common Issues and Debugging

### Frequent Issues and Solutions

**torch.compile() graph breaks:**
- **Issue**: Custom kernels causing graph breaks when used with `torch.compile()`
- **Debug**: Run with `fullgraph=True` and `TORCH_LOGS="output_code"` to inspect generated code
- **Solution**: Ensure tensor subclasses implement `__tensor_flatten__` and `__tensor_unflatten__`

**Device and data type compatibility:**
- **Issue**: Some experimental features only support specific devices (e.g., CPU-only embedding quantization)
- **Solution**: Check feature documentation for supported devices and data types
- **Example**: MPS quantization requires macOS with ARM64 architecture

**Performance analysis:**
- **Issue**: Need to benchmark optimized models
- **Tools**: Use `print_op_and_shapes.py` to identify relevant shapes for microbenchmarking
- **Profiling**: Add `--profile=profile_path` to benchmark scripts for Chrome traces

**Accuracy degradation:**
- **Issue**: Quantization/sparsity causing accuracy loss
- **Analysis**: Check scale/zero_point for quantization, mask for sparsity
- **Solution**: Consider Quantization-Aware Training (QAT) for accuracy recovery

### Common Debugging Commands

```bash
# Check CUDA availability and version
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, version: {torch.version.cuda}')"

# Check build configuration
python -c "import torchao; print(torchao.__file__)"

# Debug torch.compile issues
TORCH_LOGS="output_code" python your_script.py

# Run specific test with verbose output
pytest -v -s test/quantization/test_quant_api.py::test_specific_function

# Check for CUDA kernel compilation issues
USE_CPP=1 python setup.py develop --verbose

# Verify experimental kernels are loaded
python -c "from torchao.experimental import _check_torchao_ops_loaded; _check_torchao_ops_loaded()"

# Profile model performance
python benchmark_script.py --profile=profile_output.json
```

## Testing and Benchmarking

### Testing Infrastructure

**Test organization:**
- Unit tests: `test_base.py` for core components like `Float8Tensor`
- Integration tests: `test_integration.py` for AOTI compilation with tensor subclasses
- Numerical accuracy: `test_numerics_integration.py` for Float8 operations

**Test utilities:**
- `TorchAOBasicTestCase`: Basic tensor subclass testing
- `TorchAOCompileTestCase`: `torch.compile` compatibility testing
- SQNR assertions for minimum signal-to-quantization noise ratio

### Performance Benchmarking

**Microbenchmarks:**
- `bench_matmul.py`: Benchmark `torch._scaled_mm` function
- `bench_linear_float8.py`: Benchmark `nn.Linear` vs `Float8Linear`
- `benchmark_aq.py`: Benchmark various quantized tensor subclasses

**Model-level benchmarks:**
- **Llama**: `generate.py` for generation performance, `eval.py` for evaluation
- **SAM**: `eval_combo.py` for SAM model benchmarking
- Enable profiling with `generate_model_profile` for detailed analysis

**Continuous Integration:**
- `dashboard_perf_test.yml`: Nightly A100 benchmarks with dashboard visualization
- `torchao_experimental_test.yml`: Experimental feature validation

## Important Notes

- Always run `pre-commit run --all-files` before committing
- Use `USE_CPP=0` for faster iteration during Python-only development
- CUTLASS kernels have architecture-specific builds (SM90a, SM100a) based on CUDA version
- Git submodules (CUTLASS) are automatically initialized during build
