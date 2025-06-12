# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

torchao is PyTorch's official Architecture Optimization library that accelerates PyTorch models through advanced quantization and sparsification techniques. It provides optimization for weights, gradients, activations, and more for both inference and training with minimal code changes.

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

### Core Components

**torchao/quantization/** - Primary quantization APIs
- `quant_api.py` - Main `quantize_()` function for one-line model quantization
- `autoquant.py` - Automatic quantization selection
- Weight-only quantization (INT4/INT8), dynamic quantization, QAT support

**torchao/dtypes/** - Custom tensor subclasses with layout and dispatch registration
- `AffineQuantizedTensor` - Base quantized tensor class
- `nf4tensor.py` - NF4 (4-bit normal float) implementation for QLoRA
- `uintx/floatx/` - Unsigned integer and floating-point quantized tensors

**torchao/float8/** - High-performance float8 training
- Delivers up to 1.5x speedup on 512 GPU clusters
- `convert_to_float8_training()` - Main entry point
- Full `torch.compile` and FSDP2 compatibility

**torchao/sparsity/** - Structured and unstructured sparsity
- 2:4 semi-structured sparsity with up to 2.4x throughput improvements
- `sparse_api.py` - Main sparsity functions
- Wanda pruning, block-sparse operations

**torchao/optim/** - Memory-efficient optimizers
- `AdamW8bit`, `AdamW4bit`, `AdamWFp8` - Quantized optimizers (2-4x memory reduction)
- `CPUOffloadOptimizer` - 60% VRAM reduction via CPU offloading

**torchao/csrc/** - Custom CUDA/CPU kernels
- CUTLASS-based implementations for maximum performance
- ROCm support for AMD GPUs
- CPU kernels with AVX512 optimizations

### Key Design Principles

**Composability**: All custom dtypes work with `torch.compile`, FSDP2, and tensor parallel out-of-the-box

**Subclass Architecture**: Tensor subclasses handle layout, dispatch, and kernel registration automatically

**Hardware Optimization**: Architecture-specific optimizations (CUDA, ROCm, CPU, MPS) with automatic detection

## Build Configuration

The build system uses environment variables for configuration:

**Core Controls:**
- `USE_CPP=0|1` - Skip C++/CUDA extensions (default: 1, set to 0 for fastest dev setup)
- `USE_CPU_KERNELS=0|1` - Enable optimized CPU kernels (Linux only, default: 0)
- `DEBUG=0|1` - Debug build mode

**Experimental Features:**
- `BUILD_TORCHAO_EXPERIMENTAL=1` - Enable experimental cmake builds
- `TORCHAO_BUILD_CPU_AARCH64=1` - ARM64 CPU kernels (auto-enabled on Apple Silicon)
- `TORCHAO_BUILD_KLEIDIAI=1` - Kleidi AI library integration
- `TORCHAO_BUILD_EXPERIMENTAL_MPS=1` - MPS acceleration (macOS only)

## Integration Points

- **HuggingFace Transformers**: Built-in backend via `TorchAoConfig`
- **vLLM/SGLang**: LLM serving integration
- **TorchTune**: QLoRA and QAT recipes
- **torch.compile**: Full compiler compatibility
- **FSDP2**: Distributed training support

## Common Development Tasks

**Adding a new quantization technique:** Implement as tensor subclass in `torchao/dtypes/`, register dispatch kernels, add to `quant_api.py`

**Performance optimization:** Custom kernels go in `torchao/csrc/`, with separate extensions for different GPU architectures (SM90a, SM100a)

**Testing:** Follow existing patterns in `test/` directory, use `pytest` for individual tests

## Important Notes

- Always run `pre-commit run --all-files` before committing
- Use `USE_CPP=0` for faster iteration during Python-only development
- CUTLASS kernels have architecture-specific builds (SM90a, SM100a) based on CUDA version
- Git submodules (CUTLASS) are automatically initialized during build
