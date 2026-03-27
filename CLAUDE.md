# TorchAO

PyTorch-native library for quantization, sparsity, and low-precision training. Works with `torch.compile()` and `FSDP2`.

## Config Classes

All configs inherit from `AOBaseConfig`. Defined in `torchao/quantization/quant_api.py`:

| Config | Description |
|--------|-------------|
| `Int4WeightOnlyConfig` | int4 weight-only (most common for inference) |
| `Int8WeightOnlyConfig` | int8 weight-only |
| `Int8DynamicActivationInt8WeightConfig` | int8 weights + int8 dynamic activations |
| `Int8DynamicActivationIntxWeightConfig` | int8 activations + arbitrary int weight width |
| `Float8WeightOnlyConfig` | float8 weight-only |
| `Float8DynamicActivationFloat8WeightConfig` | float8 weights + float8 dynamic activations |
| `Float8DynamicActivationInt4WeightConfig` | float8 activations + int4 weights |
| `IntxWeightOnlyConfig` | arbitrary bit-width for edge/ExecuTorch |
| `FqnToConfig` | map module names to different configs for per-layer quantization |

### Prototype configs (in `torchao/prototype/mx_formats/`)
- `MXDynamicActivationMXWeightConfig` - MXFP8/MXFP4 (H100/B200/MI350x)
- `NVFP4DynamicActivationNVFP4WeightConfig` - NVIDIA FP4 (B200 Blackwell only)
- `NVFP4WeightOnlyConfig` - NVFP4 weight-only (B200 Blackwell only)

## Stable vs Prototype

- **Stable** (`torchao/quantization/`, `torchao/float8/`, `torchao/sparsity/`, `torchao/optim/`): API stability guaranteed. Breaking changes go through deprecation cycle.
- **Prototype** (`torchao/prototype/`): Experimental features, API may change without notice. Includes: `mx_formats/` (MXFP8, MXFP4, NVFP4), `moe_training/` (MoE mixed-precision), `awq/`, `hqq/`, `autoround/`, `quantized_training/`.

## Architecture and Contributing

For architecture details, tensor subclass design, and contributor guides, see the in-repo docs:
- [Quantization Overview](docs/source/contributing/quantization_overview.rst) - full stack walkthrough, tensor subclasses, quantization flows
- [Contributor Guide](docs/source/contributing/contributor_guide.rst) - how to add tensors, kernels, configs
- [Workflows Matrix](docs/source/workflows/index.md) - dtype x hardware status table
- [PT2E Quantization](docs/source/pt2e_quantization/index.rst) - PyTorch 2 Export quantization for deployment backends (X86, XPU, ExecuTorch)

These same files render at https://docs.pytorch.org/ao/main/contributing/index.html

## Deprecated APIs

Do not use or recommend these:
- `AffineQuantizedTensor` (AQT) in `torchao/dtypes/` - old v1 system, being removed. New tensor types inherit from `TorchAOBaseTensor` in `torchao/utils.py`
- `autoquant()` - deleted
- Layout registration system (`PlainLayout`, `Float8Layout`, `TensorCoreTiledLayout`, etc.) - deleted
- `TorchAODType` - deprecated
- `change_linear_weights_to_int4_woqtensors` - deleted, use `quantize_(model, Int4WeightOnlyConfig())`

## Development

```bash
# Setup
USE_CPP=0 pip install -e . --no-build-isolation   # CPU-only
USE_CUDA=1 pip install -e . --no-build-isolation   # With CUDA

# Lint (ruff v0.11.6, rules: F and I)
ruff check --fix && ruff format .

# Test (mirrors source structure)
pytest test/quantization/test_quant_api.py
pytest test/float8/
pytest test/prototype/mx_formats/
```

## Coding Style

- 2 spaces for indentation
- 80 character line length
- BSD 3-Clause license header required on all source files
- Match existing patterns in the file you're editing

## Commit Messages

- Do not commit without explicit request from the user
- Preserve ghstack trailers when amending commits
- If Claude or another AI tool was used, disclose in the commit message
