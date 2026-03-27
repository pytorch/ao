# TorchAO

PyTorch-native library for quantization, sparsity, and low-precision training. Works with `torch.compile()` and `FSDP2`.

## Quick Reference

```python
# Quantize a model to int4
from torchao.quantization import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig(group_size=32))

# Float8 dynamic quantization
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

# Per-layer configs (different quantization per module)
from torchao.quantization import FqnToConfig
quantize_(model, FqnToConfig({"layers.0.attn": Int4WeightOnlyConfig(), "layers.0.mlp": Float8DynamicActivationFloat8WeightConfig()}))

# Filter specific layers
quantize_(model, Int4WeightOnlyConfig(), filter_fn=lambda mod, fqn: "mlp" in fqn)

# QAT (prepare, train, then convert)
from torchao.quantization import Int8DynamicActivationIntxWeightConfig, PerGroup
from torchao.quantization.qat import QATConfig
base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4, weight_granularity=PerGroup(32))
quantize_(model, QATConfig(base_config, step="prepare"))
# ... train ...
quantize_(model, QATConfig(base_config, step="convert"))

# Float8 training (H100/B200 required)
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(model)
```

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

### Granularity

Controls how many elements share a quantization scale. Import from `torchao.quantization`:
- `PerTensor` - one scale for the whole tensor
- `PerRow` / `PerAxis` - one scale per row/axis (recommended for float8)
- `PerGroup(group_size)` - one scale per group (e.g., group_size=32 for int4)
- `PerBlock` - one scale per block
- `PerToken` - one scale per token (for activations)

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
