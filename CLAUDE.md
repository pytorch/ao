# TorchAO

PyTorch-native library for quantization, sparsity, and low-precision training.

## Config Classes

All configs inherit from `AOBaseConfig`. Defined in `torchao/quantization/quant_api.py`. Use `FqnToConfig` to apply different configs to different layers by module name.

## Stable vs Prototype

- **Stable** (`torchao/quantization/`, `torchao/float8/`, `torchao/sparsity/`, `torchao/optim/`): API stability guaranteed.
- **Prototype** (`torchao/prototype/`): Experimental, API may change without notice.

See [docs/source/workflows/index.md](docs/source/workflows/index.md) for the full dtype x hardware status matrix.

## Architecture and Contributing

- [Quantization Overview](docs/source/contributing/quantization_overview.rst) - full stack walkthrough, tensor subclasses, quantization flows
- [Contributor Guide](docs/source/contributing/contributor_guide.rst) - how to add tensors, kernels, configs
- [Inference Workflows](docs/source/workflows/inference.md) - which config to use for which hardware
- [PT2E Quantization](docs/source/pt2e_quantization/index.rst) - PyTorch 2 Export quantization for deployment backends (X86, XPU, ExecuTorch)

These render at https://docs.pytorch.org/ao/main/

## Deprecated APIs

Do not use or recommend these:
- `AffineQuantizedTensor` (AQT) in `torchao/dtypes/` - old v1 system, being removed
- `autoquant()` - deleted
- Layout registration system (`PlainLayout`, `Float8Layout`, `TensorCoreTiledLayout`, etc.) - deleted
- `TorchAODType` - deprecated
- `change_linear_weights_to_int4_woqtensors` - deleted, use `quantize_(model, Int4WeightOnlyConfig())`

New tensor types should inherit from `TorchAOBaseTensor` in `torchao/utils.py`, not AQT.

## Development

```bash
# Setup
USE_CPP=0 pip install -e . --no-build-isolation   # CPU-only
USE_CUDA=1 pip install -e . --no-build-isolation   # With CUDA

# Test (mirrors source structure)
pytest test/quantization/test_quant_api.py
pytest test/float8/
pytest test/prototype/mx_formats/
```
