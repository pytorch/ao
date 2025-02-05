# MX formats with native PyTorch POC

This is a POC of training and inference with tensors in the MX format from the OCP spec (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) in native PyTorch.

Note that the current version of the code is written for readability and
numerical correctness and not yet for optimal performance. We welcome
contributions on performance improvements.

Note that there are no BC guarantees at the moment and we plan to evolve
this code as the hardware specifics of MX-accelerated matmuls become
known.

# Current status

## user API (subject to change)

### MXTensor

This is casts between high precision and MX formats implemented in native PyTorch. Currently
only `torch.float32` and `torch.bfloat16` are supported as high precision formats.

```python
from torchao.prototype.mx_formats.mx_tensor import MXTensor
# Note: MX int8 is not implemented yet
from torchao.prototype.mx_formats.constants import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2, DTYPE_FP4
x = torch.randn(32, 32, device='cuda')

# elem_dtype can be torch.float8_e4m3fn, torch.float8_e5m2, DTYPE_FP6_E2M3, DTYPE_FP6_E3M2, DTYPE_FP4
elem_dtype = torch.float8_e4m3fn

# high precision to MX, block size defaults to 32
x_mx = MXTensor.to_mx(x, elem_dtype)

# mx back to high precision
x_hp = x_mx.to_dtype(torch.float)
```

### MXLinear

This is a module to do MX training, the MX matmul is currently emulated.

```python
from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_linear

m = torch.nn.Sequential(torch.nn.Linear(32, 32)).cuda()
elem_dtype = torch.float8_e4m3fn
swap_linear_with_mx_linear(m, elem_dtype, block_size=32)

# training loop (not shown)
```

### MXInferenceLinear

This is a module to do MX inference, weights are in MX and matmul is in high precision.

```python
from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_inference_linear

m = torch.nn.Sequential(torch.nn.Linear(32, 32)).cuda()
elem_dtype = torch.float8_e4m3fn
block_size = 32
swap_linear_with_mx_inference_linear(m, elem_dtype, block_size)

# do inference (not shown)
```

## accuracy status
* we match bitwise to other implementations of the OCP MX spec (code not in this repo), with a couple of edge cases left to resolve
* approximate numerics pass for `MXLinear` and `MXInferenceLinear` on sample inputs
* LLaMa 3 8B pretraining on 4 GPUs for 500 iterations shows that loss convergence is not meaningfully degraded (code not in this repo)

## performance status

### quant and dequant

* we have a benchmark of quantizing and dequantizing mxfp8 and mxfp4 tensors with size (1, 4096, 11008)
* latest numbers: https://gist.github.com/vkuzo/83656e4a74777cfc0915de6b27be1ff6

## testing and benchmarking

```bash
# numerical testing of custom fp4 and fp6 casts
pytest test/prototype/mx_formats/test_custom_cast.py
# testing of MXTensor
pytest test/prototype/mx_formats/test_mx_tensor.py
# testing of MXLinear and MXInferenceLinear
pytest test/prototype/mx_formats/test_mx_linear.py

# run the quant and dequant benchmark
python torchao/prototype/mx_formats/benchmarks/bench_qdq.py
```

## floating point format convenience functions

We have a convenience script which summarizes the various properties of
floating point formats:

```bash
python torchao/prototype/mx_formats/fp_format_spec.py
# example output: https://gist.github.com/vkuzo/b8e114aa83736f87d6618b16aa8588c0
```
