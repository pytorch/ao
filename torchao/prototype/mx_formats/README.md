# MX training and inference with native PyTorch

e2e training and inference with mxfp8, mxfp4, nvfp4 formats from the [MX OCP spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
in native PyTorch.  

> :warning: We are currently in prototype.  Use nightly versions of PyTorch and torchao (or build from source) for best results.

## Overall status

### mxfp8

| workflow | emulation | performance | accuracy | API polish |
| --- | --- | --- | --- | --- |
| training for `torch.nn.Linear` | ✅ | 🟡 / 🟢 | 🟢 | 🟡 |
| inference for `torch.nn.Linear` | ✅ | 🟡 / 🟢 | 🟢 | 🟡 |

### nvfp4

| workflow | emulation | performance | accuracy | API polish |
| --- | --- | --- | --- | --- |
| training for `torch.nn.Linear` | ✅ | 🔴 | 🟡 | 🟡 |
| QAT for `torch.nn.Linear` | ✅ | n/a | 🟢 | 🟡 |
| inference for `torch.nn.Linear` | ✅ | 🟡 / 🟢 | 🟢 | 🟡 |

### mxfp4

| workflow | emulation | performance | accuracy | API polish |
| --- | --- | --- | --- | --- |
| training for `torch.nn.Linear` | ✅ | 🔴 | 🟡 | 🟡 |
| QAT for `torch.nn.Linear` | planned | n/a | planned | planned |
| inference for `torch.nn.Linear` | ✅ | 🔴 | 🟢 | 🟡 |

### planned improvements

* mxfp8 support for grouped_gemm and all2all for MoE training (see https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training ).
* mxfp8, nvfp4, mxfp4 performance optimizations for inference
* polish the nvpf4 QAT recipe, and enable mxfp4 QAT
* blocked formats for faster training
* stochastic rounding and hadamard transforms for improved fp4 training numerics

## Training e2e benchmarks on NVIDIA B200

- Single-node training on 8x power limited B200 GPUs, batch size 1, sequence length 8192, steps 100, `torch.compile`, FSDP2, per-op SAC
- pytorch version: `2.9.0.dev20250815+cu128`, torchao version: `0.13.0+gite4e681be6`, torchtitan commit: `6fc499f6f5b32151a799188be2208cfb09faed30`

| Model         | Scaling                            | Peak Memory (GB)  | Median tokens/second | Speedup over baseline
| ------------- | ---------------------------------- | ------------------| -------------------- | ---------------------
| Llama3-8b     |  none (bfloat16)                   | 33.71             |  8307.5              | -
| Llama3-8b     |  float8 tensorwise (f8 all-gather) | 33.38             |  10417.0             | 25.4%
| Llama3-8b     |  mxfp8_cublas                      | 33.88             |  9969.0              | 20.0%
| Llama3-8b     |  mxfp8_cublas_rceil                | 33.88             |  9642.0              | 16.1%
| Llama3-8b     |  float8 rowwise                    | 33.72             |  8640.5              | 4.0%

**Reproducing training benchmarks**
To reproduce these benchmarks, you can follow these steps:

1. On a machine with compatible GPUs, clone torchtitan and follow local installation [steps](https://github.com/pytorch/torchtitan?tab=readme-ov-file#installation),
including [downloading a tokenizer](https://github.com/pytorch/torchtitan?tab=readme-ov-file#downloading-a-tokenizer).
2. Install torchao following these [steps](https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#installation).
3. From the `torchao/` directory, you can run the following commands to reproduce the benchmarks above:
   - bf16 + compile: `TORCHTITAN_ROOT=<path> ./benchmarks/float8/training/llama3.sh`
   - mxfp8_cublas: `TORCHTITAN_ROOT=<path> MX_RECIPE="mxfp8_cublas" ./benchmarks/float8/training/llama3.sh`
   - mxfp8_cublas_rceil: `TORCHTITAN_ROOT=<path> MX_RECIPE="mxfp8_cublas_rceil" ./benchmarks/float8/training/llama3.sh`

> :warning: For now you need to build `torchao` from source for optimal training performance. See https://github.com/pytorch/ao/issues/2932 for details.

# User API

## MX training

Below is a toy training loop. For an example real training loop, see our torchtitan integration here: https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/quantization/mx.py .

```python
import torch
from torchao.quantization import quantize_
import torchao.prototype.mx_formats
from torchao.prototype.mx_formats import MXLinearConfig, ScaleCalculationMode
from torchao.quantization.quantize_.common import KernelPreference

# low precision gemm, requires CUDA capability 10.0+
kernel_preference = KernelPreference.AUTO
# or, emulated gemm
# kernel_preference = KernelPreference.EMULATED

scale_calculation_mode = ScaleCalculationMode.FLOOR
# other supported modes: RCEIL, CEIL, EVEN

m = torch.nn.Sequential(torch.nn.Linear(32, 32)).cuda()
config = MXLinearConfig(
    elem_dtype=torch.float8_e4m3fn,
    block_size=32,
    kernel_preference=kernel_preference,
    scale_calculation_mode=scale_calculation_mode,
)
quantize_(m, config)
m = torch.compile(m, fullgraph=True)

# training loop (not shown)
```

## MX inference

```python
import copy

import torch
import torch.nn as nn
from torchao.quantization import quantize_
import torchao.prototype.mx_formats
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
    NVFP4WeightOnlyConfig,
)
from torchao.quantization.quantize_.common import KernelPreference

m = nn.Linear(32, 128, bias=False, dtype=torch.bfloat16, device="cuda")
x = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)

# mxfp8

m_mxfp8 = copy.deepcopy(m)
config = MXDynamicActivationMXWeightConfig(
    activation_dtype=torch.float8_e4m3fn,
    weight_dtype=torch.float8_e4m3fn,
    kernel_preference=KernelPreference.AUTO,
)
quantize_(m_mxfp8, config=config)
m_mxfp8 = torch.compile(m_mxfp8, fullgraph=True)
y_mxfp8 = m_mxfp8(x)

# nvfp4 dynamic quant

m_nvfp4 = copy.deepcopy(m)
config = NVFP4DynamicActivationNVFP4WeightConfig(
    use_dynamic_per_tensor_scale=True,
    use_triton_kernel=True,
)
quantize_(m_nvfp4, config=config)
m_nvfp4 = torch.compile(m_nvfp4, fullgraph=True)
y_nvfp4 = m_nvfp4(x)

# nvfp4 weight-only quant

m_nvfp4_wo = copy.deepcopy(m)
config = NVFP4WeightOnlyConfig(
    use_dynamic_per_tensor_scale=True,
)
quantize_(m_nvfp4_wo, config=config)
m_nvfp4_wo = torch.compile(m_nvfp4_wo, fullgraph=True)
y_nvfp4 = m_nvfp4_wo(x)

# mxfp4

m_mxfp4 = copy.deepcopy(m)
config = MXDynamicActivationMXWeightConfig(
    activation_dtype=torch.float4_e2m1fn_x2,
    weight_dtype=torch.float4_e2m1fn_x2,
    kernel_preference=KernelPreference.AUTO,
)
quantize_(m_mxfp4, config=config)
m_mxfp4 = torch.compile(m_mxfp4, fullgraph=True)
y_mxfp4 = m_mxfp4(x)
```

## MXTensor

This is casts between high precision and MX formats implemented in native PyTorch. Currently
only `torch.float32` and `torch.bfloat16` are supported as high precision formats.

```python
from torchao.prototype.mx_formats.mx_tensor import MXTensor
# Note: MX int8 is not implemented yet
from torchao.prototype.mx_formats.constants import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2
x = torch.randn(32, 32, device='cuda')

# elem_dtype can be torch.float8_e4m3fn, torch.float8_e5m2, DTYPE_FP6_E2M3, DTYPE_FP6_E3M2, torch.float4_e2m1fn_x2
elem_dtype = torch.float8_e4m3fn

# high precision to MX, block size defaults to 32
x_mx = MXTensor.to_mx(x, elem_dtype)

# mx back to high precision
x_hp = x_mx.to_dtype(torch.float)
```

# performance

## mxfp8 gemm

On NVIDIA B200 machines, we use the cuBLAS mxfp8 gemm exposed via the `torch._scaled_mm` op.
We observe a speedup of **up to ~2x** vs the bf16 baseline on common shapes.  To reproduce this
on supported hardware, you can run the following command:

```bash
> python benchmarks/float8/bench_matmul.py --recipe mxfp8_cublas
// example output: https://gist.github.com/vkuzo/a1ddb782e6e1c2aef0c726b3df99efbc
```

## to_mx cast across dim0 and dim1

On NVIDIA B200 machines, our to_mx kernels for mxfp8 achieve **up to 6.3 TB/s** for the dim0 cast (with torch.compile),
and **up to 3.9 TB/s** for the dim1 cast (with a triton kernel). We are actively working on improving
the performance of this cast ([details](https://github.com/pytorch/ao/issues/1768)).

To reproduce this on supported hardware, you can run the following command:

```bash
// dim0 cast with torch.compile
> python benchmarks/mx_formats/cast_bench.py --mode dim0_mx --M 16384 --K 16384
// example output: https://gist.github.com/vkuzo/06aae58de9b8aae02c82adb00eb33197

// dim1 cast with a handwritten triton kernel
> python benchmarks/mx_formats/cast_bench.py --mode dim1_mx_triton --M 16384 --K 16384
// example output: https://gist.github.com/vkuzo/7ac5fce44c9b90bfb9eae2a07b721cda
```

# accuracy

## training

* LLaMa 3 8B pretraining on 4 GPUs for 500 iterations shows that loss convergence is not meaningfully degraded (via torchtitan)

## inference

Coming soon!

# testing

```bash
pytest test/prototype/mx_formats/
```
