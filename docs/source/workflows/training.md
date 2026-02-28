# Quantized Training

For training, we support quantizing `torch.nn.Linear` layers (stable) and `torch._grouped_mm` ops (prototype).
Specifically, we quantize the matrix multiplies in the forward and backward of a linear, as follows:

```python
# high precision (baseline)
     output_bf16 =       input_bf16 @ weight_bf16.t()
 grad_input_bf16 = grad_output_bf16 @ weight_bf16
grad_weight_bf16 =   input_bf16.t() @ grad_output_bf16

# quantized (via torchao APIs, shown for fp8_rowwise, pseudocode)
     output_bf16 =       to_fp8(input_bf16) @ to_fp8(weight_bf16.t())
 grad_input_bf16 = to_fp8(grad_output_bf16) @ to_fp8(weight_bf16)
grad_weight_bf16 =   to_fp8(input_bf16.t()) @ to_fp8(grad_output_bf16)
```

We have various quantized training workflows:
* [`torchao.float8`](float8-section) (stable) for float8 rowwise training for `torch.nn.Linear`.
* [`torchao.prototype.mx_formats`](mx-section) (prototype) for mxfp8 training for `torch.nn.Linear`. This is on its way to stable.
* [`torchao.prototype.moe_training`](https://github.com/pytorch/ao/blob/main/torchao/prototype/moe_training/README.md) (prototype) for mxfp8 training for `torch._grouped_mm` for MoEs. The API will be combined with the training APIs in `torchao.prototype.mx_formats` in the future.
* [`torchao.prototype.quantized_training`](https://github.com/pytorch/ao/blob/main/torchao/prototype/quantized_training/README.md) (prototype) for int8 training for `torch.nn.functional.linear`. This is currently in prototype.

(float8-section)=
## float8

This is a workflow for accelerating training with [float8](https://arxiv.org/pdf/2209.05433.pdf) in native PyTorch.
With ``torch.compile`` on, we demonstrate e2e pretraining throughput speedups of up to [**1.5x at 512 GPU / 405B parameter count scale**](https://pytorch.org/blog/training-using-float8-fsdp2/),
and up to [**1.25x at 8 GPU / 8B parameter count scale**](#training-benchmarks).
The codebase strives to stay small, hackable, debuggable with native PyTorch tooling
and composable with key systems such as autograd, ```torch.compile``` and distributed.

### Key features

* e2e pretraining speedups of up to [**1.5x at 512 GPU / 405B parameter count scale**](https://pytorch.org/blog/training-using-float8-fsdp2/),
and up to [**1.25x at 8 GPU / 8B parameter count scale**](#training-benchmarks), with performance and accuracy validated on up to [**2k GPUs**](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/), via [torchtitan's float8 integration](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)
* seamless composability with [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html), [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html), [FSDP2 with float8 weight all-gather](https://dev-discuss.pytorch.org/t/enabling-float8-all-gather-in-fsdp2/2359), [Async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487), and [PyTorch AC](https://pytorch.org/blog/activation-checkpointing-techniques/)
* three recipes to trade off performance vs accuracy: `tensorwise` (fastest), `rowwise`, `rowwise_with_gw_hp` (most accurate)
* supports both NVIDIA and AMD hardware

ℹ️ <em>See the [feature tracker](https://github.com/pytorch/ao/issues/556) for upcoming features.</em>

### Quick Start

```{literalinclude} ../examples/float8_training_example.py
:language: python
```

(training-benchmarks)=
### e2e training benchmarks

[Torchtitan](https://github.com/pytorch/torchtitan) was used to benchmark float8 training performance.

#### NVIDIA H100

- Single-node training on 8xH100 GPUs, batch size 1, sequence length 8192, steps 100, `torch.compile`, FSDP2, per-op SAC
- pytorch version: `2.7.0a0+gitb98af95`, torchao version: `0.10.0+git890e0ac8`, torchtitan version: `0.0.2`

| Model         | Scaling                            | Peak Memory (GB)  | Median tokens/second | Speedup over baseline
| ------------- | ---------------------------------- | ------------------| -------------------- | ---------------------
| Llama3-8b     |  none (bfloat16)                   | 47.65             |  6150                | -
| Llama3-8b     |  tensorwise with float8 all-gather | 47.77             |  7689.5              | 25.03%
| Llama3-8b     |  rowwise with bfloat16 all-gather  | 47.79             |  6768                | 10.05%

#### AMD MI300x

- Single-node training on 8xMI300X GPUs, batch size 1, sequence length 8192, steps 100, `torch.compile`, FSDP2, per-op SAC
- pytorch version: `2.9.0.dev20250811+rocm6.4`, torchao version `0.13.0+git4fc4068d6`, torchtitan commit `2c8b5947991239913d67e2f7d22a255c3e2a9694`

| Model         | Scaling                            | Peak Memory (GB)  | Median tokens/second | Speedup over baseline
| ------------- | ---------------------------------- | ------------------| -------------------- | ---------------------
| Llama3-8b     |  none (bfloat16)                   | 39.09             |  5376.5              | -
| Llama3-8b     |  tensorwise with float8 all-gather | 39.07             |  6166.0              | 14.68%
| Llama3-8b     |  rowwise_with_gw_hp with bfloat16 all-gather  | 39.32             |  6100.0                | 13.46%
| Llama3-8b     |  rowwise with bfloat16 all-gather  | 39.32             |  5891.0              | 9.57%

**Important notes**:
- E2E speedups increase as M,K,N (GEMM dimensions) increase. Speedups as high as 1.5x have been measured with larger shapes ([example](https://pytorch.org/blog/training-using-float8-fsdp2/)).
- Rowwise scaling is better at handling outliers than tensorwise scaling, so these recipes are different points on the accuracy vs performance curve.

**Reproducing training benchmarks**
To reproduce these benchmarks, you can follow these steps:

1. On a machine with compatible GPUs, clone torchtitan and follow local installation [steps](https://github.com/pytorch/torchtitan?tab=readme-ov-file#installation),
including [downloading a tokenizer](https://github.com/pytorch/torchtitan?tab=readme-ov-file#downloading-a-tokenizer).
2. Install torchao following these [steps](https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#installation).
3. From the `torchao/` directory, you can run the following commands to reproduce the benchmarks above:
   - bf16 + compile: `TORCHTITAN_ROOT=<path> ./benchmarks/float8/training/llama3.sh`
   - float8 tensorwise with float8 all-gather + compile: `TORCHTITAN_ROOT=<path> FLOAT8_RECIPE_WITH_BEST_SETTINGS="tensorwise" ./benchmarks/float8/training/llama3.sh`
   - float8 rowwise with bf16 all-gather + compile: `TORCHTITAN_ROOT=<path> FLOAT8_RECIPE_WITH_BEST_SETTINGS="rowwise" ./benchmarks/float8/training/llama3.sh`

See the float8 training benchmarking [guide](https://github.com/pytorch/ao/blob/main/torchao/benchmarks/float8/training/README.md) for more details.

### Multi GPU User API

We compose with the `DTensor` based [distributed APIs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html),
such as FSDP, TP and SP. Please see the [torchtitan](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md) repository for e2e examples
on using `torchao.float8` in a distributed setting.

### Performance

A common question about float8 training is "when is float8 linear faster vs bfloat16?".  Given the M, K, N of the forward pass through your linear, you can reference the tables below for a microbenchmark based speedup estimate on NVIDIA H100:

#### tensorwise scaling

<img width="753" height="773" alt="Image" src="https://github.com/user-attachments/assets/e46c671a-ed35-41b4-b17c-50caf1629ecb" />

```lang=shell
# reproduction: run the script below
python benchmarks/float8/float8_roofline.py your_output_filename.csv --shape_gen_name sweep
```

#### rowwise scaling

<img width="755" height="778" alt="Image" src="https://github.com/user-attachments/assets/7d70ba36-f480-459f-b5c0-797895332631" />

```lang=shell
# reproduction: run the script below
python benchmarks/float8/float8_roofline.py your_output_filename.csv --shape_gen_name sweep --float8_recipe_name rowwise
```

#### rowwise_with_gw_hp scaling

<img width="750" height="797" alt="Image" src="https://github.com/user-attachments/assets/e4479abc-1aca-436d-a142-60e5e804ff10" />

```lang=shell
# reproduction: run the script below
python benchmarks/float8/float8_roofline.py your_output_filename.csv --shape_gen_name sweep --float8_recipe_name rowwise_with_gw_hp
```

#### Derivation

In a bf16 linear, assume all of the time is spent in gemms.  In a float8 linear, account for max_abs and casting overhead.  We want to know when

```
bf16_gemm_time > fp8_gemm_time + fp8_overhead_time
```

Or, equivalently,

```
bf16_gemm_time - fp8_gemm_time > fp8_overhead_time
```

There are three observations we can make about the formula above:
* LHS > 0 for large shapes, with the gemm speedup approaching 2x as M, K, N increase
* LHS < 0 for small shapes, on NVIDIA H100 + cuBLAS
* RHS > 0 for all shapes, bounded by memory bandwidth, framework overhead and compiler limitations

For small shapes, a combination of (2) and (3) leads to speedup < 1.  For medium shapes, (1) and (3) are of similar magnitude and the speedup depends on M, K, N and framework and compiler behavior.  For large shapes, (1) leads to speedup > 1.

### Testing

```bash
# run single-GPU unit tests
pytest test/float8/test_base.py

# run single-GPU compile tests
pytest test/float8/test_compile.py

# run single-GPU numerics integration tests
pytest test/float8/test_numerics_integration.py

# run a two-GPU integration test on FSDP
./test/float8/test_fsdp.sh

# run integration tests on the DTensor TP/SP integration
./test/float8/test_dtensor.sh

# run integration tests on the FSDP2 integration
python test/float8/test_fsdp2/test_fsdp2.py

# run all of these tests
./test/float8/test_everything.sh
```

### E2E training + inference flow

The first step in the E2E is to train your model and save a checkpoint. The second step is to load the checkpoint and optionally apply inference quantization before serving the model.
#### 1. Train model and save checkpoint
```python
import torch
from torch import nn
import torch.nn.functional as F

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear
from torchao.float8 import convert_to_float8_training

# create model and sample input
m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
    nn.Linear(128, 1),
).bfloat16().cuda()
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the last module
    if fqn == "1":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

# convert specified `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(m, module_filter_fn=module_filter_fn)

# enable torch.compile for competitive performance
m = torch.compile(m)

# toy training loop
for _ in range(10):
    optimizer.zero_grad()
    output = m(x)
    # use fake labels for demonstration purposes
    fake_labels = torch.ones_like(output)
    loss = F.mse_loss(output, fake_labels)
    loss.backward()
    optimizer.step()

# save the model
torch.save({
    'model': m,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')
```

#### 2. Load checkpoint and optionally apply inference quantization

There are 3 float8 inference quantization strategies that be used after training with float8: 1) weight only quantization, and 2) dynamic activation and weight quantization, and 3) static quantization.

Below is an example of dynamic activation and weight quantization. For more details, examples, and inference benchmrks, see the [torchao inference docs](inference.md).

```python
import torch

from torchao.float8.float8_linear import Float8Linear
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import quantize_
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)

# load checkpoint
checkpoint = torch.load('checkpoint.pth', weights_only=False)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])

# optional: apply dynamic float8 quantization on both activations and weights for inference
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

# run inference
x = torch.randn(1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    out = model(x)
    print(out)
```

(mx-section)=
## mxfp8

e2e training with mxfp8 from the [MX OCP spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
in native PyTorch.

> :warning: We are currently in prototype.  Use nightly versions of PyTorch and torchao (or build from source) for best results.

### Overall status

#### mxfp8

| workflow | emulation | performance | accuracy | API polish |
| --- | --- | --- | --- | --- |
| training for `torch.nn.Linear` | ✅ | 🟡 / 🟢 | 🟢 | 🟡 |
| inference for `torch.nn.Linear` | ✅ | 🟡 / 🟢 | 🟢 | 🟡 |

#### nvfp4

| workflow | emulation | performance | accuracy | API polish |
| --- | --- | --- | --- | --- |
| training for `torch.nn.Linear` | ✅ | 🔴 | 🟡 | 🟡 |
| QAT for `torch.nn.Linear` | ✅ | n/a | 🟢 | 🟡 |
| inference for `torch.nn.Linear` | ✅ | 🟡 / 🟢 | 🟢 | 🟡 |

#### mxfp4

| workflow | emulation | performance | accuracy | API polish |
| --- | --- | --- | --- | --- |
| training for `torch.nn.Linear` | ✅ | 🔴 | 🟡 | 🟡 |
| QAT for `torch.nn.Linear` | planned | n/a | planned | planned |
| inference for `torch.nn.Linear` | ✅ | 🔴 | 🟢 | 🟡 |

#### planned improvements

* mxfp8 support for grouped_gemm and all2all for MoE training (see https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training ).
* mxfp8, nvfp4, mxfp4 performance optimizations for inference
* polish the nvpf4 QAT recipe, and enable mxfp4 QAT
* blocked formats for faster training
* stochastic rounding and hadamard transforms for improved fp4 training numerics

### Training e2e benchmarks on NVIDIA B200

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

### User API

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
    elem_dtype=torch.float4_e2m1fn_x2,
    block_size=32,
    kernel_preference=kernel_preference,
    scale_calculation_mode=scale_calculation_mode,
)
quantize_(m, config)
m = torch.compile(m, fullgraph=True)

# training loop (not shown)
```

### MXTensor

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

### performance

#### mxfp8 gemm

On NVIDIA B200 machines, we use the cuBLAS mxfp8 gemm exposed via the `torch._scaled_mm` op.
We observe a speedup of **up to ~2x** vs the bf16 baseline on common shapes.  To reproduce this
on supported hardware, you can run the following command:

```bash
> python benchmarks/float8/bench_matmul.py --recipe mxfp8_cublas
// example output: https://gist.github.com/vkuzo/a1ddb782e6e1c2aef0c726b3df99efbc
```

#### quantization kernel microbenchmarks

Results for shape 16384x16384 on NVIDIA B200 with 1000W power supply:

| Mode | Time (μs) | Memory Bandwidth (GB/s) | Notes
| --- | --- | --- | --- |
| dim0_mxfp8_floor | 125.92 | 6462.00 | `to_mx` + `torch.compile`
| dim0_mxfp8_rceil | 220.10 | 3697.00 | `to_mx` + `torch.compile` (not used: https://github.com/pytorch/pytorch/issues/170635)
| dim0_mxfp8_triton_floor | 139.23 | 5844.17 | Triton kernel
| dim0_mxfp8_triton_rceil | 138.18 | 5888.83 | Triton kernel
| dim1_mxfp8_cuda_floor | 150.56 | 5404.46 | CUDA kernel
| dim1_mxfp8_cuda_rceil | 142.34 | 5716.72 | CUDA kernel

To reproduce these benchmarks:
```bash
conda run -n torch python benchmarks/mx_formats/cast_bench.py --M 16384 --K 16384 --mode <mode>
```
