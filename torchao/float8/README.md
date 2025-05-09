# torchao.float8

This is a workflow for accelerating training with [float8](https://arxiv.org/pdf/2209.05433.pdf) in native PyTorch.
With ``torch.compile`` on, we demonstrate e2e pretraining throughput speedups of up to [**1.5x at 512 GPU / 405B parameter count scale**](https://pytorch.org/blog/training-using-float8-fsdp2/),
and up to [**1.25x at 8 GPU / 8B parameter count scale**](#training-benchmarks).
The codebase strives to stay small, hackable, debuggable with native PyTorch tooling
and composable with key systems such as autograd, ```torch.compile``` and distributed.

ℹ️ <em>See the [feature tracker](https://github.com/pytorch/ao/issues/556) for upcoming features.</em>

ℹ️ <em>These APIs are training-only and float8-only, and we plan to [unify them with the rest of torchao](https://github.com/pytorch/ao/issues/894) in the future.</em>

# Single GPU User API

## float8 linear with dynamic tensorwise scaling

This is the default recipe, with a good balance of performance and accuracy.

```python
import torch
import torch.nn as nn
from torchao.float8 import convert_to_float8_training
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

# create model and sample input
m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
).bfloat16().cuda()
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

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
    y = m(x)
    y.sum().backward()
    optimizer.step()
```

## float8 linear with rowwise scaling

This is a more accurate recipe compared to tensorwise, with more granular scaling.

```python
import torch
import torch.nn as nn
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

# create model and sample input
m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
).bfloat16().cuda()
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

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

# configure rowwise scaling
config = Float8LinearConfig.from_recipe_name("rowwise")

# convert specified `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(m, config=config, module_filter_fn=module_filter_fn)

# enable torch.compile for competitive performance
m = torch.compile(m)

# toy training loop
for _ in range(10):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()
```

# Multi GPU User API

We compose with the `DTensor` based [distributed APIs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html),
such as FSDP, TP and SP. Please see the [torchtitan](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md) repository for e2e examples
on using `torchao.float8` in a distributed setting.

# Performance

A common question about float8 training is "when is float8 linear faster vs bfloat16?".  Given the M, K, N of the forward pass through your linear, you can reference the table below for a microbenchmark based speedup estimate on NVIDIA H100:

<img width="805" alt="float8_speedup" src="https://github.com/user-attachments/assets/5c5f2817-7eb7-4cab-bd03-49fe70cd31a8">

Example 1 (small shapes):
* forward input tensor size 1024x2048, linear weight size 2048x1024; M, K, N = 1024, 2048, 1024
* benchmark speedup is 0.80
* recommendation: leave this linear in bfloat16, the shapes are too small to benefit from float8 compute

Example 2 (large shapes):
* forward input tensor size 4096x8192, linear weight size 8192x16384; M, K, N = 4096, 8192, 16384
* benchmark speedup is 1.39
* recommendation: enable float8 for this linear to get a speedup

To reproduce the raw data for table above, you can run the following script

```lang=shell
python benchmarks/float8/float8_roofline.py your_output_filename.csv --shape_gen_name sweep
```

## Derivation

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

# Testing

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

# Benchmarking

```bash
# benchmark the torch._scaled_mm function on LLaMa 2 70B shapes
./benchmarks/float8/bench_matmul.py

# benchmark fw/bw of `Linear` and `Float8Linear` on LLaMa 2 70B shapes
# make sure to turn on torch.compile to get the best performance
./benchmarks/float8/bench_linear_float8.py -o ../tmp/test.txt --compile
```

### Training benchmarks

[Torchtitan](https://github.com/pytorch/torchtitan) was used to benchmark float8 training performance, for both rowwise
and tensorwise scaling. The training benchmarks were all run using:

- Single-node training on 8xH100 GPUs
- Batch size 1
- Sequence length 8192
- Steps 100
- `torch.compile`
- FSDP2
- pytorch version: `2.7.0a0+gitb98af95`
- torchao version: `0.10.0+git890e0ac8`
- torchtitan version: `0.0.2`


| Model         | Scaling                            | Activation checkpointing | Peak Memory (GB)  | Median tokens/second | Speedup over baseline
| ------------- | ---------------------------------- | ------------------------ | ------------------| -------------------- | ---------------------
| Llama3-8b     |  none (bfloat16)                   | per op SAC               | 47.65             |  6150                | -
| Llama3-8b     |  tensorwise with float8 all-gather | per op SAC               | 47.77             |  7689.5              | 25.03%
| Llama3-8b     |  rowwise with bfloat16 all-gather  | per op SAC               | 47.79             |  6768                | 10.05%

**Important notes**:
- E2E speedups increase as M,K,N (GEMM dimensions) increase. Speedups as high as 1.5x have been measured with larger shapes ([example](https://pytorch.org/blog/training-using-float8-fsdp2/)).
- Rowwise scaling is better at handling outliers than tensorwise scaling, so these recipes are different points on the accuracy vs performance curve.

**Reproducing training benchmarks**
To reproduce these benchmarks, you can follow these steps:

1. On a machine with 8 H100 GPUs, clone torchtitan and follow local installation [steps](https://github.com/pytorch/torchtitan?tab=readme-ov-file#installation),
including [downloading a tokenizer](https://github.com/pytorch/torchtitan?tab=readme-ov-file#downloading-a-tokenizer).
2. Install torchao following these [steps](https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#installation).
3. From the `torchao/float8/benchmarking/` directory, you can run the following commands to reproduce the benchmarks above:
   - bf16 + compile: `TORCHTITAN_ROOT=<path> ./float8_training_benchmark.sh`
   - float8 tensorwise with float8 all-gather + compile: `TORCHTITAN_ROOT=<path> FLOAT8_RECIPE_WITH_BEST_SETTINGS="tensorwise" ./float8_training_benchmark.sh`
   - float8 rowwise with bf16 all-gather + compile: `TORCHTITAN_ROOT=<path> FLOAT8_RECIPE_WITH_BEST_SETTINGS="rowwise" ./float8_training_benchmark.sh`

See the float8 training benchmarking [guide](.torchao/float8/benchmarking/README.md) for more details.

# E2E training + inference flow

There are two float8 inference quantization strategies that be used after training with float8: 1) weight only, and 2) dynamic activation and weight.
#### 1. Train model and save checkpoint
```python
import torch
from torch import nn

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear

import torch
import torch.nn as nn
from torchao.float8 import convert_to_float8_training
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

# create model and sample input
m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
).bfloat16().cuda()
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

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
    y = m(x)
    y.sum().backward()
    optimizer.step()

# save the model
torch.save({
    'model': m,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')
```

#### 2. Load checkpoint and optionally apply inference quantization

```python
import torch

from torchao.float8.float8_linear import Float8Linear
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import quantize_
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.quantization import (
    float8_weight_only,
)

# load checkpoint
checkpoint = torch.load('checkpoint.pth', weights_only=False)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])

# option 1: weight only quantization
quantize_(model, float8_weight_only())

# option 2: dynamic float8 quantization on both activations and weights for inference
# quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

# run inference
x = torch.randn(1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    out = model(x)
    print(out)
```

For more float8 inference performance benchmarks, see the inference docs [here](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#cuda-backend-1).
