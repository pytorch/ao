# torchao.float8

This is a workflow for accelerating training with [float8](https://arxiv.org/pdf/2209.05433.pdf) in native PyTorch.
With ``torch.compile`` on, we demonstrate e2e pretraining throughput speedups of up to [**1.5x at 512 GPU / 405B parameter count scale**](https://pytorch.org/blog/training-using-float8-fsdp2/),
and up to [**1.25x at 8 GPU / 8B parameter count scale**](#training-benchmarks).
The codebase strives to stay small, hackable, debuggable with native PyTorch tooling
and composable with key systems such as autograd, ```torch.compile``` and distributed.

## Key features

* e2e pretraining speedups of up to [**1.5x at 512 GPU / 405B parameter count scale**](https://pytorch.org/blog/training-using-float8-fsdp2/),
and up to [**1.25x at 8 GPU / 8B parameter count scale**](#training-benchmarks), with performance and accuracy validated on up to [**2k GPUs**](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/), via [torchtitan's float8 integration](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)
* seamless composability with [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html)
* seamless composability with [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html), including [FSDP2 with float8 weight all-gather](https://dev-discuss.pytorch.org/t/enabling-float8-all-gather-in-fsdp2/2359) and [Async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)
* seamless composability with [PyTorch Activation Checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/)
* three different scaling recipes to trade off performance vs accuracy: tensorwise (fastest), rowwise, rowwise_with_gw_hp (most accurate)
* supports both NVIDIA and AMD hardware

ℹ️ <em>See the [feature tracker](https://github.com/pytorch/ao/issues/556) for upcoming features.</em>

ℹ️ <em>These APIs are training-only and float8-only, and we plan to [unify them with the rest of torchao](https://github.com/pytorch/ao/issues/894) in the future.</em>

# Single GPU User API

```python
import time

import torch
import torch.nn as nn
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

# create model and sample input
M, K, N = 4096, 8192, 4096
m = nn.Sequential(
    nn.Linear(K, N, bias=False),
    nn.Linear(N, 128, bias=False),
).bfloat16().cuda()
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
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

# configure float8 recipe
# valid recipe names: "tensorwise", "rowwise", "rowwise_with_gw_hp"
config = Float8LinearConfig.from_recipe_name("tensorwise")

# convert specified `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(m, config=config, module_filter_fn=module_filter_fn)

# display converted model
print(m)

# enable torch.compile for competitive performance
m = torch.compile(m)

# warm up torch.compile for a clean training time measurement
for _ in range(1):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()

torch.cuda.synchronize()
start_time = time.time()

# toy training loop
for _ in range(10):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()

torch.cuda.synchronize()
end_time = time.time()
print("Training time:", end_time - start_time)
```

# Multi GPU User API

We compose with the `DTensor` based [distributed APIs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html),
such as FSDP, TP and SP. Please see the [torchtitan](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md) repository for e2e examples
on using `torchao.float8` in a distributed setting.

# Performance

A common question about float8 training is "when is float8 linear faster vs bfloat16?".  Given the M, K, N of the forward pass through your linear, you can reference the tables below for a microbenchmark based speedup estimate on NVIDIA H100:

### tensorwise scaling

<img width="753" height="773" alt="Image" src="https://github.com/user-attachments/assets/e46c671a-ed35-41b4-b17c-50caf1629ecb" />

```lang=shell
# reproduction: run the script below
python benchmarks/float8/float8_roofline.py your_output_filename.csv --shape_gen_name sweep
```

### rowwise scaling

<img width="755" height="778" alt="Image" src="https://github.com/user-attachments/assets/7d70ba36-f480-459f-b5c0-797895332631" />

```lang=shell
# reproduction: run the script below
python benchmarks/float8/float8_roofline.py your_output_filename.csv --shape_gen_name sweep --float8_recipe_name rowwise
```

### rowwise_with_gw_hp scaling

<img width="750" height="797" alt="Image" src="https://github.com/user-attachments/assets/e4479abc-1aca-436d-a142-60e5e804ff10" />

```lang=shell
# reproduction: run the script below
python benchmarks/float8/float8_roofline.py your_output_filename.csv --shape_gen_name sweep --float8_recipe_name rowwise_with_gw_hp
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
3. From the `torchao/benchmarks/float8/training/` directory, you can run the following commands to reproduce the benchmarks above:
   - bf16 + compile: `TORCHTITAN_ROOT=<path> ./torchtitan_benchmark.sh`
   - float8 tensorwise with float8 all-gather + compile: `TORCHTITAN_ROOT=<path> FLOAT8_RECIPE_WITH_BEST_SETTINGS="tensorwise" ./torchtitan_benchmark.sh`
   - float8 rowwise with bf16 all-gather + compile: `TORCHTITAN_ROOT=<path> FLOAT8_RECIPE_WITH_BEST_SETTINGS="rowwise" ./torchtitan_benchmark.sh`

See the float8 training benchmarking [guide](.torchao/benchmarks/float8/training/README.md) for more details.

# E2E training + inference flow

The first step in the E2E is to train your model and save a checkpoint. The second step is to load the checkpoint and optionally apply inference quantization before serving the model.
#### 1. Train model and save checkpoint
```python
import torch
from torch import nn
import torch.nn.functional as F

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear
from torchao.float8 import convert_to_float8_training
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

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

Below is an example of dynamic activation and weight quantization. For more details, examples, and inference benchmrks, see the [torchao inference docs](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md).

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
