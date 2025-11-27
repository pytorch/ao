# Low precision MoE Training

This prototype provides:

1. Quantized building block for low precision MoE training: [_to_mxfp8_then_scaled_grouped_mm](https://github.com/pytorch/ao/blob/53b5efdac921a38fd15e8d3ac8191c3927140287/torchao/prototype/moe_training/scaled_grouped_mm.py#L677). It is a differentiable drop-in replacement for `torch._grouped_mm` that dynamically quantizes inputs using the given recipe, performs a scaled grouped GEMM, then returns the results in original precision. See runnable [example](#torchao_scaled_grouped_mm-example-forward--backward-pass) of a forward and backward pass below.
    - Using MXFP8 on a B200 GPU, this provides:
        - **~1.4x - 1.8x speedups** over bfloat16 `torch._grouped_mm` for Llama4 Scout shapes
        - **~1.19 - 1.6x speedups** over bfloat16 `torch._grouped_mm` for DeepSeekV3 671b shapes
    - These benchmarks use `seq_len=8192`, `local_batch_size=16` (so `total_M = 8192 * 16 = 131,072`). We recommend using a large `total_M` dim to maximize speedup. See [benchmarks](#microbenchmarks) for more details.


2. [TorchTitan](https://github.com/pytorch/torchtitan/tree/main) integration: pretrain DeepSeekV3/Llama4 with MXFP8 grouped GEMMs by adding the flag to your training command: `--model.converters="quantize.grouped_mm.mx" --quantize.grouped_mm.mx.fqns="experts"`

3. Model conversion API to swap all `torch._grouped_mm` ops in your model definition to use torchao `_quantize_then_scaled_grouped_mm` under the hood (see [example](#model-conversion-api-example-end-to-end-training) below).


## Equivalent convergence to bfloat16 training baseline

Training runs on 64 node GB200 cluster with TorchTitan Llama4 Scout show that MXFP8 MoE training has equivalent convergence to bfloat16 training baseline. Infact, after 3,000 steps it finishes with slightly *lower* loss than bfloat16! This is consistent with our scaling experiments with [MXFP8 training for dense models](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/).

<img alt="Image" src="../../../docs/static/mxfp8_with_loss.png" />

Training and model configurations for this run:
- Model: Llama4 Scout
- Dataset: C4
- Sequence length: 8192
- Local batch size: 1
- Learning rate: 1e-4
- LR scheduler warmup steps: 2000
- Parallelisms (64 nodes of 4 devices each = 256 chips):
    - FSDP=256 (on attention layers, shared experts, dense layer FFNs) and 256/4=64 (on routed experts)
    - EP=16 (on routed experts)
- Activation checkpointing mode: `none` (ideally this should use selective per op AC but there was a bug at the time preventing us from using it).
- `torch.compile` enabled
- `mxfp8` applied to routed experts computation (grouped GEMMs)
- `mxfp8` applied to all linear layers except: `output`, `router.gate`, `attention.wk`, `attention.wv` (Wk and Wv too small to benefit from mxfp8)


## Table of Contents

- [Examples](#examples)
- [System Requirements](#system-requirements)
- [Microbenchmarks](#microbenchmarks)
- [Single MoE layer benchmarks](#benchmark-single-moe-layer-forward--backward-pass)
- [E2E training benchmarks](#end-to-end-training-benchmark-with-torchtitan-llama4-scout-vs-bfloat16-baseline)
- [Implementation Details for Developers](#implementation-details-for-developers)
- [Limitations](#limitations)

## Examples
#### _to_mxfp8_and_scaled_grouped_mm usage
```python
import torch
from torch.nn import functional as F
from torchao.prototype.moe_training import (
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.conversion_utils import MoEScalingType
from torchao.prototype.moe_training.utils import generate_jagged_offs

num_groups, total_M, N, K = 8, 131072, 8192, 5120

# A = input actvations, B = expert weights
A = torch.randn(total_M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
B = torch.randn(num_groups, N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# Token group offsets computed by router in actual MoE layer
offs = generate_jagged_offs(num_groups, total_M, device="cuda")

# Forward and backward example
out = _to_mxfp8_then_scaled_grouped_mm(
        A,
        B.transpose(-2, -1),
        offs,
)

# (Fake labels for demonstration purposes)
labels = torch.ones_like(out)
loss = F.mse_loss(out, labels)
loss.backward()
```

#### Model conversion API example: end-to-end training
```python
from torchao.prototype.moe_training.conversion_utils import MoEScalingType
import torch
from torch import nn
from torch.nn import functional as F

# This feature requires CUDA 12.8+ and SM100+
assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)

from torchao.prototype.moe_training.conversion_utils import MoETrainingConfig
from torchao.quantization.quant_api import quantize_

# This example uses torchtitan Llama4 MoE.
try:
    from torchtitan.models.moe.utils import (
        set_token_group_alignment_size_m,
    )
    from torchtitan.models.moe import MoE, MoEArgs
except ImportError:
    pytest.skip(
        "torchtitan not installed, skipping MoE tests.", allow_module_level=True
    )


# Initialize model
device = torch.device("cuda")
moe_args = MoEArgs(
    num_experts=8,
)
dim, hidden_dim = 5120, 8192
model = MoE(moe_args, dim, hidden_dim).to(torch.bfloat16).to(device)
init_std = 0.02
model.init_weights(init_std, device)

# Module filter function to define which modules to quantize
target_fqns = ["experts"]


def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
    for target_fqn in target_fqns:
        if target_fqn in cur_fqn:
            return True
    return False

# Token group sizes must be padded to multiple of MXFP8 scaling block size (1x32)
alignment_size = 32
set_token_group_alignment_size_m(alignment_size)

# Convert model to use MXFP8 scaled grouped GEMMs
config = MoETrainingConfig(scaling_type=MoEScalingType.MXFP8)
quantize_(model, config=config, filter_fn=moe_module_filter_fn)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size, seq_len = 2, 2048
for step in range(10):
    # Simulate random batch of input data
    x = torch.randn(
        batch_size, seq_len, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )

    # Forward pass
    out = model(x)

    # Compute loss with fake labels for demonstration purposes
    labels = torch.ones_like(out)
    out_loss = F.mse_loss(out, labels)
    print(f"step {step} loss: {out_loss.item()}")

    # Backward pass
    out_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## System requirements
- torchao 0.14+
- For MXFP8 MoE training, CUDA 12.8+ and SM100+ GPU arch are required.
- For FP8 rowwise MoE training, CUDA 12.4+ and SM89+ GPU arch are required.


## Microbenchmarks

**MXFP8 with Llama4 17b 16e shapes** (with G from 1 to 8 to simulate different degrees of expert parallelism)

| M,N,K,G                 | bf16_fwd_bwd_us | scaled_fwd_bwd_us | scaled_fwd_bwd_speedup |
| ----------------------- | --------------: | ----------------: | ---------------------: |
| (128000, 8192, 5120, 1) |        43140.20 |          23867.00 |                 1.808x |
| (128000, 8192, 5120, 2) |        39487.60 |          23359.00 |                 1.690x |
| (128000, 8192, 5120, 4) |        39189.20 |          23945.50 |                 1.637x |
| (128000, 8192, 5120, 8) |        37700.70 |          22170.60 |                 1.700x |

**MXFP8 with DeepSeekV3** (with G from 1 to 8 to simulate different degrees of expert parallelism)

| M,N,K,G                 | bf16_fwd_bwd_us | scaled_fwd_bwd_us | scaled_fwd_bwd_speedup |
| ----------------------- | --------------: | ----------------: | ---------------------: |
| (128000, 2048, 7168, 1) |        13064.80 |          10996.00 |                 1.188x |
| (128000, 2048, 7168, 2) |        14900.20 |          11283.40 |                 1.321x |
| (128000, 2048, 7168, 4) |        15823.60 |           9919.36 |                 1.595x |
| (128000, 2048, 7168, 8) |        14966.80 |          10397.20 |                 1.440x |


To reproduce this benchmark, on a B200 GPU machine, run the following command:
- `python benchmarks/prototype/moe_training/benchmark_scaled_grouped_mm_dq.py --compile`
- torchao: `0.14.0+gitc7b8e13da`
- torch: `2.10.0a0+gitf6de195`

### Roofline Performance Analysis

The following roofline plots provide roofline analysis and benchmarks for the following:

1. **Net Speedup vs () Size** - MXFP8 vs BF16 for forward + backward pass
2. **2D Quantization + Block Format Kernels** - Bandwidth utilization for input quantization and per-group scale conversion to blocked format
3. **3D Quantization + Block Format Kernels** - Bandwidth utilization for weight quantization and per-group scale conversion to blocked format
4. **Grouped GEMM Kernel Speedup** - MXFP8 over BF16 for 2D/3D and 2D/2D GEMM operations
5. **Kernel Breakdown** - Stacked bar chart showing actual measured times for each kernel component (quantization, conversion to blocked format, GEMM) across forward, backward input, and backward weight passes

These benchmarks were generated on **November 26, 2025** and will be updated with every change that affects performance.

Next steps for optimization:
* Improve 2D-2D MXFP8 grouped GEMM CUTLASS kernel performance (used for computing wgrad), which currently produces much lower speedups than the 2D-3D case (used for computing output and dgrad).

#### Llama4 Shapes (K=5120, N=8192, G=8)

![Llama Rooflines](../../../benchmarks/prototype/moe_training/mxfp8/llama_rooflines.png)

**Command to reproduce:**
```bash
cd benchmarks/prototype/moe_training/mxfp8
python roofline_unified.py --K=5120 --N=8192 --G=8 --power_limit_percent=100 --breakdown_M=131072 --plot_file=llama_rooflines.png
```

#### DeepSeek V3 Shapes (K=7168, N=2048, G=8)

![DeepSeek V3 Rooflines](../../../benchmarks/prototype/moe_training/mxfp8/dsv3_rooflines.png)

**Command to reproduce:**
```bash
cd benchmarks/prototype/moe_training/mxfp8
python roofline_unified.py --K=7168 --N=2048 --G=8 --power_limit_percent=100 --breakdown_M=131072 --plot_file=dsv3_rooflines.png
```

## Benchmark: single MoE layer forward + backward pass

| Model        | total_M | N    | K    | bf16 time (ms) | mxfp8 time (ms) | speedup |
|--------------|---------|------|------|---------------|-----------------|---------|
| Llama4 16e   | 131072  | 8192 | 5120 | 275.270       | 192.420         | 1.431x  |
| DeepSeekV3   | 131072  | 2048 | 7168 | 92.032        | 80.182          | 1.148x  |

To reproduce these benchmarks, on a B200 GPU machine, run the following commands:

Llama4 17b 16e shapes:
```bash
CUDA_VISIBLE_DEVICES=6 python benchmarks/prototype/moe_training/bench_moe_layer.py --recipe mxfp8 --local_batch_size=16 --dim=5120 --hidden_dim=8192 --local_num_experts=8
```

DeepSeekV3 671b shapes:
```bash
CUDA_VISIBLE_DEVICES=6 python benchmarks/prototype/moe_training/bench_moe_layer.py --recipe mxfp8 --local_batch_size=16 --dim=7168 --hidden_dim=2048 --local_num_experts=8
```



## End-to-end training benchmark with TorchTitan: Llama4 Scout vs bfloat16 baseline
- Single node benchmarks with 4xB200
- Llama4 16e default configs; FSDP=4, EP=4; AC=none; compile=True; seq_len=8192; local_bs=8
- Reduced num layers from 48 -> 2 to avoid OOM in single node setting
- TorchTitan debug model config (same as Llama4 17bx16e, but with 2 layers):


| Configuration                                                              | Throughput (Median Tokens/s) | Max Memory (GiB) | Speedup over bf16
|:---------------------------------------------------------------------------|-----------------------------:|------------------|------------------|
| bf16 baseline                                                              |                      49381.0 |           145.55 | -
| MXFP8 for Linears only                                                     |                      52038.0 |           146.62 | 1.053x
| MXFP8 for Grouped GEMMs only                                               |                      69350.0 |           144.71 | 1.404x
| MXFP8 for Linears + Grouped GEMMs                                          |                      70747.0 |           145.32 | 1.433x

#### Commands to reproduce these benchmarks:

bfloat16 baseline:
```
rm -rf /tmp/torchinductor_${USER}; CUDA_VISIBLE_DEVICES="4,5,6,7" TORCHTITAN_ROOT=/home/${USER}/torchtitan NGPU=4 EXTRA_ARGS="--metrics.log_freq=10 --training.steps=200  --parallelism.data_parallel_shard_degree=4 --parallelism.expert_parallel_degree=4 --parallelism.tensor_parallel_degree=1 --compile.enable --training.seq_len=8192 --activation_checkpoint.mode=none --model.print_after_conversion" ./llama4.sh
```

MXFP8 dense only:
```
rm -rf /tmp/torchinductor_${USER}; CUDA_VISIBLE_DEVICES="4,5,6,7" TORCHTITAN_ROOT=/home/${USER}/torchtitan NGPU=4 EXTRA_ARGS="--metrics.log_freq=10 --training.steps=200  --parallelism.data_parallel_shard_degree=4 --parallelism.expert_parallel_degree=4 --parallelism.tensor_parallel_degree=1 --compile.enable --training.seq_len=8192 --activation_checkpoint.mode=none --model.print_after_conversion --model.converters="quantize.linear.mx"" ./llama4.sh
```

MXFP8 MoE only:
```
rm -rf /tmp/torchinductor_${USER}; CUDA_VISIBLE_DEVICES="4,5,6,7" TORCHTITAN_ROOT=/home/${USER}/torchtitan NGPU=4 EXTRA_ARGS="--metrics.log_freq=10 --training.steps=200  --parallelism.data_parallel_shard_degree=4 --parallelism.expert_parallel_degree=4 --parallelism.tensor_parallel_degree=1 --compile.enable --training.seq_len=8192 --activation_checkpoint.mode=none --model.print_after_conversion --model.converters="quantize.grouped_mm.mx"" ./llama4.sh
```

MXFP8 MoE + Dense:
```
rm -rf /tmp/torchinductor_${USER}; CUDA_VISIBLE_DEVICES="4,5,6,7" TORCHTITAN_ROOT=/home/${USER}/torchtitan NGPU=4 EXTRA_ARGS="--metrics.log_freq=10 --training.steps=50  --parallelism.data_parallel_shard_degree=4 --parallelism.expert_parallel_degree=4 --parallelism.tensor_parallel_degree=1 --compile.enable --training.seq_len=8192 --activation_checkpoint.mode=none --model.print_after_conversion --model.converters="quantize.grouped_mm.mx,quantize.linear.mx"" ./llama4.sh
```


## Implementation details for developers
This prototype is specifically designed to be used on MoE models using
`torch._grouped_mm` to implement expert computation in token-choice routing,
where expert weights are implemented as 3D nn.Parameters with `num_experts` as
the leading dim.

The `MoETrainingConfig` has a module handler registered to it which will
find all nn.Parameters whose parent module matches the module filter function,
and swap their data tensor with a ScaledGroupedMMTensor.

The ScaledGroupedMMTensor is a tensor subclass which overrides the
`torch._grouped_mm` op by dispatching to a differentiable scaled grouped mm,
which performs dynamic float8 rowwise quantization on scaled grouped GEMM
operands in both the forward and backward pass.

For all other ops, ScaledGroupedMMTensor behaves like a regular torch.Tensor.

## Limitations
- The new CUDA kernel for MXFP8 quantization of the non-transposed expert weights in the backwards pass does not support TP yet.
