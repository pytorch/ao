# Low precision MoE Training

This prototype provides:

1. Quantized building block for low precision MoE training: `_scaled_grouped_mm`. It is a differentiable drop-in replacement for `torch._grouped_mm` that dynamically quantizes inputs using the given recipe, performs a scaled grouped GEMM, then returns the results in original precision. See runnable [example](#torchao_scaled_grouped_mm-example-forward--backward-pass) of a forward and backward pass below.
    - Using MXFP8 on a B200 GPU, this provides:
        - ~1.4x - 1.8x speedups over bfloat16 `torch._grouped_mm` for Llama4 17b 16e shapes (depending on the `M` dimension, i.e. batch_size * seq_len)
        - ~1.15 - 1.3x speedups over bfloat16 `torch._grouped_mm` for DeepSeekV3 671b shapes (depending on the `M` dimension, i.e. batch_size * seq_len)


2. [TorchTitan](https://github.com/pytorch/torchtitan/tree/main) integration of torchao's dynamically quantized `_scaled_grouped_mm`: pretrain DeepSeekV3/Llama4 with MXFP8 grouped GEMMs by adding the flag to your training command: `--model.converters="quantize.grouped_mm.mx" [--quantize.grouped_mm.mx.fqns="experts"]`

3. `quantize_(...)` API support for model conversion: this swaps all `torch._grouped_mm` ops in your model definition to use torchao `_scaled_grouped_mm` under the hood (see [example](#model-conversion-api-example-end-to-end-training) below).


## Table of Contents

- [Examples](#examples)
- [Performance Benchmarks](#performance-benchmarks-mxfp8)
- [System Requirements](#system-requirements)
- [Implementation Details for Developers](#implementation-details-for-developers)
- [Limitations](#limitations)

## Examples
#### torchao_scaled_grouped_mm example: forward + backward pass
```python
import torch
from torch.nn import functional as F
from torchao.prototype.moe_training import (
    _quantize_then_scaled_grouped_mm as torchao_scaled_grouped_mm
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
out = torchao_scaled_grouped_mm(
        A,
        B.transpose(-2, -1),
        offs=offs,
        scaling_type=MoEScalingType.MXFP8,
)

# (Fake labels for demonstration purposes)
labels = torch.ones_like(out)
loss = F.mse_loss(out, labels)
loss.backward()
```

#### Model conversion API example: end-to-end training
```python
import torch
from torch import nn
from torch.nn import functional as F

# this feature requires CUDA 12.8+ and SM100+
assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)

from torchao.prototype.moe_training.conversion_utils import MoETrainingConfig
from torchao.quantization.quant_api import quantize_

# this example uses torchtitan llama4 MoE, see
# this benchmark requires torchtitan
try:
    from torchtitan.distributed.expert_parallel import (
        set_token_group_alignment_size_m,
    )
    from torchtitan.models.moe import MoE, MoEArgs
except ImportError:
    pytest.skip(
        "torchtitan not installed, skipping MoE tests.", allow_module_level=True
    )


# initialize model
device = torch.device("cuda")
moe_args = MoEArgs(
    num_experts=8,
)
dim, hidden_dim = 5120, 8192
model = MoE(moe_args, dim, hidden_dim).to(torch.bfloat16).to(device)
init_std = 0.02
model.init_weights(init_std, device)

# module filter function to define which modules to quantize
target_fqns = ["experts"]


def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
    for target_fqn in target_fqns:
        if target_fqn in cur_fqn:
            return True
    return False

# Token group alignment size must be 32 for MXFP8 training
alignment_size = 32 if recipe == MoEScalingType.MXFP8 else 16
set_token_group_alignment_size_m(alignment_size)

# quantize the model
config = MoETrainingConfig()
quantize_(model, config=config, filter_fn=moe_module_filter_fn)

# training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size, seq_len = 2, 2048
for step in range(10):
    x = torch.randn(
        batch, seq, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )

    # forward pass
    out = model(x)

    # compute loss
    labels = torch.ones_like(out)
    out_loss = F.mse_loss(out, labels)
    print(f"step {step} loss: {out_loss.item()}")

    # backward pass
    out_loss.backward()
    optimizer.step()

```

## System requirements
- torchao 0.14+
- For MXFP8 MoE training, CUDA 12.8+ and SM100+ GPU arch are required.
- For FP8 rowwise MoE training, CUDA 12.4+ and SM89+ GPU arch are required.

## Performance benchmarks: MXFP8


### Single MoE layer forward + backward pass vs bfloat16 baseline

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

### Individual bfloat16 torch._grouped_mm op vs torchao_scaled_grouped_mm

**MXFP8 with Llama4 17b 16e shapes** (with G=1-8 to simulate different degrees of expert parallelism)

| M,N,K,G                 | bf16_fwd_bwd_us | scaled_fwd_bwd_us | scaled_fwd_bwd_speedup |
| ----------------------- | --------------: | ----------------: | ---------------------: |
| (128000, 8192, 5120, 1) |        43140.20 |          23867.00 |                 1.808x |
| (128000, 8192, 5120, 2) |        39487.60 |          23359.00 |                 1.690x |
| (128000, 8192, 5120, 4) |        39189.20 |          23945.50 |                 1.637x |
| (128000, 8192, 5120, 8) |        37700.70 |          22170.60 |                 1.700x |

**MXFP8 with DeepSeekV3** (with G=-8 to simulate different degrees of expert parallelism)

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


#### End-to-end training: Llama4 16e MoE layer vs bfloat16 baseline with TorchTitan
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
