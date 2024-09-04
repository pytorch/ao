# Low-bit optimizers

This folder implements:

- 8-bit optimizers as outlined in https://arxiv.org/abs/2110.02861
- 4-bit optimizers as outlined in https://arxiv.org/abs/2309.01507
- FP8 optimizers using the native `torch.float8_e4m3fn` dtype (experimental)

The implementation is fully done in Python (with tensor subclass) and relies on `torch.compile()` to generate efficient fused kernel.

## Usage

This is a drop-in replacement for `torch.optim.Adam`

```python
from torchao.prototype.low_bit_optim import Adam8bit

model = ...
optim = Adam8bit(model.parameters())
```

To use 4-bit Adam, replace the above with `Adam4bit`. Similarly for `AdamFp8`. You can also change quantization block size by passing `block_size=value` to the optimizer. By default, block size is 2048 for 8-bit and FP8 optimizers, and 128 for 4-bit optimizers.

**Other optimizers**: AdamW is also available as `AdamW8bit`, `AdamW4bit`, and `AdamWFp8`. Other optimizers can be added based on demand.

NOTE:
- The low-bit optimizers require PyTorch >= 2.3
- For FP8 optimizers on CUDA, PyTorch >= 2.4 and CUDA compute capability >= 8.9 are required.
- For 4-bit optimizers, we don't implement rank-1 normalization for quantizing 2nd moment as originally done in the paper.
- The first training step is expected to be slow since the optimizer needs to be compiled.

## Benchmarks

Fine-tune [timm](https://github.com/huggingface/pytorch-image-models)'s ViT-H (630M params) on [resisc45](https://huggingface.co/datasets/timm/resisc45) dataset. BF16 AMP, 1 epoch, batch size 8, cosine LR scheduler, 4070Ti SUPER, fixed random seed. Benchmark script is available at [benchmarks/benchmark_low_bit_adam.py](../../../benchmarks/benchmark_low_bit_adam.py).

AdamW impl      | Max memory (GB) | imgs/s | accuracy
----------------|-----------------|--------|----------
PyTorch (fused) | 12.23           | 41.8   | 94.38
bnb 8-bit       |  8.32           | 43.6   | 94.18
ao 8-bit        |  8.33           | 42.6   | 94.25
ao FP8 E4M3     |  9.27           | 44.1   | 94.40
lpmm 4-bit      |  7.72           | 46.0   | 94.29
ao 4-bit        |  7.72           | 40.0   | 94.03
lpmm 4-bit (*)  |  7.74           | 26.6   | 94.25

(*) means rank-1 normalization is used for 2nd optimizer state. Refer to [paper](https://arxiv.org/abs/2309.01507) for more details.

Fine-tune [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b) on [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset. Full BF16, 1 epoch, A100, fixed random seed. Benchmark is done with [torchtune](https://github.com/pytorch/torchtune). See [#746](https://github.com/pytorch/ao/pull/746) for more details.

AdamW impl       | Max memory (GB) | toks/s | `truthfulqa_mc2` acc | Compile time
-----------------|-----------------|--------|----------------------|-------------
Not fine-tuned   | -               | -      | 38.95                | -
PyTorch (fused)  | 52              | ~4500  | 42.12                | ~4 min
bnb 8-bit        | 39              | ~4000  | 41.98                | ~4 min
ao 8-bit         | 39              | ~4000  | 42.41                | ~12 min
ao 4-bit         | 33              | ~3600  | 42.34                | ~4 min

NOTE: lpmm's 4-bit AdamW does not support BF16 weights.

### Note on compile times

There are 2 approaches to compile optimizer step in low-bit optim:

1. Compile optim step for single param i.e. `torch.compile(single_param_adam)`
2. Compile optim step for all params i.e. `torch.compile(param_groups_adam)`

Currently Adam8bit and AdamFp8 use approach (2) (with static shape) since it is faster (but compile much slower), while Adam4bit uses approach (1) (with dynamic shape) since there are excessive memory usage for "Adam4bit + approach (2)". Approach (1) requires dynamic shape to avoid hitting recompiles limit.

## Optimizer CPU offload

This folder also implements optimizer CPU offload (i.e. ZeRO-Offload) for single GPU training. For multi-GPU training, you can use FSDP's built-in CPU offload.

```python
import torch
from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

model = ...

# only offload optimizer state
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)

# offload optimizer state AND gradients
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, offload_gradients=True, fused=True)
```

This will reduce GPU memory usage by optimizer state size, and additionally gradient size if `offload_gradients=True`. `CPUOffloadOptimizer` can wrap any base optimizer.

For saving and loading `CPUOffloadOptimizer`, it is important that you load model's weights BEFORE creating the optimizer, since we create a CPU copy of the parameters inside `CPUOffloadOptimizer.__init__()`. (TODO: we might want to have a method to synchronize CUDA and CPU params in either direction CPU->CUDA and CUDA->CPU, in case they are out of sync.)

```python
ckpt = torch.load("checkpoint.pth")

model = ...
model.load_state_dict(ckpt["model"])

optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

NOTE:
- Since the optimizer step is done on CPU, it is highly recommended to use a fast CPU optimizer, such as `torch.optim.AdamW(fused=True)` (requires PyTorch 2.4). For other optimizers, you can try `torch.compile()` their optimizer step.
- To minimize the amount of CPU<->GPU data transfer, we keep a copy of parameters and pre-allocate gradients memory on CPU. Therefore, expect your RAM usage to increase by 2x model size + optimizer state (which is 2x model size for Adam).
- It is recommended NOT to `torch.compile()` your whole model when `CPUOffloadOptimizer` is used, as it prevents us from interleaving gradient device-to-host transfer with backward pass. To minimize such impact, you can compile parts of your model separately. See [#584](https://github.com/pytorch/ao/pull/584) for more information.
- CPU optimizer step is often the bottleneck when optimizer CPU offload is used. To minimize the slowdown, it is recommended to (1) do full BF16 training (instead of AMP), so that parameters, gradients, and optimizer states are in BF16; and (2) give GPU more work per optimizer step (e.g. larger batch size with activation checkpointing, gradient accumulation).
- `offload_gradients=True` is not compatible with gradient accumulation, since we clear gradients on GPU every backward pass.
- Gradient clipping is currently not supported.

Benchmark done for `timm/vit_giant_patch14_dinov2.lvd142m` (1.1B params), eager mode, full BF16 training, activations checkpointing, batch size 32, on 4070Ti SUPER (16GB VRAM), Ryzen 5600, DDR4 RAM. DeepSpeed is untuned.

Adam offload           | Time per step | Max memory
-----------------------|---------------|------------
None                   | 1.27s/it      | 9.82 GB
DeepSpeed ZeRO-Offload | 3.13s/it      | 6.85 GB
ao                     | 1.52s/it      | 5.24 GB
ao (offload gradients) | 1.53s/it      | 4.01 GB

Ablations on AMP and `torch.compile()`

Training config     | Adam offload | Time per step | Max memory
--------------------|--------------|---------------|------------
Full BF16, compiled | None         | 1.18s/it      | 9.90 GB
Full BF16, compiled | ao           | 1.75s/it      | 5.33 GB
BF16 AMP, eager     | None         | OOM           | OOM
BF16 AMP, eager     | ao           | 2.18s/it      | 9.90 GB

## Credits

Credits to

- Tim Dettmers for creating the wonderful [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library.
- [lpmm](https://github.com/thu-ml/low-bit-optimizers) authors for their work on 4-bit optimizers.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) team for [ZeRO-Offload](https://arxiv.org/abs/2101.06840).