# Low-bit and specialized optimizers

This module implements:

- 8-bit optimizers as outlined in https://arxiv.org/abs/2110.02861
- 4-bit optimizers as outlined in https://arxiv.org/abs/2309.01507
- FP8 optimizers using the native `torch.float8_e4m3fn` dtype (experimental)
- Stochastic rounding for BF16 weight (https://arxiv.org/abs/2010.06192, experimental)
- CPU offload optimizers for single GPU training

The implementation is fully done in Python (with tensor subclass) and relies on `torch.compile()` to generate efficient fused kernel. Thus, your platform must support `torch.compile()` to use these optimizers. We only test on CPU and CUDA, so there might be bugs or errors on other platforms.

## Usage

This is a drop-in replacement for `torch.optim.Adam`

```python
from torchao.optim import Adam8bit

model = ...
optim = Adam8bit(model.parameters())
```

To use 4-bit Adam, replace the above with `Adam4bit`. Similarly for `AdamFp8`. You can also change quantization block size by passing `block_size=value` to the optimizer. By default, block size is 256 for 8-bit and FP8 optimizers, and 128 for 4-bit optimizers.

**Other optimizers**: AdamW is also available as `AdamW8bit`, `AdamW4bit`, and `AdamWFp8`. Other optimizers can be added based on demand.

NOTE:
- The low-bit optimizers require PyTorch >= 2.3
- For FP8 optimizers on CUDA, PyTorch >= 2.4 and CUDA compute capability >= 8.9 are required.
- For 4-bit optimizers, we don't implement rank-1 normalization for quantizing 2nd moment as originally done in the paper.

## Benchmarks

Fine-tune [timm](https://github.com/huggingface/pytorch-image-models)'s [ViT-H](https://huggingface.co/timm/vit_huge_patch14_224.orig_in21k) (630M params) on [resisc45](https://huggingface.co/datasets/timm/resisc45) dataset. PyTorch 2.4, BF16 AMP, compiled model, 1 epoch, batch size 8, cosine LR scheduler, 4070Ti SUPER, fixed random seed. Benchmark script is available at [benchmarks/benchmark_low_bit_adam.py](../../benchmarks/benchmark_low_bit_adam.py).

AdamW impl      | Peak memory allocated (GB) | imgs/s | accuracy
----------------|----------------------------|--------|----------
PyTorch (fused) | 12.23                      | 41.9   | 94.52
bnb 8-bit       |  8.32                      | 43.6   | 94.54
ao 8-bit        |  8.33                      | 42.5   | 94.30
ao FP8 E4M3     |  8.33                      | 43.2   | 94.13
lpmm 4-bit      |  7.72                      | 46.1   | 94.40
ao 4-bit        |  7.72                      | 42.4   | 94.13
lpmm 4-bit (*)  |  7.74                      | 26.7   | 94.10

(*) means rank-1 normalization is used for 2nd optimizer state. Refer to [paper](https://arxiv.org/abs/2309.01507) for more details.

Fine-tune [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b) on [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset. PyTorch 2.4, full BF16, 1 epoch, A100, fixed random seed. Benchmark is done with [torchtune 52d1b838](https://github.com/pytorch/torchtune/tree/52d1b838c1c35b5e75fddf8776be400adc36dff5). See [#812](https://github.com/pytorch/ao/pull/812) for more details.

AdamW impl       | Peak memory allocated (GB) | toks/s | `truthfulqa_mc2` acc
-----------------|----------------------------|--------|----------------------
Not fine-tuned   | -                          | -      | 38.95
PyTorch (fused)  | 51.6                       | 3200   | 42.61
bnb 8-bit        | 39.3                       | 3000   | 42.75
ao 8-bit         | 39.1                       | 2900   | 41.50
ao 4-bit         | 33.2                       | 2900   | 42.27

NOTE: lpmm's 4-bit AdamW does not support BF16 weights.

## Stochastic rounding for BF16 weight

BF16 only has around 3 decimal precision. This means that if weight update is smaller than 1e-3 of the weight magnitude, there will be no change to the weight (using nearest rounding). This is highly problematic for full BF16 training, where we don't keep an FP32 copy of model weights.

Note that our optimizer step calculations are always done in FP32 to ensure accurate results. The "underflow" only happens when we copy the new weight value (in FP32) to the existing BF16 weight. To combat this problem, one way is to perform **stochastic rounding** when casting FP32->BF16.
- In stochastic rounding, we will round up with the probability of `(x - round_down(x)) / (round_up(x) - round_down(x))`, and round down otherwise.
- It follows that successive weight update with stochastic rounding will correctly approximate high-precision weight update.
- Since BF16 is simply a truncation of FP32, there is an efficient implementation for FP32->BF16 stochastic rounding (the same is not true for FP32->FP16).
- More detailed discussion can be found at https://arxiv.org/abs/2010.06192. [llm.c](https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/llmc/adamw.cuh#L43) also implements this approach.

```python
# a clone of torch.optim.AdamW with extra features
from torchao.optim import _AdamW

model = ...
model_bf16 = model.bfloat16()
optim = _AdamW(model_bf16.parameters(), bf16_stochastic_round=True)
```

All of our low-bit optimizers mentioned above also support `bf16_stochastic_round` flag. Note that this flag only applies to BF16 weight.

## Optimizer CPU offload

This folder also implements optimizer CPU offload (i.e. ZeRO-Offload) for single GPU training. Only CUDA and XPU is supported. For multi-GPU training, you can use FSDP's built-in CPU offload.

```python
import torch
from torchao.optim import CPUOffloadOptimizer

model = ...

# only offload optimizer state
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)

# offload optimizer state AND gradients
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, offload_gradients=True, fused=True)
```

This will reduce GPU memory usage by optimizer state size, and additionally gradient size if `offload_gradients=True`. `CPUOffloadOptimizer` can wrap any base optimizer.

For saving and loading `CPUOffloadOptimizer`, it is important that you load model's weights BEFORE creating the optimizer, since we create a CPU copy of the parameters inside `CPUOffloadOptimizer.__init__()`. (TODO: we might want to have a method to synchronize GPU and CPU params in either direction CPU->GPU and GPU->CPU, in case they are out of sync.)

```python
ckpt = torch.load("checkpoint.pth")

model = ...
model.load_state_dict(ckpt["model"])

optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

`CPUOffloadOptimizer` is not compatible with PyTorch's built-in LR scheduler because it only acts as a wrapper around the actual optimizers (and extra logic for moving data around). To adjust the LR, you have to manually update it like follows (in fact you can use the below code for all PyTorch optimizers too):

```python
lr = ... # compute your desired LR value
for param_group in optim.param_groups: 
    if isinstance(param_group["lr"], torch.Tensor): 
        param_group["lr"].fill_(lr) 
    else: 
        param_group["lr"] = lr 
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
