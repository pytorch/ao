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
- The low-bit optimizers require PyTorch >= 2.3. FP8 optimizers require CUDA compute capability >= 8.9.
- For 4-bit optimizers, we don't implement rank-1 normalization for quantizing 2nd moment as originally done in the paper.
- **Known issue**: When learning rate is updated every step (e.g. using cosine learning rate scheduler), training speed is slower. This is because we have to convert learning rate to a CUDA tensor (which incurs expensive memory transfer cost), since torch.compile() will treat a Python float as a constant and trigger recompile whenever the value is changed.

## Benchmarks

Benchmark script for fine-tuning a [timm](https://github.com/huggingface/pytorch-image-models) model on [resisc45](https://huggingface.co/datasets/timm/resisc45) dataset is available at [benchmarks/benchmark_low_bit_adam.py](../../../benchmarks/benchmark_low_bit_adam.py).

Results for fine-tuning ViT-H (630M params) with BF16 AMP for 2 epochs, batch size 8, on 4070Ti SUPER, with fixed random seed:

Adam impl      | max memory (GB) | time taken for 2nd epoch | accuracy
---------------|-----------------|--------------------------|----------
PyTorch        | 12.94           |  8m 18s                  | 91.14
bnb 8-bit      |  8.31           |  6m 50s                  | 90.67
ao 8-bit       |  8.31           |  6m 44s                  | 90.63
ao FP8 E4M3    |  8.32           |  6m 35s                  | 90.98
lpmm 4-bit     |  7.72           |  5m 59s                  | 89.97
ao 4-bit       |  7.72           |  7m 13s                  | 90.05
lpmm 4-bit (*) |  7.73           | 11m 10s                  | 89.71

(*) means rank-1 normalization is used for 2nd optimizer state. Refer to [paper](https://arxiv.org/abs/2309.01507) for more details.

## Optimizer CPU offload

This folder also implements optimizer CPU offload (i.e. ZeRO-Offload) for single GPU training. For multi-GPU training, you can use FSDP's built-in CPU offload.

```python
import torch
from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

model = ...
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
```

This will reduce GPU memory usage by the size of optimizer state. `CPUOffloadOptimizer` can wrap any base optimizer.

NOTE:
- Since the optimizer step is done on CPU, it is highly recommended to use a fast CPU optimizer, such as `torch.optim.AdamW(fused=True)`. For other optimizers, you can try `torch.compile()` their optimizer step.
- To minimize the amount of CPU<->GPU data transfer, we keep a copy of parameters and pre-allocate gradients memory on CPU. Therefore, expect your RAM usage to increase by 2x model size + optimizer state (which is 2x model size for Adam).
- It is recommended not to `torch.compile()` your whole model when `CPUOffloadOptimizer` is used, as it prevents us from interleaving gradient device-to-host transfer with backward pass. To minimize such impact, you can compile parts of your model separately.
- CPU optimizer step is often the bottleneck when optimizer CPU offload is used. To minimize the slowdown, it is recommended to (1) do full BF16 training (instead of AMP), so that parameters, gradients, and optimizer states are in BF16; and (2) give GPU more work per optimizer step (e.g. larger batch size with activation checkpointing, gradient accumulation).

Benchmark done for `timm/vit_giant_patch14_dinov2.lvd142m` (1.1B params), eager mode, full BF16 training, on 4070Ti SUPER, Ryzen 5600, DDR4 RAM.

Adam offload           | Speed (bs=8) | Max memory (bs=8) | Speed (bs=16) | Max memory (bs=16)
-----------------------|--------------|-------------------|---------------|-------------------
None                   | 3.3 it/s     | 12.15 GB          | OOM           | OOM
DeepSpeed ZeRO-Offload | 0.5 it/s     |  9.61 GB          | 0.4 it/s      | 14.89 GB
ao                     | 1.1 it/s     |  7.59 GB          | 0.9 it/s      | 12.86 GB

## Credits

Credits to Tim Dettmers for creating the wonderful [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library, and [lpmm](https://github.com/thu-ml/low-bit-optimizers) authors for their work on 4-bit optimizers.
