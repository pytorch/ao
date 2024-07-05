# Low-bit optimizers

This folder implements:

- 8-bit optimizers as outlined in https://arxiv.org/abs/2110.02861
- 4-bit optimizers as outlined in https://arxiv.org/abs/2309.01507

The implementation is fully done in Python (with tensor subclass) and relies on `torch.compile()` to generate efficient fused kernel.

## Usage

This is a drop-in replacement for `torch.optim.Adam`

```python
from torchao.prototype.low_bit_optim import Adam8bit

model = ...
optim = Adam8bit(model.parameters())
```

To use 4-bit Adam, replace the above with `Adam4bit`. You can also change quantization block size by passing `block_size=value` to the optimizer. By default, block size is 2048 for 8-bit optimizers, and 128 for 4-bit optimizers.

**Other optimizers**: AdamW is also available as `AdamW8bit` and `AdamW4bit`. Other optimizers can be added based on demand.

NOTE:
- The low-bit optimizers require PyTorch >= 2.3
- For 4-bit optimizers, we don't implement rank-1 normalization for quantizing 2nd moment as originally done in the paper.
- **Known issue**: When learning rate is updated every step (e.g. using cosine learning rate scheduler), training speed is slower. This is because we have to convert learning rate to a CUDA tensor (which incurs expensive memory transfer cost), since torch.compile() will treat a Python float as a constant and trigger recompile whenever the value is changed.

## Benchmarks

Benchmark script for fine-tuning a [timm](https://github.com/huggingface/pytorch-image-models) model on [resisc45](https://huggingface.co/datasets/timm/resisc45) dataset is available at [benchmarks/benchmark_low_bit_adam.py](../../../benchmarks/benchmark_low_bit_adam.py).

Results for fine-tuning ViT-H (630M params) with BF16 AMP, batch size 4, 1 epoch, on 4070Ti SUPER:

Adam impl  | max memory (GB) | time taken | accuracy
-----------|-----------------|------------|----------
PyTorch    | 12.98           | 10m 08s    | 87.70
bnb 8-bit  |  8.31           |  8m 38s    | 86.22
ao 8-bit   |  8.32           | 10m 54s    | 86.67
lpmm 4-bit |  7.72           |  7m 48s    | 84.70
ao 4-bit   |  7.72           |  9m 17s    | 85.60

NOTE: time taken includes validation time, and compile time for torchao optimizers.

## Credits

Credits to Tim Dettmers for creating the wonderful [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library, and [lpmm](https://github.com/thu-ml/low-bit-optimizers) authors for their work on 4-bit optimizers.
