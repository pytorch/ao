# 8-bit optimizers

This folder implements 8-bit optimizers using dynamic tree quantization as outlined in https://arxiv.org/abs/2110.02861. The implementation is fully done in Python (with tensor subclass) and relies on `torch.compile()` to generate efficient fused kernel.

## Usage

This is a drop-in replacement for `torch.optim.Adam`

```python
from torchao.prototype.optim_8bit import AdamDTQ8bit

model = ...
optim = AdamDTQ8bit(model.parameters())
```

You can also change quantization block size (default 2048) by passing `block_size=value` to the optimizer.

**Other optimizers**: AdamW is also available as `AdamWDTQ8bit`.

NOTE: this requires PyTorch >= 2.3

## Benchmarks

Benchmark script for fine-tuning a [timm](https://github.com/huggingface/pytorch-image-models) model on [resisc45](https://huggingface.co/datasets/timm/resisc45) dataset is available at [benchmarks/benchmark_adam_8bit.py](../../../benchmarks/benchmark_adam_8bit.py).

Results for fine-tuning ViT-B with BF16 AMP, on 4070Ti SUPER:

Adam impl | max memory (GB) | training time | accuracy
----------|-----------------|---------------|----------
PyTorch   | 5.26            | 9m 11s        | 93.62%
bnb 8-bit | 4.78            | 9m 10s        | 93.06%
ao 8-bit  | 4.78            | 9m 15s        | 94.14%

**Known issue**: When learning rate is updated every step (e.g. using cosine learning rate scheduler), training speed is slower. This is because we have to convert learning rate to a CUDA tensor (which incurs expensive memory transfer cost), since torch.compile() will treat a Python float as a constant and trigger recompile whenever the value is changed

## Credits

Credits to Tim Dettmers for creating the wonderful bitsandbytes library.
