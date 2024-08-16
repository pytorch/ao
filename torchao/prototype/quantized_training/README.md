# Quantized training

This folder contains experimental work on quantized training (QT). The main difference from quantization-aware training (QAT) is that in QT, we don't keep a high-precision copy of model weights. We take inspirations from:
- Q-GaLore: [[paper](https://arxiv.org/abs/2407.08296)] [[code](https://github.com/VITA-Group/Q-GaLore)]
- AQT: [[related paper](https://arxiv.org/abs/2105.03536)] [[code](https://github.com/google/aqt)]

Typically, low-precision weights cannot be trained directly due to quantization error: a small change in the quantized weight will be round down to zero. To tackle this problem, we use **stochastic rounding** for weight update. In simple terms, stochastic rounding will round up or down randomly, but with a higher chance if it is closer to that direction. For example, 0.8 will have 80% chance of rounding up and 20% of rounding down. It also follows that on average, stochastic rounding will estimate the floating point value exactly.

In precise terms, the probability of rounding up is `x - ⌊x⌋`. Note that when the value is exactly an integer value, the probability of rounding up is zero.

There are 2 main benefits for training in this way:
1. Reduce memory footprint. Also reduce communication bandwidth in distributed setting.
2. What you train is what you serve ([WYTIWYS](https://github.com/google/aqt?tab=readme-ov-file#features)).

Currently we only support weight-only channel-wise INT8 symmetric quantization.

## INT8 weight only

In this recipe, all linear weights are quantized to INT8 using channel-wise symmetric quantization `[-127, 127]`. In the forward and backward pass, the weights are upcast to activations' dtype (e.g. BF16). Therefore, their gradients are also in activations' dtype.

Usage

```python
from torchao.prototype.quantized_training import int8_weight_only_quantized_training
from torchao.prototype.low_bit_optim import _AdamW
from torchao.quantization.quant_api import quantize_

model = ...
quantize_(model, int8_weight_only_quantized_training())

optim = _AdamW(model.parameters(), lr=3e-4)
```

Only `torch.optim.Adam` and optimizers from `torchao.prototype.low_bit_optim` are known to work with quantized training in this folder. This is because we implement stochastic rounding logic within tensor subclass instead of the optimizer. We provide `torchao.prototype.low_bit_optim._AdamW` as an alternative to `torch.optim.AdamW` specifically for this purpose.

[`benchmarks/quantized_training/pretrain_llama2.py`](../../../benchmarks/quantized_training/pretrain_llama2.py) demonstrates an end-to-end Llama2 pre-training using this INT8 quantized training.

See [#644](https://github.com/pytorch/ao/pull/644) for some early results.

TODO: investigate suboptimal memory saving when `torch.compile()` is used. Might be due to transposed weight. Memory benchamark for Llama2-1B, bs=4, seq_len=2048, activation checkpointing. 

Model           | Peak memory (GB)
----------------|-----------------
BF16 eager      | 11.06847
BF16 compile    | 10.16915
INT8 QT eager   | 10.11437
INT8 QT compile | 10.03365

## Future ideas

- INT8 activation x INT8 weight. This can potentially leverage INT8 Tensor Cores, which is 2x faster than FP16/BF16 Tensor Cores.
- INT4 weight only (with group-wise quantization). This can be used with INT4 tinygemm deployment in mind (or other optimized INT4 kernels).
- FP8 activation x FP8 weight. The current FP8 training recipe can be seen as a form of QAT, which maintains a high-precision copy of model weights. We can eliminate the high-precision copy.
