# Quantized training

This folder contains experimental work on quantized training (QT). The main difference from quantization-aware training (QAT) is that in QT, we don't keep a high-precision copy of model weights. We take inspirations from:
- Q-GaLore: [[paper](https://arxiv.org/abs/2407.08296)] [[code](https://github.com/VITA-Group/Q-GaLore)]
- AQT: [[related paper](https://arxiv.org/abs/2105.03536)] [[code](https://github.com/google/aqt)]

Currently we only support weight-only channel-wise INT8 symmetric quantization.

## INT8 weight only

In this recipe, all linear weights are quantized to INT8 using channel-wise symmetric quantization scheme `[-127, 127]`. During forward and backward, the weights are upcast to activations' dtype (e.g. BF16). Therefore, gradients are also in activations' dtype. In the optimizer step, we use **stochastic rounding** to update the weights, ensuring small weight updates can still change the weights.

Usage

```python
from torchao.prototype.quantized_training import int8_weight_only_quantized_training
from torchao.prototype.low_bit_optim import AdamW
from torchao.quantization.quant_api import quantize_

model = ...
quantize_(model, int8_weight_only_quantized_training())

optim = AdamW(model.parameters(), lr=3e-4)
```

It is recommended to use optimizers from `torchao.prototype.low_bit_optim` for quantized training, because they can automatically generate efficient fused optimizer kernel for `dequant->optimizer_step->quant` thanks to `torch.compile()`.

[`benchmarks/benchmark_int8_qt.py`](../../../benchmarks/benchmark_int8_qt.py) demonstrates an end-to-end Llama2 pre-training using this INT8 quantized training.

See [#644](https://github.com/pytorch/ao/pull/644) for some early results.
