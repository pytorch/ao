# Quantized training

This folder contains experimental work on quantized training (QT), with a focus on INT8. We take inspirations from:

- AQT: [[related paper](https://arxiv.org/abs/2105.03536)] [[code](https://github.com/google/aqt)]
- SwitchBack: [[paper](https://arxiv.org/abs/2304.13013)]
- Q-GaLore: [[paper](https://arxiv.org/abs/2407.08296)] [[code](https://github.com/VITA-Group/Q-GaLore)]
- JetFire: [[paper](https://arxiv.org/abs/2403.12422)] [[code](https://github.com/thu-ml/Jetfire-INT8Training)]

The main difference from quantization-aware training (QAT) is that in QT, we don't keep a high-precision copy of model weights. However, terminologies for INT8 training are generally not standardized yet. To be precise, we use these terms with the following meaning:

- **Quantized training**: model weights are quantized. This is a strict requirement. Does not matter what is the compute precision. Examples of this: Q-GaLore, JetFire.
- **INT8 mixed-precision training**: model weights are in original precision, while compute dtype for some or all ops is in INT8. We call it like this because it is similar to FP16/BF16 mixed-precision training. One difference is that in FP16/BF16 mixed-precision training, matmul will return FP16/BF16 outputs, while for INT8 mixed-precision training, the returned dtype is usually not INT8. Examples include Google AQT and SwitchBack.

There are 3 main benefits of using low-precision dtype for training (the extent depends on the actual strategies):

- **Memory**: reduce memory footprint by model weights, activations, gradients, and distributed communication bandwidth.
- **Speed**: speedup compute-bound ops with low-precision hardware instructions (e.g. INT8 Tensor Cores) and speedup memory-bound ops with quantized inputs/outputs.
- [What you train is what you serve](https://github.com/google/aqt?tab=readme-ov-file#features).

[`benchmarks/quantized_training/pretrain_llama2.py`](../../../benchmarks/quantized_training/pretrain_llama2.py) demonstrates an end-to-end Llama2 pre-training on single GPU for strategies implemented in this folder.

## INT8 quantized training

Typically, quantized weights cannot be trained directly due to quantization error: a small change in the quantized weight will be round down to zero. To tackle this problem, we use **stochastic rounding** for weight update. In simple terms, stochastic rounding will round up or down randomly, but with a higher chance if it is closer to that direction. For example, 0.8 will have 80% chance of rounding up and 20% of rounding down. It also follows that on average, stochastic rounding will estimate the floating point value exactly.

In precise terms, the probability of rounding up is `x - ⌊x⌋`. Note that when the value is exactly an integer value, the probability of rounding up is zero.

Currently we only support weight-only channel-wise INT8 symmetric quantization.

In this recipe, all linear weights are quantized to INT8 using channel-wise symmetric quantization `[-127, 127]`. In the forward and backward pass, the weights are upcast to activations' dtype (e.g. BF16). Therefore, their gradients are also in activations' dtype.

Usage

```python
from torchao.prototype.quantized_training import int8_weight_only_quantized_training
from torchao.prototype.low_bit_optim import _AdamW
from torchao.quantization import quantize_

model = ...
quantize_(model, int8_weight_only_quantized_training())

optim = _AdamW(model.parameters(), lr=3e-4)
```

Only `torch.optim.Adam` and optimizers from `torchao.prototype.low_bit_optim` are known to work with quantized training in this folder. This is because we implement stochastic rounding logic within tensor subclass instead of the optimizer. We provide `torchao.prototype.low_bit_optim._AdamW` as an alternative to `torch.optim.AdamW` specifically for this purpose.

See [#644](https://github.com/pytorch/ao/pull/644) for some early results.

TODO: investigate suboptimal memory saving when `torch.compile()` is used. Might be due to transposed weight. Benchamark for Llama2-1B, bs=4, seq_len=2048, activation checkpointing, 4070Ti SUPER.

Model           | Peak memory (GB) | toks/s
----------------|------------------|-------
BF16 eager      | 11.07            | 6200
BF16 compile    | 10.25            | 9000
INT8 QT eager   | 10.12            | 5600
INT8 QT compile |  9.84            | 8700

## INT8 mixed-precision

On NVIDIA GPUs, INT8 Tensor Cores is approximately 2x faster than their BF16/FP16 counterparts. In mixed-precision training, we can down-cast activations and weights dynamically to INT8 to leverage faster matmuls. However, since INT8 has very limited range [-128,127], we perform row-wise quantization, similar to how INT8 post-training quantization (PTQ) is done. Weight is still in original precision.

### Basic usage

```python
from torchao.prototype.quantized_training import int8_mixed_precision_training, Int8MixedPrecisionTrainingConfig
from torchao.quantization import quantize_

model = ...

# apply INT8 matmul to all 3 matmuls
quantize_(model, int8_mixed_precision_training())

# customize which matmul is left in original precision.
config = Int8MixedPrecisionTrainingConfig(
    output=True,
    grad_input=True,
    grad_weight=False,
)
quantize_(model, int8_mixed_precision_training(config))

# train model as usual
```

During training, there are 3 matmuls involved in each `nn.Linear` layer:
- 1 in forward: `output = input @ weight.T`
- 2 in backward:
  - `grad_input = grad_output @ weight`
  - `grad_weight = grad_output.T @ input`

You can configure which matmul to be applied with INT8 mixed-precision (shown above). If convergence is an issue, we recommend leaving `grad_weight` in original matmul precision, and also `grad_input` if the issue still persists.

Note:
- When we only apply INT8 mixed-precision in the forward pass, this can be considered QAT for INT8 dynamic activations + INT8 weight quantization (A8W8).
- When we only apply INT8 mixed-precision to `output` and `grad_input`, this is similar to SwitchBack. However, SwitchBack uses tensor-wise scaling for weight. For simplicity, we only support row-wise scaling.
- Apply stochastic rounding to INT8 quantization may improve matmul accuracy. However, from our testing, this seems to be unnecessary, thus we don't implement it at the moment.

Pre-train Llama2-1B on C4 realnewslike subset. bs=32, seq_len=2048 -> 65k tok/batch. Train for 20k steps (1.3B tokens). Using 4090. INT8 mixed precision is not applied to LM head.

Config               | Tok/s | Peak mem (GB) | Val loss
---------------------|-------|---------------|---------
BF16 (baseline)      | ~17k  | 19.47         | 2.97
INT8 mixed-precision | ~29k  | 19.47         | 2.90

See [#748](https://github.com/pytorch/ao/pull/748) for more results.

### FSDP support

Out of the box, this INT8 mixed-precision training is not compatible with FSDP2 `MixedPrecisionPolicy(param_dtype=param_dtype)`, where `param_dtype` != model dtype. As a workaround, you will need to manually specify the FSDP2's `param_dtype` in `Int8MixedPrecisionTrainingConfig`

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torchao.prototype.quantized_training import int8_mixed_precision_training, Int8MixedPrecisionTrainingConfig
from torchao.quantization import quantize_

model = ...  # FP32 model

# setup configs
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
int8mp_config = Int8MixedPrecisionTrainingConfig(fsdp_param_dtype=mp_policy.param_dtype)

# exclude LM head
quantize_(model.layers, int8_mixed_precision_training(int8mp_config))

# shard the model w/ FSDP2
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)

# train model as usual
```

## Future ideas

- Tile-wise INT8 quantization to keep quantized weight for both forward and backward pass (similar to JetFire).
- INT4 weight only (with group-wise quantization). This can be used with INT4 tinygemm deployment in mind (or other optimized INT4 kernels).
- FP8 activation x FP8 weight. The current FP8 training recipe can be seen as a form of QAT, which maintains a high-precision copy of model weights. We can eliminate the high-precision copy.
