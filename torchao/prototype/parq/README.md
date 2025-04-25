# PARQ: Piecewise-Affine Regularized Quantization

PARQ is a QAT method based on a convex regularization framework. It converges to hard quantization (i.e., STE) at its asymptotic limit.

This library applies QAT without modifying model-level code. It instead interfaces with the optimizer only, allowing a user to choose which parameters should be quantized via parameter groups. It separates QAT into the below components.

* quantization method: computing the best set of discrete, quantized values
* proximal mapping: projection of weights onto quantized values


## PARQ vs. torchao

There are two main QAT interfaces in torchao:

- Swap modules (e.g., `torch.nn.Linear`) with their quantized counterparts (e.g., `Int4WeightOnlyQATLinear`). See [Quantizer API (legacy)](../../quantization/qat#quantizer-api-legacy) for details.
- Replace instances of `torch.Tensor` with a quantized tensor subclass such as `AffineQuantizedTensor`. The [`quantize_` API](../../quantization/qat#quantize_-api-recommended) uses this method by default.

PARQ is conceptually more similar to the tensor subclass interface. It quantizes tensors through the optimizer's parameter groups, leaving the model untouched.

An example PARQ flow and its torchao equivalent are shown below. The prepare stage occurs before training, while the convert stage runs after training to produce a quantized model.

<table>
<tr>
<td align="center"><b>stage</b><td align="center"><b>PARQ</b></td><td align="center"><b>torchao</b></td>
</tr>
<tr>
<td>prepare</td>
<td valign="top">

```python
from torchao.prototype.parq.optim import QuantOptimizer
from torchao.prototype.parq.quant import UnifTorchaoQuantizer

param_groups = [
    {"params": params_quant, "quant_bits": 4, "quant_block_size": 32},
    {"params": params_no_quant},
]
base_optimizer = torch.optim.AdamW(param_groups, ...)
optimizer = QuantOptimizer(
    base_optimizer,
    UnifTorchaoQuantizer(symmetric=True),
    ProxHardQuant(),
    quant_per_channel=True,
)
```

</td>
<td valign="top">

```python
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    intx_quantization_aware_training,
)

weight_config = FakeQuantizeConfig(torch.int4, group_size=32)
quantize_(
    model,
    intx_quantization_aware_training(weight_config=weight_config),
)
```

</td>
</tr>
<tr>
<td>convert</td>
<td valign="top">

```python
config = IntXWeightOnlyConfig(
    weight_dtype=torch.int4, granularity=PerGroup(32)
)
optimizer.torchao_quantize_(model, config)
```

</td>
<td valign="top">

```python
from torchao.quantization.qat import from_intx_quantization_aware_training

quantize_(model, from_intx_quantization_aware_training())
```

</td>
</tr>
</table>

Note that `UnifTorchaoQuantizer` calls the same quantization primitives as torchao to match the numerics (see [Affine Quantization Details](../../quantization#affine-quantization-details)).

## QAT arguments

| | description | choices |
| --- | --- | --- |
| `quant-bits` | bit-width for quantized weights | 0 (ternary), 1-4 |
| `quant-method` | method for determining quantized values | `lsbq`, `uniform` |
| `quant-proxmap` | proximal mapping to project weights onto quantized values | `hard`, `parq`, `binaryrelax` |
| `anneal-start` | start epoch for QAT annealing period | (0, `total_steps` - 1) |
| `anneal-end` | end epoch for QAT annealing period | (`anneal_end`, `total_steps`) |
| `anneal-steepness` | sigmoid steepness for PARQ inverse slope schedule | 1-20 |

## Optimizer-only interface

The `QuantOptimizer` wrapper takes any `torch.optim.Optimizer` object. It is also initialized with a `Quantizer` and `ProxMap` object. Integration into new training pipelines is simple:
```python
from torchao.prototype.parq.optim import ProxPARQ, QuantOptimizer
from torchao.prototype.parq.quant import LSBQuantizer


# split params into quantizable and non-quantizable params
params_quant, params_no_wd, params_wd = split_param_groups(model)  # user-defined
param_groups = [
    {"params": params_quant, "quant_bits": 2},
    {"params": params_no_wd, "weight_decay": 0},
    {"params": params_wd},
]

# create PyTorch optimizer
base_optimizer = torch.optim.SGD(  # user-defined
    param_groups, lr=0.1, momentum=0.9, weight_decay=1e-4
)

# create quantizer and proximal map objects
quantizer = LSBQuantizer()
prox_map = ProxPARQ(anneal_start=..., anneal_end=..., steepness=...)

optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map)
```
