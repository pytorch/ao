# PARQ: Piecewise-Affine Regularized Quantization

PARQ is a QAT method based on a convex regularization framework. It converges to hard quantization (i.e., STE) at its asymptotic limit.

This library applies QAT without modifying model-level code. It instead interfaces with the optimizer only, allowing a user to choose which parameters should be quantized via parameter groups. It separates QAT into the below components.

* quantization method: computing the best set of discrete, quantized values
* proximal mapping: projection of weights onto quantized values

## QAT arguments

| | description | choices |
| --- | --- | --- |
| `quant-bits` | bit-width for quantized weights | 0 (ternary), 1—4 |
| `quant-method` | method for determining quantized values | `lsbq`, `uniform` |
| `quant-proxmap` | proximal mapping to project weights onto quantized values | `hard`, `parq`, `binaryrelax` |
| `anneal-start` | start epoch for QAT annealing period | (0, `total_steps` - 1) |
| `anneal-end` | end epoch for QAT annealing period | (`anneal_end`, `total_steps`) |
| `anneal-steepness` | sigmoid steepness for PARQ inverse slope schedule | 25—100 |

## Optimizer-only interface

The `QuantOptimizer` wrapper takes any `torch.optim.Optimizer` object. It is also initialized with a `Quantizer` and `ProxMap` object. Integration into new training pipelines is simple:
```python
from parq.optim import ProxPARQ, QuantOptimizer
from parq.quant import LSBQuantizer


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
