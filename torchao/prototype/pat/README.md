# PAT: Pruning-Aware Training

PAT is a library based on group Lasso regularization. It directly induces structured sparsity during training, removing the need for custom pruning metrics and/or multiple rounds of training.

PAT's simple optimizer-only interface supports easy integration into existing training pipelines. The code is organized into two main components:
* grouper: defines the granularity of pruning (e.g., filter, channel, layer)
* proximal mapping: projects groups of weights onto sparse values

## Optimizer-only interface

This package provides a `PruneOptimizer` that simply wraps around a base optimizer inheriting from `torch.optim.Optimizer`. The following code snippet illustrates how to set up PAT:

```python
from pat.optim import PruneOptimizer

model = torchvision.models.resnet18().cuda()

# split params into prunable and non-prunable groups
weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
others = [p for name, p in model.named_parameters() if not name.endswith("weight")]

# apply row-wise group Lasso regularization to the weights
param_groups = [
    {
        "params": weights",
        "group_type": "Dim0Grouper",
        "prox_type": "ProxGroupLasso",
    },
    {"params": others},
]

# create base optimizer (SGD, Adam or AdamW)
base_optimizer = torch.optim.SGD(
    param_groups, lr=0.1, momentum=0.9, weight_decay=1e-4
)

# create PruneOptimizer
optimizer = PruneOptimizer(base_optimizer, warmup_steps=10, reg_lambda=2e-4)
```

After creating `PruneOptimizer`, one can use it as a regular PyTorch optimizer.

## Grouper and proximal mapping combinations

PAT supports various combinations of groupers and proximal mappings. The following table summarizes the available options:
| grouper | proximal mapping | description |
|---|---|---|
| `AttentionHeadGrouperDim{0,1}` | `ProxGroupLasso` | Structured pruning of attention heads. |
| `ConvFilterGrouper` | `ProxGroupLasso` | Structured pruning of convolutional filters. |
| `Dim0Grouper` | `ProxGroupLasso` | Row-wise group Lasso regularization. |
| `Dim1Grouper` | `ProxGroupLasso` | Column-wise group Lasso regularization. |
| `ElemGrouper` | ProxLasso | Unstructured pruning that induces elementwise sparsity. |
| `LayerGrouper` | `ProxGroupLasso` | Structured pruning that induces layer-wise sparsity. |
| `SVDGrouper` | `ProxNuclearNorm` | Induces low-rank structure in weight matrices with an SVD-based proximal mapping. |

## Pruning configuration

Pruning configs are dictionaries that define which parameter groups to prune and how to prune them. Each key-value pair in the config maps to a prunable parameter group of `PruneOptimizer`. The keys are used to match model parameters, while the values specify the pruning granularity and proximal map. The key can be one of the following types:

- parameter name (string): e.g., `blocks.0.attn.qkv.weight`
- regex pattern (string): e.g., `:.*attn\.qkv\.weight`
- module type and parameter name suffix ((class, string) tuple): e.g., `(torch.nn.Linear, 'weight')`

## Unstructured pruning on 1.3B OLMo models

The goal of the following experiments was to compare PAT pruned LLMs with "dense" models of equivalent nonzero parameter count. For example, a 1.3B model pruned to ~58% sparsity would be compared to a 760M model trained from scratch on the same token budget. The dense models have better inference efficiency than models with unstructured sparsity, but this is a good sanity check for PAT.

We borrowed the setup from AllenAI's OLMo models. The table below is Table 1 of [this paper](https://arxiv.org/abs/2412.04403) from AllenAI.
<img src="https://github.com/user-attachments/assets/c66b8f3b-702d-478b-9a94-9718ea0b0583" style="width:80%" />

The two plots show that the PAT pruned 1.3B models (blue curve) reach much better training loss and mean test accuracy on 8 reasoning benchmarks (ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, OpenBookQA, PIQA, Social IQa, WinoGrande) across different sparsity levels.
![](https://github.com/user-attachments/assets/b04347bd-6f16-44ca-85b9-8591349a9b31)
![](https://github.com/user-attachments/assets/91820a7f-519b-4415-ba68-f510df1e18e9)

