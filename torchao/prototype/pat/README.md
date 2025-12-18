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
        "group_type": "pat.group.Dim0Grouper",
        "prox_type": "pat.prox.ProxGroupLasso",
        "reg_lambda": 2e-4,
    },
    {"params": others},
]

# create base optimizer (SGD, Adam or AdamW)
base_optimizer = torch.optim.SGD(
    param_groups, lr=0.1, momentum=0.9, weight_decay=1e-4
)

# create PruneOptimizer
optimizer = PruneOptimizer(base_optimizer)
```

After creating `PruneOptimizer`, one can use it as a regular PyTorch optimizer.

## Pruning configuration

Pruning configs are dictionaries that define which parameter groups to prune and how to prune them. Each key-value pair in the config maps to a prunable parameter group of `PruneOptimizer`. The keys are used to match model parameters, while the values specify the pruning granularity and proximal map. The key can be one of the following types:

- parameter name (string): e.g., `blocks.0.attn.qkv.weight`
- regex pattern (string): e.g., `:.*attn\.qkv\.weight`
- module type and parameter name suffix ((class, string) tuple): e.g., `(torch.nn.Linear, 'weight')`
