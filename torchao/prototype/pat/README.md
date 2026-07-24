# PAT: Pruning-Aware Training

PAT is a library based on proximal gradient methods. It directly induces structured sparsity or low-rank structure during training, removing the need for custom pruning metrics and multiple rounds of training.

PAT's optimizer-only interface supports integration into existing training pipelines. The code is organized around two components:

* grouper: defines how a parameter is viewed as groups of weights or singular values
* proximal map: projects those groups toward sparse or low-rank values

## Optimizer-only interface

This package provides a `PruneOptimizer` that wraps a base optimizer inheriting from `torch.optim.Optimizer`. The following code illustrates how to set up PAT:

```python
from torchao.prototype.pat.optim import PruneOptimizer

model = torchvision.models.resnet18().cuda()

# Split parameters into prunable and non-prunable groups.
weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
others = [p for name, p in model.named_parameters() if not name.endswith("weight")]

# Apply row-wise group Lasso regularization to the weights.
param_groups = [
    {
        "params": weights,
        "group_type": "Dim0Grouper",
        "prox_type": "ProxGroupLasso",
        "reg_lambda": 2e-4,
    },
    {"params": others},
]

base_optimizer = torch.optim.SGD(
    param_groups, lr=0.1, momentum=0.9, weight_decay=1e-4
)
optimizer = PruneOptimizer(base_optimizer)
```

After creating `PruneOptimizer`, use it as a regular PyTorch optimizer.

## Pruning configuration

Pruning configs are dictionaries that define which parameter groups to prune and how to prune them. Each key-value pair maps to a prunable parameter group of `PruneOptimizer`. The keys match model parameters, while the values specify the pruning granularity and proximal map. A key can be one of the following types:

- parameter name (string): for example, `blocks.0.attn.qkv.weight`
- regex pattern (string): for example, `:.*attn\.qkv\.weight`
- module type and parameter name suffix (`(class, string)` tuple): for example, `(torch.nn.Linear, "weight")`

## Groupers and proximal maps

A pruning entry pairs a **grouper** with a **proximal map**. The grouper reshapes a tensor into `(n_groups, group_size)`, or exposes singular values for an SVD grouper, and the proximal map is then applied to that view.

### Groupers (`torchao.prototype.pat.group`)

| Grouper | Group structure |
| --- | --- |
| `Dim0Grouper` / `Dim1Grouper` | One group per row or column of a 2-D weight |
| `ElemGrouper` | Whole tensor as one group with per-element pruning |
| `LayerGrouper` | Whole tensor as one group for layer-level pruning |
| `KElementGrouper(k)` | `(numel / k, k)` blocks of `k` consecutive elements |
| `ConvFilterGrouper` | One group per `(c_out, c_in)` filter slice of a Conv2d kernel |
| `AttentionHeadGrouperDim0(num_heads)` | One group per attention head along dimension 0 |
| `AttentionHeadGrouperDim1(num_heads)` | One group per attention head along dimension 1 |
| `SVDGrouper` | Decompose `W = U diag(s) Vh` and expose its singular values |
| `PackedSVDGrouper(npack)` | Apply SVD independently to each of `npack` sub-tensors |

### Proximal maps (`torchao.prototype.pat.optim`)

| Proximal map | Behavior |
| --- | --- |
| `ProxLasso` | Soft-threshold each element for magnitude-based sparsity |
| `ProxGroupLasso` | Soft-threshold each group's L2 norm to zero whole groups |
| `ProxNuclearNorm` | Soft-threshold singular values to shrink rank smoothly |
| `MinSparsityConstraint(min_sparsity)` | Hard-zero the smallest-L2 `ceil(min_sparsity * n_groups)` groups |
| `MinRankConstraint(min_sparsity)` | Hard-zero the smallest `ceil(min_sparsity * k)` singular values in each matrix |
| `NMSparseConstraint(n_nonzero)` | Keep the largest-magnitude `n_nonzero` elements per group |

### Recipes

| Goal | Grouper | Proximal map | Notes |
| --- | --- | --- | --- |
| **2:4 sparsity** | `KElementGrouper(k=4)` | `NMSparseConstraint(n_nonzero=2)` | Keeps two nonzero elements in each four-element block |
| **Row sparsity** | `Dim0Grouper` | `MinSparsityConstraint` or `ProxGroupLasso` | Use the hard constraint for an exact target or group Lasso for a smooth regularization knob |
| **Column sparsity** | `Dim1Grouper` | `MinSparsityConstraint` or `ProxGroupLasso` | Drops input channels of a Linear layer |
| **Conv filter sparsity** | `ConvFilterGrouper` | `MinSparsityConstraint` or `ProxGroupLasso` | Drops complete `(c_out, c_in)` filter slices |
| **Attention head sparsity** | `AttentionHeadGrouperDim0` and/or `AttentionHeadGrouperDim1` | `MinSparsityConstraint` | Configure matching `num_heads` for the selected projections |
| **Low-rank approximation, smooth** | `SVDGrouper` or `PackedSVDGrouper` | `ProxNuclearNorm` | Uses regularization-controlled rank decay |
| **Low-rank approximation, exact target** | `SVDGrouper` or `PackedSVDGrouper` | `MinRankConstraint(min_sparsity)` | Zeros `ceil(min_sparsity * k)` and retains `k - ceil(min_sparsity * k)` singular values per matrix |

`MinSparsityConstraint`, `MinRankConstraint`, and `NMSparseConstraint` are hard-zero maps: they ignore `reg_lambda` and `gamma` and are driven by their target argument. Set `min_sparsity_schedule: true` to ramp a minimum-sparsity or minimum-rank target cubically from the end of warmup to `healing_start_step`.

SVD decompositions can dominate optimizer cost on wide tensors. Set `prox_freq: N` on a parameter group to run its grouper and proximal map every `N` optimizer steps. SVD groups using the hard `MinRankConstraint` reapply the proximal map during healing by default so the base optimizer cannot refill removed singular values. Soft SVD maps such as `ProxNuclearNorm` retain their existing behavior unless `prox_through_heal: true` is set explicitly, while `MinRankConstraint` can opt out with `prox_through_heal: false`. Setting `prox_through_heal: true` on a non-SVD grouper is invalid and rejected during optimizer construction.

## Unstructured pruning on 1.3B OLMo models

The goal of the following experiments was to compare PAT pruned LLMs with "dense" models of equivalent nonzero parameter count. For example, a 1.3B model pruned to ~58% sparsity would be compared to a 760M model trained from scratch on the same token budget. The dense models have better inference efficiency than models with unstructured sparsity, but this is a good sanity check for PAT.

We borrowed the setup from AllenAI's OLMo models. The table below is Table 1 of [this paper](https://arxiv.org/abs/2412.04403) from AllenAI.
<img src="https://github.com/user-attachments/assets/c66b8f3b-702d-478b-9a94-9718ea0b0583" style="width:80%" />

The two plots show that the PAT pruned 1.3B models (blue curve) reach much better training loss and mean test accuracy on 8 reasoning benchmarks (ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, OpenBookQA, PIQA, Social IQa, WinoGrande) across different sparsity levels.
![](https://github.com/user-attachments/assets/b04347bd-6f16-44ca-85b9-8591349a9b31)
![](https://github.com/user-attachments/assets/91820a7f-519b-4415-ba68-f510df1e18e9)
