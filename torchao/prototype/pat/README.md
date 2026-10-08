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

## Direct projection

`PruneOptimizer(..., latent_weights=False)` and `build_prune_optimizer(..., latent_weights=False)` step directly from projected parameters instead of restoring PAT's dense latent weights. The default remains `True`. Direct projection removes PAT's latent copy but changes the optimization trajectory; it does not remove the base optimizer's momentum or AdamW moments. Gradients or momentum may revive zeros during pruning; the existing healing mask freezes them once healing starts.

The mode is a constructor setting, not part of the delegated optimizer state dict. When resuming, reconstruct the same mode before loading both model and optimizer checkpoints. Cross-mode resume is unsupported: direct checkpoints do not contain the latent weights needed by default-mode restoration.

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
| `AttentionHeadGrouperDim0(num_heads=..., head_dim=...)` | One group per attention head along dimension 0; specify exactly one argument |
| `AttentionHeadGrouperDim1(num_heads=..., head_dim=...)` | One group per attention head along dimension 1; specify exactly one argument |
| `SVDGrouper` | Decompose `W = U diag(s) Vh` and expose its singular values |
| `PackedSVDGrouper(npack)` | Apply SVD independently to each of `npack` sub-tensors |

Attention-head groupers accept exactly one positive integer `num_heads` or `head_dim`, and the packed dimension must be divisible by it. Existing positional `num_heads` calls remain supported. For heterogeneous layer widths, use a fixed `head_dim` in the pruning config so each tensor derives its own head count; for example, `group_type: AttentionHeadGrouperDim0` with `head_dim: 64`.

### Proximal maps (`torchao.prototype.pat.optim`)

| Proximal map | Behavior |
| --- | --- |
| `ProxLasso` | Soft-threshold each element for magnitude-based sparsity |
| `ProxGroupLasso` | Soft-threshold each group's L2 norm to zero whole groups |
| `ProxNuclearNorm` | Soft-threshold singular values to shrink rank smoothly |
| `MinSparsityConstraint(min_sparsity)` | Hard-zero the smallest-L2 `ceil(min_sparsity * n_groups)` groups in each tensor |
| `GlobalMinSparsityConstraint(min_sparsity)` | Hard-zero one jointly scored group-count budget across all tensors in an optimizer parameter group |
| `CoupledMinSparsityConstraint(min_sparsity)` | Hard-zero the same shared-channel indices across all tensors with one `couple_key` |
| `MinRankConstraint(min_sparsity)` | Hard-zero the smallest `ceil(min_sparsity * k)` singular values in each matrix |
| `NMSparseConstraint(n_nonzero)` | Keep the largest-magnitude `n_nonzero` elements per group |

### Recipes

| Goal | Grouper | Proximal map | Notes |
| --- | --- | --- | --- |
| **2:4 sparsity** | `KElementGrouper(k=4)` | `NMSparseConstraint(n_nonzero=2)` | Keeps two nonzero elements in each four-element block |
| **Row sparsity** | `Dim0Grouper` | `MinSparsityConstraint` or `ProxGroupLasso` | Use the hard constraint for an exact target or group Lasso for a smooth regularization knob |
| **Column sparsity** | `Dim1Grouper` | `MinSparsityConstraint` or `ProxGroupLasso` | Drops input channels of a Linear layer |
| **Conv filter sparsity** | `ConvFilterGrouper` | `MinSparsityConstraint` or `ProxGroupLasso` | Drops complete `(c_out, c_in)` filter slices |
| **Attention head sparsity** | `AttentionHeadGrouperDim0` and/or `AttentionHeadGrouperDim1` | `MinSparsityConstraint` | Configure `num_heads` or fixed `head_dim` for the selected projections |
| **Global structured sparsity** | Any supported non-SVD grouper | `GlobalMinSparsityConstraint` | Shares one group-count budget across every tensor in the optimizer parameter group; `score_type: rms` is the recommended default |
| **Shared residual-channel sparsity** | `Dim1Grouper` on readers and `Dim0Grouper` on writers | `CoupledMinSparsityConstraint` | One caller-defined cluster uses the same selected channel indices everywhere |
| **Low-rank approximation, smooth** | `SVDGrouper` or `PackedSVDGrouper` | `ProxNuclearNorm` | Uses regularization-controlled rank decay |
| **Low-rank approximation, exact target** | `SVDGrouper` or `PackedSVDGrouper` | `MinRankConstraint(min_sparsity)` | Zeros `ceil(min_sparsity * k)` and retains `k - ceil(min_sparsity * k)` singular values per matrix |

`MinSparsityConstraint`, `GlobalMinSparsityConstraint`, `CoupledMinSparsityConstraint`, `MinRankConstraint`, and `NMSparseConstraint` are hard-zero maps: they ignore `reg_lambda` and `gamma` and are driven by their target argument. Set `min_sparsity_schedule: true` to ramp a minimum-sparsity or minimum-rank target cubically from the end of warmup to `healing_start_step`. Regardless of `prox_freq` alignment, PAT applies a hard constraint once at `healing_start_step - 1` so healing freezes the final target mask rather than an earlier mask.

`GlobalMinSparsityConstraint` computes one budget as `ceil(min_sparsity * total_groups)` for each optimizer parameter group, not one budget per tensor and not a parameter-count budget. It jointly ranks the groups exposed by the configured grouper, and all parameters in that optimizer group must produce scores on the same device. Use `score_type: rms` by default when tensors have different group sizes because it normalizes L2 magnitude by `sqrt(group_size)`. Raw `l2` tends to favor retaining larger groups because their norms grow with group size, while `param_cost` divides by the full group size and more strongly favors removing groups that save more parameters. Padded `KElementGrouper` views are rejected because padding would distort both scoring and accounting; choose a `k` that divides every grouped dimension.

For recipes whose initialization scales magnitudes by the inverse square root of the number of groups, set the optional `score_group_count_ref` positive integer in a global pruning group. This multiplies each tensor's scores by `sqrt(n_groups / score_group_count_ref)` before joint selection. It is separate from RMS normalization by group size, is not automatically enabled by `head_dim`, and is disabled by default. Different positive reference magnitudes produce the same mathematical ranking because the reference contributes one common scale factor across candidates.

For DTensor parameters, global selection requires full materialization of every grouped tensor on every rank before the shared top-k decision, followed by scattering the selected masks back to the original placements. The current internal `CACHE_FULL_TENSORS` policy gathers each DTensor once and retains all dense copies until selection and write-back finish, so peak dense memory is the sum of the tensors in the optimizer parameter group. A future lower-memory policy could release each copy after scoring, but would need to gather each DTensor again to apply the selected mask. Every rank gathers the same tensors and performs the same deterministic selection, so masks are expected to agree, and CPU Gloo regression tests exercise actual multi-rank execution, including subset meshes.

### Coupled shared-axis pruning

Global pruning shares a budget but can remove different indices in each tensor. Coupled pruning sums squared norms at each shared channel index, scores the combined norm (`rms`, `l2`, or `param_cost`), and removes one index set of size `ceil(min_sparsity * channels)` from every tensor with the same `couple_key`, even across optimizer parameter groups.

```python
import torch
from torchao.prototype.pat.optim import PruneOptimizer

reader = torch.nn.Parameter(torch.randn(4, 8))  # input-channel axis
writer = torch.nn.Parameter(torch.randn(8, 4))  # output-channel axis
shared = {
    "prox_type": "CoupledMinSparsityConstraint",
    "couple_key": "residual",
    "min_sparsity": 0.25,
}
groups = [
    {**shared, "params": [reader], "group_type": "Dim1Grouper"},
    {**shared, "params": [writer], "group_type": "Dim0Grouper"},
]
optimizer = PruneOptimizer(torch.optim.SGD(groups, lr=0.01), healing_start_step=8)
for p in (reader, writer):
    p.grad = torch.zeros_like(p)
optimizer.step()
```

An existing non-SVD prox can add a nested `coupled` dictionary containing `couple_key`, `group_type`, and `min_sparsity`. Its own per-parameter/global projection runs first, followed by coupling. The nested stage inherits only `min_sparsity_schedule`, `reg_lambda`, and `score_type`; specify its own target and optional `prox_freq` (default 1). All members must agree on effective target, schedule, score type, and coupled frequency. Scheduled coupling requires a finite `healing_start_step`; the final hard mask is applied at `healing_start_step - 1` even off cadence. Metrics count each participating parameter once using final literal zeros, and retain cached totals on skipped updates. SVD/coupled composition and padded layouts are unsupported.

The caller must enumerate every relevant tensor and its channel axis. Coupling does not discover residual dependencies, handle normalization/bias dependencies automatically, resize modules, or produce a compact inference model.

For DTensors, coupled selection materializes every clustered tensor on each mesh participant, selects one shared mask, and redistributes locally with `src_data_rank=None`. Peak dense memory is proportional to the sum of clustered tensors. Subset-mesh nonparticipants skip processing, and replica dimensions do not duplicate counts. CPU Gloo tests cover the materializing reference on a subset excluding rank zero, including uneven row sharding and cached nonparticipant metrics. These correctness tests do not establish NCCL/FSDP2 throughput or model-training performance.

SVD decompositions can dominate optimizer cost on wide tensors. Set `prox_freq: N` on a parameter group to run its grouper and proximal map every `N` optimizer steps. SVD groups using the hard `MinRankConstraint` reapply the proximal map during healing by default so the base optimizer cannot refill removed singular values. Soft SVD maps such as `ProxNuclearNorm` retain their existing behavior unless `prox_through_heal: true` is set explicitly, while `MinRankConstraint` can opt out with `prox_through_heal: false`. Setting `prox_through_heal: true` on a non-SVD grouper is invalid and rejected during optimizer construction.

## Unstructured pruning on 1.3B OLMo models

The goal of the following experiments was to compare PAT pruned LLMs with "dense" models of equivalent nonzero parameter count. For example, a 1.3B model pruned to ~58% sparsity would be compared to a 760M model trained from scratch on the same token budget. The dense models have better inference efficiency than models with unstructured sparsity, but this is a good sanity check for PAT.

We borrowed the setup from AllenAI's OLMo models. The table below is Table 1 of [this paper](https://arxiv.org/abs/2412.04403) from AllenAI.
<img src="https://github.com/user-attachments/assets/c66b8f3b-702d-478b-9a94-9718ea0b0583" style="width:80%" />

The two plots show that the PAT pruned 1.3B models (blue curve) reach much better training loss and mean test accuracy on 8 reasoning benchmarks (ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, OpenBookQA, PIQA, Social IQa, WinoGrande) across different sparsity levels.
![](https://github.com/user-attachments/assets/b04347bd-6f16-44ca-85b9-8591349a9b31)
![](https://github.com/user-attachments/assets/91820a7f-519b-4415-ba68-f510df1e18e9)
