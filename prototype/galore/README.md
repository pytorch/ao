## Fused GaLore Adam (WIP)

### Various fused implementations of `Adam` update step per [Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)

This is an initial attempt at optimizing the update step of the `GaLore Adam` optimizer.

#### Overview

The `GaLore` `Adam` optimizer introduces additional ops to the traditional `adam` update step.

Specifically:

1.  `grad` is projected to low rank --> additional matmul
2.  `adam` states are updated with `grad` elementwise (same as `Adam` except in low-rank)
3.  normalized `grad` is projected to full rank --> additional matmul
4.  `params` are updated with the normalized full rank grad

#### Installation

```
pip install --editable .
```

#### Implementation

See `galore_fused/NOTE.md` for implementation details

#### Next Steps

- [ ] Implement `FusedGaLoreOptimizer`
- [ ] `Cutlass` - given fixed GEMM shape, experiment with `Cutlass` GEMMs (`split-k`, `stream-k`, fast `tensorops`). Interestingly, profiling `torch.matmul` for down projection shows that `cuBlas` dispatches to a `Cutlass` kernel of shape `128x128x16`.
- [ ] Repeat with `AdamW8bit` - pure `triton` implementation of `bitsandbytes` `AdamW8bit`
- [ ] More detailed analysis of `torch.compile` performance
