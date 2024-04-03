# Prototype

### Experimental kernels and utilities for quantization

#### Code structure

- `galore` - fused kernels for memory-efficient pre-training / fine-tuning per the [GaLore algorithm](https://arxiv.org/abs/2403.03507)
  - `galore/kernels` - `triton` kernels that fuse various steps of the `GaLore` algorithm
  - `galore/docs` - implementation notes and discussion of issues faced in kernel design.
