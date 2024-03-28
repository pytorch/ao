# Prototype

### Experimental kernels and utilities for quantization

#### Code structure

- `galore` - fused kernels for memory-efficient pre-training / fine-tuning per the [GaLore algorithm](https://arxiv.org/abs/2403.03507)
- `cutlass` - python utils for defining mixed-type `cutlass` kernels and quant ops.
- `triton` - composable `triton` kernels for quantization ops
