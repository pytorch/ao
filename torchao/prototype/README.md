# Prototype

### Experimental kernels and utilities for efficient inference and training

> The goal isn't to reproduce all emerging methodologies but to extract common components across prevalent, proven paradigms that can be modularized and composed with the `torch` stack as well as other OSS ML frameworks.

#### Code structure

- `galore` - fused kernels for memory-efficient pre-training / fine-tuning per the [GaLore algorithm](https://arxiv.org/abs/2403.03507)
  - `galore/kernels` - `triton` kernels that fuse various steps of the `GaLore` algorithm
  - `galore/docs` - implementation notes and discussion of issues faced in kernel design.
- [`quant_llm`](quant_llm) - FP16 x Floatx mixed matmul kernel per [FP6-LLM](https://arxiv.org/abs/2401.14112)
- [`low_bit_optim`](low_bit_optim) - re-implementation of 8-bit optimizers from [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and 4-bit optimizers from [lpmm](https://github.com/thu-ml/low-bit-optimizers).

#### Roadmap

- `hqq`, `awq`, `marlin`,`QuaRot`, and other well-researched methodologies for quantized fine-tuning and inference.
  - ideally, techniques that are both **theoretically sound** and have **practical hardware-aware implementations**
  - AWQ and GPTQ are good examples.
- `cutlass` / `triton` utilities for common quantization ops (numeric conversion, quant / dequant, mixed type gemm, etc.)
  - goal is to create a set of kernels and components that can expedite the implementation & optimization across the spectrum of quantization, fine-tuning, and inference patterns.
