# FP6-LLM

This is a FP16 x FP6 mixed matmul kernel optimized for io bound workloads per [FP6-LLM](https://arxiv.org/abs/2401.14112). The actual CUDA kernel is located under [csrc/cuda/fp6_llm/](../../csrc/cuda/fp6_llm/). This module provides helper functions to quantize FP32 weights to FP6 and facility to convert existing models to FP6.

## Usage

```python
from torchao.prototype.fp6_llm import convert_fp6_llm

model = ...
convert_fp6_llm(model)  # convert model in-place, replacing nn.Linear modules with Fp6LlmLinear

# fully compatible with torch.compile()
model.compile(mode="max-autotune", fullgraph=True)
```

## TODO

- [ ] Compile CUDA kernel for Windows
- [ ] Merge FP5 from upstream

## Credits

Credits to FP6-LLM authors

- Paper: https://arxiv.org/abs/2401.14112
- Code: https://github.com/usyd-fsalab/fp6_llm
