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

It's also possible to pre-process the weight and call the kernel directly.

```python
import torch
from torchao.prototype.fp6_llm import to_scaled_tc_float6_e3m2
from torchao.ops import fp6_llm_linear

fp32_weight = torch.randn(1024, 512).cuda()

# pre-process the weight. this will quantize the weight to FP6 and pack it in a special
# layout for tensor cores. refer to paper for more details.
fp6_weight, scales = to_scaled_tc_float6_e3m2(fp32_weight)

fp16_act = torch.randn(1, 512).cuda().half()
outputs = fp6_llm_linear(fp16_act, fp6_weight, scales)  # shape (1, 1024)
```

**NOTE**: since this kernel's computation dtype is FP16, it is recommended to convert the model to FP16 (instead of BF16) before applying quantization.

## TODO

- [ ] Compile CUDA kernel for Windows
- [ ] Merge FP5 from upstream

## Credits

Credits to FP6-LLM authors

- Paper: https://arxiv.org/abs/2401.14112
- Code: https://github.com/usyd-fsalab/fp6_llm
