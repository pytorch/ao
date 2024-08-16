# Quant-LLM

This is a FP16 x FPx mixed matmul kernel optimized for io bound workloads per [FP6-LLM](https://arxiv.org/abs/2401.14112). The actual CUDA kernel is located under [csrc/cuda/fp6_llm/](../../csrc/cuda/fp6_llm/). This module provides helper functions to quantize FP32/FP16/BF16 weights to FPx and integration with torchao API.

## Usage

```python
from torchao.quantization.quant_api import quantize_
from torchao.prototype.quant_llm import fp6_llm_weight_only, quant_llm_fpx_weight_only

model = ...
model.half()  # not necessary, but recommeneded to maintain accuracy
quantize_(model, fp6_llm_weight_only())  # convert nn.Lineaer.weight to FP6 E3M2 in-place

# for generic FPx EyMz where x = 1 + y + z
# quantize_(model, quant_llm_fpx_weight_only(2, 2))  # use FP5 E2M2 instead

# fully compatible with torch.compile()
model.compile(mode="max-autotune", fullgraph=True)
```

It's also possible to pre-process the weight and call the kernel directly.

```python
import torch
from torchao.prototype.quant_llm import to_scaled_tc_fpx
from torchao.ops import quant_llm_linear

fp32_weight = torch.randn(1024, 512).cuda()
ebits, mbits = 3, 2

# pre-process the weight. this will quantize the weight to FP6 and pack it in a special
# layout for tensor cores. refer to paper for more details.
fp6_weight, scales = to_scaled_tc_fpx(fp32_weight, ebits, mbits)

fp16_act = torch.randn(1, 512).cuda().half()
outputs = quant_llm_linear(ebits, mbits, fp16_act, fp6_weight, scales)  # shape (1, 1024)
```

**NOTE**:
- Since this kernel's computation dtype is FP16, it is recommended to convert the model to FP16 (instead of BF16) before applying quantization and use FP16 for activations.
- Only FP6 E3M2 and FP5 E2M2 are tested and enabled in the official repo. We additionally enable support for FP6 E2M3 and FP5 E3M1.
- On most hardware, this kernel is faster than FP16 linear for batch size from 1 to 128, and slower for batch size larger than or equal to 256. See https://github.com/usyd-fsalab/fp6_llm/issues/8 for a detailed discussion. See https://github.com/pytorch/ao/pull/223 for some microbenchmark results.

## End-to-End benchmarks

Benchmarks are run on a machine with a single 4070Ti SUPER GPU using the scripts in [_models/llama](../../_models/llama). tokens/s is measured using [generate.py](../../_models/llama/generate.py) which generates text in a latency optimized way (batchsize=1). wikitext perplexity is measured using [eval.py](../../_models/llama/eval.py) which uses [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness). The model used is [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

FPx quantization is run with `--precision float16`. The rest uses the default precision of `bfloat16`.

Quantization        | wikitext perplexity | tokens/s
--------------------|---------------------|----------
INT8                | 12.21               |  87.45
INT4-256 (tinygemm) | --                  | 157.10
FP6 E3M2            | 12.34               | 106.76
FP6 E2M3            | 12.23               | 106.77
FP5 E3M1            | 12.55               | 122.69
FP5 E2M2            | 12.47               | 122.66
FP4 E3M0            | 14.58               | 145.55
FP4 E2M1            | 15.01               | 146.05
FP3 E2M0            | 74625.18            | 164.49

## Credits

Credits to FP6-LLM authors

- Paper: https://arxiv.org/abs/2401.14112
- Code: https://github.com/usyd-fsalab/fp6_llm
