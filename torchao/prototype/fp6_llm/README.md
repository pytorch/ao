# FP6-LLM

This is a FP16 x FP6 mixed matmul kernel optimized for io bound workloads per [FP6-LLM](https://arxiv.org/abs/2401.14112). The actual CUDA kernel is located under [csrc/cuda/fp6_llm/](../../csrc/cuda/fp6_llm/). This module provides helper functions to quantize FP32 weights to FP6 and facility to convert existing models to FP6.

## Usage

```python
from torchao.quantization.quant_api import quantize
from torchao.prototype.fp6_llm import fp6_llm_weight_only

model = ...
quantize(model, fp6_llm_weight_only())  # convert nn.Lineaer.weight to FP6 in-place

# fully compatible with torch.compile()
model.compile(mode="max-autotune", fullgraph=True)
```

It's also possible to pre-process the weight and call the kernel directly.

```python
# TODO: update
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

**NOTE**: since this kernel's computation dtype is FP16, it is recommended to convert the model to FP16 (instead of BF16) before applying quantization and use FP16 for activations.

## Benchmark results

Benchmarks are run on a machine with a single 4070Ti SUPER GPU using the scripts in [_models/llama](../../_models/llama). tokens/s is measured using [generate.py](../../_models/llama/generate.py) which generates text in a latency optimized way (batchsize=1). wikitext perplexity is measured using [eval.py](../../_models/llama/eval.py) which uses [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness). The model used is [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

FPx quantization is run with `--precision float16`. The rest uses the default precision of `bfloat16`.

Quantization        | wikitext perplexity | tokens/s
--------------------|---------------------|----------
INT8                | 12.21               |  87.45
INT4-256 (tinygemm) | 76266957.87 (bug)   | 157.10
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
