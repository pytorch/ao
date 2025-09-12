# Quant-LLM

This is a FP16 x Floatx mixed matmul kernel optimized for io bound workloads per [FP6-LLM](https://arxiv.org/abs/2401.14112). The actual CUDA kernel is located under [csrc/cuda/fp6_llm/](../../csrc/cuda/fp6_llm/). This module provides helper functions to quantize FP32/FP16/BF16 weights to Floatx and integration with torchao API.

This kernel was originally designed for FP16, but was extended to work for BF16 by @tobiasvanderwerff.

## Usage

```python
from torchao.quantization import (
    quantize_,
    FPXWeightOnlyConfig,
)

model = ...
# model can have dtype float16 or bfloat16

# for generic Floatx EyMz where x = 1 + y + z
# fp6 with ebits = 3 and mbits = 2
quantize_(model, FPXWeightOnlyConfig(3, 2))

# fully compatible with torch.compile()
model.compile(mode="max-autotune", fullgraph=True)
```

It's also possible to pre-process the weight and call the kernel directly.

```python
import torch
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear

fp32_weight = torch.randn(1024, 512).cuda()
ebits, mbits = 3, 2

# pre-process the weight. this will quantize the weight to FP6 and pack it in a special
# layout for tensor cores. refer to paper for more details.
fp6_weight, scales = to_scaled_tc_floatx(fp32_weight, ebits, mbits)

fp16_act = torch.randn(1, 512).cuda().half()
outputs = quant_llm_linear(ebits, mbits, fp16_act, fp6_weight, scales)  # shape (1, 1024)
```

**NOTE**:
- The kernel works for both FP16 and BF16 input activations
- Only FP6 E3M2 and FP5 E2M2 are tested and enabled in the official repo. We additionally enable support for FP6 E2M3 and FP5 E3M1.
- On most hardware, this kernel is faster than FP16 linear for batch size from 1 to 128, and slower for batch size larger than or equal to 256. See https://github.com/usyd-fsalab/fp6_llm/issues/8 for a detailed discussion. See https://github.com/pytorch/ao/pull/223 and https://github.com/pytorch/ao/pull/1147 for some microbenchmark results.
- The kernel is supported for >=SM80 (Ampere generation) as well as SM75 (Turing generation) GPUs. However, SM75 support requires manual compilation of the C++/CUDA extensions (see the installation instructions in the [README](https://github.com/pytorch/ao/blob/main/README.md#installation) for details).

## End-to-End benchmarks

Benchmarks are run on a machine with a single 4070Ti SUPER GPU using the scripts in [_models/llama](../../_models/llama). tokens/s is measured using [generate.py](../../_models/llama/generate.py) which generates text in a latency optimized way (batchsize=1). wikitext perplexity is measured using [eval.py](../../_models/llama/eval.py) which uses [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness). The model used is [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

Floatx quantization is run with `--precision float16`. The rest uses the default precision of `bfloat16`.

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
