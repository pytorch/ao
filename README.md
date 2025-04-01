# torchao: PyTorch Architecture Optimization

[![](https://dcbadge.vercel.app/api/server/gpumode?style=flat)](https://discord.gg/gpumode)

[Introduction](#introduction) | [Inference](#inference) | [Training](#training)  | [Composability](#composability) | [Custom Kernels](#custom-kernels) | [Alpha Features](#alpha-features) | [Installation](#installation) | [Integrations](#integrations) | [Videos](#videos) | [License](#license) | [Citation](#citation)

## Introduction

torchao: PyTorch library for custom data types & optimizations. Quantize and sparsify weights, gradients, optimizers & activations for inference and training.

From the team that brought you the fast series
* 9.5x inference speedups for Image segmentation models with [sam-fast](https://pytorch.org/blog/accelerating-generative-ai)
* 10x inference speedups for Language models with [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2)
* 3x inference speedup for Diffusion models with [sd-fast](https://pytorch.org/blog/accelerating-generative-ai-3)

torchao works for training too, with [up to 1.5x e2e speedups](https://pytorch.org/blog/training-using-float8-fsdp2/) on large scale (512 GPU / 405B parameter count) pretraining jobs with `torchao.float8`!

torchao just works with `torch.compile()` and `FSDP2` over most PyTorch models on Huggingface out of the box.

## Inference

### Post Training Quantization

Quantizing and Sparsifying your models is a 1 liner that should work on any model with an `nn.Linear` including your favorite HuggingFace model. You can find a more comprehensive usage instructions [here](torchao/quantization/), sparsity [here](/torchao/_models/sam/README.md) and a HuggingFace inference example [here](scripts/hf_eval.py)

For inference, we have the option of
1. Quantize only the weights: works best for memory bound models
2. Quantize the weights and activations: works best for compute bound models
2. Quantize the activations and weights and sparsify the weight

```python
from torchao.quantization.quant_api import (
    quantize_,
    Int8DynamicActivationInt8WeightConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig
)
quantize_(m, Int4WeightOnlyConfig())
```

For gpt-fast `Int4WeightOnlyConfig()` is the best option at bs=1 as it **2x the tok/s and reduces the VRAM requirements by about 65%** over a torch.compiled baseline.

If you don't have enough VRAM to quantize your entire model on GPU and you find CPU quantization to be too slow then you can use the device argument like so `quantize_(model, Int8WeightOnlyConfig(), device="cuda")` which will send and quantize each layer individually to your GPU.

If you see slowdowns with any of these techniques or you're unsure which option to use, consider using [autoquant](./torchao/quantization/README.md#autoquantization) which will automatically profile layers and pick the best way to quantize each layer.

```python
model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
```

We also provide a developer facing API so you can implement your own quantization algorithms so please use the excellent [HQQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/hqq) algorithm as a motivating example.

### KV Cache Quantization

We've added kv cache quantization and other features in order to enable long context length (and necessarily memory efficient) inference.

In practice these features alongside int4 weight only quantization allow us to **reduce peak memory by ~55%**, meaning we can Llama3.1-8B inference with a **130k context length with only 18.9 GB of peak memory.** More details can be found [here](torchao/_models/llama/README.md)

## Training

### Quantization Aware Training

Post-training quantization can result in a fast and compact model, but may also lead to accuracy degradation. We recommend exploring Quantization Aware Training (QAT) to overcome this limitation. In collaboration with Torchtune, we've developed a QAT recipe that demonstrates significant accuracy improvements over traditional PTQ, recovering **96% of the accuracy degradation on hellaswag and 68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization (PTQ). And we've provided a full recipe [here](https://pytorch.org/blog/quantization-aware-training/). For more details, please see the [QAT README](./torchao/quantization/qat/README.md).

```python
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationInt4WeightConfig,
)
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    FromIntXQuantizationAwareTrainingConfig,
    IntXQuantizationAwareTrainingConfig,
)

# Insert fake quantization
activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
weight_config = FakeQuantizeConfig(torch.int4, group_size=32)
quantize_(
    my_model,
    IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
)

# Run training... (not shown)

# Convert fake quantization to actual quantized operations
quantize_(my_model, FromIntXQuantizationAwareTrainingConfig())
quantize_(my_model, Int8DynamicActivationInt4WeightConfig(group_size=32))
```

### Float8

[torchao.float8](torchao/float8) implements training recipes with the scaled float8 dtypes, as laid out in https://arxiv.org/abs/2209.05433.

With ``torch.compile`` on, current results show throughput speedups of up to **1.5x on up to 512 GPU / 405B parameter count scale** ([details](https://pytorch.org/blog/training-using-float8-fsdp2/))

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m, module_filter_fn=...)
```

And for an end-to-minimal training recipe of pretraining with float8, you can check out [torchtitan](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md). 

#### Blog posts about float8 training

* [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
* [Efficient Pre-training of Llama 3-like model architectures using torchtitan on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
* [Float8 in PyTorch](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)


### Sparse Training

We've added support for semi-structured 2:4 sparsity with **6% end-to-end speedups on ViT-L**. Full blog [here](https://pytorch.org/blog/accelerating-neural-network-training/)

The code change is a 1 liner with the full example available [here](torchao/sparsity/training/)

```python
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-efficient optimizers

ADAM takes 2x as much memory as the model params so we can quantize the optimizer state to either 8 or 4 bit effectively reducing the optimizer VRAM requirements by 2x or 4x respectively over an fp16 baseline

```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
```

In practice, we are a tiny bit slower than expertly written kernels but the implementations for these optimizers were written in a **few hundred lines of PyTorch code** and compiled so please use them or copy-paste them for your quantized optimizers. Benchmarks [here](https://github.com/pytorch/ao/tree/main/torchao/optim)

We also have support for [single GPU CPU offloading](https://github.com/pytorch/ao/tree/main/torchao/optim#optimizer-cpu-offload) where both the gradients (same size as weights) and the optimizers will be efficiently sent to the CPU. This alone can **reduce your VRAM requirements by 60%**

```python
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

## Composability

1. `torch.compile`: A key design principle for us is composability as in any new dtype or layout we provide needs to work with our compiler. It shouldn't matter if the kernels are written in pure PyTorch, CUDA, C++, or Triton - things should just work! So we write the dtype, layout, or bit packing logic in pure PyTorch and code-generate efficient kernels.
3. [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md): Historically most quantization has been done for inference, there is now a thriving area of research combining distributed algorithms and quantization.

The best example we have combining the composability of lower bit dtype with compile and fsdp is [NF4](torchao/dtypes/nf4tensor.py) which we used to implement the [QLoRA](https://www.youtube.com/watch?v=UvRl4ansfCg) algorithm. So if you're doing research at the intersection of this area we'd love to hear from you.

## Custom Kernels

We've added support for authoring and releasing [custom ops](./torchao/csrc/) that do not graph break with `torch.compile()` so if you love writing kernels but hate packaging them so they work all operating systems and cuda versions, we'd love to accept contributions for your custom ops. We have a few examples you can follow

1. [fp6](torchao/dtypes/floatx) for 2x faster inference over fp16 with an easy to use API `quantize_(model, FPXWeightOnlyConfig(3, 2))`
2. [2:4 Sparse Marlin GEMM](https://github.com/pytorch/ao/pull/733) 2x speedups for FP16xINT4 kernels even at batch sizes up to 256
3. [int4 tinygemm unpacker](https://github.com/pytorch/ao/pull/415) which makes it easier to switch quantized backends for inference

If you believe there's other CUDA kernels we should be taking a closer look at please leave a comment on [this issue](https://github.com/pytorch/ao/issues/697)


## Alpha features

Things we're excited about but need more time to cook in the oven

1. [MX](torchao/prototype/mx_formats) training and inference support with tensors using the [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) data types, which can be described as groupwise scaled float8/float6/float4/int8, with the scales being constrained to powers of two. This work is prototype as the hardware support is not available yet.
2. [Int8 Quantized Training](https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training): We're trying out full int8 training. This is easy to use with `quantize_(model, int8_weight_only_quantized_training())`. This work is prototype as the memory benchmarks are not compelling yet.
3. [IntX](https://github.com/pytorch/ao/tree/main/torchao/dtypes/uintx): We've managed to support all the ints by doing some clever bitpacking in pure PyTorch and then compiling it. This work is prototype as unfortunately without some more investment in either the compiler or low-bit kernels, int4 is more compelling than any smaller dtype
4. [Bitnet](https://github.com/pytorch/ao/blob/main/torchao/prototype/dtypes/bitnet.py): Mostly this is very cool to people on the team. This is prototype because how useful these kernels are is highly dependent on better hardware and kernel support.

## Installation

`torchao` makes liberal use of several new features in Pytorch, it's recommended to use it with the current nightly or latest stable version of PyTorch.

Stable release from Pypi which will default to CUDA 12.4

```Shell
pip install torchao
```

Stable Release from the PyTorch index
```Shell
pip install torchao --extra-index-url https://download.pytorch.org/whl/cu124 # full options are cpu/cu118/cu124/cu126
```

Nightly Release
```Shell
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126 # full options are cpu/cu118/cu126/cu128
```

For *most* developers you probably want to skip building custom C++/CUDA extensions for faster iteration

```Shell
USE_CPP=0 pip install -e .
```

## OSS Integrations

We're also fortunate to be integrated into some of the leading open-source libraries including
1. Hugging Face transformers with a [builtin inference backend](https://huggingface.co/docs/transformers/main/quantization/torchao) and [low bit optimizers](https://github.com/huggingface/transformers/pull/31865)
2. Hugging Face diffusers best practices with torch.compile and torchao in a standalone repo [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao)
3. Mobius HQQ backend leveraged our int4 kernels to get [195 tok/s on a 4090](https://github.com/mobiusml/hqq#faster-inference)
4. [TorchTune](https://github.com/pytorch/torchtune) for our QLoRA and QAT recipes
5. [torchchat](https://github.com/pytorch/torchchat) for post training quantization
6. SGLang for LLM serving: [usage](https://github.com/sgl-project/sglang/blob/4f2ee48ed1c66ee0e189daa4120581de324ee814/docs/backend/backend.md?plain=1#L83) and the major [PR](https://github.com/sgl-project/sglang/pull/1341).

## Videos
* [Keynote talk at GPU MODE IRL](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
* [Low precision dtypes at PyTorch conference](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
* [Slaying OOMs at the Mastering LLM's course](https://www.youtube.com/watch?v=UvRl4ansfCg)
* [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
* [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
* [Cohere for AI community talk](https://www.youtube.com/watch?v=lVgrE36ZUw0)


## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.

# Citation

If you find the torchao library useful, please cite it in your work as below.

```bibtex
@software{torchao,
  title = {torchao: PyTorch native quantization and sparsity for training and inference},
  author = {torchao maintainers and contributors},
  url = {https://github.com/pytorch/torchao},
  license = {BSD-3-Clause},
  month = oct,
  year = {2024}
}
```
