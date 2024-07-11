# torchao: PyTorch Architecture Optimization

[![](https://dcbadge.vercel.app/api/server/cudamode?style=flat)](https://discord.gg/cudamode)

[Introduction](#introduction) | [Inference](#inference) | [Training](#training) | [Dtypes](#newer-dtypes) | [Composability](#composability) | [Installation](#installation) |  [Community Contributions](#community-contributions) | [How to contribute](#how-to-contribute)

## Introduction

torchao is a library to create and integrate high-performance custom data types, layouts and kernels into your PyTorch workflows with up to **2x speedups** with **65% less VRAM** for [inference](#inference) and support for [training](#training)

All with no intrusive code changes and minimal accuracy degradation.

## Benchmarks

### Inference

#### Without intrusive code changes

Quantizing your models is a 1 liner that should work on any model with an `nn.Linear` including your favorite HuggingFace model. You can find a more comprehensive usage instructions [here](torchao/quantization/) and a HuggingFace inference example [here](scripts/hf_eval.py)

```python
from torchao.quantization.quant_api import quantize, int4_weight_only
m = quantize(m, int4_weight_only())
```

Benchmarks are run on a machine with a single A100 GPU using the script in `_models/llama` which generates text in a latency-optimized way (batchsize=1)

The models used were `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Meta-Llama-3-8B`.

| Model       | Technique          | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ------------------ | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (bfloat16)    | 12.212              |  105.14       | 1389.35                 | 13.88            | 13.21           |
|             | int8dq             | 12.262              |    9.20       |   60.93                 |  8.33            |  6.62           |
|             | int8wo             | 12.204              |  150.18       |  994.40                 |  8.95            |  6.62           |
|             | int4wo-64          | 12.843              |  199.86       |  746.66                 |  4.50            |  3.74           |
|             | int4wo-64-GPTQ     | 12.489              |  199.86       |  746.66                 |  4.50            |  3.74           |
|             | autoquant          | 12.204              |  159.22       | 1069.87                 |  8.91            |  6.72           |
| Llama-3-8B  | Base (bfloat16)    | N/A                 |   94.97       | 1425.55                 | 16.43            | 15.01           |
|             | int8dq             | N/A                 |    8.44       |   63.45                 |  8.98            |  7.52           |
|             | int8wo             | N/A                 |  139.76       | 1051.02                 | 10.42            |  7.52           |
|             | int4wo-64          | N/A                 |  179.44       |  757.60                 |  6.62            |  4.22           |
|             | autoquant          | N/A                 |  137.71       | 1037.74                 | 11.08            |  7.54           |

note: Int8 dynamic quantization works best on compute bound as opposed to memory bound models. Some relatable examples might be [SAM](https://github.com/pytorch-labs/segment-anything-fast) which is compute bound vs Llama at batchsize=1 which is memory bound.

For int4 we make heavy use of [tinygemm](https://github.com/pytorch/ao/blob/cb3bd8c674f2123af232a0231b5e38ddafa756a8/torchao/dtypes/aqt.py#L526) of `torch.ops.aten._weight_int4pack_mm` to bitpack into a layout optimized for tensor cores

And a quick crash course on inference quantization to help parse the above table. Int4 quantization is an ambiguous term because there's the dtype in which a layer is represented and then the dtype in which the computation is done. For example, if you're using Weight-Only (wo) int4 quantization that means that the layer will be upcasted to a larger dtype like fp16 so an int4 matrix multiplication is defined as `F.linear(input, weight.to(input.dtype))`. Dynamic quantization (DQ) primarily targets activations, enabling on-the-fly quantization from higher precision formats like bf16 to lower precision formats such as int8. This process, when supported by hardware, allows for direct computation, such as performing `F.linear(input, weight)`. Naive quantization algorithms are also notoriously sensitive to outliers so we also typically set a group size that applies a scale factor per group of 64 elements in the case of `int4wo64`.


#### With intrusive code changes

In some cases we rewrote popular GenAI models to be significantly faster in native PyTorch as in no C++/CUDA to achieve at the time SOTA inference performance. These involve more intrusive code changes.

* 9.5x speedups for Image segmentation models with [sam-fast](https://pytorch.org/blog/accelerating-generative-ai) compared to vanilla [sam](https://github.com/facebookresearch/segment-anything).
* 1.16x speedup when composing int8 quantization with 2:4 sparsity against the accelerated baseline `bfloat16` dtype and `torch.compile="max_autotune"`.

| Model Type | Technique                                                                                            | img/s | memory (MiB) | mIoU (coco2017 val) | relative speedup | relative accuracy |
|------------|------------------------------------------------------------------------------------------------------|-------|--------------|---------------------|------------------|-------------------|
| ViT-h      | sam (float32, eager)                                                                                 |  2.78 | 28806        | 0.58                | baseline         | baseline          |
|            | sam (bfloat16, eager)                                                                                | 14.85 | 14424        | 0.58                | **5.34x**        | **100%**          |
|            | sam-fast (bfloat16, max-autotune)                                                                    | 22.75 | 15172        | 0.58                | **8.18x**        | **100%**          |
|            | int8 dynamic quant (attn + mlp)                                                                      | 24.91 | 15154        | 0.58                | **8.96x**        | **100%**          |
|            | 2:4 sparsity (mlp only)                                                                              | 24.81 | 15632        | 0.57                | **8.92x**        | **98%**           |
|            | int8 dynamic quant (attn)<br>int8 dynamic quant + 2:4 sparsity (mlp lin1)<br>2:4 sparsity (mlp lin2) | 26.46 | 14865        | 0.57                | **9.52x**        | **98%**           |

The relative speedup is measured purely across the image encoder (ViT) of the model, where we apply our model optimizations. Benchmarks ran on an NVIDIA-A100-80GB with batch_size=32

* 10x speedups for Language models with [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2)
* 3x speedup for Diffusion models with [sd-fast](https://pytorch.org/blog/accelerating-generative-ai-3)

### Training

We've added support for semi-structured 2:4 sparsity with 6% end to end speedups on ViT-L

The code change is a 1 liner with the full example available [here](torchao/sparsity/training/)

```python
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

## Newer dtypes

* [MX](torchao/prototype/mx_formats) implementing training and inference support with tensors using the [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) data types, which can be described as groupwise scaled float8/float6/float4/int8, with the scales being constrained to powers of two. This work is prototype as the hardware support is not available yet.
* [nf4](torchao/dtypes/nf4tensor.py) which was used to [implement QLoRA](https://github.com/pytorch/torchtune/blob/main/docs/source/tutorials/qlora_finetune.rst) one of the most popular finetuning algorithms without writing custom Triton or CUDA code. Accessible talk [here](https://x.com/HamelHusain/status/1800315287574847701)
* [fp6](torchao/prototype/quant_llm/) for 2x faster inference over fp16 with an easy to use API `quantize(model, fp6_llm_weight_only())`

## Composability

A key design principle for us is composability as in any new dtype or layout we provide needs to work with `torch.compile()` and needs to work with `FSDP`. It shouldn't matter if the kernels are written in pure PyTorch, CUDA, C++, or Triton - things should just work! And here is our current strategy
1. Write the dtype, layout or bit packing logic in pure PyTorch and code-generate efficient kernels with torch.compile. You can inspect those kernels with `TORCH_LOGS="output_code" python your_code.py` and check if a single kernel is being generated and if any unnecessary buffers are being created
2. However once you get a kernel, how do you know how good it is? The best way is to benchmark the compiler generated code with the best kernel on the market. But packaging custom CPP/CUDA kernels that work on multiple devices is tedious but we've abstracted all the tedium from you with our [custom ops support](./torchao/csrc/) so if you love writing kernels but hate packaging, we'd love to accept contributions for your custom ops. One key benefit is a kernel written as a custom op will just work with no graph breaks with `torch.compile()`. Compilers are great at optimizations like fusions and overhead reduction but it's challenging for compilers to rewrite the math of an algorithm such that it's faster but also numerically stable so we are betting on both compilers and custom ops
3. Finally while historically most quantization has been done for inference, there is now a thriving area of research combining distributed algorithms and quantization. One popular example is [NF4](torchao/dtypes/nf4tensor.py) which was used to implement the QLoRA algorithm. The NF4 tensor also contains semantics for how it should be sharded over multiple devices so it composes with FSDP. We gave an accessible talk on [how to do this](https://x.com/HamelHusain/status/1800315287574847701).


### Installation

`torchao` makes liberal use of several new features in Pytorch, it's recommended to use it with the current nightly or latest stable version of PyTorch.

#### Install torch

Install torch stable

```
pip install torch
```

Or torch nightlies

```
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

#### Install torchao

Stable release from Pypi which will default to CUDA 12.1

```Shell
pip install torchao
```

Stable Release from the PyTorch index
```Shell
pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121 # full options are cpu/cu118/cu121/cu124
```

Nightly Release
```Shell
pip install --pre torchao-nightly --index-url https://download.pytorch.org/whl/nightly/cu121 # full options are cpu/cu118/cu121/cu124
```

From source
```Shell
git clone https://github.com/pytorch/ao
cd ao
python setup.py install
```

## Community Contributions

* [jeromeku](https://github.com/jeromeku) has implemented
    * [GaLore](torchao/prototype/galore/) a drop for the Adam Optimizer that allows you to finetune llama 7b on a single 4090 card with up to 70% speedups relative to eager PyTorch
    * [DoRA](torchao/prototype/dora) a newer replacement for QLoRA with more promising convergence characteristics
    * [Fused int4/fp16 Quant Matmul](torchao/prototype/hqq) which is particularly useful for compute bound kernels showing 4x speedups over tinygemm for larger batch sizes such as 512
* [gau-nernst](https://github.com/gau-nernst) fp6 kernels that are 4x faster than fp16 [torchao/prototype/quant_llm](torchao/prototype/quant_llm)
* [vayuda](https://github.com/vayuda) with generic bitpacking kernels that were code generated using pure PyTorch [prototype/common](torchao/prototype/common)
* [andreaskopf](https://github.com/andreaskoepf) and [melvinebenezer](https://github.com/melvinebenezer) with [1 bit LLMs](torchao/prototype/dtypes) Bitnet 1.58 bitpacked into uint2 and fully code-generated with torch.compile

## Blogs and Videos
* [Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity](https://pytorch.org/blog/accelerating-neural-network-training/)
* [https://mobiusml.github.io/whisper-static-cache-blog/](https://mobiusml.github.io/whisper-static-cache-blog/)
* [Slaying OOMs at the Mastering LLM's course](https://x.com/HamelHusain/status/1800315287574847701)
* [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
* [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)

## How to contribute

This repository is currently under heavy development
* If you have suggestions on the API or use cases you'd like to be covered, please open an [issue](https://github.com/pytorch/ao/issues)
* If you'd like to co-develop the library with us please join us on #torchao on [discord.gg/cudamode](https://discord.gg/cudamode) - there are a lot of dtypes out there and we could use a lot more hands to make them go brrr

If you're contributing a feature to ao
```Shell
pip install -r dev-requirements.txt
python setup.py develop
```

For *most* developers you probably want to skip building custom C++/CUDA extensions for faster iteration

```shell
USE_CPP=0 python setup.py install
```

## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
