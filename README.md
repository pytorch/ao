# torchao: PyTorch Architecture Optimization

[![](https://dcbadge.vercel.app/api/server/cudamode?style=flat)](https://discord.gg/cudamode)


## Introduction

torchao is a library which makes it easy to integrate and create high performance kernels with custom data types and layouts with up to
* **30% speedups** for training
* **2x speedups** with **65%** less VRAM for inference

All with no intrusive code changes and minimal accuracy degradation.

## Benchmarks

### Inference

#### Without intrusive code changes

Quantizing your models is a 1 liner that should work on any model with `nn.Linear` including your favorite HuggingFace model. You can find a more comprehensive usage example [here](torchao/quantization/)

```python
from torchao.quantization.quant_api import quantize
m = quantize(m, "int4wo")
```

Benchmarks are run on a machine with a single A100 GPU using the script in `_models/llama` which generates text in a latency-optimized way (batchsize=1)

The models used were `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Meta-Llama-3-8B`.

| Model       | Technique          | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ------------------ | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (bfloat16)    | 12.212              |  105.02       | 1387.78                 | 13.21            | 13.90           |
|             | int8dq             | 12.262              |  9.40         | 62.26                   | 6.62             | 8.61            |
|             | int8wo             | 12.204              |  147.03       | 973.54                  | 6.62             | 8.95            |
|             | int4wo-64          | 12.843              |  199.81       | 746.45                  | 3.74             | 4.75            |
|             | int4wo-64-GPTQ     | 12.489              |  199.81       | 746.45                  | 3.74             | 4.75            |
| Llama-3-8B  | Base (bfloat16)    | N/A                 |  94.91        | 1424.58                 | 15.01            | 16.43           |
|             | int8dq             | N/A                 |  8.41         | 63.23                   | 7.52             | 9.24            |
|             | int8wo             | N/A                 |  136.75       | 1028.38                 | 7.52             | 10.42           |
|             | int4wo-64          | N/A                 |  179.41       | 757.45                  | 4.22             | 6.88            |

note: Int8 dynamic quantization works best on compute bound models like [SAM](https://github.com/pytorch-labs/segment-anything-fast) whereas Llama with batchsize=1 tends to be memory bound, thus the rather low performance.

And a quick crash course on inference quantization to help parse the above table. Int4 quantization is an ambiguous term because there's the dtype in which a layer is represented and then the dtype in which the computation is done. For example, if you're using Weight-Only (wo) int4 quantization that means that the layer will be upcasted to a larger dtype like fp16 so an int4 matrix multiplication is defined as `F.linear(input, weight.to(input.dtype))` whereas if it's possible to perform the computation using the smaller dtype directly pending support by a hardware vendor then that means you can perform `F.linear(input, weight)` directly and this is what we refer to as Dynamic-Quantization (dq). Naive quantization algorithms are also notoriously sensitive to outliers so we also typically set a group size that applies a scale factor per group of 64 elements in the case of `int4wo64`.


#### With intrusive code changes

In some cases we rewrote popular GenAI models to be significantly faster in native PyTorch as in no C++/CUDA to achieve at the time SOTA inference performance. These involve more intrusive code changes.

* 8x speedups for Image segmentation models with [sam-fast](https://pytorch.org/blog/accelerating-generative-ai)
* 10x speedups for Language models with [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2)
* 3x speedup for Diffusion models with [sd-fast](https://pytorch.org/blog/accelerating-generative-ai-3)

### Training

We've added support for semi-structured 2:4 sparsity with over 30% speedups on ViT-L

The code change is a 1 liner with the full example available [here](torchao/sparsity/training/)


```python
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

For VIT-L MLP shapes on a NVIDIA A100 we see the following results:

|                     |   act24   |   dense   |   w24    | s24_inp_sparsify24 | s24_inp_clone |
|---------------------|-----------|-----------|----------|--------------------|---------------|
| f16 (44160,1024,4096,1024) |  11881.0  |  11534.3  |  9204.7  |        255.1        |      125.8    |

Times are in microseconds (us).

## Newer dtypes

[MX](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats) implementing training and inference support with tensors using the [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) data types, which can be described as groupwise scaled float8/float6/float4/int8, with the scales being constrained to powers of two. This work is prototype as the hardware support is not available yet.

[nf4](https://github.com/pytorch/ao/blob/main/torchao/dtypes/nf4tensor.py) which was used to [implement QLoRA](https://github.com/pytorch/torchtune/blob/main/docs/source/tutorials/qlora_finetune.rst) one of the most popular finetuning algorithms without writing custom Triton or CUDA code. Accessible talk [here](https://x.com/HamelHusain/status/1800315287574847701)

[tinygemm](https://github.com/pytorch/ao/blob/cb3bd8c674f2123af232a0231b5e38ddafa756a8/torchao/dtypes/aqt.py#L526) we make heavy use of `torch.ops.aten._weight_int4pack_mm` to bitpack into a layout optimized for tensor cores

## Composability

A key design principle for us is composability as in any new dtype or layout we provide needs to work with `torch.compile()` and needs to work with `FSDP`. It shouldn't matter if the kernels are written are pure PyTorch, CUDA, C++, or Triton - things should just work! And here is our current strategy
1. Write the dtype, layout or bit packing logic in pure PyTorch and code-generate efficient kernels with torch.compile. You can inspect those kernels with `TORCH_LOGS="output_code" python your_code.py` and check if a single kernel is being generated and if any unnecessary buffers are being created
2. However once you get a kernel, how do you know how good it is? The best way is to benchmark the code-generated code with the best kernel on the market. But packaging custom CPP/CUDA kernels that work on multiple devices is tedious but we've abstracted all the tedium from you with our [custom ops support](./torchao/csrc/) so if you love writing kernels but hate packaging, we'd love to accept contributions for your custom ops. One key benefit is a kernel written as a custom op will just work with no graph breaks with `torch.compile()`. Compilers are great at optimizations like fusions and overhead reduction but it's challenging for compilers to rewrite the math of an algorithm such that it's faster but also numerically stable so we are betting on both compilers and custom ops
3. Finally while historically most quantization has been done for inference there is now a thriving area of research combining lower dtypes and sharding. One popular example is [NF4](torchao/dtypes/nf4tensor.py) which is used to create the QLoRA algorithm and you can define the semantics for how custom tensors should be sharded over multiple devices. We gave an accessible talk on [how to do this](https://x.com/HamelHusain/status/1800315287574847701).

## Get Started

### Installation
`torchao` makes liberal use of several new features in Pytorch, it's recommended to use it with the current nightly or latest stable version of PyTorch.

Stable Release
```Shell
pip install torchao --extra-index-url https://download.pytorch.org/whl/test/cu121 # full options are cpu/cu118/cu121/cu124
```

Nightly Release
```Shell
pip install --pre torchao-nightly --index-url https://download.pytorch.org/whl/nightly/cu121 # full options are cpu/cu118/cu121/cu124
```

## Community Contributions

* [jeromeku](https://github.com/jeromeku) has implemented
    * [GaLore](torchao/prototype/galore/) a drop for the Adam Optimizer that allows you to finetune llama 7b on a single 4090 card with up to 70% speedups relative to eager PyTorch
    * [DoRA](torchao/prototype/dora) a newer replacement for QLoRA with more promising convergence characteristics
    * [Fused int4/fp16 Quant Matmul](torchao/prototype/hqq) which is particularly useful for compute bound kernels showing 4x speedups over tinygemm for larger batch sizes such as 512
* [gau-nernst](https://github.com/gau-nernst) fp6 kernels that are 4x faster than fp16 [torchao/prototype/fp6_llm](torchao/prototype/fp6_llm)
* [vayuda](https://github.com/vayuda) with generic bitpacking kernels that were code generated using pure PyTorch [prototype/common](torchao/prototype/common)
* [andreaskopf](https://github.com/andreaskoepf) and [melvinebenezer](https://github.com/melvinebenezer) with [1 bit LLMs](torchao/prototype/dtypes) Bitnet 1.58 bitpacked into uin2 and fully code-generated with torch.compile

## How to contribute

This repository is currently under heavy development
* If you have suggestions on the API or use cases you'd like to be covered, please open an [issue](https://github.com/pytorch/ao/issues)
* If you'd like to co-develop the library with us please join us on #torchao on [discord.gg/cudamode](https://discord.gg/cudamode) - there are a lot of dtypes out there and we could use a lot more hands to make them go brrr

Installation instructions

```Shell
git clone https://github.com/pytorch/ao
cd ao
python setup.py install
```

If you're contributing a feature ao
```Shell
pip install -r dev-requirements.txt
python setup.py develop
```

For *most* developers you probably want to skip building custom C++/CUDA extensions for faster iteration cycles

```shell
USE_CPP=0 python setup.py install
```

## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
