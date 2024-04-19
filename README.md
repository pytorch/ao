# torchao: PyTorch Architecture Optimization

**Note: This repository is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an github issue**

## Introduction

torchao is a PyTorch native library for optimizing your models using lower precision dtypes, techniques like quantization and sparsity and performant kernels.

## Our Goals
torchao embodies PyTorch’s design philosophy [details](https://pytorch.org/docs/stable/community/design.html), especially "usability over everything else". Our vision for this repository is the following:

* Composability: Native solutions for optimization techniques that compose with both `torch.compile` and `FSDP` 
    * For example, for QLoRA for new dtypes support
* Interoperability: Work with the rest of the PyTorch ecosystem such as torchtune, gpt-fast and ExecuTorch
* Transparent Benchmarks: Regularly run performance benchmarking of our APIs across a suite of Torchbench models and across hardware backends
* Heterogeneous Hardware: Efficient kernels that can run on CPU/GPU based server (w/ torch.compile) and mobile backends (w/ ExecuTorch).
* Infrastructure Support: Release packaging solution for kernels and a CI/CD setup that runs these kernels on different backends. 

## Key Features
The library provides
1. Support for lower precision [dtypes](./torchao/dtypes) such as nf4, uint4 that are torch.compile friendly
2. [Quantization algorithms](./torchao/quantization) such as dynamic quant, smoothquant, GPTQ that run on CPU/GPU and Mobile.
  * Int8 dynamic activation quantization
  * Int8 and int4 weight-only quantization
  * Int8 dynamic activation quantization with int4 weight quantization
  * [GPTQ](https://arxiv.org/abs/2210.17323) and [Smoothquant](https://arxiv.org/abs/2211.10438)
  * High level `autoquant` API and kernel auto tuner targeting SOTA performance across varying model shapes on consumer/enterprise GPUs.
3. [Sparsity algorithms](./torchao/sparsity) such as Wanda that help improve accuracy of sparse networks
4. Integration with other PyTorch native libraries like [torchtune](https://github.com/pytorch/torchtune) and [ExecuTorch](https://github.com/pytorch/executorch)

## Interoperability with PyTorch Libraries

torchao has been integrated with other repositories to ease usage

* [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md) is integrated with 8 and 4 bit weight-only quantization techniques with and without GPTQ.
* [Executorch](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#quantization) is integrated with GPTQ for both 8da4w (int8 dynamic activation, with int4 weight) and int4 weight only quantization.

## Success stories
Our kernels have has been used to achieve SOTA inference performance on

1. Image segmentation models with [sam-fast](pytorch.org/blog/accelerating-generative-ai)
2. Language models with [gpt-fast](pytorch.org/blog/accelerating-generative-ai-2)
3. Diffusion models with [sd-fast](pytorch.org/blog/accelerating-generative-ai-3)


## Installation

**Note: this library makes liberal use of several new features in pytorch, its recommended to use it with the current pytorch nightly if you want full feature coverage. If not, the subclass APIs may not work, though the module swap api's will still work.**

1. From PyPI:
```Shell
pip install torchao
```

2. From Source:

```Shell
git clone https://github.com/pytorch-labs/ao
cd ao
pip install -e .
```

## Get Started
To try out our APIs, you can check out API examples in [quantization](./torchao/quantization) (including `autoquant`), [sparsity](./torchao/sparsity), [dtypes](./torchao/dtypes).

## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
