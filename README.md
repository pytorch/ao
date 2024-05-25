# torchao: PyTorch Architecture Optimization

[![](https://dcbadge.vercel.app/api/server/cudamode?style=flat)](https://discord.gg/cudamode)

This repository is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an [issue](https://github.com/pytorch/ao/issues)

## Introduction
`torchao` is a PyTorch library for quantization and sparsity.

## Get Started

### Installation
`torchao` makes liberal use of several new features in pytorch, it's recommended to use it with the current nightly or latest stable version of PyTorch.

Stable Release
```Shell
pip install torchao
```

Nightly Release
```Shell
pip install torchao-nightly
```

From source

```Shell
git clone https://github.com/pytorch/ao
cd ao
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

There are two options;
-If you plan to be developing the library run:
```Shell
python setup.py develop
```

If you want to install from source run
```Shell
python setup.py install
```

** Note:
Since we are building pytorch c++/cuda extensions by default, running `pip install .` will
not work.

### Quantization

```python
import torch
import torchao

# inductor settings which improve torch.compile performance for quantized modules
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True

# Plug in your model and example input
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
input = torch.randn(32,32, dtype=torch.bfloat16, device='cuda')

# perform autoquantization and compilation
q_model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
q_model(input)
```

### Sparsity

```python
import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.ao.pruning import WeightNormSparsifier

# bfloat16 CUDA model
model = torch.nn.Sequential(torch.nn.Linear(64, 64)).cuda().to(torch.bfloat16)

# Accuracy: Finding a sparse subnetwork
sparse_config = []
for name, mod in model.named_modules():
   if isinstance(mod, torch.nn.Linear):
      sparse_config.append({"tensor_fqn": f"{name}.weight"})

sparsifier = WeightNormSparsifier(sparsity_level=1.0,
                                 sparse_block_shape=(1,4),
                                 zeros_per_block=2)

# attach FakeSparsity
sparsifier.prepare(model, sparse_config)
sparsifier.step()
sparsifier.squash_mask()
# now we have dense model with sparse weights

# Performance: Accelerated sparse inference
for name, mod in model.named_modules():
   if isinstance(mod, torch.nn.Linear):
      mod.weight = torch.nn.Parameter(to_sparse_semi_structured(mod.weight))
```

To learn more try out our APIs, you can check out API examples in
* [quantization](./torchao/quantization)
* [sparsity](./torchao/sparsity)
* [dtypes](./torchao/dtypes)


## Supported Features
1. [Quantization algorithms](./torchao/quantization)
    - [Int8 weight-only](https://github.com/pytorch/ao/blob/main/torchao/quantization/weight_only.py) quantization
    - [Int4 weight-only](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu) quantization
    - [GPTQ](https://github.com/pytorch/ao/blob/main/torchao/quantization/GPTQ.py) and [Smoothquant](https://github.com/pytorch/ao/blob/main/torchao/quantization/smoothquant.py) for low latency inference
    - High level [torchao.autoquant API](https://github.com/pytorch/ao/blob/main/torchao/quantization/autoquant.py) and [kernel autotuner](https://github.com/pytorch/ao/blob/main/torchao/kernel/autotuner.py) targeting SOTA performance across varying model shapes on consumer and enterprise GPUs
2. [Sparsity algorithms](./torchao/sparsity) such as Wanda that help improve accuracy of sparse networks
3. Support for lower precision [dtypes](./torchao/dtypes) such as
    - [nf4](https://github.com/pytorch/ao/blob/main/torchao/dtypes/nf4tensor.py) which was used to [implement QLoRA](https://github.com/pytorch/torchtune/blob/main/docs/source/tutorials/qlora_finetune.rst) without writing custom Triton or CUDA code
    - [uint4](https://github.com/pytorch/ao/blob/main/torchao/dtypes/uint4.py)
4. [Bleeding Edge Kernels](./torchao/prototype/) for experimental kernels without backwards compatibility guarantees
    - [GaLore](https://github.com/pytorch/ao/tree/main/torchao/prototype/galore) for memory efficient finetuning
    - [fused HQQ Gemm Kernel](https://github.com/pytorch/ao/tree/main/torchao/prototype/hqq) for compute bound workloads

## Our Goals

* Composability with `torch.compile`: We rely heavily on `torch.compile` to write pure PyTorch code and codegen efficient kernels. There are however limits to what a compiler can do so we don't shy away from writing our custom CUDA/Triton kernels
* Composability with `FSDP`: The new support for FSDP per parameter sharding means engineers and researchers alike can experiment with different quantization and distributed strategies concurrently.
* Performance: We measure our performance on every commit using an A10G. We also regularly run performance benchmarks on the [torchbench](https://github.com/pytorch/benchmark) suite
* Heterogeneous Hardware: Efficient kernels that can run on CPU/GPU based server (w/ torch.compile) and mobile backends (w/ ExecuTorch).
* Packaging kernels should be easy: We support custom [CUDA and Triton extensions](./torchao/csrc/) so you can focus on writing your kernels and we'll ensure that they work on most operating systems and devices

## Integrations

torchao has been integrated with other libraries including

* [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md) leverages our 8 and 4 bit weight-only quantization techniques with optional support for GPTQ
* [Executorch](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#quantization) leverages our GPTQ implementation for both 8da4w (int8 dynamic activation with int4 weight) and int4 weight-only quantization.
* [HQQ](https://github.com/mobiusml/hqq/blob/master/hqq/backends/torchao.py) leverages our int4mm kernel for low latency inference

## Success stories
Our kernels have been used to achieve SOTA inference performance on

* Image segmentation models with [sam-fast](https://pytorch.org/blog/accelerating-generative-ai)
* Language models with [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2)
* Diffusion models with [sd-fast](https://pytorch.org/blog/accelerating-generative-ai-3)

## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
