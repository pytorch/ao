# torchao: PyTorch Architecture Optimization

[![](https://dcbadge.vercel.app/api/server/gpumode?style=flat&label=TorchAO%20in%20GPU%20Mode)](https://discord.com/channels/1189498204333543425/1205223658021458100)


[Introduction](#introduction) | [Inference](#inference) | [Training](#training)  | [Installation](#installation) |[Composability](#composability) | [Prototype Features](#prototype-features)  | [Integrations](#integrations) | [Videos](#videos) | [For Developers](#for-developers) | [License](#license) | [Citation](#citation)

## Introduction

`torchao` accelerates PyTorch models with minimal code changes through advanced quantization and sparsification techniques. Optimize weights, gradients, activations, and more for both inference and training.

From the team that brought you the fast series
* 9.5x inference speedups for Image segmentation models with [sam-fast](https://pytorch.org/blog/accelerating-generative-ai)
* 10x inference speedups for Language models with [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2)
* 3x inference speedup for Diffusion models with [sd-fast](https://pytorch.org/blog/accelerating-generative-ai-3)

`torchao` isn't just for inference - it delivers substantial speedups at scale, from [up to 1.5x speedups](https://pytorch.org/blog/training-using-float8-fsdp2/) on 512 GPU clusters, to [1.34-1.43x speedups](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) on 2K H200 clusters with the latest `torchao.float8` rowwise

`torchao` works out-of-the-box with `torch.compile()` and `FSDP2` across most Hugging Face PyTorch models


## Inference

`torchao` delivers substantial performance gains with minimal code changes:

### Performance Highlights

- **INT4 Weight-Only Quantization**: 2x throughput (180 vs 107 tokens/sec) with 60% less memory (6.88 GB vs 16.43 GB) on LLaMA-3-7B
- **Float8 Dynamic Quantization**: 53.88% speedup on Flux.1-Dev* and 27.33% speedup on CogVideoX-5b on H100 GPU with preserved quality
- **INT4 + 2:4 Sparsity**: 2.4x throughput (226 vs 95 tokens/sec) with 80% memory reduction (5.3GB vs 16.4GB) on LLaMA-3-8B

[View detailed benchmarks](torchao/quantization/README.md) |  [Learn about sparsity](torchao/sparsity/README.md)

### Getting Started with Quantization

Quantize any model with `nn.Linear` layers (including HuggingFace models) in just one line:

#### Option 1: Direct TorchAO API

```python
from torchao.quantization.quant_api import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig(group_size=128, use_hqq=True))
```

#### Option 2: HuggingFace Integration

```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization.quant_api import Int4WeightOnlyConfig

# Create quantization configuration
quantization_config = TorchAoConfig(quant_type=Int4WeightOnlyConfig(group_size=128, use_hqq=True))

# Load and automatically quantize
quantized_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

### Deployment with vLLM

Deploy quantized models with one command:

```shell
vllm serve pytorch/Phi-4-mini-instruct-int4wo-hqq --tokenizer microsoft/Phi-4-mini-instruct -O3
```

**Benefits**: 67% VRAM reduction and 12-20% speedup on A100 GPUs while maintaining quality.

[Step-by-step quantization guide](https://huggingface.co/pytorch/Phi-4-mini-instruct-int4wo-hqq#quantization-recipe) | [Pre-quantized models](https://huggingface.co/pytorch)

## Training

### Quantization Aware Training

Post-training quantization can result in a fast and compact model, but may also lead to accuracy degradation. We recommend exploring Quantization Aware Training (QAT) to overcome this limitation. In collaboration with [Torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat), we've developed a QAT recipe that demonstrates significant accuracy improvements over traditional PTQ, recovering **96% of the accuracy degradation on hellaswag and 68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization (PTQ). And we've provided a full recipe [here](https://pytorch.org/blog/quantization-aware-training/). For more details, please see the [QAT README](./torchao/quantization/qat/README.md).

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

[torchao.float8](torchao/float8) implements training recipes with the scaled float8 dtypes, as laid out in https://arxiv.org/abs/2209.05433

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
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear

swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-efficient optimizers

Optimizers like ADAM can consume substantial GPU memory - 2x as much as the model parameters themselves. TorchAO provides two approaches to reduce this overhead:

1. **Quantized optimizers**: Reduce optimizer state memory by 2-4x by quantizing to lower precision


```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
```
Our quantized optimizers are implemented in just a few hundred lines of PyTorch code and compiled for efficiency. While slightly slower than specialized kernels, they offer an excellent balance of memory savings and performance. See detailed [benchmarks here](https://github.com/pytorch/ao/tree/main/torchao/optim).

2. **CPU offloading**: Move optimizer state and gradients to CPU memory

For maximum memory savings, we support [single GPU CPU offloading](https://github.com/pytorch/ao/tree/main/torchao/optim#optimizer-cpu-offload) that efficiently moves both gradients and optimizer state to CPU memory. This approach can **reduce your VRAM requirements by 60%** with minimal impact on training speed:

```python
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

## Installation

`torchao` makes liberal use of several new features in PyTorch, it's recommended to use it with the current nightly or latest stable version of PyTorch, see [getting started](https://pytorch.org/get-started/locally/) for more details.

Install the stable release (recommended):
```bash
pip install torchao
```

Other options:
```bash
# Nightly build
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu124

# Different CUDA versions
pip install torchao --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torchao --index-url https://download.pytorch.org/whl/cpu    # CPU only

```

### Development Install
```
USE_CPP=0 python setup.py develop  # Skip C++/CUDA extensions
```

## Composability
`torch.compile`: A key design principle for us is composability - any custom dtype or memory layout should work with our compiler. We enable kernel implementations in PyTorch, CUDA, C++, or Triton. This allows researchers and engineers to start with high-level dtype and layout logic in pure PyTorch, then progressively optimize performance by implementing lower-level kernels as needed, while maintaining compatibility with the compile infrastructure.

[FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md): Historically most quantization has been done for inference, there is now a thriving area of research combining distributed algorithms and quantization.

The best example we have combining the composability of lower bit dtype with compile and fsdp is [NF4](torchao/dtypes/nf4tensor.py) which we used to implement the [QLoRA](https://www.youtube.com/watch?v=UvRl4ansfCg) algorithm. So if you're doing research at the intersection of this area we'd love to hear from you.

Our framework makes it straightforward to add tensor parallel support to your custom quantized tensor subclass. Check out our [tensor parallel tutorial](tutorials/developer_api_guide/tensor_parallel.py) to see how a quantized tensor subclass can be extended to support column and row-wise tensor sharding while maintaining compatibility with `torch.compile`.

## Prototype Features

The [prototype](torchao/prototype/README.md) directory contains experimental and upcoming features including:

- [MX Training & Inference](torchao/prototype/mx_formats/README.md): MX formats with a native PyTorch POC
- [Int8 Quantized Training](torchao/prototype/quantized_training/README.md): Low-precision training methods
- And more experimental features in development

These features are under active development and may change. We welcome contributions from researchers and developers!

> ⚠️ Note: Features in the prototype directory do not have BC guarantees and are subject to change.

## OSS Integrations

We're also fortunate to be integrated into some of the leading open-source libraries including
1. Hugging Face transformers with a [builtin inference backend](https://huggingface.co/docs/transformers/main/quantization/torchao) and [low bit optimizers](https://github.com/huggingface/transformers/pull/31865)
2. Hugging Face diffusers best practices with torch.compile and torchao in a standalone repo [diffusers-torchao](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/torchao.md)
3. Mobius HQQ backend leveraged our int4 kernels to get [195 tok/s on a 4090](https://github.com/mobiusml/hqq#faster-inference)
4. [TorchTune](https://pytorch.org/torchtune/main/tutorials/qlora_finetune.html?highlight=qlora) for our QLoRA and QAT recipes
5. VLLM for LLM serving: [usage](https://docs.vllm.ai/en/latest/features/quantization/torchao.html)
6. SGLang for LLM serving: [usage](https://docs.sglang.ai/backend/server_arguments.html#server-arguments) and the major [PR](https://github.com/sgl-project/sglang/pull/1341).
7. Axolotl for [Quantization Aware Training](https://docs.axolotl.ai/docs/qat.html) and [post-training quantization](https://docs.axolotl.ai/docs/quantize.html)

## Videos
* [Keynote talk at GPU MODE IRL](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
* [Low precision dtypes at PyTorch conference](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
* [Slaying OOMs at the Mastering LLM's course](https://www.youtube.com/watch?v=UvRl4ansfCg)
* [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
* [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
* [Cohere for AI community talk](https://www.youtube.com/watch?v=lVgrE36ZUw0)


## For Developers

### Custom Kernels

We've added support for authoring and releasing [custom ops](./torchao/csrc/) that do not graph break with `torch.compile()`. We have a few examples you can follow

1. [fp6](torchao/dtypes/floatx/README.md) for 2x faster inference over fp16 with an easy to use API `quantize_(model, fpx_weight_only(3, 2))`
2. [2:4 Sparse Marlin GEMM](https://github.com/pytorch/ao/pull/733) 2x speedups for FP16xINT4 kernels even at batch sizes up to 256
3. [int4 tinygemm unpacker](https://github.com/pytorch/ao/pull/415) which makes it easier to switch quantized backends for inference

If you believe there's other CUDA kernels we should be taking a closer look at please leave a comment on [this issue](https://github.com/pytorch/ao/issues/697) or feel free to contribute directly to the repo.


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
