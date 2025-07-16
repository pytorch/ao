<div align="center">

# TorchAO

</div>

### PyTorch-Native Training-to-Serving Model Optimization
- Pre-train Llama-3.1-70B **1.5x faster** with float8 training
- Recover **77% of quantized perplexity degradation** on Llama-3.2-3B with QAT
- Quantize Llama-3-8B to int4 for **1.89x faster** inference with **58% less memory**

<div align="center">

[![](https://img.shields.io/badge/CodeML_%40_ICML-2025-blue)](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf)
[![](https://dcbadge.vercel.app/api/server/gpumode?style=flat&label=TorchAO%20in%20GPU%20Mode)](https://discord.com/channels/1189498204333543425/1205223658021458100)
[![](https://img.shields.io/github/contributors-anon/pytorch/ao?color=yellow&style=flat-square)](https://github.com/pytorch/ao/graphs/contributors)
[![](https://img.shields.io/badge/torchao-documentation-blue?color=DE3412)](https://docs.pytorch.org/ao/stable/index.html)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)

[Latest News](#-latest-news) | [Overview](#-overview) | [Quick Start](#-quick-start)  | [Integrations](#-integrations) | [Inference](#-inference) | [Training](#-training) | [Videos](#-videos) | [Citation](#-citation)

</div>


## 📣 Latest News

- [Jun 25] Our [TorchAO paper](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf) was accepted to CodeML @ ICML 2025!
- [May 25] QAT is now integrated into [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning ([docs](https://docs.axolotl.ai/docs/qat.html))!
- [Apr 25] Float8 rowwise training yielded [1.34-1.43x training speedup](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) at 2k H100 GPU scale
- [Apr 25] TorchAO is added as a [quantization backend to vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html) ([docs](https://docs.vllm.ai/en/latest/features/quantization/torchao.html))!
- [Mar 25] Our [2:4 Sparsity paper](https://openreview.net/pdf?id=O5feVk7p6Y) was accepted to SLLM @ ICLR 2025!
- [Jan 25] Our [integration with GemLite and SGLang](https://pytorch.org/blog/accelerating-llm-inference/) yielded 1.1-2x faster inference with int4 and float8 quantization across different batch sizes and tensor parallel sizes
- [Jan 25] We added [1-8 bit ARM CPU kernels](https://pytorch.org/blog/hi-po-low-bit-operators/) for linear and embedding ops

<details>
  <summary>Older news</summary>

- [Nov 24] We achieved [1.43-1.51x faster pre-training](https://pytorch.org/blog/training-using-float8-fsdp2/) on Llama-3.1-70B and 405B using float8 training
- [Oct 24] TorchAO is added as a quantization backend to HF Transformers!
- [Sep 24] We officially launched TorchAO. Check out our blog [here](https://pytorch.org/blog/pytorch-native-architecture-optimization/)!
- [Jul 24] QAT [recovered up to 96% accuracy degradation](https://pytorch.org/blog/quantization-aware-training/) from quantization on Llama-3-8B
- [Jun 24] Semi-structured 2:4 sparsity [achieved 1.1x inference speedup and 1.3x training speedup](https://pytorch.org/blog/accelerating-neural-network-training/) on the SAM and ViT models respectively
- [Jun 24] Block sparsity [achieved 1.46x training speeedup](https://pytorch.org/blog/speeding-up-vits/) on the ViT model with <2% drop in accuracy

</details>


## 🌅 Overview

TorchAO is a PyTorch-native model optimization framework leveraging quantization and sparsity to provide an end-to-end, training-to-serving workflow
for AI models. TorchAO works out-of-the-box with `torch.compile()` and `FSDP2` across most HuggingFace PyTorch models. Key features include:
* Float8 [training](torchao/float8/README.md) and [inference](https://docs.pytorch.org/ao/main/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html) for speedups without compromising accuracy
* [MX training and inference](torchao/prototype/mx_formats/README.md), provides MX tensor formats based on native PyTorch MX dtypes (prototype)
* [Quantization-Aware Training (QAT)](torchao/quantization/qat/README.md) for mitigating quantization degradation
* [Post-Training Quantization (PTQ)](torchao/quantization/README.md) for int4, int8, fp6 etc, with matching kernels targeting a variety of backends including CUDA, ARM CPU, and XNNPACK
* [Sparsity](torchao/sparsity/README.md), includes different techniques such as 2:4 sparsity and block sparsity

Check out our [docs](https://docs.pytorch.org/ao/main/) for more details!

From the team that brought you the fast series:
* 9.5x inference speedups for Image segmentation models with [sam-fast](https://pytorch.org/blog/accelerating-generative-ai)
* 10x inference speedups for Language models with [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2)
* 3x inference speedup for Diffusion models with [sd-fast](https://pytorch.org/blog/accelerating-generative-ai-3) (new: [flux-fast](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/))
* 2.7x inference speedup for FAIR’s Seamless M4T-v2 model with [seamlessv2-fast](https://pytorch.org/blog/accelerating-generative-ai-4/)


## 🚀 Quick Start

First, install TorchAO. We recommend installing the latest stable version:
```
pip install torchao
```

<details>
  <summary>Other installation options</summary>

  ```
  # Nightly
  pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126
  
  # Different CUDA versions
  pip install torchao --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
  pip install torchao --index-url https://download.pytorch.org/whl/cpu    # CPU only

  # For developers
  USE_CUDA=1 python setup.py develop
  ```

</details>

Quantize your model weights to int4!
```
from torchao.quantization import Int4WeightOnlyConfig, quantize_
quantize_(model, Int4WeightOnlyConfig(group_size=32))
```
Compared to a `torch.compiled` bf16 baseline, your quantized model should be significantly smaller and faster on a single A100 GPU:
```
int4 model size: 1.25 MB
bfloat16 model size: 4.00 MB
compression ratio: 3.2

bf16 mean time: 30.393 ms
int4 mean time: 4.410 ms
speedup: 6.9x
```
For the full model setup and benchmark details, check out our [quick start guide](https://docs.pytorch.org/ao/stable/quick_start.html). Alternatively, try quantizing your favorite model using our [HuggingFace space](https://huggingface.co/spaces/pytorch/torchao-my-repo)!


## 🔗 Integrations

TorchAO is integrated into some of the leading open-source libraries including:

* HuggingFace transformers with a [builtin inference backend](https://huggingface.co/docs/transformers/main/quantization/torchao) and [low bit optimizers](https://github.com/huggingface/transformers/pull/31865)
* HuggingFace diffusers best practices with `torch.compile` and TorchAO in a standalone repo [diffusers-torchao](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/torchao.md)
* Mobius HQQ backend leveraged our int4 kernels to get [195 tok/s on a 4090](https://github.com/mobiusml/hqq#faster-inference)
* TorchTune for our [QLoRA](https://docs.pytorch.org/torchtune/main/tutorials/qlora_finetune.html), [QAT](https://docs.pytorch.org/torchtune/main/recipes/qat_distributed.html), and [float8 quantized fine-tuning](https://github.com/pytorch/torchtune/pull/2546) recipes
* TorchTitan for [float8 pre-training](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)
* VLLM for LLM serving: [usage](https://docs.vllm.ai/en/latest/features/quantization/torchao.html), [detailed docs](https://docs.pytorch.org/ao/main/torchao_vllm_integration.html)
* SGLang for LLM serving: [usage](https://docs.sglang.ai/backend/server_arguments.html#server-arguments) and the major [PR](https://github.com/sgl-project/sglang/pull/1341).
* Axolotl for [QAT](https://docs.axolotl.ai/docs/qat.html) and [PTQ](https://docs.axolotl.ai/docs/quantize.html)


## 🔎 Inference

TorchAO delivers substantial performance gains with minimal code changes:

- **Int4 weight-only**: [1.89x throughput with 58.1% less memory](torchao/quantization/README.md) on Llama-3-8B
- **Float8 dynamic quantization**: [1.54x and 1.27x speedup on Flux.1-Dev* and CogVideoX-5b respectively](https://github.com/sayakpaul/diffusers-torchao) on H100 with preserved quality
- **Int4 + 2:4 Sparsity**: [2.37x throughput with 67.7% memory reduction](torchao/sparsity/README.md) on Llama-3-8B

Quantize any model with `nn.Linear` layers in just one line (Option 1), or load the quantized model directly from HuggingFace using our integration with HuggingFace transformers (Option 2):

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

#### Deploy quantized models in vLLM with one command:

```shell
vllm serve pytorch/Phi-4-mini-instruct-int4wo-hqq --tokenizer microsoft/Phi-4-mini-instruct -O3
```

With this quantization flow, we achieve **67% VRAM reduction and 12-20% speedup** on A100 GPUs while maintaining model quality. For more detail, see this [step-by-step quantization guide](https://huggingface.co/pytorch/Phi-4-mini-instruct-int4wo-hqq#quantization-recipe). We also release some pre-quantized models [here](https://huggingface.co/pytorch).

## 🚅 Training

### Quantization-Aware Training

Post-training quantization can result in a fast and compact model, but may also lead to accuracy degradation. We recommend exploring Quantization-Aware Training (QAT) to overcome this limitation, especially for lower bit-width dtypes such as int4. In collaboration with [TorchTune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat), we've developed a QAT recipe that demonstrates significant accuracy improvements over traditional PTQ, recovering **96% of the accuracy degradation on hellaswag and 68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization (PTQ). For more details, please refer to the [QAT README](torchao/quantization/qat/README.md) and the [original blog](https://pytorch.org/blog/quantization-aware-training/):

```python
from torchao.quantization import quantize_
from torchao.quantization.qat import FakeQuantizeConfig, IntXQuantizationAwareTrainingConfig
activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
weight_config = FakeQuantizeConfig(torch.int4, group_size=32)
qat_config = IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
quantize_(my_model, qat_config)
```

Users can also combine LoRA + QAT to speed up training by [1.89x](https://dev-discuss.pytorch.org/t/speeding-up-qat-by-1-89x-with-lora/2700) compared to vanilla QAT using this [fine-tuning recipe](https://github.com/pytorch/torchtune/blob/main/recipes/qat_lora_finetune_distributed.py).


### Float8

[torchao.float8](torchao/float8) implements training recipes with the scaled float8 dtypes, as laid out in https://arxiv.org/abs/2209.05433. With ``torch.compile`` on, current results show throughput speedups of up to **1.5x on up to 512 GPU / 405B parameter count scale** ([details](https://pytorch.org/blog/training-using-float8-fsdp2/)):

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m)
```

Our float8 training is integrated into [TorchTitan's pre-training flows](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md) so users can easily try it out. For more details, check out these blog posts about our float8 training support:
* [Accelerating Large Scale Training and Convergence with PyTorch Float8 Rowwise on Crusoe 2K H200s](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)
* [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
* [Efficient Pre-training of Llama 3-like model architectures using torchtitan on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
* [Float8 in PyTorch](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)


### Sparse Training

We've added support for semi-structured 2:4 sparsity with **6% end-to-end speedups on ViT-L**. Full blog [here](https://pytorch.org/blog/accelerating-neural-network-training/). The code change is a 1 liner with the full example available [here](torchao/sparsity/training/):

```python
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-efficient optimizers

Optimizers like ADAM can consume substantial GPU memory - 2x as much as the model parameters themselves. TorchAO provides two approaches to reduce this overhead:

**1. Quantized optimizers**: Reduce optimizer state memory by 2-4x by quantizing to lower precision

```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
```
Our quantized optimizers are implemented in just a few hundred lines of PyTorch code and compiled for efficiency. While slightly slower than specialized kernels, they offer an excellent balance of memory savings and performance. See detailed [benchmarks here](https://github.com/pytorch/ao/tree/main/torchao/optim).

**2. CPU offloading**: Move optimizer state and gradients to CPU memory

For maximum memory savings, we support [single GPU CPU offloading](https://github.com/pytorch/ao/tree/main/torchao/optim#optimizer-cpu-offload) that efficiently moves both gradients and optimizer state to CPU memory. This approach can **reduce your VRAM requirements by 60%** with minimal impact on training speed:

```python
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

<!--
## For Developers

### Composability
`torch.compile`: A key design principle for us is composability - any custom dtype or memory layout should work with our compiler. We enable kernel implementations in PyTorch, CUDA, C++, or Triton. This allows researchers and engineers to start with high-level dtype and layout logic in pure PyTorch, then progressively optimize performance by implementing lower-level kernels as needed, while maintaining compatibility with the compile infrastructure.

[FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md): Historically most quantization has been done for inference, there is now a thriving area of research combining distributed algorithms and quantization.

The best example we have combining the composability of lower bit dtype with compile and fsdp is [NF4](torchao/dtypes/nf4tensor.py) which we used to implement the [QLoRA](https://www.youtube.com/watch?v=UvRl4ansfCg) algorithm. So if you're doing research at the intersection of this area we'd love to hear from you.

Our framework makes it straightforward to add tensor parallel support to your custom quantized tensor subclass. Check out our [tensor parallel tutorial](tutorials/developer_api_guide/tensor_parallel.py) to see how a quantized tensor subclass can be extended to support column and row-wise tensor sharding while maintaining compatibility with `torch.compile`.

### Custom Kernels

We've added support for authoring and releasing [custom ops](./torchao/csrc/) that do not graph break with `torch.compile()`. We have a few examples you can follow

1. [fp6](torchao/dtypes/floatx/README.md) for 2x faster inference over fp16 with an easy to use API `quantize_(model, fpx_weight_only(3, 2))`
2. [2:4 Sparse Marlin GEMM](https://github.com/pytorch/ao/pull/733) 2x speedups for FP16xINT4 kernels even at batch sizes up to 256
3. [int4 tinygemm unpacker](https://github.com/pytorch/ao/pull/415) which makes it easier to switch quantized backends for inference

If you believe there's other CUDA kernels we should be taking a closer look at please leave a comment on [this issue](https://github.com/pytorch/ao/issues/697) or feel free to contribute directly to the repo.
-->


## 🎥 Videos
* [Keynote talk at GPU MODE IRL](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
* [Low precision dtypes at PyTorch conference](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
* [Slaying OOMs at the Mastering LLM's course](https://www.youtube.com/watch?v=UvRl4ansfCg)
* [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
* [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
* [Cohere for AI community talk](https://www.youtube.com/watch?v=lVgrE36ZUw0)


## 💬 Citation

If you find the torchao library useful, please cite it in your work as below.

<!-- TODO: update to cite CodeML paper after Jul 2025 -->
```bibtex
@software{torchao,
  title={TorchAO: PyTorch-Native Training-to-Serving Model Optimization},
  author={torchao},
  url={https://github.com/pytorch/torchao},
  license={BSD-3-Clause},
  month={oct},
  year={2024}
}
```
