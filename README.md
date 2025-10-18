<div align="center">

# TorchAO

</div>

### PyTorch-Native Training-to-Serving Model Optimization
- Pre-train Llama-3.1-70B **1.5x faster** with float8 training
- Recover **67% of quantized accuracy degradation** on Gemma3-4B with QAT
- Quantize Llama-3-8B to int4 for **1.89x faster** inference with **58% less memory**

<div align="center">

[![](https://img.shields.io/badge/CodeML_%40_ICML-2025-blue)](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf)
[![](https://dcbadge.vercel.app/api/server/gpumode?style=flat&label=TorchAO%20in%20GPU%20Mode)](https://discord.com/channels/1189498204333543425/1205223658021458100)
[![](https://img.shields.io/github/contributors-anon/pytorch/ao?color=yellow&style=flat-square)](https://github.com/pytorch/ao/graphs/contributors)
[![](https://img.shields.io/badge/torchao-documentation-blue?color=DE3412)](https://docs.pytorch.org/ao/stable/index.html)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)

[Latest News](#-latest-news) | [Overview](#-overview) | [Quick Start](#-quick-start)  | [Installation](#-installation) | [Integrations](#-integrations) | [Inference](#-inference) | [Training](#-training) | [Videos](#-videos) | [Citation](#-citation)

</div>


## 📣 Latest News

- [Oct 20] MXFP8 MoE training prototype achieved **~1.45x speedup** for MoE layer in Llama4 Scout, and **~1.25x** speedup for MoE layer in DeepSeekV3 671b - with comparable numerics to bfloat16! Check out the [docs](./torchao/prototype/moe_training/) to try it out.
- [Sept 25] MXFP8 training achieved [1.28x speedup on Crusoe B200 cluster](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/) with virtually identical loss curve to bfloat16!
- [Sept 19] [TorchAO Quantized Model and Quantization Recipes Now Available on Huggingface Hub](https://pytorch.org/blog/torchao-quantized-models-and-quantization-recipes-now-available-on-huggingface-hub/)!
- [Jun 25] Our [TorchAO paper](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf) was accepted to CodeML @ ICML 2025!
- [May 25] QAT is now integrated into [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning ([docs](https://docs.axolotl.ai/docs/qat.html))!
- [Apr 25] Float8 rowwise training yielded [1.34-1.43x training speedup](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) at 2k H100 GPU scale
- [Apr 25] TorchAO is added as a [quantization backend to vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html) ([docs](https://docs.vllm.ai/en/latest/features/quantization/torchao.html))!

<details>
  <summary>Older news</summary>

- [Mar 25] Our [2:4 Sparsity paper](https://openreview.net/pdf?id=O5feVk7p6Y) was accepted to SLLM @ ICLR 2025!
- [Jan 25] Our [integration with GemLite and SGLang](https://pytorch.org/blog/accelerating-llm-inference/) yielded 1.1-2x faster inference with int4 and float8 quantization across different batch sizes and tensor parallel sizes
- [Jan 25] We added [1-8 bit ARM CPU kernels](https://pytorch.org/blog/hi-po-low-bit-operators/) for linear and embedding ops
- [Nov 24] We achieved [1.43-1.51x faster pre-training](https://pytorch.org/blog/training-using-float8-fsdp2/) on Llama-3.1-70B and 405B using float8 training
- [Oct 24] TorchAO is added as a quantization backend to HF Transformers!
- [Sep 24] We officially launched TorchAO. Check out our blog [here](https://pytorch.org/blog/pytorch-native-architecture-optimization/)!
- [Jul 24] QAT [recovered up to 96% accuracy degradation](https://pytorch.org/blog/quantization-aware-training/) from quantization on Llama-3-8B
- [Jun 24] Semi-structured 2:4 sparsity [achieved 1.1x inference speedup and 1.3x training speedup](https://pytorch.org/blog/accelerating-neural-network-training/) on the SAM and ViT models respectively
- [Jun 24] Block sparsity [achieved 1.46x training speeedup](https://pytorch.org/blog/speeding-up-vits/) on the ViT model with <2% drop in accuracy

</details>


## 🌅 Overview

TorchAO is an easy to use quantization library for native PyTorch. TorchAO works out-of-the-box with `torch.compile()` and `FSDP2` across most HuggingFace PyTorch models. Key features include:
* Float8 [training](torchao/float8/README.md) and [inference](https://docs.pytorch.org/ao/main/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html) for speedups without compromising accuracy
* [MX training and inference](torchao/prototype/mx_formats/README.md), provides MX tensor formats based on native PyTorch MX dtypes (prototype)
* [Low precision MoE training](torchao/prototype/moe_training/README.md) provides training speedups with comparable numerics to bfloat16 training.
* [Quantization-Aware Training (QAT)](torchao/quantization/qat/README.md) for mitigating quantization degradation
* [Post-Training Quantization (PTQ)](torchao/quantization/README.md) for int4, int8, fp6 etc, with matching kernels targeting a variety of backends including CUDA, ARM CPU, and XNNPACK
* [Sparsity](torchao/sparsity/README.md), includes different techniques such as 2:4 sparsity and block sparsity

Check out our [docs](https://docs.pytorch.org/ao/main/) for more details!

## 🚀 Quick Start

First, install TorchAO. We recommend installing the latest stable version:
```bash
pip install torchao
```

Quantize your model weights to int4!
```python
from torchao.quantization import Int4WeightOnlyConfig, quantize_
quantize_(model, Int4WeightOnlyConfig(group_size=32, int4_packing_format="tile_packed_to_4d", int4_choose_qparams_algorithm="hqq"))
```
See our [quick start guide](https://docs.pytorch.org/ao/stable/quick_start.html) for more details.

## 🛠 Installation

To install the latest stable version:
```bash
pip install torchao
```

<details>
  <summary>Other installation options</summary>

  ```
  # Nightly
  pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128

  # Different CUDA versions
  pip install torchao --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
  pip install torchao --index-url https://download.pytorch.org/whl/cu129  # CUDA 12.9
  pip install torchao --index-url https://download.pytorch.org/whl/cpu    # CPU only

  # For developers
  # Note: the `--no-build-isolation` flag is required.
  USE_CUDA=1 pip install -e . --no-build-isolation
  USE_CPP=0 pip install -e . --no-build-isolation
  ```

</details>

Please see the [torchao compability table](https://github.com/pytorch/ao/issues/2919) for version requirements for dependencies.

## 🔗 Integrations

TorchAO is integrated into some of the leading open-source libraries including:

* Unsloth for QAT, blog post coming soon!
* HuggingFace transformers with a [builtin inference backend](https://huggingface.co/docs/transformers/main/quantization/torchao) and [low bit optimizers](https://github.com/huggingface/transformers/pull/31865)
* HuggingFace diffusers best practices with `torch.compile` and TorchAO in a standalone repo [diffusers-torchao](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/torchao.md)
* vLLM for LLM serving: [usage](https://docs.vllm.ai/en/latest/features/quantization/torchao.html), [detailed docs](https://docs.pytorch.org/ao/main/torchao_vllm_integration.html)
* Integration with [FBGEMM](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai) for SOTA kernels on server GPUs
* Integration with [ExecuTorch](https://github.com/pytorch/executorch/) for edge device deployment
* Axolotl for [QAT](https://docs.axolotl.ai/docs/qat.html) and [PTQ](https://docs.axolotl.ai/docs/quantize.html)
* TorchTitan for [float8 pre-training](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)
* HuggingFace PEFT for LoRA using TorchAO as their [quantization backend](https://huggingface.co/docs/peft/en/developer_guides/quantization#torchao-pytorch-architecture-optimization)
* TorchTune for our NF4 [QLoRA](https://docs.pytorch.org/torchtune/main/tutorials/qlora_finetune.html), [QAT](https://docs.pytorch.org/torchtune/main/recipes/qat_distributed.html), and [float8 quantized fine-tuning](https://github.com/pytorch/torchtune/pull/2546) recipes
* SGLang for LLM serving: [usage](https://docs.sglang.ai/advanced_features/quantization.html#online-quantization)

## 🔎 Inference

TorchAO delivers substantial performance gains with minimal code changes:

- **Int4 weight-only**: [1.73x speedup with 65% less memory](https://huggingface.co/pytorch/gemma-3-12b-it-INT4) for Gemma3-12b-it on H100 with slight impact on accuracy
- **Float8 dynamic quantization**: [1.5-1.6x speedup on gemma-3-27b-it](https://huggingface.co/pytorch/gemma-3-27b-it-FP8/blob/main/README.md#results-h100-machine) and [1.54x and 1.27x speedup on Flux.1-Dev* and CogVideoX-5b respectively](https://github.com/sayakpaul/diffusers-torchao) on H100 with preserved quality
- **Int8 activation quantization and int4 weight quantization**: Quantized Qwen3-4B running with 14.8 tokens/s with 3379 MB memory usage on iPhone 15 Pro through [ExecuTorch](https://huggingface.co/pytorch/Qwen3-4B-INT8-INT4#running-in-a-mobile-app)
- **Int4 + 2:4 Sparsity**: [2.37x throughput with 67.7% memory reduction](torchao/sparsity/README.md) on Llama-3-8B

Following is our recommended flow for quantization and deployment:
```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

# Create quantization configuration
quantization_config = TorchAoConfig(quant_type=Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

# Load and automatically quantize
quantized_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

Alternative quantization API to use when the above doesn't work is `quantize_` API in [quick start guide](https://docs.pytorch.org/ao/main/quick_start.html).

Serving with vllm on 1xH100 machine:
```shell
# Server
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve pytorch/Qwen3-32B-FP8 --tokenizer Qwen/Qwen3-32B -O3
```

```shell
# Client
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "pytorch/Qwen3-32B-FP8",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 32768
}'
```

We also support deployment to edge devices through ExecuTorch, for more detail, see [quantization and serving guide](https://docs.pytorch.org/ao/main/serving.html). We also release pre-quantized models [here](https://huggingface.co/pytorch).

## 🚅 Training

### Quantization-Aware Training

Post-training quantization can result in a fast and compact model, but may also lead to accuracy degradation. We recommend exploring Quantization-Aware Training (QAT) to overcome this limitation, especially for lower bit-width dtypes such as int4. In collaboration with [TorchTune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat), we've developed a QAT recipe that demonstrates significant accuracy improvements over traditional PTQ, recovering **96% of the accuracy degradation on hellaswag and 68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization (PTQ). For more details, please refer to the [QAT README](torchao/quantization/qat/README.md) and the [original blog](https://pytorch.org/blog/quantization-aware-training/):

```python
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig
from torchao.quantization.qat import QATConfig

# prepare
base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
quantize_(my_model, QATConfig(base_config, step="prepare"))

# train model (not shown)

# convert
quantize_(my_model, QATConfig(base_config, step="convert"))
```

Users can also combine LoRA + QAT to speed up training by [1.89x](https://dev-discuss.pytorch.org/t/speeding-up-qat-by-1-89x-with-lora/2700) compared to vanilla QAT using this [fine-tuning recipe](https://github.com/pytorch/torchtune/blob/main/recipes/qat_lora_finetune_distributed.py).


### Quantized training

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

<details>
  <summary>Other features (sparse training, memory efficient optimizers)</summary>

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

</details>

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
  url={https://github.com/pytorch/ao},
  license={BSD-3-Clause},
  month={oct},
  year={2024}
}
```
