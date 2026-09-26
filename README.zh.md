<div align="center">

# TorchAO

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

</div>

### PyTorch 原生全流程模型架构优化库（从训练到推理部署）
- 利用 Float8 低精度训练将 Llama-3.1-70B 预训练速度提升 **1.5 倍**
- 通过 QAT（量化感知训练）在 Gemma3-4B 上挽回 **67% 的量化精度损失**
- 将 Llama-3-8B 权重量化为 Int4，实现 **1.89 倍**推理提速并**降低 58% 显存占用**

<div align="center">

[![](https://img.shields.io/badge/CodeML_%40_ICML-2025-blue)](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf)
[![](https://dcbadge.vercel.app/api/server/gpumode?style=flat&label=TorchAO%20in%20GPU%20Mode)](https://discord.com/channels/1189498204333543425/1205223658021458100)
[![](https://img.shields.io/github/contributors-anon/pytorch/ao?color=yellow&style=flat-square)](https://github.com/pytorch/ao/graphs/contributors)
[![](https://img.shields.io/badge/torchao-documentation-blue?color=DE3412)](https://docs.pytorch.org/ao/stable/index.html)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)

[最新动态](#-最新动态) | [项目概览](#-项目概览) | [快速上手](#-快速上手)  | [安装指南](#-安装指南) | [生态集成](#-生态集成) | [推理部署](#-推理部署) | [训练优化](#-训练优化) | [视频讲座](#-视频讲座) | [引用本项目](#-引用本项目)

</div>


## 📣 最新动态

- [2025/10] QAT 现已深度集成至 [Unsloth](https://docs.unsloth.ai/new/quantization-aware-training-qat)，支持全量微调与 LoRA 微调！可通过[此 Colab 笔记本](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_%284B%29_Instruct-QAT.ipynb)即刻体验。
- [2025/10] MXFP8 MoE 训练原型在 Llama4 Scout 的 MoE 层实现了 **~1.45 倍加速**，在 DeepSeekV3 671B 的 MoE 层实现了 **~1.25 倍加速**——数值表现与 bfloat16 完全对齐！查阅[说明文档](./torchao/prototype/moe_training/)体验试用。
- [2025/09] MXFP8 训练在 Crusoe B200 集群上实现 [1.28 倍训练提速](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)，损失收敛曲线与 bfloat16 几乎完全一致！
- [2025/09] [TorchAO 量化模型与量化配方现已正式发布于 Huggingface Hub](https://pytorch.org/blog/torchao-quantized-models-and-quantization-recipes-now-available-on-huggingface-hub/)！
- [2025/06] 我们的 [TorchAO 论文](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf) 被 CodeML @ ICML 2025 录用！


<details>
  <summary>查看更早的历史动态</summary>

- [2025/05] QAT 现已集成至 [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) 微调框架（[查看文档](https://docs.axolotl.ai/docs/qat.html)）！
- [2025/04] Float8 按行（Rowwise）训练在 2,000 张 H100 GPU 规模下带来 [1.34-1.43 倍的训练加速](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)。
- [2025/04] TorchAO 正式作为[量化后端接入 vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html)（[查看文档](https://docs.vllm.ai/en/latest/features/quantization/torchao.html)）！
- [2025/03] 我们的 [2:4 稀疏化论文](https://openreview.net/pdf?id=O5feVk7p6Y) 被 SLLM @ ICLR 2025 录用！
- [2025/01] 结合 GemLite 与 SGLang 的优化方案在不同 Batch Size 和张量并行尺寸下，通过 Int4 和 Float8 量化实现了 1.1-2 倍的推理提速。
- [2025/01] 为线性层与 Embedding 算子新增了 [1-8 bit ARM CPU 高性能算子内核](https://pytorch.org/blog/hi-po-low-bit-operators/)。
- [2024/11] 在 Llama-3.1-70B 和 405B 上借助 Float8 训练实现了 [1.43-1.51 倍的预训练提速](https://pytorch.org/blog/training-using-float8-fsdp2/)。
- [2024/10] TorchAO 正式作为量化后端接入 Hugging Face Transformers！
- [2024/09] TorchAO 正式开源发布。查看我们的[官方博客](https://pytorch.org/blog/pytorch-native-architecture-optimization/)！
- [2024/07] QAT 在 Llama-3-8B 上[挽回了高达 96% 因量化导致的精度下降](https://pytorch.org/blog/quantization-aware-training/)。
- [2024/06] 半结构化 2:4 稀疏化在 SAM 和 ViT 模型上分别实现了 1.1 倍的推理提速和 1.3 倍的训练提速。
- [2024/06] 块稀疏化（Block Sparsity）在 ViT 模型上实现了 1.46 倍的训练提速，精度损失小于 2%。

</details>


## 🌅 项目概览

TorchAO 是一个专为 PyTorch 原生生态打造、极简易用的模型架构优化与量化库。TorchAO 开箱支持 `torch.compile()` 与 `FSDP2` 分布式训练，并广泛兼容绝大多数 HuggingFace PyTorch 模型。

如需详细了解面向不同硬件平台与数据类型的稳定版及原型优化工作流，请参阅[工作流指南文档](https://docs.pytorch.org/ao/main/workflows/index.html)。

更多详细技术细节请参阅[官方文档](https://docs.pytorch.org/ao/main/)！

## 🚀 快速上手

首先安装 TorchAO，推荐安装最新的稳定发布版本：
```bash
pip install torchao
# 可选 - 安装用于 float8 和 nvfp4 推理算子内核的 MSLK
pip install mslk --index-url https://download.pytorch.org/whl/cu130
# 可选 - 安装用于 mxfp8 MoE 训练内核的 apache-tvm-ffi 与 cutedsl
pip install apache-tvm-ffi
pip install nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2 nvidia-cutlass-dsl-libs-cu13==4.5.2
```

将你的模型权重一键量化至 Int4：
```python
import torch
from torchao.quantization import Int4WeightOnlyConfig, quantize_
quantize_(model, Int4WeightOnlyConfig(group_size=32, int4_packing_format="tile_packed_to_4d", int4_choose_qparams_algorithm="hqq"))
```
更多进阶范例请查阅[快速入门指南](https://docs.pytorch.org/ao/stable/quick_start.html)。

## 🛠 安装指南

安装最新稳定版本：
```bash
pip install torchao
```

<details>
  <summary>其他安装选项（Nightly、特定 CUDA、开发模式等）</summary>

  ```bash
  # Nightly 每日构建版
  pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128

  # 指定不同 CUDA 版本
  pip install torchao --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
  pip install torchao --index-url https://download.pytorch.org/whl/cu129  # CUDA 12.9
  pip install torchao --index-url https://download.pytorch.org/whl/xpu    # XPU
  pip install torchao --index-url https://download.pytorch.org/whl/cpu    # 纯 CPU 环境

  # 开发者源码可编辑安装
  # 注意：必须传入 `--no-build-isolation` 参数
  USE_CUDA=1 pip install -e . --no-build-isolation
  USE_XPU=1 pip install -e . --no-build-isolation
  USE_CPP=0 pip install -e . --no-build-isolation
  ```

</details>

关于各底层依赖的版本要求，请参阅 [TorchAO 兼容性矩阵](https://github.com/pytorch/ao/issues/2919)。

### 可选依赖组件

[MSLK](https://github.com/meta-pytorch/mslk) 是一个可选的运行时加速依赖，为 TorchAO 中的部分核心工作流提供高性能算子。稳定版 TorchAO 请搭配稳定版 MSLK，Nightly 版 TorchAO 搭配 Nightly 版 MSLK：
```bash
# 稳定版
pip install mslk --index-url https://download.pytorch.org/whl/cu130

# Nightly 构建版
pip install --pre mslk --index-url https://download.pytorch.org/whl/nightly/cu130
```

`apache-tvm-ffi` 与 `nvidia-cutlass-dsl` 用于 MoE 架构下的 MXFP8 训练算子内核：

```bash
pip install apache-tvm-ffi
pip install nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2 nvidia-cutlass-dsl-libs-cu13==4.5.2
```

## 🔎 推理部署

TorchAO 仅需极少量的代码修改即可带来显著的性能提升：

- **Int4 仅权重量化 (Int4 weight-only)**：在 H100 上对 Gemma3-12b-it 实现了 [1.73 倍加速并减少 65% 内存占用](https://huggingface.co/pytorch/gemma-3-12b-it-INT4)，对精度影响微乎其微。
- **Float8 动态量化 (Float8 dynamic quantization)**：在 H100 上对 gemma-3-27b-it 带来 [1.5-1.6 倍提速](https://huggingface.co/pytorch/gemma-3-27b-it-FP8/blob/main/README.md#results-h100-machine)，并在保持生成质量的同时对 Flux.1-Dev 和 CogVideoX-5b 分别实现 [1.54 倍和 1.27 倍加速](https://github.com/sayakpaul/diffusers-torchao)。
- **Int8 激活量化与 Int4 权重量化**：通过 [ExecuTorch](https://huggingface.co/pytorch/Qwen3-4B-INT8-INT4#running-in-a-mobile-app) 在 iPhone 15 Pro 手机端运行量化后的 Qwen3-4B，推理速度达 14.8 tokens/s，内存占用仅 3379 MB。

以下是我们推荐的模型量化与部署调用流：
```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

# 1. 创建量化配置
quantization_config = TorchAoConfig(quant_type=Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

# 2. 加载并自动量化
quantized_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

若上述高级接口在特定模型上受限，可使用[快速入门指南](https://docs.pytorch.org/ao/main/quick_start.html)中介绍的原生 `quantize_` API。

在单张 H100 机器上使用 vLLM 进行服务化部署：
```shell
# 启动服务端
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve pytorch/Qwen3-32B-FP8 --tokenizer Qwen/Qwen3-32B -O3
```

```shell
# 客户端测试调用
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "pytorch/Qwen3-32B-FP8",
  "messages": [
    {"role": "user", "content": "请简要介绍大语言模型。"}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 32768
}'
```

针对扩散生成模型，可通过 Hugging Face diffusers 进行量化：

```python
import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig
from torchao.quantization.granularity import PerGroup

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig(granularity=PerGroup(128)))}
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

我们还支持通过 ExecuTorch 部署至移动端与边缘设备，更多细节请查阅[量化与部署指南](https://docs.pytorch.org/ao/main/serving.html)。我们还在 [Hugging Face 组织主页](https://huggingface.co/pytorch) 上直接提供了预量化好的模型。

## 🚅 训练优化

### 量化感知训练 (QAT)

训练后量化（PTQ）能够产出高紧凑、高吞吐的模型，但可能带来精度下降。我们建议探索量化感知训练（QAT）以克服此瓶颈，尤其是在 Int4 等超低位宽场景下。通过与 [TorchTune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat) 的紧密合作，我们开发出表现卓越的 QAT 配方，相较传统 PTQ 显著挽回了精度损失：在 Llama3 上**挽回了 Hellaswag 任务上 96% 的精度降幅，并恢复了 Wikitext 上 68% 的困惑度损失**。更多细节请参考 [QAT 文档](torchao/quantization/qat/README.md) 与[官方技术博客](https://pytorch.org/blog/quantization-aware-training/)：

```python
import torch
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig, PerGroup
from torchao.quantization.qat import QATConfig

# 1. 准备阶段 (Prepare)
base_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),
)
quantize_(my_model, QATConfig(base_config, step="prepare"))

# 2. 模型训练微调阶段 (代码略)

# 3. 转换导出阶段 (Convert)
quantize_(my_model, QATConfig(base_config, step="convert"))
```

用户还可以将 LoRA 与 QAT 深度融合，利用此[分布式微调配方](https://github.com/pytorch/torchtune/blob/main/recipes/qat_lora_finetune_distributed.py)相较原生 QAT 获得 [1.89 倍的训练提速](https://dev-discuss.pytorch.org/t/speeding-up-qat-by-1-89x-with-lora/2700)。


### 低精度量化训练 (Quantized training)

[torchao.float8](torchao/float8) 基于论文 https://arxiv.org/abs/2209.05433 实现了带缩放因子的 Float8 低精度训练方案。在开启 `torch.compile` 的前提下，实测在**高达 512 张 GPU / 405B 参数量级规模**上带来了高达 **1.5 倍的训练吞吐提速**（[技术细节](https://pytorch.org/blog/training-using-float8-fsdp2/)）：

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m)
```

我们的 Float8 训练能力已内置集成至 [TorchTitan 预训练管线](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md) 中，便于开发者开箱试用。如需了解更多细节，请参阅关于 Float8 训练支持的官方博客系列：
* [在 Crusoe 2K H200 集群上使用 PyTorch Float8 Rowwise 加速大规模训练与收敛](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)
* [使用 Float8 与 FSDP2 为模型训练全面加速](https://pytorch.org/blog/training-using-float8-fsdp2/)
* [在 Amazon SageMaker 上利用 TorchTitan 高效预训练类 Llama 3 架构大模型](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
* [PyTorch 中的 Float8 技术架构解析](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)

<details>
  <summary>更多特性（显存高效优化器）</summary>

### 显存高效优化器 (Memory-efficient optimizers)

类似 ADAM 的优化器可能消耗巨大的 GPU 显存——往往达到模型参数本身显存占用的 2 倍。TorchAO 提供了两种方案来有效降低此开销：

**1. 低位宽量化优化器**：通过量化优化器状态将状态显存占用削减 2-4 倍：

```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # 亦可替换为 4bit 或 fp8 版本的 AdamW4bit / AdamWFp8
```
我们的量化优化器完全由数百行纯 PyTorch 代码实现并通过编译达到极高效率。尽管略逊于深度手写特化算子内核，但其在显存节省与运行速度之间达成了绝佳平衡。详见 [基准测试与说明](https://github.com/pytorch/ao/tree/main/torchao/optim)。

**2. CPU 内存卸载 (CPU offloading)**：将优化器状态和梯度卸载至系统内存

为了榨干每一寸显存，我们支持[单 GPU CPU 内存卸载技术](https://github.com/pytorch/ao/tree/main/torchao/optim#optimizer-cpu-offload)，高效地将梯度与优化器状态动态卸载至系统 CPU 内存。该方案在对训练速度影响极小的情况下，**可降低高达 60% 的显存占用需求**：

```python
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

</details>


## 🔗 生态集成

TorchAO 已深度集成至众多业界顶尖的开源项目与框架中：

* **Unsloth** 现已全面支持 QAT：[阅读博客](https://docs.unsloth.ai/new/quantization-aware-training-qat) 与 [操作指南](https://docs.unsloth.ai/new/quantization-aware-training-qat#qat--lora-finetuning)。
* **HuggingFace Transformers**：提供[内置推理后端](https://huggingface.co/docs/transformers/main/quantization/torchao)与[低位宽优化器](https://github.com/huggingface/transformers/pull/31865)。
* **HuggingFace Diffusers**：在独立仓库 [diffusers-torchao](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/torchao.md) 中提供基于 `torch.compile` 与 TorchAO 的[最佳实践](https://huggingface.co/docs/diffusers/main/en/quantization/torchao)。
* **vLLM** 大模型服务框架：[使用指南](https://docs.vllm.ai/en/latest/features/quantization/torchao.html) 与 [详细文档](https://docs.pytorch.org/ao/main/torchao_vllm_integration.html)。
* 与 [MSLK](https://github.com/meta-pytorch/MSLK) 深度协同，为服务器 GPU 提供 SOTA 高性能算子支持。
* 与 [ExecuTorch](https://github.com/pytorch/executorch/) 深度协同，实现边缘与移动设备部署。
* **Axolotl**：支持基于 TorchAO 的 [QAT](https://docs.axolotl.ai/docs/qat.html) 与 [PTQ](https://docs.axolotl.ai/docs/quantize.html)。
* **TorchTitan**：内置支持 [Float8 预训练](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)。
* **HuggingFace PEFT**：在 LoRA 微调中使用 TorchAO 作为[底层量化后端](https://huggingface.co/docs/peft/en/developer_guides/quantization#torchao-pytorch-architecture-optimization)。
* **TorchTune**：基于 TorchAO 构建 NF4 [QLoRA](https://docs.pytorch.org/torchtune/main/tutorials/qlora_finetune.html)、[QAT](https://docs.pytorch.org/torchtune/main/recipes/qat_distributed.html) 以及 [Float8 量化微调](https://github.com/pytorch/torchtune/pull/2546) 配方。
* **SGLang** 大模型服务框架：[使用指南](https://docs.sglang.ai/advanced_features/quantization.html#online-quantization)。

## 🎥 视频讲座

* [GPU MODE IRL 峰会主题演讲](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
* [PyTorch 大会低精度数据类型专题分享](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
* [Mastering LLM 课程：解决大模型显存 OOM 难题](https://www.youtube.com/watch?v=UvRl4ansfCg)
* [CUDA MODE：进阶大模型量化技术精讲](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
* [Chip Huyen 主持的 GPU 优化前沿研讨会](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
* [Cohere for AI 社区技术讲座](https://www.youtube.com/watch?v=lVgrE36ZUw0)


## 💬 引用本项目

如果您在学术研究或项目中使用了 TorchAO，请按照如下格式引用：

```bibtex
@misc{or2025torchao,
  title={TorchAO: PyTorch-Native Training-to-Serving Model Optimization},
  author={Andrew Or and Apurva Jain and Daniel Vega-Myhre and Jesse Cai and Charles David Hernandez and Zhenrui Zheng and Driss Guessous and Vasiliy Kuznetsov and Christian Puhrsch and Mark Saroufim and Supriya Rao and Thien Tran and Aleksandar Samardžić},
  year={2025},
  eprint={2507.16099},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.16099},
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
