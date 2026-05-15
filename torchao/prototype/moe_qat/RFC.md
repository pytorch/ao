# [RFC] MoE Quantization-Aware Training (QAT) in TorchAO

## Motivation

Quantization-aware training for MoE models is becoming a production standard.
**DeepSeek-V4** applies MXFP4 QAT to its MoE expert weights during post-training.
**Kimi K2 Thinking** employs INT4 QAT on its MoE components during post-training,
achieving lossless 2× speedup (as reported in the model card). There is also an open request for MXFP4 QAT support
for **GPT-OSS** (a 120B MoE model) in torchao
(issue [#3547](https://github.com/pytorch/ao/issues/3547)).

TorchAO's current QAT support partially addresses this need: it handles
`nn.Linear` layers (Q, K, V, O projections in attention) via module swap
(`FakeQuantizedLinear`). However, the MoE expert weights — where the majority
of parameters reside — are 3D `nn.Parameter` tensors routed through ops such
as `torch._grouped_mm`, `torch.bmm`, and `F.linear`, which fall entirely
outside the existing QAT path.

`torchao.prototype.moe_training` has already established a parameter-swap pattern
(tensor subclass on `param.data`) for low-precision MoE training that composes with
FSDP2, Expert Parallel, and `torch.compile`. This RFC proposes extending QAT to MoE
models using the same parameter-swap approach.

## Goal

Deliver MoE QAT support using the parameter-swap pattern from `moe_training`, starting with
FP8 row-wise weight-only fake quantization and designed for future extension to other precisions
and activation QAT.

### Features Plan

| Precision | Weight-only QAT | Activation+Weight QAT |
|-----------|----------------|----------------------|
| FP8 rowwise | Planned | Planned |
| Int4 weight-only | Planned | — |
| Int8 dynamic activation + IntX weight | Planned | Planned |
| MXFP8 / MXFP4 | Planned | Planned |
| NVFP4 | Planned | Planned |

Additionally, we plan to support a dual-precision QAT path: fake-quantize from the original
precision to an ultra-low precision (e.g., MXFP4) for storage, then dequantize to a low
precision (e.g., MXFP8) for computation. This decouples storage precision from compute
precision and enables reusing the existing low-precision training pipeline which is usually customized by users. This is inspired by **DeepSeek-V4**, where the path is FP32 → MXFP4 (1×32) → FP8 (128×128).

### Compatibility

- **FSDP2**, **Expert Parallel**, **torch.compile**: planned
- **Target hardware**: not hardware-specific. The current infrastructure is used for CPU, GPU, and XPU support. Ascend-NPU support is planned.
- **Target model**: not model-specific

## Design

### Architecture

Unlike dense layers which are `nn.Linear`, expert layers in MoE models
are user-defined classes:
- `Qwen3MoeExperts` in `unsloth/Qwen3-4B-instruct-25507`
- `GptOssExperts` in `openai/gpt-oss-20b`
- `DeepseekV4Experts` in `deepseek-ai/DeepSeek-V4-Flash` and `deepseek-ai/DeepSeek-V4-Pro`

To provide a model-agnostic MoE QAT support, we choose to adopt a parameter-swap
approach rather than the module-swap
approach used by dense QAT:

1. **Injection**: `QATConfig`'s handler swaps `nn.Linear` and `nn.Embedding` modules for
   their fake-quantized counterparts. For expert layers, the swapping logic is fundamentally
   different — it must operate at the parameter level (wrapping `nn.Parameter.data`) rather
   than the module level, since expert layer classes vary across models. We therefore define
   `MoEQATConfig` (extending `QATConfig`) with its own handler registered via
   `register_quantize_module_handler`.

2. **Fake quantization**: the wrapper subclass intercepts computation ops via
   `__torch_function__`, applies fake quantization, and delegates to the standard op.
   We reuse the existing `FakeQuantizeConfig` hierarchy (config types,
   `_infer_fake_quantize_configs`, and quantization primitives) but discard the
   `FakeQuantizer` module system — attaching an `nn.Module` to a tensor subclass
   is awkward and may cause incompatibilities with `torch.compile` and FSDP2.
   
   1. **Stateless**: FP8 row-wise, MX, and NVFP4. Fake quantization is a stateless
      function driven by config, stored as a static method on the subclass.

   2. **Stateful**: `IntxFakeQuantizeConfig` only (range learning). We have not yet
      found a clean design for this case — trainable states must be visible to
      `model.parameters()` so the optimizer can find and update them, but tensor subclass
      attributes are not reachable through the module hierarchy.
      A possible approach: store `nn.Parameter` states on the wrapper, register them
      on the parent module, and deregister on unwrap. This requires custom code for
      `torch.compile`, checkpointing, and FSDP2 compatibility.

   3. **Deferred**: slicing, indexing, and dimension-manipulation ops preserve the
      wrapper without applying fake quantization. Quantization is deferred until
      computation time, avoiding double quantization.

3. **Parameter-level traversal with handler dispatch**: `quantize_` passes each module
   that matches `filter_fn` to the handler of `MoEQATConfig`. The handler walks the
   module's parameters (including sub-modules), filters them via `config.params_filter_fn`,
   and replaces each matching `nn.Parameter` using a handler selected from a registry.
   Handlers for different `FakeQuantizeConfig` types are registered in a dictionary via a
   decorator. This two-level traversal pattern is reusable for MoE training, QAT, and
   inference.

4. **PTQ in the convert step**: applying PTQ during convert is closer to MoE inference
   quantization. A related RFC for MoE inference quantization
   ([#4355](https://github.com/pytorch/ao/issues/4355)) is under discussion in TorchAO.
   We will reuse that infrastructure if available. If not, MoE PTQ will be implemented
   first and reused in MoE QAT.



### Precision in the Forward Propagation

Dense QAT uses different forward-precision strategies for different configs, and we follow
the same choices:

- **Float8, Int4, IntX**: fully fake-quantized forward — quantize to low precision,
  dequantize back, run the matmul in high precision. STE with clamp.
- **MX, NVFP4**: real low-precision quantization and matmul in forward, dequantized
  high-precision matmul in backward. Clamp is bypassed in STE.

MoE QAT reuses the same underlying quantization primitives and follows the same
precision strategy per config type.


### Class Hierarchy

```
AOBaseConfig
└── QATConfig
    └── MoEQATConfig       [NEW]  — parameter-level prepare/convert for MoE

TorchAOBaseTensor
└── FakeQuantizedWeightWrapperBaseTensor     [NEW]  — shared plumbing
    ├── Float8FakeQuantizedWeightWrapperTensor  [NEW]  — FP8 row-wise
    ├── Int4FakeQuantizedWeightWrapperTensor      (planned)
    ├── IntXFakeQuantizedWeightWrapperTensor      (planned)
    ├── MXFakeQuantizedWeightWrapperTensor        (planned)
    └── NVFP4FakeQuantizedWeightWrapperTensor     (planned)
```

**Assumption**: expert weights are stored as 3D `nn.Parameter` tensors of shape
`[num_experts, in_features, out_features]`. This holds for HF MoE models using
`@use_experts_implementation`.



### Dispatch Path

```
quantize_(model, MoEQATConfig(weight_config=Float8FakeQuantizeConfig(...), step="prepare"))
  → _moe_qat_config_transform()
    → walk parameters
      → wrap param.data with Float8FakeQuantizedWeightWrapperTensor

Model forward (any of the following computation ops on a wrapped weight):
  torch._grouped_mm(A, wrapped_weight, offs=offs)   [HF grouped_mm path]
  torch.bmm(A, wrapped_weight)                      [HF batched_mm path / Llama4 eager]
  torch.mm(A, wrapped_weight)                       [per-expert 2D matmul]
  torch.matmul(A, wrapped_weight)
  F.linear(A, wrapped_weight)                       [shared experts]
  torch.addmm(bias, A, wrapped_weight)

    → Float8FakeQuantizedWeightWrapperTensor.__torch_function__
      → fake-quantize weight (FP8 quantize → dequant, STE gradient)
      → standard op with fake-quantized weight

Convert (after training):
  quantize_(model, MoEQATConfig(step="convert"))
    → _moe_qat_config_transform()
      → walk parameters
        → unwrap FakeQuantizedWeightWrapperBaseTensor
          → nn.Parameter(param.data.to_tensor(), requires_grad=param.requires_grad)
```

### Two-Step Workflow

```python
from torchao.quantization import quantize_
from torchao.prototype.moe_qat import MoEQATConfig
from torchao.quantization.qat import Float8FakeQuantizeConfig
from torchao.quantization.granularity import PerRow

# Prepare: wrap expert weights with fake-quantized tensor subclass
qat_config = MoEQATConfig(
    weight_config=Float8FakeQuantizeConfig(
        dtype=torch.float8_e4m3fn,
        granularity=PerRow(),
    ),
    step="prepare",
)
quantize_(model, qat_config)

# Train / fine-tune
train(model)

# Convert: unwrap back to regular tensors
quantize_(model, MoEQATConfig(step="convert"))
```

## Deferred

- Applying real PTQ during the convert step
- Range learning (stateful quantizers with learnable scale/zero_point)





