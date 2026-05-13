# [RFC] MoE Quantization-Aware Training (QAT) in TorchAO

## Motivation

TorchAO's QAT support targets `nn.Linear` modules via module swap (`FakeQuantizedLinear`).
MoE models use 3D expert weight `nn.Parameter` tensors routed through ops such as
`torch._grouped_mm`, `torch.bmm`, and `F.linear`, which fall entirely outside the
existing QAT path. Meanwhile, `torchao.prototype.moe_training`
has established a parameter-swap pattern (tensor subclass on `param.data`) for low-precision
MoE training that composes with FSDP2, Expert Parallel, and `torch.compile`. This RFC proposes
extending QAT to MoE models using the same parameter-swap approach.

## Goal

Deliver MoE QAT support using the parameter-swap pattern from `moe_training`, starting with
FP8 row-wise weight-only fake quantization and designed for future extension to other precisions
and activation QAT.

## Design

### Architecture

Contrary to dense layers which are usually `nn.Linear`, expert layers in MoE models are usually user-defined classes,
such as `Qwen3MoeExperts` in `unsloth/Qwen3-4B-instruct-25507` and `GptOssExperts` in `openai/gpt-oss-20b`.
To provide a model-agnostic MoE QAT support, we choose to adopt a parameter-swap approach rather than the module-swap
approach used by dense QAT:

1. **Injection**: a `MoEQATConfig` (extending `QATConfig`) uses a dedicated handler that
   wraps 3D expert weight `nn.Parameter.data` with a fake-quantized tensor subclass.

2. **Fake quantization**: the wrapper subclass intercepts computation ops via
   `__torch_function__` and applies fake quantization before delegating to the
   standard op. We reuse the `FakeQuantizeConfig` hierarchy from the existing QAT
   infrastructure (config types, `_infer_fake_quantize_configs`, and the underlying
   quantization primitives), but give up the `FakeQuantizer` module system.
   `FakeQuantizer` is an `nn.Module` — attaching a module to a tensor subclass is
   awkward and may cause potential incompatibilities with `torch.compile` and FSDP2.
   
   1. **Stateless fake quantization**: This case is relevant with FP8 row-wise,
      MX formats, and NVFP4 fake-quantization. The fake quantization logic is expressed
      as stateless functions driven by config, stored directly as static methods on the
      wrapper tensor subclass. 
   
   2. **Stateful fake quantization**: This case is only relevant with `IntxFakeQuantizeConfig` in
      the current `FakeQuantizeConfig` hierarchy. The design for this case remains to be discussed.
      The key point is where to store the trainable quantization states and how the quantization
      logic accesses them. A first attempt may be:
      * Separate the logic and state of fake quantization.
      * The logic is implemented in the same way as in stateless fake quantization.
      * Trainable states are stored as `nn.Parameter` fields in the wrapper tensor. They
        are registered on the parent module so optimizers can see them.
      * The wrapper tensor stores a reference to its parent module. It deregisters
        trainable states from the parent module when the wrapper tensor is unwrapped.

      The shortcoming is that customized code is needed for this design to be compatible with
      `torch.compile`, checkpointing, and FSDP2.
   
   3. **Deferred fake quantization**: slicing, indexing, and dimension-manipulation ops preserve
   the wrapper subclass without applying fake quantization. Quantization is deferred to
   computation time, avoiding double quantization.

3. **Parameter-level traversal & filtering**: when `MoEQATConfig` is passed to `quantize_`, it
   starts a module-level recursive traversal, passing modules picked up by `filter_fn` to the
   handler of `MoEQATConfig`. The handler in turn invokes a parameter-level recursive traversal
   that scans through parameters of this module and its sub-modules, uses a filter function carried
   by `MoEQATConfig` to pick up `nn.Parameter`, and replaces it using a handler chosen based on
   `weight_config` carried by `MoEQATConfig`. A function `_replace_params_with_custom_fn_if_matches_filter`
   is implemented for this recursion with filtering and replacing logic injected into it via arguments.
   This could be a common pattern for the support of MoE training, MoE QAT, and MoE inference.

4. **Handler registration**: handlers for the parameter-level traversal targeting different
   fake-quantization configs are registered in a dictionary via a decorator. The handler of
   `MoEQATConfig` relies on the dictionary to dispatch handlers.

5. **PTQ in the convert step**: applying PTQ during convert is closer to MoE inference
   quantization. We will reuse the infrastructure if MoE inference quantization support
   already exists in TorchAO. If not, MoE PTQ will be implemented first and reused in
   MoE QAT.



### Precision in the Forward Propagation

Dense QAT uses different forward-precision strategies for different configs, and we follow
the same choices:

- **Float8, Int4, IntX**: fully fake-quantized forward — quantize to low precision then
  immediately dequantize back to the original dtype, running the matmul in high precision.
  The STE carries gradients through the quantize-dequantize step, including any clamp.
- **MX, NVFP4**: quantized forward, fake-quantized backward — real low-precision quantization
  and low-precision matmul in the forward pass, dequantized high-precision matmul in the
  backward pass. The clamp is applied during forward quantization but bypassed in the
  backward STE.

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

### Key Design Decisions

1. **Parameter swap, not module swap**: avoids per-model module implementations.
   Compatible with all HF MoE models using `@use_experts_implementation`.

2. **Config-driven, not quantizer-driven**: `FakeQuantizeConfig` dataclasses
   drive the fake quantization logic via plain functions, compatible with
   `torch.compile`, FSDP2, and pickling. No `FakeQuantizer` nn.Modules stored
   on the tensor.

3. **`detach` semantics**: properly detaches `_data` (`requires_grad=False`),
   unlike `moe_training` which preserves `requires_grad=True` on the wrapper.

4. **`copy_` preserves self-reference**: returns the original wrapper for
   in-place semantics (`x.copy_(y) is x`).

## Features Plan

| Precision | Weight-only QAT | Activation+Weight QAT |
|-----------|----------------|----------------------|
| FP8 rowwise | Implemented | Planned |
| Int4 weight-only | Planned | — |
| Int8 dynamic act + IntX weight | Planned | Planned |
| MXFP8 / MXFP4 | Planned | Planned |
| NVFP4 | Planned | Planned |

## Compatibility

- **FSDP2**: supported via the standard wrapper subclass contract: `__tensor_flatten__`, `__tensor_unflatten__`  and `fsdp_pre_all_gather`, `fsdp_post_all_gather` hooks.
- **Expert Parallel**: same parameter-swap pattern as `moe_training`
- **torch.compile**: plain function calls in `__torch_function__`, proven by `moe_training` tests
- **Target hardware**: not hardware-specific. The current infrastructure is used for CPU/GPU/XPU support. Ascend-NPU support is planned.
- **Target model**: not model-specific 

## Deferred

- Applying real PTQ during the convert step
- Range learning (stateful quantizers with learnable scale/zero_point)

## To be Discussed

### Design for stateful fake quantization

What is a suitable design for stateful fake-quantization in
`FakeQuantizedWeightWrapperBaseTensor`? The trainable parameters (e.g., scale,
zero_point for range learning) must be visible to `model.parameters()` so the
optimizer can update them, but tensor subclass attributes are not reachable via
the `nn.Module` hierarchy.


### Fragility of deferred fake quantization after view operations
If complex view operations (such as transpose, permute, index, slice) are applied,
the deferred fake-quantization may cause ambiguity about the desired dimension
along which the fake-quantization should apply. `PerRow(dim=-1)` is positional —
it depends on the tensor's current shape, which may have been rearranged by
intervening view ops. Is this a critical concern in practice? If so, how to
avoid this ambiguity?

