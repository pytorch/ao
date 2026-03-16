# FP8 Dynamic Activation + FP8 Weight Quantization: Flow & Calling Stack

Covers the full lifecycle of:
```python
quantize_(model, Float8DynamicActivationFloat8WeightConfig())
output = model(input)
```

---

## Table of Contents

1. [Configuration Class](#1-configuration-class)
2. [Phase 1 — Quantization](#2-phase-1--quantization)
   - [Calling Stack](#21-calling-stack)
   - [Data Transformation](#22-data-transformation)
   - [Float8Tensor Layout](#23-float8tensor-layout)
3. [Phase 2 — Inference](#3-phase-2--inference)
   - [Calling Stack](#31-calling-stack)
   - [Dynamic Activation Quantization](#32-dynamic-activation-quantization-at-runtime)
   - [Kernel Paths](#33-kernel-paths)
4. [Key Formulas](#4-key-formulas)
5. [End-to-End Summary Table](#5-end-to-end-summary-table)

---

## 1. Configuration Class

**File**: `torchao/quantization/quant_api.py:1501`

```python
@dataclass
class Float8DynamicActivationFloat8WeightConfig(AOBaseConfig):
    activation_dtype: torch.dtype = e4m3_dtype        # float8_e4m3fn
    weight_dtype: torch.dtype = e4m3_dtype            # float8_e4m3fn
    granularity: Optional[Union[FP8Granularity,
                  List[FP8Granularity]]] = None        # PerTensor / PerRow / blockwise
    packing_format: Float8PackingFormat = PLAIN        # PLAIN or DENSE (for NPU)
    mm_config: Optional[Float8MMConfig] = None        # matmul precision options
    activation_value_lb: Optional[float] = None       # clip activation min
    activation_value_ub: Optional[float] = None       # clip activation max
    kernel_preference: KernelPreference = AUTO        # AUTO / TORCH / MSLK
    set_inductor_config: bool = True
    version: int = 2
```

`__post_init__` normalizes `granularity` into `[weight_granularity, activation_granularity]` and sets
default `mm_config` if not provided.

### Granularity Options

| Granularity | Weight Scale Shape | Description |
|-------------|-------------------|-------------|
| `PerTensor` | scalar | Single scale for the entire matrix |
| `PerRow` | `(N,)` | One scale per output row |
| `PerTensor` + `PerRow` | weight PerTensor, act PerRow | Mixed mode |
| 128×128 block | `(N/128, K/128)` | Fine-grained block scaling |

---

## 2. Phase 1 — Quantization

### 2.1 Calling Stack

```
quantize_(model, config)                                         quant_api.py:434
  │  Iterates all modules, applies filter_fn (default: _is_linear)
  └─> _replace_with_custom_fn_if_matches_filter(...)             quant_api.py:496
        │  For each matched nn.Linear:
        └─> handler = _QUANTIZE_CONFIG_HANDLER[Float8DynamicActivationFloat8WeightConfig]
              │  Registered via @register_quantize_module_handler       transform_module.py:19
              └─> _float8_dynamic_activation_float8_weight_transform(module, config)
                                                                   quant_api.py:1635
                    │  1. Check hardware (CUDA SM >= 8.9 or MI300+)    line:1641
                    │  2. Set torch.compile inductor flags              line:1645
                    │  3. Unwrap any existing Float8Linear wrappers     line:1652
                    └─> _float8_dynamic_activation_float8_weight_quantize_tensor(weight, config)
                                                                   quant_api.py:1562
                          │  Extracts granularity config                line:1574
                          │  Builds QuantizeTensorToFloat8Kwargs        line:1604
                          │    (stores activation quant params inside weight tensor)
                          │
                          │  PLAIN packing format branch               line:1611
                          └─> Float8Tensor.from_hp(weight, block_size, ...)
                                                float8_tensor.py:166
                                │
                                │  1. Compute block_size from granularity  line:177
                                │  2. Choose quantization kernel            line:180
                                │     (MSLK or PyTorch)
                                │
                                │  PyTorch kernel path:
                                ├─> _choose_scale_float8(weight, block_size, dtype)
                                │                            quant_primitives.py:2224
                                │     │  max_abs = max(|weight|) per block
                                │     │  scale = max_abs / finfo(fp8_dtype).max  → float32
                                │     └─> Returns scale tensor
                                │
                                └─> _quantize_affine_float8(weight, block_size, scale, dtype)
                                                             quant_primitives.py:2322
                                      │  weight_fp32 = weight.to(float32)
                                      │  scaled = weight_fp32 / scale_expanded
                                      │  clamped = clamp(scaled, -max_val, max_val)
                                      └─> Returns qdata in float8 dtype

                          └─> Float8Tensor(qdata, scale, block_size, act_quant_kwargs, ...)
                                # act_quant_kwargs carries activation quant params
                                # into inference time

                    └─> module.weight = nn.Parameter(quantized_tensor, requires_grad=False)
                          # Original float16/bfloat16 weight is replaced in-place
                          # nn.Linear module itself is NOT replaced — only its weight
```

### 2.2 Data Transformation

```
Original weight:  Tensor (N, K)  float16/bfloat16
        │
        ▼  _choose_scale_float8  (per PerTensor or PerRow or 128×128 block)
scale:  float32 scalar  OR  (N,)  OR  (N/128, K/128)
        │
        ▼  _quantize_affine_float8
        │    qdata = clamp(weight_fp32 / scale, -448.0, 448.0).to(float8_e4m3fn)
        │
qdata:  Tensor (N, K)  float8_e4m3fn   ← stored in Float8Tensor.qdata
        │
        ▼  wrapped into Float8Tensor with act_quant_kwargs embedded
```

**Memory saving**: 16-bit → 8-bit = **2× compression** on weights.
Unlike INT4, FP8 does **not** pack multiple values — one float8 value per element.

### 2.3 `Float8Tensor` Layout

**File**: `torchao/quantization/quantize_/workflows/float8/float8_tensor.py:82`

This is a `TorchAOBaseTensor` (`torch.Tensor`) subclass. After `quantize_`, every `nn.Linear.weight` is one of these.

| Attribute | Type | Description |
|-----------|------|-------------|
| `qdata` | `Tensor (N, K)` float8 | Quantized weight values |
| `scale` | `Tensor` float32 | Scale factor(s); shape depends on granularity |
| `block_size` | metadata | Quantization granularity, e.g. `[1, K]` for PerRow |
| `mm_config` | metadata | `Float8MMConfig` — matmul precision/accumulation options |
| `act_quant_kwargs` | metadata | `QuantizeTensorToFloat8Kwargs` — **activation quantization params**, carried into inference |
| `kernel_preference` | metadata | `KernelPreference.AUTO/TORCH/MSLK` |
| `dtype` | metadata | Original high-precision dtype (float16 / bfloat16) |

> **Key design**: `act_quant_kwargs` is stored on the **weight** tensor so that at inference time,
> `F.linear` can find the activation quantization recipe without any changes to `nn.Linear.forward`.

---

## 3. Phase 2 — Inference

### 3.1 Calling Stack

```
model(input)
  └─> nn.Linear.forward(input)
        └─> F.linear(input, weight, bias)
              │  weight is Float8Tensor → __torch_function__ intercepts
              └─> TorchAOBaseTensor.__torch_function__(F.linear, ...)
                                                              utils.py:979
                    └─> _dispatch__torch_function__(cls, func, types, args, kwargs)
                                                              utils.py:667
                          │  Looks up handler in _TORCH_FN_TABLE
                          └─> Float8Tensor._TORCH_FN_TABLE[F.linear]
                                                  float8_tensor.py:258
                                └─> handler(func, types, args, kwargs)
                                      │  input_tensor = args[0]   (raw float16/bfloat16)
                                      │  weight_tensor = args[1]  (Float8Tensor)
                                      └─> _float8_addmm_impl(input_tensor, weight.t(), bias)
                                                              float8_tensor.py:318

                                            ┌─────────────────────────────────────────────┐
                                            │  DYNAMIC ACTIVATION QUANTIZATION  (line 327)│
                                            │                                             │
                                            │  act_quant_kwargs =                         │
                                            │      weight_tensor.act_quant_kwargs          │
                                            │                                             │
                                            │  input_tensor =                             │
                                            │    _choose_quant_func_and_quantize_tensor(  │
                                            │      input_tensor, act_quant_kwargs         │
                                            │    )                                        │
                                            │    → Float8Tensor.from_hp(input, ...)       │
                                            │      (same scale+quantize as weight phase)  │
                                            │    → input is now a Float8Tensor            │
                                            └─────────────────────────────────────────────┘

                                            │  Unpack both tensors:
                                            │    a_data  = input_tensor.qdata   (float8)
                                            │    a_scale = input_tensor.scale
                                            │    b_data  = weight_tensor.qdata  (float8)
                                            │    b_scale = weight_tensor.scale
                                            │
                                            ├─── KERNEL PATH A: MSLK  (line 369)
                                            │    if kernel_preference == MSLK:
                                            │    ├─ RowWise scale →
                                            │    │    torch.ops.mslk.f8f8bf16_rowwise(
                                            │    │        a_data, b_data, a_scale, b_scale)
                                            │    └─ TensorWise scale →
                                            │         torch.ops.mslk.f8f8bf16(
                                            │             a_data, b_data, a_scale, b_scale)
                                            │
                                            ├─── KERNEL PATH B: 128×128 Blockwise  (line 425)
                                            │    if blockwise granularity:
                                            │         blockwise_fp8_gemm(
                                            │             a_data, a_scale, b_data, b_scale)
                                            │                    kernel/blockwise_quantization.py:231
                                            │         → Triton kernel: tl.dot(a,b)*a_s*b_s
                                            │
                                            └─── KERNEL PATH C: Standard (default)  (line 440)
                                                 addmm_float8_unwrapped_inference(
                                                     a_data, a_scale,
                                                     b_data, b_scale,
                                                     bias, output_dtype)
                                                              float8/inference.py:85
                                                   └─> torch._scaled_mm(
                                                             a_data, b_data,
                                                             scale_a=a_scale,
                                                             scale_b=b_scale,
                                                             bias=bias,
                                                             out_dtype=bfloat16
                                                         )
                                                         # → cublasLt float8 GEMM (CUDA)
                                                         # Fused: matmul + dequant + bias
                                                         # Output: bfloat16 or float32
```

> **Note**: `aten.linear.default` (used inside `torch.compile`) is also registered to the same
> handler via `@implements(aten.linear.default)` at `float8_tensor.py:258`.

### 3.2 Dynamic Activation Quantization at Runtime

This is the key difference from weight-only quantization. At every forward pass:

```
input (batch, K)  float16/bfloat16
        │
        ▼  _choose_quant_func_and_quantize_tensor()   quantize_tensor_kwargs.py:33
             calls Float8Tensor.from_hp(input, act_block_size, ...)
        │
        ▼  _choose_scale_float8 — computes scale from THIS batch's input
        │    scale = max(|input|) / finfo(fp8).max      ← computed per forward call
        ▼  _quantize_affine_float8
             a_data = clamp(input / scale, -448, 448).to(float8_e4m3fn)
        │
        ▼  Float8Tensor(a_data, a_scale, ...)
        │
        ▼  Passed to kernel as (a_data, a_scale)
```

The scale is **recomputed at every forward call** — hence "dynamic". This is more accurate than
static quantization (where scale is calibrated once) at the cost of a small runtime overhead.

### 3.3 Kernel Paths

| Path | Trigger | Kernel | Notes |
|------|---------|--------|-------|
| **MSLK RowWise** | `KernelPreference.MSLK` + PerRow granularity | `torch.ops.mslk.f8f8bf16_rowwise` | Optimized for row-scaled FP8 |
| **MSLK TensorWise** | `KernelPreference.MSLK` + PerTensor granularity | `torch.ops.mslk.f8f8bf16` | Optimized for tensor-scaled FP8 |
| **Blockwise Triton** | 128×128 block granularity | Triton `blockwise_fp8_gemm_kernel` | Per-block fused scaling in GEMM |
| **Standard (default)** | Everything else | `torch._scaled_mm` → cuBLASLt | PyTorch built-in; hardware-fused float8 matmul |

---

## 4. Key Formulas

### Weight Quantization (done once, at `quantize_` time)
```
scale_w  = max(|weight|_per_block) / finfo(float8_e4m3fn).max   # ≈ 448.0
qweight  = clamp(weight_fp32 / scale_w, -448.0, 448.0).to(float8)
```

### Activation Quantization (done every forward pass)
```
scale_a  = max(|input|_per_block) / finfo(float8_e4m3fn).max
qinput   = clamp(input_fp32 / scale_a, -448.0, 448.0).to(float8)
```

### Inference (inside the kernel)
```
output = (qinput * scale_a) @ (qweight * scale_w).T + bias
       ≡ torch._scaled_mm(qinput, qweight.T, scale_a=scale_a, scale_b=scale_w)
```
The kernel dequantizes **implicitly** — no materialization of a full float16 weight matrix.

---

## 5. End-to-End Summary Table

| Step | Function / Class | File | Line | What Happens |
|------|-----------------|------|------|--------------|
| **Quantization** | | | | |
| Entry | `quantize_()` | `quant_api.py` | 434 | Walks model, finds `nn.Linear` |
| Config dispatch | `_QUANTIZE_CONFIG_HANDLER` lookup | `transform_module.py` | 13 | Maps config type → handler |
| Transform | `_float8_dynamic_activation_float8_weight_transform()` | `quant_api.py` | 1635 | HW check, calls weight quantizer |
| Weight quantize | `_float8_dynamic_activation_float8_weight_quantize_tensor()` | `quant_api.py` | 1562 | Builds `act_quant_kwargs`, calls `from_hp` |
| Tensor creation | `Float8Tensor.from_hp()` | `float8_tensor.py` | 166 | Computes scale, quantizes weight |
| Scale compute | `_choose_scale_float8()` | `quant_primitives.py` | 2224 | `scale = max_abs / fp8_max` |
| Quantize | `_quantize_affine_float8()` | `quant_primitives.py` | 2322 | `qdata = clamp(w/scale).to(fp8)` |
| Storage | `Float8Tensor` | `float8_tensor.py` | 82 | Holds `qdata`, `scale`, `act_quant_kwargs` |
| Replace weight | `module.weight = nn.Parameter(...)` | `quant_api.py` | 1657 | `nn.Linear` weight swapped in-place |
| **Inference** | | | | |
| Intercept | `TorchAOBaseTensor.__torch_function__` | `utils.py` | 979 | Catches `F.linear(input, fp8_weight, bias)` |
| Dispatch | `Float8Tensor._TORCH_FN_TABLE[F.linear]` | `float8_tensor.py` | 258 | Routes to `_float8_addmm_impl` |
| Act. quantize | `_choose_quant_func_and_quantize_tensor()` | `quantize_tensor_kwargs.py` | 33 | Quantizes input to FP8 **at runtime** |
| AddMM | `_float8_addmm_impl()` | `float8_tensor.py` | 318 | Selects kernel, runs matmul |
| Default kernel | `addmm_float8_unwrapped_inference()` | `float8/inference.py` | 85 | Calls `torch._scaled_mm` → cuBLASLt |
| Blockwise kernel | `blockwise_fp8_gemm()` | `kernel/blockwise_quantization.py` | 231 | Triton kernel with per-block scaling |
| MSLK kernel | `torch.ops.mslk.f8f8bf16[_rowwise]` | — | — | Vendor-optimized FP8 GEMM |

### Comparison with INT4 Weight-Only

| Aspect | FP8 Dynamic (this doc) | INT4 Weight-Only |
|--------|------------------------|------------------|
| Weight dtype | `float8_e4m3fn` | Packed int4 (8 per int32) |
| Activation | Quantized **dynamically** per forward call | Full precision (no quantization) |
| Module replaced | No — only weight tensor replaced | No — only weight tensor replaced |
| Dispatch mechanism | `__torch_function__` / `__torch_dispatch__` | Same |
| Scale per weight | PerTensor / PerRow / 128×128 block | Per group of 64 columns |
| Memory saving | ~2× | ~4× |
| Accuracy | Higher (FP8 is closer to FP16) | Lower (INT4 is coarser) |
| Hardware requirement | CUDA SM ≥ 8.9 (Ada/Hopper) or MI300 | XPU / NPU |
