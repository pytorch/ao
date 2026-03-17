# Int4 Weight-Only Quantization: Flow & Calling Stack

Covers the full lifecycle of:
```python
quantize_(model, Int4WeightOnlyConfig(group_size=64, int4_packing_format="plain_int32"))
output = model(input)
```

---

## Table of Contents

1. [Configuration Classes](#1-configuration-classes)
2. [Phase 1 — Quantization](#2-phase-1--quantization)
   - [Calling Stack](#21-calling-stack)
   - [Data Transformation](#22-data-transformation)
   - [Int4PlainInt32Tensor Layout](#23-int4plainint32tensor-layout)
3. [Phase 2 — Inference](#3-phase-2--inference)
   - [Calling Stack](#31-calling-stack)
   - [Dispatch Mechanism](#32-dispatch-mechanism)
   - [Device-Specific Kernel Signatures](#33-device-specific-kernel-signatures)
4. [End-to-End Summary Table](#4-end-to-end-summary-table)

---

## 1. Configuration Classes

### `Int4WeightOnlyConfig`
**File**: `torchao/quantization/quant_api.py:745`

```python
@dataclass
class Int4WeightOnlyConfig(AOBaseConfig):
    group_size: int = 128                    # quantization granularity (we use 64)
    set_inductor_config: bool = True         # auto-configure torch.compile inductor
    int4_packing_format: Int4PackingFormat = Int4PackingFormat.PLAIN
    int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = TINYGEMM
    version: int = 2
```

### `Int4PackingFormat` Enum
**File**: `torchao/quantization/quantize_/workflows/int4/int4_packing_format.py:12`

| Value | Description |
|-------|-------------|
| `PLAIN_INT32` | 8 int4 values packed into one int32; no special tile layout |
| `PLAIN` | Default tinygemm-optimized tile layout |

With `PLAIN_INT32`, each `int32` stores 8 adjacent int4 values:
```
int32 bits:  [28:25][24:21][20:17][16:13][12:9][8:5][4:1][0]
              val7   val6   val5   val4   val3  val2  val1 val0
```

### `Int4ChooseQParamsAlgorithm` (TINYGEMM default)
**File**: `torchao/quantization/quantize_/workflows/int4/int4_choose_qparams_algorithm.py:16`

```
scale      = (max_val - min_val) / (quant_max - quant_min)
zero_point = min_val + scale * mid_point
# For uint4: quant_min=0, quant_max=15
```

---

## 2. Phase 1 — Quantization

### 2.1 Calling Stack

```
quantize_(model, config)                                    quant_api.py:434
  │  Iterates all modules, applying filter_fn (default: _is_linear)
  └─> _replace_with_custom_fn_if_matches_filter(...)        quant_api.py:183
        │  For each matched nn.Linear module:
        └─> handler = _QUANTIZE_CONFIG_HANDLER[Int4WeightOnlyConfig]
              │  Registered via @register_quantize_module_handler decorator
              │                                             transform_module.py:19
              └─> _int4_weight_only_transform(module, config)
                                                            quant_api.py:837
                    │  Optionally sets torch.compile inductor flags
                    └─> _int4_weight_only_quantize_tensor(weight, config)
                                                            quant_api.py:779
                          │  block_size = [1, group_size] = [1, 64]
                          │  Branches on int4_packing_format == PLAIN_INT32
                          └─> Int4PlainInt32Tensor.from_hp(weight, block_size)
                                                  int4_plain_int32_tensor.py:87
                                │  Dispatches by device type
                                ├─> _from_hp_xpu(cls, w, block_size)    line:102
                                │     1. choose_qparams_affine(ASYMMETRIC, uint4)
                                │        → scale (K/64, N), zero_point (K/64, N)
                                │     2. quantize_affine(...)
                                │        → int_data (N, K), values in [0, 15]
                                │     3. pack: [lo | hi<<4] → uint8 → int32
                                │     4. aten._convert_weight_to_int4pack(packed, 8)
                                │        → qdata (N, K/8) int32
                                │     5. return Int4PlainInt32Tensor(qdata, scale, zp, ...)
                                │
                                └─> _from_hp_npu(cls, w, block_size)    line:161
                                      1. choose_qparams_affine(SYMMETRIC, int4)
                                         quant_min=-8, quant_max=7
                                         → scale (K/64, N), zero_point (K/64, N)
                                      2. quantize_affine(...)
                                         → int_data (N, K), values in [-8, 7]
                                      3. pack similarly
                                      4. npu.npu_convert_weight_to_int4pack(...)
                                         → qdata (N, K/8) int32
                                      5. return Int4PlainInt32Tensor(qdata, scale, zp, ...)

                    └─> module.weight = nn.Parameter(new_weight, requires_grad=False)
                          # The original float16/bfloat16 Tensor is replaced in-place
```

### 2.2 Data Transformation

```
Original weight:  Tensor (N, K)  float16/bfloat16
        │
        ▼  choose_qparams_affine  (per group of 64 columns, ASYMMETRIC for XPU)
scale:       (K/64, N)   float16      — one scale per group
zero_point:  (K/64, N)   float16/int8 — one offset per group
        │
        ▼  quantize_affine
int_data:    (N, K)      int32        — raw int4 values, range [0,15] (XPU) or [-8,7] (NPU)
        │
        ▼  pack two adjacent int4s into one byte
packed:      (N, K/2)    uint8
        │
        ▼  aten._convert_weight_to_int4pack  (or npu equivalent)
qdata:       (N, K/8)    int32        — 8 int4 values per int32  ← stored in tensor subclass
```

**Memory saving**: 16-bit → 4-bit = **4× compression** on weights (plus small scale/zp overhead).

### 2.3 `Int4PlainInt32Tensor` Layout

**File**: `torchao/quantization/quantize_/workflows/int4/int4_plain_int32_tensor.py:26`

This is a `TorchAOBaseTensor` (i.e., `torch.Tensor`) subclass. After `quantize_`, every `nn.Linear.weight` is one of these.

| Attribute | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `qdata` | `(N, K/8)` | `int32` | Packed int4 weights |
| `scale` | `(K/group_size, N)` | `float16` | Per-group scale |
| `zero_point` | `(K/group_size, N)` | `int8` | Per-group zero point |
| `block_size` | — | metadata | `[1, 64]` — quantization granularity |
| `shape` | — | metadata | Original `(N, K)` — preserved for reshape |

---

## 3. Phase 2 — Inference

### 3.1 Calling Stack

```
model(input)
  └─> nn.Linear.forward(input)
        └─> F.linear(input, weight, bias)
              │  weight is Int4PlainInt32Tensor → __torch_function__ intercepts
              └─> TorchAOBaseTensor.__torch_function__(F.linear, ...)
                                                            utils.py:979
                    └─> _dispatch__torch_dispatch__(cls, func, types, args, kwargs)
                                                            utils.py:667
                          │  Looks up registered handler in _TORCH_FN_TABLE
                          └─> Int4PlainInt32Tensor._TORCH_FN_TABLE[F.linear]
                                                  int4_plain_int32_tensor.py:250
                                └─> handler(func, types, args, kwargs)
                                      │  Dispatches by input device
                                      │
                                      ├─> _linear_xpu(input, weight_tensor, bias)
                                      │                               line:269
                                      │     1. (optional) input *= weight.act_pre_scale
                                      │     2. act_mat = input.reshape(-1, K)
                                      │     3. groupsize = weight.block_size[1]  # 64
                                      │     4. ──── KERNEL ────
                                      │        y = aten._weight_int4pack_mm_with_scales_and_zeros(
                                      │              act_mat,          # (batch, K)
                                      │              weight.qdata,     # (N, K/8)
                                      │              groupsize,        # 64
                                      │              weight.scale,     # (K/64, N)
                                      │              weight.zero_point # (K/64, N)
                                      │            )                   # → (batch, N)
                                      │     5. y = y[:, :N].reshape(*orig_shape[:-1], N)
                                      │     6. if bias: y += bias
                                      │     7. return y.to(orig_dtype)
                                      │
                                      └─> _linear_npu(input, weight_tensor, bias)
                                                                      line:318
                                            1. (optional) input *= weight.act_pre_scale
                                            2. align scale/zp dtype to activation dtype
                                            3. act_mat = input.reshape(-1, K)
                                            4. groupsize = weight.block_size[1]  # 64
                                            5. ──── KERNEL ────
                                               y = npu.npu_weight_quant_batchmatmul(
                                                     x=act_mat,                  # (batch, K)
                                                     weight=qdata.T,             # (K/8, N)
                                                     antiquant_scale=scale,      # (K/64, N)
                                                     antiquant_offset=zero_point,# (K/64, N)
                                                     antiquant_group_size=64,
                                                     bias=bias,
                                                   )                             # → (batch, N)
                                            6. y = y[:, :N].reshape(*orig_shape[:-1], N)
                                            7. return y.to(orig_dtype)
```

> **Note**: `aten.linear.default` (used inside `torch.compile`) is also registered to the same handler via `@implements(aten.linear.default)` at `int4_plain_int32_tensor.py:250`.

### 3.2 Dispatch Mechanism

`Int4PlainInt32Tensor` inherits from `TorchAOBaseTensor` (`utils.py:783`), which sets up two dispatch tables:

| Table | Triggered by | Registration decorator |
|-------|-------------|----------------------|
| `_TORCH_FN_TABLE` | `F.linear(...)` eager mode | `@implements_torch_function(torch.nn.functional.linear)` |
| `_ATEN_OP_TABLE` | `aten.linear.default` torch.compile/C++ | `@implements(aten.linear.default)` |

Both point to the **same handler function** at `int4_plain_int32_tensor.py:252`.

### 3.3 Device-Specific Kernel Signatures

**XPU** (`torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros`):
```python
# input:  (batch, K)  float16
# weight: (N, K/8)    int32   — packed int4, row-major
# scale:  (K/64, N)   float16
# zp:     (K/64, N)   int8/float16
# output: (batch, N)  float16
#
# Semantics: dequant_w = (weight_int4 - zp) * scale  [per group]
#            output = input @ dequant_w.T
```

**NPU** (`torch.ops.npu.npu_weight_quant_batchmatmul`):
```python
# x:                   (batch, K)   float16/bfloat16
# weight:              (K/8, N)     int32   — transposed vs XPU
# antiquant_scale:     (K/64, N)    float16/bfloat16
# antiquant_offset:    (K/64, N)    float16/bfloat16
# antiquant_group_size: 64
# bias:                (N,) optional
# output:              (batch, N)
```

The kernel dequantizes on-the-fly during GEMM — quantized weights **never** materialize as full float tensors.

---

## 4. End-to-End Summary Table

| Step | Function / Class | File | Line | What Happens |
|------|-----------------|------|------|--------------|
| Entry | `quantize_()` | `quant_api.py` | 434 | Walks model, finds `nn.Linear` |
| Routing | `_QUANTIZE_CONFIG_HANDLER` lookup | `transform_module.py` | 13 | Maps config type → handler |
| Transform | `_int4_weight_only_transform()` | `quant_api.py` | 837 | Calls quantize, replaces `module.weight` |
| Quantize | `_int4_weight_only_quantize_tensor()` | `quant_api.py` | 779 | Selects packing format branch |
| Tensor creation | `Int4PlainInt32Tensor.from_hp()` | `int4_plain_int32_tensor.py` | 87 | Computes scale/zp, packs int4 |
| XPU packing | `_from_hp_xpu()` | `int4_plain_int32_tensor.py` | 102 | ASYMMETRIC uint4, aten pack op |
| NPU packing | `_from_hp_npu()` | `int4_plain_int32_tensor.py` | 161 | SYMMETRIC int4, npu pack op |
| Storage | `Int4PlainInt32Tensor` | `int4_plain_int32_tensor.py` | 26 | Holds `qdata`, `scale`, `zero_point` |
| — | — | — | — | — |
| Inference dispatch | `TorchAOBaseTensor.__torch_function__` | `utils.py` | 979 | Intercepts `F.linear` |
| Handler lookup | `_TORCH_FN_TABLE[F.linear]` | `utils.py` | 667 | Routes to registered handler |
| Device dispatch | handler in `int4_plain_int32_tensor.py` | `int4_plain_int32_tensor.py` | 252 | Picks XPU or NPU path |
| XPU kernel | `aten._weight_int4pack_mm_with_scales_and_zeros` | `int4_plain_int32_tensor.py` | 303 | Int4 GEMM, dequant on-the-fly |
| NPU kernel | `npu.npu_weight_quant_batchmatmul` | `int4_plain_int32_tensor.py` | 365 | Int4 GEMM, dequant on-the-fly |
