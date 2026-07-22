# Issues Found in TorchAO Inference Examples

## 1. `int4_weight_only.py` - dtype mismatch

**File:** `docs/source/examples/inference/int4_weight_only.py`

### Problem
The example creates a model with default dtype (float32), but uses `int4_packing_format="tile_packed_to_4d"` which requires bfloat16.

### Description
The `Int4TilePackedTo4dTensor` only supports bfloat16 weights. When running the example, users get:
```
Only bfloat16 is supported for Int4TilePackedTo4dTensor, got torch.float32
```

This happens in `torchao/quantization/quantize_/workflows/int4/int4_tile_packed_to_4d_tensor.py:112-114`:
```python
assert hp_tensor.dtype == torch.bfloat16, (
    f"Only bfloat16 is supported for Int4TilePackedTo4dTensor, got {hp_tensor.dtype}"
)
```

### Environment
- Config: `Int4WeightOnlyConfig(int4_packing_format="tile_packed_to_4d")`
- Supported: bfloat16
- Unsupported: float32, float16

### Steps to Approach to Solve
Add `dtype=torch.bfloat16` to the model creation:
```python
model = nn.Sequential(nn.Linear(2048, 2048, device="cuda", dtype=torch.bfloat16))
```

---

## 2. `float8_dynamic_activation_int4_weight.py` - missing dependency

**File:** `docs/source/examples/inference/float8_dynamic_activation_int4_weight.py`

### Problem
The example requires the `mslk` package (Meta's quantization library) which is not installed by default and not mentioned in the example.

### Description
When running the example:
```
ImportError: Requires mslk >= 1.0.0
```

The `Float8DynamicActivationInt4WeightConfig` uses MSLK kernels for quantization. The check is in `torchao/quantization/quantize_/workflows/int4/int4_tensor.py:140`.

### Environment
- Package: MSLK (https://github.com/pytorch/MSLK)
- Version required: >= 1.0.0
- Not included in default TorchAO installation

### Steps to Approach to Solve
1. Document the dependency at the top of the example
2. Or provide an alternative config that works without MSLK

---

## 3. `float8_dynamic_activation_float8_weight.py` - hardware requirement

**File:** `docs/source/examples/inference/float8_dynamic_activation_float8_weight.py`

### Problem
Requires CUDA compute capability ≥8.9 (Ada/Hopper) or MI300+ or XPU. The example will fail on older GPUs without clear indication why.

### Description
When running on unsupported hardware:
```
Float8 dynamic quantization requires CUDA compute capability ≥8.9 or MI300+ or XPU.
```

This is a hardware constraint check in the float8 quantization code path.

### Environment
- Required compute capability: sm_89 (Ada) or higher
- Also supports: sm_90 (Hopper), MI300+, XPU
- Unsupported: sm_80 (Ampere - A100)

### Steps to Approach to Solve
Add a note at the top of the example indicating hardware requirements.

---

## 4. `uintx_weight_only.py` - missing dependency

**File:** `docs/source/examples/inference/uintx_weight_only.py`

### Problem
Requires `gemlite` package which is not installed by default and not documented in the example.

### Description
When running the example:
```
gemlite is required. Install with: pip install gemlite
```

This config is in the prototype/quantization module, indicating it's experimental and has external dependencies.

### Environment
- Package: gemlite
- Module: `torchao.prototype.quantization.UIntxWeightOnlyConfig`

### Steps to Approach to Solve
Document the dependency at the top of the example:
```python
# Requires: pip install gemlite
```

---

## 5. `int8_dynamic_activation_uintx_weight.py` - missing dependency

**File:** `docs/source/examples/inference/int8_dynamic_activation_uintx_weight.py`

### Problem
Same as #4 - requires `gemlite` package not documented in the example.

### Description
When running the example:
```
gemlite is required. Install with: pip install gemlite
```

### Environment
- Package: gemlite
- Module: `torchao.prototype.quantization.Int8DynamicActivationUIntxWeightConfig`

### Steps to Approach to Solve
Document the dependency at the top of the example.

---

## 6. `nvfp4_dynamic_activation_nvfp4_weight.py` - NVFP4 dynamic mode requires Blackwell architecture

**File:** `docs/source/examples/inference/nvfp4_dynamic_activation_nvfp4_weight.py`

### Problem
Running this example on non-Blackwell GPU hardware fails with an unhelpful assertion error:
```
AssertionError: NVFP4 DYNAMIC mode is only supported on sm100+ machines
```

### Description
NVFP4 (NVIDIA 4-bit Floating Point) is a 4-bit floating point format using E2M1 encoding with block scaling. The example uses `NVFP4DynamicActivationNVFP4WeightConfig` with default settings, which invokes the "DYNAMIC" quantization path (when `step=None`).

The dynamic quantization path requires Blackwell architecture (sm100) because it computes activation scales at runtime. This is implemented in `torchao/prototype/mx_formats/inference_workflow.py:309-311`:

```python
elif step is None:
    # Dynamic quantization
    assert is_sm_at_least_100(), (
        "NVFP4 DYNAMIC mode is only supported on sm100+ machines"
    )
```

The error message "sm100+" is cryptic and provides no context about what it means or which GPUs are affected.

### Environment
- **NVFP4 format:** NVIDIA 4-bit floating point with E2M1 encoding
- **Supported GPUs:** sm100 (Blackwell - B200, GB200)
- **Unsupported GPUs:**
  - sm_80: Ampere (A100)
  - sm_90: Hopper (H100)
  - sm_89: Ada (RTX 4090)

### Steps to Approach to Solve
1. Add hardware requirement documentation at the top of the example:
   ```python
   # Requires NVIDIA Blackwell (sm100) architecture
   # NOT supported on: Ampere (A100), Hopper (H100), Ada (RTX 4090)
   ```

2. Add a pre-check with informative error message before the assertion fails

3. Consider offering a fallback path for non-Blackwell GPUs using static quantization (PREPARE/CONVERT steps)

### Contrast with `nvfp4_weight_only.py`
The sibling file `nvfp4_weight_only.py` works on all GPUs because it uses weight-only quantization (static path via PREPARE/CONVERT steps) which does not require runtime dynamic activation scaling.

---

## 7. Missing Copyright Headers

Some example files have copyright headers while others don't, causing inconsistency.

### Problem
Inconsistent licensing headers across example files.

### Description
Only 3 files have copyright headers:
- `int8_dynamic_activation_int4_weight.py`
- `uintx_weight_only.py`
- `int8_dynamic_activation_uintx_weight.py`

All other 12 files lack headers.

### Environment
All inference examples in `docs/source/examples/inference/`

### Steps to Approach to Solve
Add consistent copyright headers to all files:
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
```

---

## Summary Table

| Example File | Issue | Severity |
|-------------|-------|----------|
| `int4_weight_only.py` | dtype mismatch (float32 vs bfloat16) | High |
| `float8_dynamic_activation_int4_weight.py` | missing `mslk` dependency | High |
| `float8_dynamic_activation_float8_weight.py` | hardware ≥sm_89 required | Medium |
| `uintx_weight_only.py` | missing `gemlite` dependency | High |
| `int8_dynamic_activation_uintx_weight.py` | missing `gemlite` dependency | High |
| `nvfp4_dynamic_activation_nvfp4_weight.py` | hardware ≥sm_100 (Blackwell) required | Medium |
| All files | Inconsistent copyright headers | Low |