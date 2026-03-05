# Design: TransformerEngine MXFP8 Grouped GEMM Integration in TorchAO

## 1. Motivation

TorchAO's `torchao.prototype.moe_training` provides MXFP8-quantized grouped
GEMM for Mixture-of-Experts (MoE) training.  The existing implementation uses
torchao's own Triton/CUDA quantization kernels and CUTLASS grouped GEMM
(via `torch._scaled_grouped_mm`), targeting SM100 (Blackwell) hardware.

NVIDIA's **TransformerEngine** (TE) ships its own highly-optimized MXFP8
quantizer (`MXFP8Quantizer` + `split_quantize`) and grouped GEMM kernel
(`general_grouped_gemm`).  This PR adds TE as an alternative backend that users
can opt into via a single configuration flag, without changing any model code.

### Why both?

| | TorchAO native (AUTO) | TransformerEngine (TE) |
|---|---|---|
| Quantization | Triton + CUDA custom kernels | TE `MXFP8Quantizer` + `split_quantize` |
| GEMM | CUTLASS via `torch._scaled_grouped_mm` | cuBLASLt via TE `general_grouped_gemm` |
| Hardware | SM100+ (Blackwell) | SM90+ (Hopper/Blackwell) |
| Dependency | torchao only | torchao + `transformer_engine` |
| torch.compile | Supported (native ops) | Supported (custom ops with fake impls) |

Users choose at config time; the model code is untouched.

---

## 2. MXFP8 Quantization and GEMM Layouts

### 2.1 MXFP8: 1D block scaling (OCP MX specification)

MXFP8 quantizes tensors using **1D blocks of 32 consecutive elements**, each
sharing a single E8M0 scale (8-bit shared exponent).  This follows the
[OCP Microscaling (MX) specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
with `block_size=32`.

For a 2D matrix `[M, K]`:

- **Rowwise**: blocks of 32 along the last dimension (K).
  Each row is divided into `K/32` blocks.  Scale shape: `[M, K/32]`.
- **Columnwise**: blocks of 32 along the first dimension (M).
  Each column is divided into `M/32` blocks.  Scale shape: `[M/32, K]`.

Both are 1D vectors (`[1×32]` or `[32×1]`), not 2D tiles.

### 2.2 GEMM layouts and scale alignment

TE's `general_grouped_gemm(A, B, C, layout="XY")` uses cuBLASLt with
column-major convention.  In PyTorch (row-major) terms, the layout computes:

- **TN**: `C = B @ A^T`
- **NN**: `C = B @ A`
- **NT**: `C = B^T @ A`

Each layout dictates which scale orientation cuBLASLt expects per operand:

```
A_scales: rowwise_usage = transa,    columnwise_usage = !transa
B_scales: rowwise_usage = !transb,   columnwise_usage = transb
```

Where "rowwise" = scales along last dim of stored tensor, "columnwise" = scales
along first dim of stored tensor.

TE's layout choices ensure scales are always along the **contraction dimension**,
which gives the best numerical accuracy (the 32 elements sharing a scale
participate in the same dot-product reduction, so the shared exponent
approximation is tightest).

**Forward — layout="TN" (default)**

```
general_grouped_gemm(
    A = weight [N, K],     # rowwise scales → along K
    B = input  [m, K],     # rowwise scales → along K
    layout="TN"
)
out = B @ A^T = input[m, K] @ weight[N, K]^T = [m, K] @ [K, N] = [m, N]

Contraction dim = K (last dim of both A and B) → rowwise covers it ✓
```

**DGRAD — layout="NN"**

```
general_grouped_gemm(
    A = weight [N, K],     # columnwise scales → along N
    B = grad   [m, N],     # rowwise scales → along N
    layout="NN"
)
out = B @ A = grad[m, N] @ weight[N, K] = [m, N] @ [N, K] = [m, K]

Contraction dim = N (first dim of A, last dim of B)
  → A needs columnwise (along N, first dim) ✓
  → B needs rowwise (along N, last dim) ✓
```

**WGRAD — layout="NT"**

```
general_grouped_gemm(
    A = input  [m, K],     # columnwise scales → along m
    B = grad   [m, N],     # columnwise scales → along m
    layout="NT"
)
out = B^T @ A = grad[m, N]^T @ input[m, K] = [N, m] @ [m, K] = [N, K]

Contraction dim = m (first dim of both A and B)
  → A needs columnwise (along m, first dim) ✓
  → B needs columnwise (along m, first dim) ✓
```

---

## 3. Design Overview

### 3.1 Integration point: `KernelPreference` enum

TorchAO already has a `KernelPreference` enum that controls which kernel
backend is used for quantized computation:

```
KernelPreference
├── AUTO        # Best available (CUTLASS / Triton / CUDA)
├── TORCH       # torch-native kernels
├── MSLK        # MSLK library
├── EMULATED    # Dequant → BF16 GEMM (for CI / debug)
└── TE          # ← NEW: TransformerEngine MXFP8 kernels
```

This enum is threaded through:
  `MXFP8TrainingOpConfig` → `MXFP8TrainingWeightWrapperTensor.__torch_function__`
  → `_to_mxfp8_then_scaled_grouped_mm` → `_MXFP8GroupedMM` (autograd Function)
  → `_compute_dgrad` / `_compute_wgrad`

Adding `TE` to this enum gives us a clean dispatch point at each stage of
the forward/backward pass.

### 3.2 Architecture diagram

```
User code (unchanged)
  │
  │  torch._grouped_mm(activations, expert_weights, offs=offsets)
  │
  ▼
MXFP8TrainingWeightWrapperTensor.__torch_function__
  │
  │  Intercepts _grouped_mm, dispatches to:
  ▼
_quantize_then_scaled_grouped_mm(A, B_t, config)
  │
  │  Reads config.kernel_preference
  ▼
_to_mxfp8_then_scaled_grouped_mm(A, B_t, ..., kernel_preference)
  │
  ▼
_MXFP8GroupedMM.apply(...)   ← torch.autograd.Function
  │
  ├─ kernel_preference == AUTO
  │    ├─ forward:  torchao quant → torch._scaled_grouped_mm
  │    ├─ dgrad:    torchao quant → torch._scaled_grouped_mm
  │    └─ wgrad:    torchao quant → torch._scaled_grouped_mm
  │
  ├─ kernel_preference == EMULATED
  │    ├─ forward:  to_mx → emulated BF16 grouped mm
  │    ├─ dgrad:    to_mx → emulated BF16 grouped mm
  │    └─ wgrad:    to_mx → emulated BF16 grouped mm
  │
  └─ kernel_preference == TE          ← NEW
       ├─ forward:  te_moe::gemm_fwd   (TE MXFP8Quantizer → general_grouped_gemm)
       ├─ dgrad:    te_moe::gemm_dgrad  (TE MXFP8Quantizer → general_grouped_gemm)
       └─ wgrad:    te_moe::gemm_wgrad  (TE MXFP8Quantizer → general_grouped_gemm)
```

---

## 4. File Changes

### 4.1 `torchao/quantization/quantize_/common/kernel_preference.py`

Added `TE = "te"` to the `KernelPreference` enum with docstring explaining
the TE dependency and what kernels it uses.

### 4.2 `torchao/prototype/moe_training/te_grouped_mm.py` (new file)

Contains all TE-specific logic, isolated behind an import guard:

```python
try:
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False
```

**MXFP8 quantization helpers:**

| Function | Purpose |
|---|---|
| `_mxfp8_quantize_inputs` | Pad to block-32 multiples, then `split_quantize` rowwise |
| `_mxfp8_quantize_weights` | Transpose [E,K,N]→[E,N,K], quantize each expert rowwise |
| `_mxfp8_quantize_weights_dgrad` | Same transpose, quantize row+col, switch to col-only |
| `_mxfp8_quantize_wgrad` | Pad both inputs & grads, quantize row+col, switch inputs to col-only |

**Custom ops (torch.library):**

Each op is registered with `@torch.library.custom_op` and has a
`.register_fake` implementation for torch.compile tracing:

| Op | Signature | GEMM Layout | Purpose |
|---|---|---|---|
| `te_moe::gemm_fwd` | `(A, B_t, offs, out_dtype, use_fp8) → out` | TN | Forward: `out[i] = input[i] @ weight[i]^T` |
| `te_moe::gemm_dgrad` | `(grad_out, B_t, offs, out_dtype, use_fp8) → grad_A` | NN | Backward: `grad_A[i] = grad_out[i] @ weight[i]` |
| `te_moe::gemm_wgrad` | `(A, grad_out, offs, out_dtype, use_fp8) → grad_B_t` | NT | Backward: `wgrad[i] = grad_out[i]^T @ A[i]` |

Each op internally:
1. Converts cumulative offsets → per-expert `m_splits` (GPU→CPU sync)
2. Pads per-expert token chunks to multiples of 32
3. Quantizes via TE's `MXFP8Quantizer` / `split_quantize`
4. Calls TE's `general_grouped_gemm`
5. Unpads the output if padding was applied

### 4.3 `torchao/prototype/moe_training/mxfp8_grouped_mm.py` (modified)

Three changes to the existing `_MXFP8GroupedMM` autograd function:

1. **`forward()`**: Early-return branch when `kernel_preference == TE`:
   calls `te_gemm_fwd`, saves tensors for backward, returns.

2. **`backward()`**: Updated SM100 assertion to also allow TE mode.

3. **`_compute_dgrad()`** and **`_compute_wgrad()`**: Early-return branches
   that call `te_gemm_dgrad` / `te_gemm_wgrad` respectively.

The TE branches are self-contained — they don't interact with the existing
quantization or scale-rearrangement logic.

### 4.4 `torchao/prototype/moe_training/config.py` (modified)

- Added `MXFP8TrainingRecipe.MXFP8_TE` enum value
- Added factory case in `MXFP8TrainingOpConfig.from_recipe()`:
  ```python
  MXFP8_TE → MXFP8TrainingOpConfig(kernel_preference=KernelPreference.TE, ...)
  ```

### 4.5 `torchao/prototype/moe_training/__init__.py` (modified)

Exported: `is_te_available`, `te_gemm_fwd`, `te_gemm_dgrad`, `te_gemm_wgrad`.

### 4.6 `test/prototype/moe_training/test_te_grouped_mm.py` (new file)

Test coverage:
- Custom ops directly (fwd, dgrad, wgrad) in both MXFP8 and BF16 modes
- Full forward+backward through `_to_mxfp8_then_scaled_grouped_mm`
  with `KernelPreference.TE` vs BF16 reference
- Recipe creation (`MXFP8_TE`)
- Model conversion via `quantize_()` with TE recipe
- `torch.compile` compatibility (fullgraph=True)

---

## 5. Data Flow: Forward + Backward

### Forward

```
input_act [M, K]  (bf16)      weight_t [E, K, N]  (bf16)      offs [E]
     │                               │                           │
     └──────────────┬────────────────┘                           │
                    ▼                                            │
          te_moe::gemm_fwd ──────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │  _offs_to_m_splits(offs)      │
    │  → m_splits = [n1, n2, ...]   │
    │                               │
    │  _pad_for_mxfp8(A, m_splits)  │
    │  → padded_A, padded_splits    │
    │                               │
    │  MXFP8Quantizer (rowwise)     │
    │  → inputmats_fp8              │
    │                               │
    │  weight_t [E,K,N] → [E,N,K]   │
    │  MXFP8Quantizer (rowwise)     │
    │  → weights_fp8                │
    │                               │
    │  general_grouped_gemm         │
    │    layout=TN, single_output   │
    │  → padded_out                 │
    │                               │
    │  _unpad_mxfp8_output          │
    │  → out [M, N]                 │
    └───────────────────────────────┘
```

### Backward (DGRAD)

```
grad_output [M, N]    weight_t [E, K, N]    offs [E]
     │                      │                  │
     └──────────┬───────────┘                  │
                ▼                              │
      te_moe::gemm_dgrad ─────────────────────┘
                │
   MXFP8Quantizer (grad: rowwise, weight: row+col → col-only)
   general_grouped_gemm (layout=NN, grad=True)
   → grad_A [M, K]
```

### Backward (WGRAD)

```
input_act [M, K]    grad_output [M, N]    offs [E]
     │                    │                  │
     └──────────┬─────────┘                  │
                ▼                            │
      te_moe::gemm_wgrad ───────────────────┘
                │
   _pad_for_mxfp8 (both A and grad_output)
   MXFP8Quantizer (both: row+col; inputs switched to col-only)
   general_grouped_gemm (layout=NT, grad=True)
   → wgrad [E, N, K] → transpose → grad_weight_t [E, K, N]
```

---

## 6. Usage

### 6.1 Programmatic (TorchAO)

```python
from torchao.prototype.moe_training.config import (
    MXFP8TrainingOpConfig,
    MXFP8TrainingRecipe,
)
from torchao.quantization.quant_api import quantize_

config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_TE)

def moe_filter(mod, fqn):
    return "experts" in fqn

quantize_(model, config=config, filter_fn=moe_filter)
```

Or directly:

```python
from torchao.quantization.quantize_.common import KernelPreference

config = MXFP8TrainingOpConfig(kernel_preference=KernelPreference.TE)
```

### 6.2 TorchTitan CLI

```bash
--model.converters="quantize.grouped_mm.mx"
--quantize.grouped_mm.mx.recipe_name="mxfp8_te"
--quantize.grouped_mm.mx.fqns="experts"
```

This routes through:
```
torchtitan recipe_name="mxfp8_te"
  → MXFP8TrainingRecipe.MXFP8_TE
  → MXFP8TrainingOpConfig(kernel_preference=KernelPreference.TE)
  → _swap_params → MXFP8TrainingWeightWrapperTensor(config=...)
  → __torch_function__ intercept
  → _MXFP8GroupedMM.forward(kernel_preference=TE)
  → te_moe::gemm_fwd
```

---

## 7. Known Caveats

1. **GPU→CPU sync**: `_offs_to_m_splits` calls `offs.tolist()` which is a
   device-to-host synchronization. This happens 3 times per MoE layer per
   training step (fwd, dgrad, wgrad). It's required because TE's
   `general_grouped_gemm` accepts `m_splits` as `List[int]`.

2. **Per-expert quantizer allocation**: Each call allocates `num_experts`
   `MXFP8Quantizer` objects. This is lightweight (Python-side only; the
   heavy work is in the C++ `split_quantize` / `quantize_impl`).

3. **Padding overhead**: Dynamic MoE routing produces arbitrary per-expert
   token counts. MXFP8 requires multiples of 32. Padding adds zero-rows
   per expert chunk. With HybridEP's `pad_multiple=32`, this is a no-op
   at the dispatch level, but the TE ops still do the check internally.

4. **BF16 mode**: The custom ops support `use_fp8=False` for non-quantized
   grouped GEMM via TE's kernel. This is currently only used for testing;
   the `MXFP8_TE` recipe always sets `use_fp8=True`.

5. **TE version dependency**: Requires `transformer_engine` with
   `MXFP8Quantizer` support (TE 2.x+). The import guard provides a clear
   error message if TE is missing.

---

## 8. Testing Plan

| Test | What it validates |
|---|---|
| `test_te_gemm_fwd` | Forward custom op (MXFP8 + BF16) vs BF16 ref |
| `test_te_gemm_dgrad` | DGRAD custom op vs BF16 ref |
| `test_te_gemm_wgrad` | WGRAD custom op vs BF16 ref |
| `test_te_integrated_fwd_bwd` | Full autograd path through `_MXFP8GroupedMM` with TE |
| `test_mxfp8_te_recipe` | Recipe enum → config creation |
| `test_te_model_conversion` | `quantize_()` parameter swap + forward/backward |
| `test_te_compile` | `torch.compile(fullgraph=True)` traces without breaks |
| **TorchTitan 2-node** | End-to-end DeepSeek-V3 training with TE grouped GEMM |

---

## 9. Future Work

- **Fused quantize-GEMM**: If TE exposes a fused quantize+GEMM API in the
  future, the custom ops can be updated internally without changing the
  dispatch interface.

- **3D x 3D grouped GEMM**: The current TE path only handles the 2D x 3D
  (routed experts) case. Adding 3D x 3D (shared experts / BMM) support
  is a natural extension.

- **Performance profiling**: Benchmarking TE vs CUTLASS grouped GEMM across
  different expert counts, sequence lengths, and hidden dims to provide
  guidance on when each backend is preferable.
