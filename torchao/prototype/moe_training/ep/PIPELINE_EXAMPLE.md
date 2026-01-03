# Complete EP + MXFP8 Grouped GEMM Pipeline

## Overview

This document demonstrates the complete end-to-end pipeline integrating the EP autograd functions with the existing MXFP8 grouped GEMM kernel.

## Pipeline Architecture

```
Forward Pass:
bf16 input → [a2a_dispatch] → MXTensor → [permute] → MXTensor →
[_MXFP8GroupedMM] → bf16 → [unpermute] → bf16 → [a2a_combine] → bf16 output

Backward Pass:
bf16 grad ← [a2a_dispatch] ← bf16 ← [permute] ← bf16 ←
[_MXFP8GroupedMM] ← MXTensor ← [unpermute] ← MXTensor ← [a2a_combine] ← bf16 grad
```

## Complete Example

See [test_integration.py](../../../test/prototype/moe_training/ep/test_integration.py) for a working test.

```python
import torch
import torch.distributed as dist
from torchao.prototype.moe_training.ep import (
    a2a_dispatch,
    permute,
    unpermute,
    a2a_combine,
)
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)

# Setup
tokens = 64
hidden_dim = 128
num_experts = 4
world_size = 2

# Inputs
input_tensor = torch.randn(
    tokens, hidden_dim, dtype=torch.bfloat16, requires_grad=True
)

expert_weights = torch.randn(
    num_experts, hidden_dim, hidden_dim, dtype=torch.bfloat16, requires_grad=True
)
# Convert to column-major layout for grouped GEMM
expert_weights_t = expert_weights.transpose(-2, -1).contiguous().transpose(-2, -1)

# Routing info
input_splits = [32, 32]  # Per-rank token counts
output_splits = [28, 36]  # After routing
permuted_indices = torch.randperm(tokens)  # Token routing order
padded_shape = torch.Size([tokens + 1, hidden_dim])
group_offsets = torch.tensor([16, 32, 48, 64], dtype=torch.int32)  # Expert boundaries

# ========== Forward Pass ==========

# 1. Pink: Dispatch with MXFP8 quantization
mx_dispatched = a2a_dispatch(input_tensor, output_splits, input_splits)
# Output: MXTensor with qdata (fp8) and scale (e8m0)

# 2. Green: Permute tokens to experts
mx_permuted = permute(mx_dispatched, permuted_indices, padded_shape)
# Output: MXTensor (permuted)

# 3. White: MXFP8 Grouped GEMM
gemm_output = _to_mxfp8_then_scaled_grouped_mm(
    mx_permuted,           # Input: MXTensor (accepts quantized input!)
    expert_weights_t,      # Weights: bf16 (will be quantized internally)
    offs=group_offsets,    # Expert boundaries
    block_size=32,         # MXFP8 block size
)
# Output: bf16

# 4. Purple: Unpermute tokens back
unpermuted = unpermute(gemm_output, permuted_indices, padded_shape)
# Output: bf16

# 5. Blue: Combine with all-to-all
final_output = a2a_combine(unpermuted, output_splits, input_splits)
# Output: bf16

# ========== Backward Pass (automatic!) ==========

loss = final_output.sum()
loss.backward()

# Gradients are computed automatically:
# - input_tensor.grad: bf16
# - expert_weights.grad: bf16
```

## Key Integration Points

### 1. a2a_dispatch → permute
```python
mx_dispatched = a2a_dispatch(...)  # Returns MXTensor
mx_permuted = permute(mx_dispatched, ...)  # Accepts MXTensor ✅
```

### 2. permute → _MXFP8GroupedMM
```python
mx_permuted = permute(...)  # Returns MXTensor
gemm_output = _to_mxfp8_then_scaled_grouped_mm(mx_permuted, ...)  # Accepts MXTensor ✅
```
The GEMM function checks `isinstance(A, MXTensor)` and extracts `.qdata` and `.scale`.

### 3. _MXFP8GroupedMM → unpermute
```python
gemm_output = _to_mxfp8_then_scaled_grouped_mm(...)  # Returns bf16
unpermuted = unpermute(gemm_output, ...)  # Accepts bf16 ✅
```

### 4. unpermute → a2a_combine
```python
unpermuted = unpermute(...)  # Returns bf16
final_output = a2a_combine(unpermuted, ...)  # Accepts bf16 ✅
```

### 5. a2a_combine backward → unpermute backward
```python
# In backward pass:
# a2a_combine.backward receives bf16 gradient
# Quantizes to MXTensor
# Passes MXTensor to unpermute.backward ✅
```

### 6. unpermute backward → _MXFP8GroupedMM backward
```python
# In backward pass:
# unpermute.backward receives MXTensor gradient
# Permutes the MXTensor components
# Passes MXTensor to GEMM.backward ✅
```
The GEMM backward checks `isinstance(grad_out, MXTensor)` and extracts `.qdata` and `.scale`.

### 7. _MXFP8GroupedMM backward → permute backward
```python
# In backward pass:
# GEMM.backward returns bf16 gradients
# permute.backward receives bf16 gradient ✅
```

### 8. permute backward → a2a_dispatch backward
```python
# In backward pass:
# permute.backward returns bf16 gradient
# a2a_dispatch.backward receives bf16 gradient ✅
```

## Backward Pass Flow Detail

```python
loss.backward()
# ↓
# Blue a2a_combine.backward(grad=ones_like(final_output))
#   - Receives: bf16 gradient
#   - Quantizes to MXTensor
#   - All-to-all
#   - Returns: MXTensor gradient
# ↓
# Purple unpermute.backward(grad=MXTensor)
#   - Receives: MXTensor gradient
#   - Adds padding, permutes qdata and scale separately
#   - Returns: MXTensor gradient
# ↓
# White _MXFP8GroupedMM.backward(grad=MXTensor)
#   - Receives: MXTensor gradient (checks isinstance!)
#   - Extracts qdata and scale
#   - Computes grad_A and grad_B_t
#   - Returns: bf16 gradients
# ↓
# Green permute.backward(grad=bf16)
#   - Receives: bf16 gradient
#   - Unpermutes, removes padding
#   - Returns: bf16 gradient
# ↓
# Pink a2a_dispatch.backward(grad=bf16)
#   - Receives: bf16 gradient
#   - Inverse all-to-all
#   - Returns: bf16 gradient
```

## Benefits

1. **Communication Efficiency**: Quantizes before all-to-all (pink forward, blue backward)
2. **Computation Efficiency**: GEMM operates on quantized data
3. **Flexibility**: GEMM can accept pre-quantized or high-precision inputs
4. **Automatic Differentiation**: Full PyTorch autograd support

## Testing

Run the integration test:
```bash
conda run -n torch env PYTHONPATH=/home/dev/ao:$PYTHONPATH \
    python test/prototype/moe_training/ep/test_integration.py
```

This test validates:
- ✅ MXTensor passes through forward pipeline correctly
- ✅ MXTensor passes through backward pipeline correctly
- ✅ Gradients are computed for both input and weights
- ✅ All dtype transitions work as expected
