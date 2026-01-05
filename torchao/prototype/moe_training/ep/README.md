# Expert Parallelism (EP) Autograd Functions with MXFP8

This directory contains custom PyTorch autograd functions for efficient expert parallelism in MoE (Mixture of Experts) training with selective MXFP8 quantization.

## Overview

The implementation follows the architecture diagram where each color represents a separate autograd function:

```
Forward:  bf16 -> [PINK: a2a_dispatch] -> MXTensor -> [GREEN: permute] -> MXTensor ->
          [WHITE: GEMM] -> bf16 -> [PURPLE: unpermute] -> bf16 -> [BLUE: a2a_combine] -> bf16

Backward: bf16 <- [PINK: a2a_dispatch] <- bf16 <- [GREEN: permute] <- bf16 <-
          [WHITE: GEMM] <- bf16 <- [PURPLE: unpermute] <- MXTensor <- [BLUE: a2a_combine] <- bf16
```

## Key Design Principles

1. **Tensor Subclass Compatibility**: The autograd functions support passing `MXTensor` (a tensor subclass) through the backward pass, even when the forward pass outputs a different dtype (bf16).

2. **Selective Quantization**: MXFP8 quantization is applied strategically:
   - **a2a_dispatch**: Quantizes in forward to reduce communication
   - **a2a_combine**: Quantizes in backward to reduce communication
   - **permute/unpermute**: Work with quantized tensors where appropriate

3. **Shape Consistency**: PyTorch autograd only requires shape matching between forward output and backward grad_input. The dtype and tensor subclass can differ.

## Autograd Functions

### 1. Pink: `a2a_dispatch` ([a2a_dispatch.py](a2a_dispatch.py))

**Forward:**
- Input: bf16 tensor
- Quantize to MXFP8 using `triton_to_mxfp8_dim0()`
- All-to-all on qdata and scales separately
- Output: `MXTensor` (wrapping qdata + scales)

**Backward:**
- Input: bf16 gradient
- Inverse all-to-all (no quantization)
- Output: bf16 gradient

**Usage:**
```python
from torchao.prototype.moe_training.ep import a2a_dispatch

mx_output = a2a_dispatch(
    input,              # bf16 tensor
    output_splits,      # list[int]
    input_splits,       # list[int]
    group=dist.group.WORLD,
)
```

### 2. Green: `permute` ([permute.py](permute.py))

**Forward:**
- Input: `MXTensor` (qdata + scales)
- Permute qdata and scales separately based on routing indices
- Add padding row for alignment
- Output: `MXTensor` (permuted)

**Backward:**
- Input: bf16 gradient
- Unpermute using saved indices
- Output: bf16 gradient

**Usage:**
```python
from torchao.prototype.moe_training.ep import permute

# For MXFP8 pipeline (use_mxfp8=True)
(
    padded_shape,
    mx_permuted,
    permuted_indices,
    num_tokens_per_expert_padded,
    offsets,
) = permute(
    mx_tensor,               # MXTensor
    num_tokens_per_expert,   # torch.Tensor
    ep_degree,               # int
    num_local_experts,       # int
    alignment,               # int (block size)
    use_mxfp8=True,
)

# For BF16 pipeline (use_mxfp8=False)
(
    input_shape,
    permuted,
    permuted_indices,
    offsets,
) = permute(
    tensor,                  # bf16 tensor
    num_tokens_per_expert,   # torch.Tensor
    ep_degree,               # int
    num_local_experts,       # int
    alignment,               # int (block size)
    use_mxfp8=False,
)
```

### 3. Purple: `unpermute` ([unpermute.py](unpermute.py))

**Forward:**
- Input: bf16 tensor
- Unpermute using saved indices
- Remove padding row
- Output: bf16 tensor

**Backward:**
- Input: `MXTensor` (qdata + scales from downstream)
- Permute qdata and scales separately
- Add padding row
- Output: `MXTensor` (permuted)

**Usage:**
```python
from torchao.prototype.moe_training.ep import unpermute

# For MXFP8 pipeline (use_mxfp8=True)
output = unpermute(
    input,              # bf16 tensor
    permuted_indices,   # torch.Tensor
    output_shape,       # torch.Size
    use_mxfp8=True,
)

# For BF16 pipeline (use_mxfp8=False)
output = unpermute(
    input,              # bf16 tensor
    permuted_indices,   # torch.Tensor
    output_shape,       # torch.Size
    use_mxfp8=False,
)
```

### 4. Blue: `a2a_combine` ([a2a_combine.py](a2a_combine.py))

**Forward:**
- Input: bf16 tensor
- All-to-all in bf16 (no quantization)
- Output: bf16 tensor

**Backward:**
- Input: bf16 gradient
- Quantize to MXFP8 using `triton_to_mxfp8_dim0()`
- Inverse all-to-all on qdata and scales separately
- Output: `MXTensor` (wrapping grad qdata + scales)

**Usage:**
```python
from torchao.prototype.moe_training.ep import a2a_combine

output = a2a_combine(
    input,              # bf16 tensor
    output_splits,      # list[int]
    input_splits,       # list[int]
    group=dist.group.WORLD,
)
```

## Complete Pipeline Example

```python
import torch
import torch.distributed as dist
from torchao.prototype.moe_training.ep import (
    a2a_dispatch,
    permute,
    unpermute,
    a2a_combine,
)

# Forward pass
x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)

# 1. Dispatch with quantization
mx_dispatched = a2a_dispatch(x, output_splits, input_splits, group)

# 2. Permute (MXFP8 mode)
(
    padded_shape,
    mx_permuted,
    permuted_indices,
    num_tokens_per_expert_padded,
    offsets,
) = permute(
    mx_dispatched,
    num_tokens_per_expert_group,
    ep_degree,
    num_local_experts,
    block_size,
    use_mxfp8=True,
)

# 3. GEMM (existing mxfp8 grouped GEMM - outputs bf16)
out = mxfp8_grouped_gemm(mx_permuted, weights, offsets)

# 4. Unpermute (MXFP8 mode)
padded_output_shape = torch.Size([padded_shape[0], out.shape[-1]])
out_unpermuted = unpermute(
    out, permuted_indices, padded_output_shape, use_mxfp8=True
)

# 5. Combine
final_out = a2a_combine(out_unpermuted, output_splits, input_splits, group)

# Backward pass is automatic via autograd!
loss = final_out.sum()
loss.backward()
```

## Implementation Notes

### MXTensor Handling

Since `MXTensor` only supports limited operations, the implementations work with `qdata` and `scale` components separately:

```python
# Extract components
qdata = mx_tensor.qdata
scales = mx_tensor.scale

# Perform operations separately
qdata_result = some_operation(qdata)
scales_result = some_operation(scales)

# Wrap back into MXTensor
result = MXTensor(
    qdata_result,
    scales_result,
    elem_dtype=mx_tensor._elem_dtype,
    block_size=mx_tensor.block_size,
    orig_dtype=mx_tensor._orig_dtype,
    kernel_preference=mx_tensor.kernel_preference,
    act_quant_kwargs=mx_tensor.act_quant_kwargs,
    is_swizzled_scales=mx_tensor._is_swizzled_scales,
)
```

### Backward Pass Tensor Subclass Detection

The backward methods check if they received an `MXTensor` and handle it accordingly:

```python
def backward(ctx, grad_output):
    if isinstance(grad_output, MXTensor):
        # Access internal quantized components
        qdata = grad_output.qdata
        scales = grad_output.scale
        # ... work with quantized data
    else:
        # Handle regular bf16 tensor
        # ...
```

### Communication Considerations

- All-to-all operations are asynchronous and require explicit waits
- NCCL doesn't support `torch.float8_e8m0fnu`, so scales are temporarily viewed as `torch.uint8` for communication
- Split sizes must be provided as lists for the collective operations

## Testing

See [test_autograd_tensor_subclass.py](../../../test_autograd_tensor_subclass.py) for a proof-of-concept demonstration that PyTorch autograd supports passing tensor subclasses through the backward pass.

## References

- MXTensor implementation: `torchao/prototype/mx_formats/mx_tensor.py`
- MXFP8 communication kernels: `torchao/prototype/moe_training/kernels/mxfp8/comms.py`
- Original permute/unpermute: `torchtitan/torchtitan/models/moe/utils.py`
