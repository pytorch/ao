# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import triton

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

if not (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
):
    pytest.skip("Test requires CUDA 12.8+ with SM >= 100", allow_module_level=True)

from torchao.prototype.moe_training.ep.kernels import generate_permute_indices
from torchao.prototype.moe_training.ep.permute import (
    _triton_permute_bwd,
    _triton_permute_bwd_kernel,
)


@pytest.mark.parametrize(
    "num_tokens",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "hidden_dim",
    [
        1024,
    ],
)
@pytest.mark.parametrize("num_local_experts", [2, 4, 8])
@pytest.mark.parametrize("ep_degree", [1, 2, 4])
@pytest.mark.parametrize(
    "alignment",
    [
        32,
    ],
)
def test_triton_permute_bwd(
    num_tokens, hidden_dim, num_local_experts, ep_degree, alignment
):
    device = "cuda"

    # Generate realistic permutation indices using generate_permute_indices
    # Simulate token distribution across experts
    tokens_per_expert_group = torch.randint(
        0,
        num_tokens // (num_local_experts * ep_degree) + 1,
        (ep_degree * num_local_experts,),
        device=device,
        dtype=torch.int32,
    )

    # Calculate padded length as in _Permute.forward
    x_padded_per_expert = num_tokens + num_local_experts * alignment
    padded_max_len = ((x_padded_per_expert + alignment - 1) // alignment) * alignment

    # Generate permutation indices
    permuted_indices, m_sizes, m_offsets = generate_permute_indices(
        tokens_per_expert_group,
        num_local_experts,
        ep_degree,
        padded_max_len,
        alignment,
    )

    # Get actual permuted size (may include padding)
    permuted_rows = permuted_indices.shape[0]
    original_rows = num_tokens
    original_cols = hidden_dim

    # Create gradient output tensor (this would come from upstream in backward pass)
    grad_output = torch.randn(
        permuted_rows, original_cols, device=device, dtype=torch.bfloat16
    )

    # PyTorch native implementation (from _Permute.backward, lines 144-150)
    # This is the reference implementation that was commented out
    grad_input_ref = grad_output.new_zeros((original_rows, original_cols))
    # Filter out padding indices (-1) when scattering
    valid_mask = permuted_indices != -1
    valid_indices = permuted_indices[valid_mask]
    grad_input_ref[valid_indices, :] = grad_output[valid_mask, :]

    # Triton kernel implementation
    grad_input_triton = _triton_permute_bwd(
        grad_output,
        permuted_indices,
        original_rows,
        original_cols,
    )

    # Compare results
    torch.testing.assert_close(
        grad_input_triton,
        grad_input_ref,
        rtol=0,
        atol=0,
        msg="Triton permute backward kernel output does not match PyTorch reference",
    )


def test_triton_permute_bwd_non_contiguous_inputs():
    """The backward op must handle a non-contiguous gradient and index tensor."""
    device = "cuda"
    rows, cols = 300, 384
    idx = torch.randperm(rows, device=device, dtype=torch.int32)
    idx = torch.cat([idx, torch.full((20,), -1, device=device, dtype=torch.int32)])
    grad = torch.randn(cols, idx.numel(), device=device, dtype=torch.bfloat16).t()
    # Every other element of a twice-as-long tensor: same values, stride 2.
    idx_strided = torch.stack([idx, torch.zeros_like(idx)], dim=1)[:, 0]
    assert not grad.is_contiguous() and not idx_strided.is_contiguous()

    ref = _triton_permute_bwd(grad.contiguous(), idx, rows, cols)
    assert torch.equal(_triton_permute_bwd(grad, idx, rows, cols), ref)
    assert torch.equal(
        _triton_permute_bwd(grad.contiguous(), idx_strided, rows, cols), ref
    )


def test_triton_permute_bwd_kernel_out_of_range_indices():
    """Indices outside [0, rows) are skipped like the -1 padding value.

    generate_permute_indices never emits them, but a bad index must not write
    outside the output. The kernel writes into rows [guard, guard + rows) of a
    larger buffer, so a stray write lands in the guard rows and is detected.
    """
    device = "cuda"
    rows, cols, guard = 64, 256, 8
    idx = [0, 5, rows, rows + 3, -1, -2, -7, 63]
    idx = torch.tensor(idx, device=device, dtype=torch.int32)
    grad = torch.randn(idx.numel(), cols, device=device, dtype=torch.bfloat16)

    sentinel = 7.0
    buf = torch.full(
        (guard + rows + guard, cols), sentinel, device=device, dtype=torch.bfloat16
    )
    out = buf[guard : guard + rows]
    out.zero_()
    block_rows, block_cols = 256, 256
    grid = (triton.cdiv(idx.numel(), block_rows), triton.cdiv(cols, block_cols))
    _triton_permute_bwd_kernel[grid](
        grad,
        idx,
        out,
        idx.numel(),
        cols,
        rows,
        cols,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        PADDING_VALUE=-1,
    )

    assert (buf[:guard] == sentinel).all(), "wrote before the output"
    assert (buf[guard + rows :] == sentinel).all(), "wrote past the output"
    valid = (idx >= 0) & (idx < rows)
    expected = torch.zeros_like(out)
    expected[idx[valid].long()] = grad[valid]
    assert torch.equal(out, expected)
