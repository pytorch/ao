# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# We need to skip before doing any imports which would use triton, since
# triton won't be available on CPU builds
if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

from torchao.prototype.moe_training.ep.kernels import generate_permute_indices
from torchao.prototype.moe_training.ep.permute import _triton_permute_bwd


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
