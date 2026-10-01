# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

if not (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
):
    pytest.skip("Test requires CUDA 12.8+ with SM >= 100", allow_module_level=True)

from torchao.prototype.moe_training.ep.kernels import (
    _fill_indices_kernel,
    generate_permute_indices,
)
from torchao.prototype.moe_training.ep.permute import _triton_permute_bwd


@pytest.mark.parametrize(
    "experts_per_rank,num_ranks,counts,max_len,alignment",
    [
        (2, 2, [4, 4, 4, 4], 12, 4),
        (4, 2, [3, 3, 3, 3, 3, 3, 3, 3], 20, 2),
        (1, 4, [8, 8, 8, 8], 16, 8),
    ],
)
def test_generate_permute_indices_respects_max_len(
    experts_per_rank, num_ranks, counts, max_len, alignment
):
    token_counts = torch.tensor(counts, dtype=torch.int32, device="cuda")
    start_indices = torch.cumsum(token_counts, 0) - token_counts
    total = torch.clamp_min(token_counts.view(num_ranks, -1).sum(0), alignment)
    m_sizes = ((total + alignment - 1) // alignment * alignment).to(torch.int32)
    write_offsets = torch.cumsum(m_sizes, 0) - m_sizes

    sentinel = -999
    guarded_output = torch.full(
        (max_len + 64,), sentinel, dtype=torch.int32, device="cuda"
    )
    _fill_indices_kernel[(min(experts_per_rank, 1024),)](
        token_counts,
        start_indices,
        write_offsets,
        guarded_output[:max_len],
        experts_per_rank,
        num_ranks,
        max_len,
        BLOCK_SIZE=128,
    )
    torch.cuda.synchronize()
    assert torch.all(guarded_output[max_len:] == sentinel)

    gpu_indices, _, _ = generate_permute_indices(
        token_counts, experts_per_rank, num_ranks, max_len, alignment
    )
    cpu_indices, _, _ = generate_permute_indices(
        token_counts.cpu(),
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        use_cpu=True,
    )
    torch.testing.assert_close(gpu_indices.cpu(), cpu_indices)


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
