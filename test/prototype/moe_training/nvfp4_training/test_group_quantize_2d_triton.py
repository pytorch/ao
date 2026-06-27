"""Tests for dense-expert grouped NVFP4 2D weight quantization."""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.utils import is_sm_at_least_100, torch_version_at_least

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
        triton_group_weight_quantize_2d,
    )
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )


requires_grouped_kernel = pytest.mark.skipif(
    not (
        has_triton()
        and is_sm_at_least_100()
        and torch_version_at_least("2.10.0")
    ),
    reason="requires Triton, PyTorch 2.10+, and SM100+",
)


@requires_grouped_kernel
@pytest.mark.parametrize(
    "shape",
    [pytest.param((1, 128, 256), id="one-tile"), pytest.param((3, 256, 512), id="multi-tile")],
)
@torch.no_grad()
def test_group_quantize_2d_matches_independent_experts(shape):
    """Every expert must match an independent launch of the established 2D op."""
    torch.manual_seed(42)
    weights = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    global_amax = weights.float().abs().amax(dim=(1, 2))

    actual = triton_group_weight_quantize_2d(
        weights, global_amax, num_tensors=shape[0]
    )
    expected_by_expert = [
        triton_weight_quantize_2d(weights[e], global_amax[e])
        for e in range(shape[0])
    ]

    for output_idx, grouped_output in enumerate(actual):
        expected = torch.stack(
            [outputs[output_idx] for outputs in expected_by_expert]
        )
        torch.testing.assert_close(grouped_output, expected, atol=0, rtol=0)


@requires_grouped_kernel
def test_group_quantize_2d_register_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    E, M, N = 3, 256, 512
    with FakeTensorMode():
        weights = torch.empty((E, M, N), dtype=torch.bfloat16, device="cuda")
        global_amax = torch.empty((E,), dtype=torch.float32, device="cuda")
        qa, sfa, qa_t, sfa_t = triton_group_weight_quantize_2d(
            weights, global_amax, num_tensors=E
        )

    assert qa.shape == (E, M, N // 2)
    assert qa.dtype == torch.uint8
    assert sfa.shape == (E, M // 128, N // 64, 32, 16)
    assert sfa.dtype == torch.float8_e4m3fn
    assert qa_t.shape == (E, N, M // 2)
    assert qa_t.dtype == torch.uint8
    assert sfa_t.shape == (E, N // 128, M // 64, 32, 16)
    assert sfa_t.dtype == torch.float8_e4m3fn
