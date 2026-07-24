"""Tests for dense-expert grouped NVFP4 2D weight quantization."""

import pytest
import torch
from torch.utils._triton import has_triton

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
        triton_group_weight_quantize_2d,
    )
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )


requires_grouped_kernel = pytest.mark.skipif(
    not (has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0")),
    reason="requires Triton, PyTorch 2.10+, and SM100+",
)

_CORRECTNESS_SHAPES = [
    pytest.param((1, 128, 256), id="one-tile"),
    pytest.param((3, 256, 512), id="multi-tile"),
    *[
        pytest.param(
            (shape.experts, shape.m, shape.n),
            id=f"deepseek-{shape.model}-{shape.projection}",
        )
        for shape in get_deepseek_v3_weight_shapes(factorized_experts=2)
    ],
]


@requires_grouped_kernel
@pytest.mark.parametrize("shape", _CORRECTNESS_SHAPES)
@torch.no_grad()
def test_group_quantize_2d_matches_independent_experts(shape):
    """Every expert must match an independent launch of the established 2D op."""
    torch.manual_seed(42)
    weights = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    global_amax = weights.float().abs().amax(dim=(1, 2))

    actual = triton_group_weight_quantize_2d(weights, global_amax, num_tensors=shape[0])
    expected_by_expert = [
        triton_weight_quantize_2d(weights[e], global_amax[e]) for e in range(shape[0])
    ]

    for output_idx, grouped_output in enumerate(actual):
        expected = torch.stack([outputs[output_idx] for outputs in expected_by_expert])
        torch.testing.assert_close(grouped_output, expected, atol=0, rtol=0)


@requires_grouped_kernel
@torch.no_grad()
def test_group_quantize_2d_matches_torch_oracle():
    """Rowwise codes and scales match nvfp4_quantize on aligned 16x16 blocks."""
    torch.manual_seed(42)
    E, M, N = 2, 128, 256
    weights = torch.randn(
        (E, M // 16, N), dtype=torch.bfloat16, device="cuda"
    ).repeat_interleave(16, dim=1)
    global_amax = weights.float().abs().amax(dim=(1, 2))

    actual_codes, actual_scales, _, _ = triton_group_weight_quantize_2d(
        weights, global_amax, num_tensors=E
    )

    for expert in range(E):
        expected_scales, expected_codes = nvfp4_quantize(
            weights[expert],
            per_tensor_scale=per_tensor_amax_to_scale(global_amax[expert]),
        )
        expected_scales = to_blocked(expected_scales).view_as(actual_scales[expert])
        torch.testing.assert_close(
            actual_scales[expert], expected_scales, atol=0, rtol=0
        )
        actual_unpacked = torch.stack(
            (actual_codes[expert] & 0xF, actual_codes[expert] >> 4), dim=-1
        )
        expected_unpacked = torch.stack(
            (expected_codes & 0xF, expected_codes >> 4), dim=-1
        )
        torch.testing.assert_close(
            actual_unpacked >> 3, expected_unpacked >> 3, atol=0, rtol=0
        )
        magnitude_diff = (
            (actual_unpacked & 0x7).to(torch.int16)
            - (expected_unpacked & 0x7).to(torch.int16)
        ).abs()
        assert magnitude_diff.max().item() <= 1


@requires_grouped_kernel
@torch.no_grad()
def test_group_quantize_2d_large_expert_offset():
    """The last expert remains addressable when its input base exceeds int32."""
    E, M, N = 65, 8192, 8192
    weights = torch.empty((E, M, N), dtype=torch.bfloat16, device="cuda")
    weights[-1].fill_(1.0)
    global_amax = torch.ones((E,), dtype=torch.float32, device="cuda")

    actual = triton_group_weight_quantize_2d(weights, global_amax, num_tensors=E)
    expected = triton_weight_quantize_2d(weights[-1], global_amax[-1])

    for grouped_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(grouped_output[-1], expected_output, atol=0, rtol=0)


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


@requires_grouped_kernel
@pytest.mark.parametrize("invalid_amax", ["cpu", "noncontiguous"])
def test_group_quantize_2d_validates_global_amax_storage(invalid_amax):
    weights = torch.empty((2, 128, 128), dtype=torch.bfloat16, device="cuda")
    if invalid_amax == "cpu":
        global_amax = torch.empty((2,), dtype=torch.float32)
        error = "same device as A"
    else:
        global_amax = torch.empty((4,), dtype=torch.float32, device="cuda")[::2]
        error = "contiguous"

    with pytest.raises(ValueError, match=error):
        triton_group_weight_quantize_2d(weights, global_amax, num_tensors=2)
