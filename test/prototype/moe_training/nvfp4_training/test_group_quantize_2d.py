"""Tests for dense-expert grouped NVFP4 2D weight quantization."""

import pytest
import torch
from torch.utils._triton import has_triton

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from test.prototype.moe_training.nvfp4_training._assertions import (
    assert_codes_bitwise,
    assert_scales_bitwise,
)
from test.prototype.moe_training.nvfp4_training.nvfp4_reference import (
    reference_group_weight_quantize_2d,
)
from test.prototype.moe_training.nvfp4_training.test_quantize_2d import (
    _assert_scales_match_up_to_rounding_ties,
)
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_cutedsl import (
    cutedsl_group_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
    cutedsl_weight_quantize_2d,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_HAS_TRITON = has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0")
if _HAS_TRITON:
    from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
        triton_group_weight_quantize_2d,
    )
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )


requires_grouped_kernel = pytest.mark.skipif(
    not _HAS_TRITON, reason="requires Triton, PyTorch 2.10+, and SM100+"
)
_skip_no_triton = requires_grouped_kernel
_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

_KERNELS = [
    pytest.param("triton", marks=_skip_no_triton, id="triton"),
    pytest.param("cutedsl", marks=_skip_no_cutedsl, id="cutedsl"),
]

_CORRECTNESS_SHAPES = [
    pytest.param((1, 128, 256), id="one-tile"),
    pytest.param((3, 256, 512), id="multi-tile"),
    pytest.param((4, 512, 256), id="multi-tile-per-expert"),
    # M % 128 but not % 256: the 128-row CuteDSL supertile, more than one per expert.
    pytest.param((2, 384, 256), id="multi-supertile-128row"),
    *[
        pytest.param(
            (shape.experts, shape.m, shape.n),
            id=f"deepseek-{shape.model}-{shape.projection}",
        )
        for shape in get_deepseek_v3_weight_shapes(factorized_experts=2)
    ],
]


def _skip_if_unsupported_shape(kernel: str, M: int) -> None:
    """Both backends need M % 128: Triton's BLOCK_M and the CuteDSL supertile floor."""
    if M % 128 != 0:
        pytest.skip("group weight quantize requires out_features % 128 == 0")


def _group_quantize(kernel, weights, global_amax, num_tensors):
    if kernel == "triton":
        return triton_group_weight_quantize_2d(weights, global_amax, num_tensors)
    return cutedsl_group_weight_quantize_2d(weights, global_amax, num_tensors)


def _quantize_expert(kernel, weights, global_amax, e):
    # The CuteDSL entry point rejects views with a nonzero byte offset, so the per-expert
    # slices are cloned rather than passed as views.
    if kernel == "triton":
        return triton_weight_quantize_2d(weights[e], global_amax[e])
    return cutedsl_weight_quantize_2d(weights[e].clone(), global_amax[e].clone())


def group_weight_quantize_2d_ref(weights, global_amax, scale_shape):
    """PyTorch NVFP4 reference for grouped 2D weight quantization."""
    expected_codes = []
    expected_scales = []
    for expert in range(weights.shape[0]):
        scales, codes = nvfp4_quantize(
            weights[expert],
            per_tensor_scale=per_tensor_amax_to_scale(global_amax[expert]),
        )
        expected_codes.append(codes)
        expected_scales.append(to_blocked(scales).view(scale_shape))
    return torch.stack(expected_codes), torch.stack(expected_scales)


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("shape", _CORRECTNESS_SHAPES)
@torch.no_grad()
def test_group_quantize_2d_matches_independent_experts(kernel, shape):
    """Every expert must match an independent launch of the established 2D op."""
    _skip_if_unsupported_shape(kernel, shape[1])
    torch.manual_seed(42)
    weights = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    global_amax = weights.float().abs().amax(dim=(1, 2))

    actual = _group_quantize(kernel, weights, global_amax, shape[0])
    expected_by_expert = [
        _quantize_expert(kernel, weights, global_amax, e) for e in range(shape[0])
    ]

    for output_idx, grouped_output in enumerate(actual):
        expected = torch.stack([outputs[output_idx] for outputs in expected_by_expert])
        torch.testing.assert_close(grouped_output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_quantize_2d_uses_each_experts_own_amax(kernel):
    """A misindexed expert must not survive: give every expert a distinct global scale.

    ``test_group_quantize_2d_matches_independent_experts`` draws every expert from the same
    distribution, so its per-expert amaxes are nearly equal and a wrong expert index would
    barely perturb the codes. Scaling amax by the expert index makes the scales an order of
    magnitude apart, so any offset error shows up as saturated or flushed codes.
    """
    E, M, N = 9, 256, 128
    _skip_if_unsupported_shape(kernel, M)
    torch.manual_seed(7)
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device="cuda")
    global_amax = weights.float().abs().amax(dim=(1, 2)) * torch.arange(
        1, E + 1, device="cuda", dtype=torch.float32
    )

    actual = _group_quantize(kernel, weights, global_amax, E)
    expected_by_expert = [
        _quantize_expert(kernel, weights, global_amax, e) for e in range(E)
    ]

    for output_idx, grouped_output in enumerate(actual):
        expected = torch.stack([outputs[output_idx] for outputs in expected_by_expert])
        torch.testing.assert_close(grouped_output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_quantize_2d_matches_torch_oracle(kernel):
    """Rowwise codes and scales match nvfp4_quantize on aligned 16x16 blocks.

    Scales are compared to within one E4M3 step, not bitwise: mx_formats' nvfp4_quantize
    multiplies by a reciprocal and applies an E4M3_EPS floor, where both kernels follow
    TE's correctly-rounded div_rn with no floor. The two are mathematically equal and can
    land on adjacent representable values. ``test_group_quantize_2d_vs_transformer_engine_reference``
    is the bitwise scale contract; ``test_cutedsl_group_quantize_2d_matches_triton`` pins
    the backends to each other.
    """
    torch.manual_seed(42)
    E, M, N = 2, 256, 256
    weights = torch.randn(
        (E, M // 16, N), dtype=torch.bfloat16, device="cuda"
    ).repeat_interleave(16, dim=1)
    global_amax = weights.float().abs().amax(dim=(1, 2))

    actual_codes, actual_scales, _, _ = _group_quantize(kernel, weights, global_amax, E)
    expected_codes, expected_scales = group_weight_quantize_2d_ref(
        weights, global_amax, actual_scales.shape[1:]
    )

    for expert in range(E):
        _assert_scales_match_up_to_rounding_ties(
            actual_scales[expert],
            expected_scales[expert],
            f"expert {expert} rowwise SF vs mx_formats oracle",
        )
        actual_unpacked = torch.stack(
            (actual_codes[expert] & 0xF, actual_codes[expert] >> 4), dim=-1
        )
        expected_unpacked = torch.stack(
            (expected_codes[expert] & 0xF, expected_codes[expert] >> 4), dim=-1
        )
        torch.testing.assert_close(
            actual_unpacked >> 3, expected_unpacked >> 3, atol=0, rtol=0
        )
        magnitude_diff = (
            (actual_unpacked & 0x7).to(torch.int16)
            - (expected_unpacked & 0x7).to(torch.int16)
        ).abs()
        assert magnitude_diff.max().item() <= 1


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("shape", _CORRECTNESS_SHAPES)
@torch.no_grad()
def test_group_quantize_2d_vs_transformer_engine_reference(kernel, shape):
    """Both backends must reproduce TransformerEngine's per-expert 16x16 arithmetic.

    Scales and codes bitwise for every expert and both directions.
    """
    E, M, N = shape
    _skip_if_unsupported_shape(kernel, M)
    torch.manual_seed(11)
    W = torch.randn((E, M, N), dtype=torch.bfloat16, device="cuda")
    amax = W.float().abs().amax(dim=(1, 2))

    codes, sf, t_codes, t_sf = _group_quantize(kernel, W, amax, E)
    ref_codes, ref_sf, ref_t_codes, ref_t_sf = reference_group_weight_quantize_2d(
        W, amax, E
    )
    assert_scales_bitwise(sf, ref_sf, "rowwise SF")
    assert_scales_bitwise(t_sf, ref_t_sf, "colwise SF")
    assert_codes_bitwise(codes, ref_codes, "rowwise codes")
    assert_codes_bitwise(t_codes, ref_t_codes, "colwise codes")


@_skip_no_triton
@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_group_quantize_2d_matches_triton():
    """The two backends are byte-for-byte interchangeable, codes as well as scales."""
    torch.manual_seed(3)
    E, M, N = 3, 256, 512
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device="cuda")
    global_amax = weights.float().abs().amax(dim=(1, 2))

    cutedsl = _group_quantize("cutedsl", weights, global_amax, E)
    triton_out = _group_quantize("triton", weights, global_amax, E)
    for name, c, t in zip(("q", "sf", "qt", "sft"), cutedsl, triton_out):
        assert torch.equal(c, t), f"{name} differs between backends"


@requires_grouped_kernel
@torch.no_grad()
def test_group_quantize_2d_large_expert_offset():
    """The last expert remains addressable when its input base exceeds int32.

    Triton-only: this guards the ``program_id(2).to(tl.int64)`` cast. The CuteDSL path reaches
    every tensor through TMA, whose coordinates stay decomposed per dimension, so it has no
    flattened index to overflow -- and the 8.7 GB allocation is not worth repeating for it.
    """
    E, M, N = 65, 8192, 8192
    weights = torch.empty((E, M, N), dtype=torch.bfloat16, device="cuda")
    weights[-1].fill_(1.0)
    global_amax = torch.ones((E,), dtype=torch.float32, device="cuda")

    actual = triton_group_weight_quantize_2d(weights, global_amax, num_tensors=E)
    expected = triton_weight_quantize_2d(weights[-1], global_amax[-1])

    for grouped_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(grouped_output[-1], expected_output, atol=0, rtol=0)


@pytest.mark.parametrize("kernel", _KERNELS)
def test_group_quantize_2d_register_fake_shapes(kernel):
    from torch._subclasses.fake_tensor import FakeTensorMode

    E, M, N = 3, 256, 512
    with FakeTensorMode():
        weights = torch.empty((E, M, N), dtype=torch.bfloat16, device="cuda")
        global_amax = torch.empty((E,), dtype=torch.float32, device="cuda")
        qa, sfa, qa_t, sfa_t = _group_quantize(kernel, weights, global_amax, E)

    assert qa.shape == (E, M, N // 2)
    assert qa.dtype == torch.uint8
    assert sfa.shape == (E, M // 128, N // 64, 32, 16)
    assert sfa.dtype == torch.float8_e4m3fn
    assert qa_t.shape == (E, N, M // 2)
    assert qa_t.dtype == torch.uint8
    assert sfa_t.shape == (E, N // 128, M // 64, 32, 16)
    assert sfa_t.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("invalid_amax", ["cpu", "noncontiguous"])
def test_group_quantize_2d_validates_global_amax_storage(kernel, invalid_amax):
    weights = torch.empty((2, 256, 128), dtype=torch.bfloat16, device="cuda")
    if invalid_amax == "cpu":
        global_amax = torch.empty((2,), dtype=torch.float32)
        error = "same device as A"
    else:
        global_amax = torch.empty((4,), dtype=torch.float32, device="cuda")[::2]
        error = "contiguous"

    with pytest.raises(ValueError, match=error):
        _group_quantize(kernel, weights, global_amax, 2)


@_skip_no_cutedsl
def test_cutedsl_group_quantize_2d_requires_out_features_128():
    weights = torch.empty((2, 192, 256), dtype=torch.bfloat16, device="cuda")
    global_amax = torch.ones((2,), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="out_features"):
        cutedsl_group_weight_quantize_2d(weights, global_amax, 2)
