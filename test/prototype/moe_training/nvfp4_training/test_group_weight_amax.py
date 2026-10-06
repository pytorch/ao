"""Tests for the grouped per-expert NVFP4 weight amax kernel.

The op replaces a plain PyTorch reduction in the grouped weight path, so the contract is
bit-exact agreement with ``W.float().abs().amax(dim=(1, 2))`` -- including NaN.
"""

import pytest
import torch
from torch.utils._triton import has_triton

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from torchao.utils import torch_version_at_least

_HAS_TRITON = has_triton() and torch_version_at_least("2.10.0")
if _HAS_TRITON:
    from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
        triton_group_weight_amax,
    )

requires_grouped_kernel = pytest.mark.skipif(
    not _HAS_TRITON, reason="requires Triton and PyTorch 2.10+"
)

_CORRECTNESS_SHAPES = [
    pytest.param((1, 128, 256), id="one-expert"),
    pytest.param((3, 256, 512), id="multi-expert"),
    pytest.param((3, 17, 33), id="ragged-tail"),
    *[
        pytest.param(
            (shape.experts, shape.m, shape.n),
            id=f"deepseek-{shape.model}-{shape.projection}",
        )
        for shape in get_deepseek_v3_weight_shapes(factorized_experts=2)
    ],
]


def group_weight_amax_ref(weights):
    """PyTorch reference for the per-expert global amax."""
    return weights.float().abs().amax(dim=(1, 2))


@requires_grouped_kernel
@pytest.mark.parametrize("shape", _CORRECTNESS_SHAPES)
@torch.no_grad()
def test_group_weight_amax_matches_torch_reduction(shape):
    """Bit-exact with the PyTorch reduction; ``ragged-tail`` covers the load mask."""
    torch.manual_seed(42)
    weights = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    actual = triton_group_weight_amax(weights, shape[0])

    torch.testing.assert_close(actual, group_weight_amax_ref(weights), atol=0, rtol=0)


@requires_grouped_kernel
@torch.no_grad()
def test_group_weight_amax_uses_each_experts_own_values():
    """A misindexed expert must not survive: give every expert a distinct magnitude.

    ``test_group_weight_amax_matches_torch_reduction`` draws every expert from the same
    distribution, so its per-expert amaxes are nearly equal and a wrong expert index would
    barely perturb the result. Scaling by the expert index makes them an order of
    magnitude apart, so any offset error shows up immediately.
    """
    E, M, N = 9, 256, 128
    torch.manual_seed(7)
    scale = torch.arange(1, E + 1, device="cuda", dtype=torch.bfloat16).view(E, 1, 1)
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device="cuda") * scale

    actual = triton_group_weight_amax(weights, E)

    torch.testing.assert_close(actual, group_weight_amax_ref(weights), atol=0, rtol=0)


@requires_grouped_kernel
@torch.no_grad()
def test_group_weight_amax_propagates_nan_per_expert():
    """NaN must reach the affected expert only.

    ``tl.max`` does not propagate NaN, so the kernel re-injects it after the reduction.
    Without that, a NaN weight silently yields a finite scale and the quantized expert
    looks healthy.
    """
    E, M, N = 4, 256, 128
    torch.manual_seed(11)
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device="cuda")
    weights[2, 3, 4] = float("nan")

    actual = triton_group_weight_amax(weights, E)

    assert torch.isnan(actual[2])
    unaffected = [0, 1, 3]
    torch.testing.assert_close(
        actual[unaffected],
        group_weight_amax_ref(weights)[unaffected],
        atol=0,
        rtol=0,
    )


@requires_grouped_kernel
@torch.no_grad()
def test_group_weight_amax_large_expert_offset():
    """The last expert remains addressable when its input base exceeds int32.

    Guards the ``program_id(1).to(tl.int64)`` cast. Only the last expert is filled, so
    the other experts' (uninitialized) amaxes are not asserted on.
    """
    E, M, N = 65, 8192, 8192
    weights = torch.empty((E, M, N), dtype=torch.bfloat16, device="cuda")
    weights[-1].fill_(1.0)

    actual = triton_group_weight_amax(weights, num_tensors=E)

    assert actual[-1].item() == 1.0


@requires_grouped_kernel
def test_group_weight_amax_register_fake_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode

    E, M, N = 3, 256, 512
    with FakeTensorMode():
        weights = torch.empty((E, M, N), dtype=torch.bfloat16, device="cuda")
        global_amax = triton_group_weight_amax(weights, E)

    assert global_amax.shape == (E,)
    assert global_amax.dtype == torch.float32


@requires_grouped_kernel
@pytest.mark.parametrize(
    "invalid,error",
    [
        ("dtype", "bfloat16"),
        ("ndim", "3-D"),
        ("noncontiguous", "contiguous"),
        ("num_tensors", "experts"),
    ],
)
def test_group_weight_amax_validates_inputs(invalid, error):
    weights = torch.empty((2, 256, 128), dtype=torch.bfloat16, device="cuda")
    num_tensors = 2
    if invalid == "dtype":
        weights = weights.float()
    elif invalid == "ndim":
        weights = weights[0]
    elif invalid == "noncontiguous":
        weights = weights.transpose(1, 2)
    else:
        num_tensors = 3

    with pytest.raises(ValueError, match=error):
        triton_group_weight_amax(weights, num_tensors)
