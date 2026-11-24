# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from torchao.float8.float8_training_tensor import LinearMMConfig
from torchao.float8.fsdp_utils import WeightWithDynamicFloat8CastTensor


@pytest.mark.parametrize("scale_value", [None, 0.0, 2.0])
def test_precomputed_scale_flatten_round_trip(scale_value):
    scale = None if scale_value is None else torch.tensor(scale_value)
    weight = WeightWithDynamicFloat8CastTensor(
        torch.randn(4, 4), LinearMMConfig(), torch.float8_e4m3fn, scale
    )
    names, context = weight.__tensor_flatten__()
    restored = type(weight).__tensor_unflatten__(
        {name: getattr(weight, name) for name in names},
        context,
        weight.shape,
        weight.stride(),
    )
    assert restored._precomputed_scale is scale

    fake_weight = FakeTensorMode().from_tensor(weight)
    if scale is None:
        assert fake_weight._precomputed_scale is None
    else:
        assert isinstance(fake_weight._precomputed_scale, FakeTensor)
        assert fake_weight._precomputed_scale.shape == scale.shape


def test_precomputed_fake_scale_flatten_is_not_data_dependent():
    with FakeTensorMode():
        weight = WeightWithDynamicFloat8CastTensor(
            torch.empty(4, 4),
            LinearMMConfig(),
            torch.float8_e4m3fn,
            torch.empty(()),
        )
        names, _ = weight.__tensor_flatten__()
    assert names == ["_tensor", "_precomputed_scale"]


def test_precomputed_scale_export():
    class Double(torch.nn.Module):
        def forward(self, weight):
            return weight * 2

    weight = WeightWithDynamicFloat8CastTensor(
        torch.ones(4, 4), LinearMMConfig(), torch.float8_e4m3fn, torch.tensor(2.0)
    )
    exported = torch.export.export(Double(), (weight,))
    torch.testing.assert_close(exported.module()(weight), torch.full((4, 4), 2.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_precomputed_cuda_scale_flatten_does_not_synchronize():
    weight = WeightWithDynamicFloat8CastTensor(
        torch.randn(4, 4, device="cuda"),
        LinearMMConfig(),
        torch.float8_e4m3fn,
        torch.tensor(2.0, device="cuda"),
    )
    old_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        names, _ = weight.__tensor_flatten__()
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
    assert names == ["_tensor", "_precomputed_scale"]
