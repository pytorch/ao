# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy

import pytest
import torch
from torch import nn

from torchao import optim
from torchao.optim.subclass_4bit import OptimState4bit
from torchao.optim.subclass_8bit import OptimState8bit
from torchao.optim.subclass_fp8 import OptimStateFp8


@pytest.mark.parametrize(
    "subclass,signed",
    [
        (OptimState4bit, False),
        (OptimState4bit, True),
        (OptimState8bit, False),
        (OptimState8bit, True),
        (OptimStateFp8, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8192,), (64, 128)])
@pytest.mark.parametrize("method", ["clone", "torch_clone", "deepcopy"])
def test_quantized_state_clone(subclass, signed, dtype, shape, method):
    torch.manual_seed(19)
    kwargs = {} if subclass is OptimStateFp8 else {"signed": signed}
    source = subclass.zeros(shape, block_size=128, dtype=dtype, **kwargs)
    values = torch.randn(shape)
    if not signed:
        values = values.abs()
    source.copy_(values)
    original = {name: getattr(source, name).clone() for name in source.tensor_attrs}
    if method == "clone":
        cloned = source.clone(memory_format=torch.contiguous_format)
    elif method == "torch_clone":
        cloned = torch.clone(source, memory_format=torch.preserve_format)
    else:
        repeated = copy.deepcopy([source, source])
        cloned = repeated[0]
        assert cloned is repeated[1]

    assert type(cloned) is type(source)
    assert cloned is not source
    assert cloned.shape == source.shape
    assert cloned.dtype == source.dtype
    assert cloned.device == source.device
    assert cloned.__tensor_flatten__()[1] == source.__tensor_flatten__()[1]
    torch.testing.assert_close(cloned.dequantize(), source.dequantize(), rtol=0, atol=0)
    for name in source.tensor_attrs:
        actual = getattr(cloned, name)
        expected = getattr(source, name)
        assert actual.data_ptr() != expected.data_ptr()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        actual.zero_()
        assert torch.equal(expected.view(torch.uint8), original[name].view(torch.uint8))


def _clone_test_model(dtype):
    return nn.Sequential(nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 8)).to(dtype)


def _clone_test_step(model, optimizer, inputs, target):
    optimizer.zero_grad()
    loss = (model(inputs).float() - target).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


@pytest.mark.parametrize(
    "optim_name",
    [
        "Adam8bit",
        "AdamW8bit",
        "Adam4bit",
        "AdamW4bit",
        "AdamFp8",
        "AdamWFp8",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantized_optimizer_snapshot_resume(optim_name, dtype, tmp_path):
    torch.manual_seed(29)
    model = _clone_test_model(dtype)
    optimizer_cls = getattr(optim, optim_name)
    optimizer = optimizer_cls(model.parameters(), lr=1e-3)
    inputs = torch.randn(4, 64).to(dtype)
    target = torch.randn(4, 8)
    for _ in range(2):
        _clone_test_step(model, optimizer, inputs, target)

    snapshot = copy.deepcopy(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    )
    state_id = snapshot["optimizer"]["param_groups"][0]["params"][0]
    snapshot_moment = snapshot["optimizer"]["state"][state_id]["exp_avg"]
    original_moment = optimizer.state[model[0].weight]["exp_avg"]
    frozen = {
        name: getattr(snapshot_moment, name).clone()
        for name in snapshot_moment.tensor_attrs
    }
    assert snapshot_moment.dtype == dtype
    for name in snapshot_moment.tensor_attrs:
        assert (
            getattr(snapshot_moment, name).data_ptr()
            != getattr(original_moment, name).data_ptr()
        )
    path = tmp_path / "snapshot.pt"
    torch.save(snapshot, path)

    _clone_test_step(model, optimizer, inputs * 2, target + 1)
    for name in snapshot_moment.tensor_attrs:
        assert torch.equal(
            getattr(snapshot_moment, name).view(torch.uint8),
            frozen[name].view(torch.uint8),
        )
    checkpoint = torch.load(path, weights_only=False)
    resumed, reference = _clone_test_model(dtype), _clone_test_model(dtype)
    resumed.load_state_dict(snapshot["model"])
    reference.load_state_dict(checkpoint["model"])
    resumed_optimizer = optimizer_cls(resumed.parameters())
    reference_optimizer = optimizer_cls(reference.parameters())
    resumed_optimizer.load_state_dict(snapshot["optimizer"])
    reference_optimizer.load_state_dict(checkpoint["optimizer"])
    for _ in range(2):
        actual = _clone_test_step(resumed, resumed_optimizer, inputs, target)
        expected = _clone_test_step(reference, reference_optimizer, inputs, target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_param, expected_param in zip(
            resumed.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
