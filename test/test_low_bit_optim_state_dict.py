# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch import nn

from torchao import optim

_OPTIMIZERS = ["Adam8bit", "AdamW8bit", "Adam4bit", "AdamW4bit", "AdamFp8", "AdamWFp8"]


def _model():
    return nn.Sequential(nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 8))


def _groups(model):
    return [
        {"params": model[0].parameters(), "lr": 1e-3},
        {"params": model[2].parameters(), "lr": 3e-4},
    ]


def _step(model, optimizer, inputs, target):
    optimizer.zero_grad()
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


@pytest.mark.parametrize("optim_name", _OPTIMIZERS)
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.bfloat16])
def test_restore_native_scalar_lr(optim_name, amsgrad, default_dtype, tmp_path):
    torch.manual_seed(413)
    reference = _model()
    native_cls = (
        torch.optim.AdamW if optim_name.startswith("AdamW") else torch.optim.Adam
    )
    native = native_cls(_groups(reference), amsgrad=amsgrad, weight_decay=0.03)
    inputs, target = torch.randn(4, 64), torch.randn(4, 8)
    for _ in range(2):
        _step(reference, native, inputs, target)
    path = tmp_path / "native.pt"
    torch.save(
        {"model": reference.state_dict(), "optimizer": native.state_dict()}, path
    )
    checkpoint = torch.load(path, weights_only=True)
    resumed = _model()
    resumed.load_state_dict(checkpoint["model"])
    optimizer = getattr(optim, optim_name)(_groups(resumed))
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        optimizer.load_state_dict(checkpoint["optimizer"])
    finally:
        torch.set_default_dtype(previous_dtype)

    for group, original in zip(
        optimizer.param_groups, checkpoint["optimizer"]["param_groups"]
    ):
        assert isinstance(group["lr"], torch.Tensor)
        assert group["lr"].dtype == torch.float32
        assert group["lr"].item() == pytest.approx(original["lr"])
        assert isinstance(original["lr"], float)
    native_scheduler = torch.optim.lr_scheduler.StepLR(native, step_size=1, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(3):
        expected = _step(reference, native, inputs, target)
        actual = _step(resumed, optimizer, inputs, target)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual_param, expected_param in zip(
            resumed.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(
                actual_param, expected_param, rtol=1e-5, atol=1e-6
            )
        native_scheduler.step()
        scheduler.step()


@pytest.mark.parametrize("optim_name", _OPTIMIZERS)
@pytest.mark.parametrize("scalar_lr", [False, True])
def test_restore_quantized_state_lr(optim_name, scalar_lr, tmp_path):
    torch.manual_seed(719)
    reference = _model()
    original = getattr(optim, optim_name)(_groups(reference), weight_decay=0.03)
    inputs, target = torch.randn(4, 64), torch.randn(4, 8)
    for _ in range(2):
        _step(reference, original, inputs, target)
    saved = original.state_dict()
    if scalar_lr:
        for group in saved["param_groups"]:
            group["lr"] = group["lr"].item()
    path = tmp_path / "quantized.pt"
    torch.save({"model": reference.state_dict(), "optimizer": saved}, path)
    checkpoint = torch.load(path, weights_only=False)
    resumed = _model()
    resumed.load_state_dict(checkpoint["model"])
    optimizer = getattr(optim, optim_name)(_groups(resumed))
    optimizer.load_state_dict(checkpoint["optimizer"])
    for source_group, restored_group in zip(
        original.param_groups, optimizer.param_groups
    ):
        torch.testing.assert_close(restored_group["lr"], source_group["lr"])
    source_state = original.state[reference[0].weight]["exp_avg"]
    restored_state = optimizer.state[resumed[0].weight]["exp_avg"]
    assert type(restored_state) is type(source_state)
    assert hasattr(restored_state, "dequantize")
    torch.testing.assert_close(
        restored_state.dequantize(), source_state.dequantize(), rtol=0, atol=0
    )
    for _ in range(3):
        expected = _step(reference, original, inputs, target)
        actual = _step(resumed, optimizer, inputs, target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_param, expected_param in zip(
            resumed.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
