# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch

from torchao.quantization import Int8WeightOnlyConfig, quantize_
from torchao.quantization.quant_api import swap_conv2d_1x1_to_linear


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_pointwise_conv_swap_training(batched, bias, frozen, training):
    torch.manual_seed(11)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, 1, bias=bias, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Conv2d(8, 2, 1, dtype=torch.float64),
    ).train(training)
    model[0].weight.requires_grad_(not frozen)
    reference = copy.deepcopy(model)
    original_bias = model[0].bias
    swap_conv2d_1x1_to_linear(model)
    assert isinstance(model[0].mod, torch.nn.Linear)
    assert model[0].mod.weight.requires_grad == (not frozen)
    assert model[0].mod.bias is original_bias
    assert model[0].training == model[0].mod.training == training

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    shape = (2, 4, 5, 7) if batched else (4, 5, 7)
    for _ in range(3):
        x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = model(x)
        expected = reference(reference_x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-12, rtol=1e-12)
        for actual_parameter, expected_parameter in zip(
            model.parameters(), reference.parameters()
        ):
            if not expected_parameter.requires_grad:
                assert actual_parameter.grad is None
            else:
                torch.testing.assert_close(
                    actual_parameter.grad.reshape(expected_parameter.shape),
                    expected_parameter.grad,
                    atol=1e-12,
                    rtol=1e-12,
                )
        optimizer.step()
        reference_optimizer.step()
        for actual_parameter, expected_parameter in zip(
            model.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(
                actual_parameter.reshape(expected_parameter.shape),
                expected_parameter,
                atol=1e-12,
                rtol=1e-12,
            )


@pytest.mark.parametrize(
    "kwargs",
    [{"stride": 2}, {"padding": 1}, {"groups": 2}, {"groups": 4}],
    ids=["stride", "padding", "groups", "depthwise"],
)
@pytest.mark.parametrize("custom_filter", [False, True])
def test_pointwise_conv_swap_retains_spatial_or_grouped_conv(kwargs, custom_filter):
    conv = torch.nn.Conv2d(4, 4, 1, **kwargs)
    model = torch.nn.Sequential(conv)
    x = torch.randn(2, 4, 5, 7)
    expected = model(x)
    filter_fn = (
        (lambda module, fqn: isinstance(module, torch.nn.Conv2d))
        if custom_filter
        else None
    )
    swap_conv2d_1x1_to_linear(model, filter_fn=filter_fn)
    assert model[0] is conv
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("padding", ["same", "valid"])
def test_pointwise_conv_swap_root_and_quantize(batched, padding):
    torch.manual_seed(5)
    conv = torch.nn.Conv2d(4, 8, 1, padding=padding, dilation=2).eval()
    x = torch.randn((2, 4, 5, 7) if batched else (4, 5, 7))
    expected = conv(x)
    result = swap_conv2d_1x1_to_linear(conv)
    assert isinstance(result.mod, torch.nn.Linear)
    torch.testing.assert_close(result(x), expected)
    quantize_(result, Int8WeightOnlyConfig())
    actual = result(x)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
