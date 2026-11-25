# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.prototype.qat import (
    MXFakeQuantizeConfig,
    MXFakeQuantizedLinear,
    NVFP4FakeQuantizeConfig,
    NVFP4FakeQuantizedLinear,
)
from torchao.prototype.qat.nvfp4 import _NVFP4QuantizedForwardFakeQuantizedBackward
from torchao.utils import is_sm_at_least_100


def _make_linear(quant_format, device, dtype, bias):
    if quant_format.startswith("mx"):
        elem_dtype = {
            "mxfp4": torch.float4_e2m1fn_x2,
            "mxfp8_e4m3": torch.float8_e4m3fn,
            "mxfp8_e5m2": torch.float8_e5m2,
        }[quant_format]
        config = MXFakeQuantizeConfig(dtype=elem_dtype)
        linear_cls, tensor_cls = MXFakeQuantizedLinear, MXTensor
    else:
        if device != "cuda" or not is_sm_at_least_100():
            pytest.skip("Native NVFP4 QAT forward requires Blackwell")
        config = NVFP4FakeQuantizeConfig(
            use_per_tensor_scale=quant_format == "nvfp4_scaled"
        )
        linear_cls, tensor_cls = NVFP4FakeQuantizedLinear, NVFP4Tensor
    return (
        linear_cls(
            128,
            256,
            bias=bias,
            activation_config=config,
            weight_config=config,
            device=device,
            dtype=dtype,
        ),
        tensor_cls,
    )


@pytest.mark.parametrize(
    "quant_format",
    ["mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4", "nvfp4_scaled"],
)
@pytest.mark.parametrize(
    "device,dtype", [("cpu", torch.float32), ("cuda", torch.bfloat16)]
)
@pytest.mark.parametrize("input_shape", [(64, 128), (2, 32, 128)])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("weight_grad", [False, True])
@pytest.mark.parametrize("bias_grad", [None, False, True])
def test_qat_partial_gradients(
    quant_format, device, dtype, input_shape, input_grad, weight_grad, bias_grad
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    linear, tensor_cls = _make_linear(
        quant_format, device, dtype, bias_grad is not None
    )
    reference = copy.deepcopy(linear)
    linear.weight.requires_grad_(weight_grad)
    if linear.bias is not None:
        linear.bias.requires_grad_(bias_grad)
    x_ref = torch.randn(input_shape, device=device, dtype=dtype, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_(input_grad)
    y_ref = reference(x_ref)
    grad_output = torch.randn_like(y_ref)
    y_ref.backward(grad_output)

    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        y = linear(x)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
    # dW needs the quantized input; dX needs the quantized weight.
    expected_shapes = []
    if weight_grad:
        expected_shapes.append((64, 128))
    if input_grad:
        expected_shapes.append((256, 128))
    assert [tuple(t.shape) for t in saved] == expected_shapes
    assert all(isinstance(t, tensor_cls) for t in saved)

    dequantized = []
    original_dequantize = tensor_cls.dequantize

    def dequantize(tensor, *args, **kwargs):
        dequantized.append(tuple(tensor.shape))
        return original_dequantize(tensor, *args, **kwargs)

    with (
        patch.object(tensor_cls, "dequantize", dequantize),
        patch("torch.mm", wraps=torch.mm) as mm,
    ):
        if y.requires_grad:
            y.backward(grad_output)
        assert mm.call_count == int(input_grad) + int(weight_grad)
    assert sorted(dequantized) == sorted(expected_shapes)
    for actual, expected, needed in [
        (x.grad, x_ref.grad, input_grad),
        (linear.weight.grad, reference.weight.grad, weight_grad),
        (linear.bias.grad, reference.bias.grad, bias_grad)
        if linear.bias is not None
        else (None, None, False),
    ]:
        if needed:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual is None


@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("weight_grad", [False, True])
@pytest.mark.parametrize("bias_grad", [False, True])
def test_nvfp4_backward_partial_gradients_cpu(input_grad, weight_grad, bias_grad):
    # Exercise the real dequantization/backward on CPU without a native FP4 GEMM.
    x = NVFP4Tensor.to_nvfp4(torch.randn(64, 128))
    w = NVFP4Tensor.to_nvfp4(torch.randn(256, 128))
    grad_output = torch.randn(64, 256)
    expected_x = grad_output @ w.dequantize() if input_grad else None
    expected_w = grad_output.t() @ x.dequantize() if weight_grad else None
    ctx = SimpleNamespace(
        saved_tensors=(x if weight_grad else None, w if input_grad else None),
        needs_input_grad=(input_grad, weight_grad, bias_grad, False, False),
        _fwd_used_autocast=False,
        _dtype=torch.bfloat16,
    )
    with patch("torch.mm", wraps=torch.mm) as mm:
        grads = _NVFP4QuantizedForwardFakeQuantizedBackward.backward(ctx, grad_output)
    assert mm.call_count == int(input_grad) + int(weight_grad)
    for actual, expected in zip(
        grads,
        (expected_x, expected_w, grad_output.sum(0) if bias_grad else None, None, None),
    ):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("quant_format", ["mxfp4", "nvfp4_scaled"])
@pytest.mark.parametrize(
    "input_grad,weight_grad", [(True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize("use_autocast", [False, True])
def test_qat_partial_gradients_compile(
    quant_format, input_grad, weight_grad, use_autocast
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    weight_dtype = torch.float32 if use_autocast else torch.bfloat16
    linear, _ = _make_linear(quant_format, "cuda", weight_dtype, True)
    linear.weight.requires_grad_(weight_grad)
    reference = copy.deepcopy(linear)
    compiled = torch.compile(linear, fullgraph=True)
    x_ref = torch.randn(
        2, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=input_grad
    )
    x = x_ref.detach().clone().requires_grad_(input_grad)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    )
    with autocast:
        y_ref = reference(x_ref)
        y = compiled(x)
    torch.testing.assert_close(y, y_ref)
    grad_output = torch.randn_like(y_ref)
    y_ref.backward(grad_output)
    y.backward(grad_output)
    for actual, expected in [
        (x.grad, x_ref.grad),
        (linear.weight.grad, reference.weight.grad),
        (linear.bias.grad, reference.bias.grad),
    ]:
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
