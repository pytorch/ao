# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest

import torch
import torch.nn as nn
from torchao.prototype.mx_formats.constants import SUPPORTED_ELEM_DTYPES

from torchao.prototype.mx_formats.mx_linear import (
    MXInferenceLinear,
    MXLinear,
    swap_linear_with_mx_inference_linear,
    swap_linear_with_mx_linear,
)

from torchao.quantization.utils import compute_error
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

# trying to outsmart flake8
__has_cuda = torch.cuda.is_available()
IS_CUDA_GE_89 = __has_cuda and torch.cuda.get_device_capability() >= (8, 9)

torch.manual_seed(2)

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(2, 4), (1, 2, 4), (1, 1, 2, 4)])
def test_linear_eager(elem_dtype, bias, input_shape):
    """
    Smoke test for training linear module with mx weight
    """
    grad_shape = list(input_shape)
    grad_shape[-1] = 6

    m = nn.Sequential(
        nn.Linear(4, 6, bias=bias, device="cuda"),
    )
    m_mx = copy.deepcopy(m)
    block_size = 2
    swap_linear_with_mx_linear(m_mx, elem_dtype, block_size)

    x_ref = torch.randn(*input_shape, device="cuda").requires_grad_()
    x = copy.deepcopy(x_ref)
    g = torch.randn(*grad_shape, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = m(x_ref)
        y_mx = m_mx(x)

    y_ref.backward(g)
    y_mx.backward(g)

    y_sqnr = compute_error(y_ref, y_mx)
    w_g_sqnr = compute_error(m[0].weight.grad, getattr(m_mx, "0").weight.grad)
    x_g_sqnr = compute_error(x_ref.grad, x.grad)

    if elem_dtype is torch.float8_e4m3fn:
        assert y_sqnr >= 18.0
        assert w_g_sqnr >= 18.0
        assert x_g_sqnr >= 14.0
    else:
        assert y_sqnr >= 8.0
        assert w_g_sqnr >= 10.0
        assert x_g_sqnr >= 8.0


# TODO(future): enable compile support
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_activation_checkpointing():
    input_shape = (2, 4)
    grad_shape = (2, 6)
    elem_dtype = torch.float8_e4m3fn

    m = nn.Sequential(
        nn.Linear(4, 6, bias=True, device="cuda"),
        nn.Linear(6, 6, bias=True, device="cuda"),
    )
    block_size = 2
    swap_linear_with_mx_linear(m, elem_dtype, block_size)

    x = torch.randn(*input_shape, device="cuda").requires_grad_()
    g = torch.randn(*grad_shape, device="cuda")
    y = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
    y.backward(g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("bias", [False, True])
def test_linear_compile(elem_dtype, bias):
    """
    Verify that compile does not change numerics of MX linear fw + bw
    """
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not IS_CUDA_GE_89:
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
    input_shape = (2, 4)
    grad_shape = (2, 6)
    m_mx = nn.Sequential(
        nn.Linear(4, 6, bias=bias, device="cuda"),
    )
    block_size = 2
    swap_linear_with_mx_linear(m_mx, elem_dtype, block_size)
    m_mx_c = copy.deepcopy(m_mx)
    m_mx_c = torch.compile(m_mx_c, fullgraph=True)

    x_ref = torch.randn(*input_shape, device="cuda").requires_grad_()
    x = copy.deepcopy(x_ref)
    g = torch.randn(*grad_shape, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = m_mx(x_ref)
        y = m_mx_c(x)
    torch.testing.assert_close(y_ref, y, atol=0, rtol=0)

    y_ref.backward(g)
    y.backward(g)
    w_g_ref = m_mx[0].weight.grad
    w_g = getattr(m_mx_c, "0").weight.grad
    # TODO(future): investigate why we can't match with rtol=0 atol=0
    # after moving to torchao repo. Technically compile does not give
    # bit exactness guarantees, but there also might be a bug lurking
    # around.
    torch.testing.assert_close(w_g_ref, w_g, atol=0.02, rtol=0.02)

    x_g_ref = x_ref.grad
    x_g = x.grad
    # TODO(future): investigate why we can't match with rtol=0 atol=0
    # after moving to torchao repo. Technically compile does not give
    # bit exactness guarantees, but there also might be a bug lurking
    # around.
    torch.testing.assert_close(x_g_ref, x_g, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(2, 4), (1, 2, 4), (1, 1, 2, 4)])
def test_inference_linear(elem_dtype, bias, input_shape):
    """
    Smoke test for inference linear module with mx weight
    """
    m = nn.Sequential(nn.Linear(4, 6, bias=bias, dtype=torch.bfloat16))
    m = m.cuda()
    m_mx = copy.deepcopy(m)
    block_size = 2
    swap_linear_with_mx_inference_linear(m_mx, elem_dtype, block_size)

    x = torch.randn(*input_shape, device="cuda", dtype=torch.bfloat16)
    y_ref = m(x)
    y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)
    if elem_dtype is torch.float8_e4m3fn:
        assert sqnr >= 20.0
    else:
        assert sqnr >= 11.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_inference_compile_simple(elem_dtype):
    """
    Smoke test for inference compile
    """
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not IS_CUDA_GE_89:
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
    m = nn.Sequential(nn.Linear(4, 6, bias=False, dtype=torch.bfloat16))
    m = m.cuda()
    m_mx = copy.deepcopy(m)
    block_size = 2
    swap_linear_with_mx_inference_linear(m_mx, elem_dtype, block_size)
    m_mx = torch.compile(m_mx, fullgraph="true")

    x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    y_ref = m(x)
    y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)
    if elem_dtype is torch.float8_e4m3fn:
        assert sqnr >= 20.0
    else:
        assert sqnr >= 13.5


def test_filter_fn():
    m1 = nn.Sequential(
        nn.Linear(32, 32),
        nn.Linear(32, 32),
    )
    m2 = copy.deepcopy(m1)
    filter_fn = lambda mod, fqn: fqn != "1"  # noqa: E731

    swap_linear_with_mx_linear(m1, torch.float8_e4m3fn, 32, filter_fn)
    assert type(m1[0]) == MXLinear
    assert type(m1[1]) == torch.nn.Linear

    swap_linear_with_mx_inference_linear(
        m2, torch.float8_e4m3fn, 32, filter_fn
    )  # noqa: E501
    assert type(m2[0]) == MXInferenceLinear
    assert type(m2[1]) == torch.nn.Linear
