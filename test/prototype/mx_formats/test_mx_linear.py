# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools

import pytest
import torch
import torch.nn as nn

from torchao.prototype.mx_formats.config import (
    MXLinearConfig,
    MXLinearRecipeName,
)
from torchao.prototype.mx_formats.constants import DTYPE_FP4, SUPPORTED_ELEM_DTYPES
from torchao.prototype.mx_formats.mx_linear import (
    MXInferenceLinear,
    MXLinear,
    swap_linear_with_mx_inference_linear,
    swap_linear_with_mx_linear,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    is_sm_at_least_89,
    is_sm_at_least_100,
)

torch.manual_seed(2)

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


# source: https://stackoverflow.com/a/22638709
@pytest.fixture(autouse=True)
def run_around_tests():
    # 1. before test - set up (currently do nothing)
    # 2. run test
    yield
    # 3. after test - teardown
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "elem_dtype", itertools.product(SUPPORTED_ELEM_DTYPES, repeat=3)
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(4, 8), (1, 4, 8), (1, 1, 4, 8)])
def test_linear_eager(elem_dtype, bias, input_shape):
    """
    Smoke test for training linear module with mx weight, compares the following:
    * baseline: float32
    * experiment: emulated MX
    """
    # elem_dtype is a tuple of (input, weight, gradient) dtypes.
    grad_shape = list(input_shape)
    grad_shape[-1] = 8

    m = nn.Sequential(
        nn.Linear(8, 8, bias=bias, device="cuda"),
    )
    m_mx = copy.deepcopy(m)
    config = MXLinearConfig(
        block_size=4,
        elem_dtype=elem_dtype[0],
        elem_dtype_weight_override=elem_dtype[1],
        elem_dtype_grad_output_override=elem_dtype[2],
    )
    swap_linear_with_mx_linear(m_mx, config=config)

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

    if elem_dtype == (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn):
        assert y_sqnr >= 18.0
        assert w_g_sqnr >= 18.0
        assert x_g_sqnr >= 12.0
    else:
        assert y_sqnr >= 8.0
        assert w_g_sqnr >= 10.0
        assert x_g_sqnr >= 8.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="CUDA capability >= 10.0 required for mxfloat8"
)
@pytest.mark.parametrize(
    "recipe_name",
    [
        MXLinearRecipeName.MXFP8_CUBLAS,
        MXLinearRecipeName.MXFP8_CUTLASS,
        MXLinearRecipeName.MXFP4_CUTLASS,
    ],
)
@pytest.mark.parametrize("mkn", [(128, 256, 512), (256, 512, 128), (512, 128, 256)])
def test_linear_eager_emulated_vs_real_gemm(recipe_name, mkn):
    M, K, N = 128, 128, 128
    M, K, N = mkn

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").requires_grad_()
    x_copy = copy.deepcopy(x)
    g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    m_emulated = nn.Sequential(
        nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    m_real = copy.deepcopy(m_emulated)

    elem_dtype = torch.float8_e4m3fn
    if recipe_name == MXLinearRecipeName.MXFP4_CUTLASS:
        elem_dtype = DTYPE_FP4

    config_emulated = MXLinearConfig(block_size=32, elem_dtype=elem_dtype)
    config_real = MXLinearConfig.from_recipe_name(recipe_name)

    swap_linear_with_mx_linear(m_emulated, config=config_emulated)
    swap_linear_with_mx_linear(m_real, config=config_real)

    y_emulated = m_emulated(x)
    y_emulated.backward(g)

    y_real = m_real(x_copy)
    y_real.backward(g)

    with torch.no_grad():
        y_sqnr = compute_error(y_real, y_emulated)
        w_sqnr = compute_error(m_real[0].weight.grad, m_emulated[0].weight.grad)
        g_sqnr = compute_error(x_copy.grad, x.grad)
        assert y_sqnr > 100.0, f"y_sqnr {y_sqnr} too low!"
        assert w_sqnr > 100.0, f"w_sqnr {w_sqnr} too low!"
        assert g_sqnr > 100.0, f"g_sqnr {g_sqnr} too low!"


# TODO(future): enable compile support
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_activation_checkpointing():
    input_shape = (2, 4)
    grad_shape = (2, 8)
    elem_dtype = torch.float8_e4m3fn

    m = nn.Sequential(
        nn.Linear(4, 8, bias=True, device="cuda"),
        nn.Linear(8, 8, bias=True, device="cuda"),
    )
    config = MXLinearConfig(block_size=4, elem_dtype=elem_dtype)
    swap_linear_with_mx_linear(m, config=config)

    x = torch.randn(*input_shape, device="cuda").requires_grad_()
    g = torch.randn(*grad_shape, device="cuda")
    y = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
    y.backward(g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    is_sm_at_least_100(), reason="triton does not work yet on CUDA capability 10.0"
)
@pytest.mark.parametrize(
    "recipe_name",
    ["mxfp8_emulated", "mxfp4_emulated", "mxfp8_cutlass", "mxfp4_cutlass"],
)
@pytest.mark.parametrize("bias", [False, True])
# TODO(future PR): figure out why torch.compile does not match eager when
# autocast is on
def test_linear_compile(recipe_name, bias):
    """
    Verify that compile does not change numerics of MX linear fw + bw
    """
    if recipe_name in ["mxfp8_emulated", "mxfp8_cutlass"]:
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")

    if recipe_name in ["mxfp8_cutlass", "mxfp4_cutlass"]:
        if not is_sm_at_least_100():
            pytest.skip("CUDA capability >= 10.0 required for MX gemms")

    if bias and recipe_name in ["mxfp8_cutlass", "mxfp4_cutlass"]:
        # TODO(future PR): fix this, things are clearly broken with bias=True
        pytest.skip("this test is broken for cutlass recipes with bias=True")

    M, K, N = 128, 256, 512
    input_shape = (M, K)
    grad_shape = (M, N)
    m_mx = nn.Sequential(
        nn.Linear(K, N, bias=bias, device="cuda"),
    )
    config = MXLinearConfig.from_recipe_name(recipe_name)
    swap_linear_with_mx_linear(m_mx, config=config)
    m_mx_c = copy.deepcopy(m_mx)
    m_mx_c = torch.compile(m_mx_c, fullgraph=True, backend="inductor")

    x_ref = torch.randn(*input_shape, device="cuda").requires_grad_()
    x = copy.deepcopy(x_ref)
    g = torch.randn(*grad_shape, device="cuda")

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
    m = nn.Sequential(nn.Linear(4, 8, bias=bias, dtype=torch.bfloat16))
    m = m.cuda()
    m_mx = copy.deepcopy(m)
    config = MXLinearConfig(block_size=4, elem_dtype=elem_dtype)
    swap_linear_with_mx_inference_linear(m_mx, config=config)

    x = torch.randn(*input_shape, device="cuda", dtype=torch.bfloat16)
    y_ref = m(x)
    y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)
    if elem_dtype is torch.float8_e4m3fn:
        assert sqnr >= 20.0
    else:
        assert sqnr >= 11.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    is_sm_at_least_100(), reason="triton does not work yet on CUDA capability 10.0"
)
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_inference_compile_simple(elem_dtype):
    """
    Smoke test for inference compile
    """
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
    m = nn.Sequential(nn.Linear(4, 8, bias=False, dtype=torch.bfloat16))
    m = m.cuda()
    m_mx = copy.deepcopy(m)
    config = MXLinearConfig(block_size=4, elem_dtype=elem_dtype)
    swap_linear_with_mx_inference_linear(m_mx, config=config)
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

    config = MXLinearConfig(block_size=32)
    swap_linear_with_mx_linear(m1, config=config, filter_fn=filter_fn)
    assert type(m1[0]) == MXLinear
    assert type(m1[1]) == torch.nn.Linear

    swap_linear_with_mx_inference_linear(m2, config=config, filter_fn=filter_fn)  # noqa: E501
    assert type(m2[0]) == MXInferenceLinear
    assert type(m2[1]) == torch.nn.Linear
