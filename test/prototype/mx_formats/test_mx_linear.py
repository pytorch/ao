# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn as nn

from torchao.prototype.mx_formats.config import (
    MXLinearConfig,
    MXLinearRecipeName,
)
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    SUPPORTED_ELEM_DTYPES,
)
from torchao.prototype.mx_formats.mx_linear import (
    MXInferenceLinear,
    MXLinear,
    swap_linear_with_mx_inference_linear,
)
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_89,
    is_sm_at_least_100,
)

torch.manual_seed(2)

if not TORCH_VERSION_AT_LEAST_2_8:
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
    "elem_dtype",
    (
        # test each dtype
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn),
        (DTYPE_FP6_E3M2, DTYPE_FP6_E3M2, DTYPE_FP6_E3M2),
        (DTYPE_FP6_E2M3, DTYPE_FP6_E2M3, DTYPE_FP6_E2M3),
        (DTYPE_FP4, DTYPE_FP4, DTYPE_FP4),
        # only test one type of mixed-dtype overrides, to save testing time
        (torch.float8_e4m3fn, DTYPE_FP4, DTYPE_FP4),
    ),
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(128, 256), (1, 128, 256), (1, 1, 128, 256)])
@pytest.mark.parametrize("use_fp8_dim1_cast_triton_kernel", [False, True])
def test_linear_eager_vs_hp(
    elem_dtype, bias, input_shape, use_fp8_dim1_cast_triton_kernel
):
    """
    Smoke test for training linear module with mx weight, compares the following:
    * baseline: float32
    * experiment: emulated MX
    """
    if use_fp8_dim1_cast_triton_kernel:
        if elem_dtype != (
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
        ):
            pytest.skip("unsupported configuration")
        elif not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")

    # elem_dtype is a tuple of (input, weight, gradient) dtypes.
    grad_shape = list(input_shape)
    grad_shape[-1] = 256

    m = nn.Sequential(
        nn.Linear(256, 256, bias=bias, device="cuda", dtype=torch.bfloat16),
    )
    m_mx = copy.deepcopy(m)
    config = MXLinearConfig(
        block_size=4,
        elem_dtype=elem_dtype[0],
        elem_dtype_weight_override=elem_dtype[1],
        elem_dtype_grad_output_override=elem_dtype[2],
        use_fp8_dim1_cast_triton_kernel=use_fp8_dim1_cast_triton_kernel,
    )
    quantize_(m_mx, config)

    x_ref = torch.randn(
        *input_shape, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    x = copy.deepcopy(x_ref)
    g = torch.randn(*grad_shape, device="cuda")

    y_ref = m(x_ref)
    y_mx = m_mx(x)

    assert y_mx.dtype == x.dtype

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

    quantize_(m_emulated, config=config_emulated)
    quantize_(m_real, config=config_real)

    y_emulated = m_emulated(x)
    y_emulated.backward(g)

    y_real = m_real(x_copy)
    y_real.backward(g)

    with torch.no_grad():
        y_sqnr = compute_error(y_real, y_emulated)
        w_sqnr = compute_error(m_real[0].weight.grad, m_emulated[0].weight.grad)
        g_sqnr = compute_error(x_copy.grad, x.grad)
        assert y_sqnr > 90.0, f"y_sqnr {y_sqnr} too low!"
        assert w_sqnr > 90.0, f"w_sqnr {w_sqnr} too low!"
        assert g_sqnr > 90.0, f"g_sqnr {g_sqnr} too low!"


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
    quantize_(m, config=config)

    x = torch.randn(*input_shape, device="cuda").requires_grad_()
    g = torch.randn(*grad_shape, device="cuda")
    y = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
    y.backward(g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "recipe_name",
    [
        "mxfp8_emulated",
        "mxfp4_emulated",
        "mxfp8_cublas",
        "mxfp8_cutlass",
        "mxfp4_cutlass",
    ],
)
@pytest.mark.parametrize("bias", [False, True])
# TODO(future PR): figure out why torch.compile does not match eager when
# autocast is on
@pytest.mark.parametrize("use_fp8_dim1_cast_triton_kernel", [False, True])
def test_linear_compile(hp_dtype, recipe_name, bias, use_fp8_dim1_cast_triton_kernel):
    """
    Verify that compile does not change numerics of MX linear fw + bw
    """
    if recipe_name in ["mxfp8_emulated", "mxfp8_cutlass"]:
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")

    if recipe_name in ["mxfp8_cublas", "mxfp8_cutlass", "mxfp4_cutlass"]:
        if not is_sm_at_least_100():
            pytest.skip("CUDA capability >= 10.0 required for MX gemms")

    if bias and recipe_name in ["mxfp8_cublas", "mxfp8_cutlass", "mxfp4_cutlass"]:
        # TODO(future PR): fix this, things are clearly broken with bias=True
        pytest.skip("this test is broken for non-emulated recipes with bias=True")

    if use_fp8_dim1_cast_triton_kernel:
        if recipe_name not in ("mxfp8_emulated", "mxfp8_cublas", "mxfp8_cutlass"):
            pytest.skip("unsupported configuration")
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
        if hp_dtype != torch.bfloat16:
            pytest.skip("unsupported configuration")

    if hp_dtype == torch.bfloat16 and recipe_name != "mxfp8_cublas":
        # TODO(future PR): properly enable float32 + bfloat16 for every
        # recipe, this needs a cleanup of out_dtype (needs to match in-hp-dtype, even
        # if the underlying gemm kernel only supports bf16 output)
        pytest.skip("unsupported configuration")

    M, K, N = 128, 256, 512
    input_shape = (M, K)
    grad_shape = (M, N)
    m_mx = nn.Sequential(
        nn.Linear(K, N, bias=bias, device="cuda", dtype=hp_dtype),
    )
    config = MXLinearConfig.from_recipe_name(recipe_name)
    config.use_fp8_dim1_cast_triton_kernel = use_fp8_dim1_cast_triton_kernel

    quantize_(m_mx, config=config)
    m_mx_c = copy.deepcopy(m_mx)
    m_mx_c = torch.compile(m_mx_c, fullgraph=True, backend="inductor")

    x_ref = torch.randn(*input_shape, device="cuda", dtype=hp_dtype).requires_grad_()
    x = copy.deepcopy(x_ref)
    g = torch.randn(*grad_shape, device="cuda", dtype=hp_dtype)

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
        assert sqnr >= 11.5


def test_filter_fn():
    m1 = nn.Sequential(
        nn.Linear(32, 32),
        nn.Linear(32, 32),
    )
    m2 = copy.deepcopy(m1)
    filter_fn = lambda mod, fqn: isinstance(mod, torch.nn.Linear) and fqn != "1"  # noqa: E731

    config = MXLinearConfig(block_size=32)
    quantize_(m1, config=config, filter_fn=filter_fn)
    assert type(m1[0]) == MXLinear
    assert type(m1[1]) == torch.nn.Linear

    swap_linear_with_mx_inference_linear(m2, config=config, filter_fn=filter_fn)  # noqa: E501
    assert type(m2[0]) == MXInferenceLinear
    assert type(m2[1]) == torch.nn.Linear


def test_training_print_str():
    m = nn.Sequential(nn.Linear(32, 32))
    config = MXLinearConfig()
    quantize_(m, config=config)
    s = str(m)
    assert "bl_sz=32" in s
    assert "kernel=emulated" in s


def test_inference_print_str():
    m = nn.Sequential(nn.Linear(32, 32))
    config = MXLinearConfig()
    swap_linear_with_mx_inference_linear(m, config=config)
    s = str(m)
    assert "bl_sz=32" in s
    assert "kernel=emulated" in s
