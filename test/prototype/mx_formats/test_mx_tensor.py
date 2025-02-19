# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    SUPPORTED_ELEM_DTYPES,
)
from torchao.prototype.mx_formats.custom_cast import pack_uint4
from torchao.prototype.mx_formats.mx_tensor import (
    E8M0_EXPONENT_NAN_VAL,
    MXTensor,
    ScaleCalculationMode,
    to_dtype,
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


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # source: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test  # noqa: E501

    # setup (currently do nothing)

    # tests will run here
    yield

    # teardown
    # avoid dynamo cache limit issues
    torch._dynamo.reset()


def _test_mx(
    data_hp, elem_dtype, block_size, scale_calculation_mode=ScaleCalculationMode.FLOOR
):
    data_mx = MXTensor.to_mx(data_hp, elem_dtype, block_size, scale_calculation_mode)
    data_mx_dq = data_mx.to_dtype(data_hp.dtype)

    def assert_sqnr_gt_threshold(orig, new, threshold):
        sqnr = compute_error(orig, new)
        if torch.all(torch.isnan(sqnr)):
            # if both operands are full of zeroes, sqnr is nan and this is ok
            # test for this explicitly
            assert torch.all(orig == 0) and torch.all(new == 0)
        else:
            assert sqnr >= threshold

    if elem_dtype is torch.float8_e4m3fn:
        assert_sqnr_gt_threshold(data_hp, data_mx_dq, 18.0)
    else:
        assert_sqnr_gt_threshold(data_hp, data_mx_dq, 14.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_hello_world(elem_dtype):
    data = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    block_size = 2
    _test_mx(data, elem_dtype, block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("scale_calculation_mode", [s for s in ScaleCalculationMode])
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_realistic_numerics(elem_dtype, scale_calculation_mode):
    data = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    block_size = 32
    _test_mx(data, elem_dtype, block_size, scale_calculation_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_all_zeros(elem_dtype):
    data = torch.zeros(4, 4, device="cuda", dtype=torch.bfloat16)
    block_size = 2
    _test_mx(data, elem_dtype, block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_some_zeros(elem_dtype):
    data = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    data[0, :] = 0.0
    data[:, 2] = 0.0
    block_size = 2
    _test_mx(data, elem_dtype, block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_exponent_nan_in(elem_dtype):
    """
    If high precision block values has a NaN, the exponent block
    value is set to is NaN
    """
    tensor_hp = torch.tensor(
        [float("nan"), 1, 2, 3, 4, 5], device="cuda", dtype=torch.bfloat16
    )
    block_size = 2
    tensor_mx = MXTensor.to_mx(tensor_hp, elem_dtype, block_size)
    assert torch.all(tensor_mx._scale_e8m0[0] == E8M0_EXPONENT_NAN_VAL)
    assert not torch.any(tensor_mx._scale_e8m0[1:] == E8M0_EXPONENT_NAN_VAL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_exponent_nan_out(elem_dtype):
    """
    If block exponent value is NaN, the MX tensor block value is NaN
    """
    scale_e8m0_bits = torch.tensor(
        [E8M0_EXPONENT_NAN_VAL, 23, 42], dtype=torch.uint8, device="cuda"
    )
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_bits = torch.tensor([0, 1, 2, 3, 4, 5], dtype=elem_dtype, device="cuda")  # noqa: E501
    elif elem_dtype in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2):
        data_bits = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.uint8, device="cuda")  # noqa: E501
    elif elem_dtype == DTYPE_FP4:
        data_bits = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.uint8, device="cuda")  # noqa: E501
        data_bits = pack_uint4(data_bits)
    else:
        raise AssertionError("unsupported")
    block_size = 2
    use_fp4_custom_triton_dequant_kernel = False
    tensor_mx = MXTensor(
        scale_e8m0_bits,
        data_bits,
        elem_dtype,
        block_size,
        torch.float,
        use_fp4_custom_triton_dequant_kernel,
        MXGemmKernelChoice.EMULATED,
    )
    tensor_hp = tensor_mx.to_dtype(torch.float)
    assert torch.all(torch.isnan(tensor_hp[0:1]))
    assert not torch.any(torch.isnan(tensor_hp[2:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_ranks(elem_dtype):
    """
    The reshaping logic works for various ranks
    """
    B = 2
    shapes = ((B * 4,), (B * 4, 2), (B * 4, 2, 2), (B * 4, 2, 2, 2))
    for s in shapes:
        tensor_hp = torch.randn(*s, device="cuda", dtype=torch.bfloat16)
        _test_mx(tensor_hp, elem_dtype, B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_block_sizes(elem_dtype):
    """
    Smoke test for various block sizes
    """
    for B in (1, 2, 32):
        if B == 1 and elem_dtype == DTYPE_FP4:
            pytest.skip("unsupported configuration")
        tensor_hp = torch.randn(B, device="cuda", dtype=torch.bfloat16)
        _test_mx(tensor_hp, elem_dtype, B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("fp4_triton", [False, True])
def test_transpose(elem_dtype, fp4_triton):
    """
    Verify that transposing an MX tensor works
    """
    if elem_dtype != DTYPE_FP4 and fp4_triton:
        pytest.skip("unsupported configuration")
    elif fp4_triton and is_sm_at_least_100():
        pytest.skip("triton does not work yet on CUDA capability 10.0")

    M, K = 128, 256
    block_size = 32
    tensor_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    tensor_mx = MXTensor.to_mx(
        tensor_hp,
        elem_dtype,
        block_size,
        use_fp4_custom_triton_dequant_kernel=fp4_triton,
    )
    tensor_mx_dq_t = tensor_mx.to_dtype(tensor_hp.dtype).t()

    tensor_mx_t = tensor_mx.t()
    tensor_mx_t_dq = tensor_mx_t.to_dtype(tensor_hp.dtype)

    assert tensor_mx_dq_t.shape == tensor_mx_t_dq.shape
    torch.testing.assert_close(tensor_mx_dq_t, tensor_mx_t_dq, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_cast_autograd(elem_dtype):
    x = torch.arange(8, device="cuda").bfloat16().requires_grad_()
    grad = torch.arange(8, device="cuda").bfloat16() * 0.5
    block_size = 8
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_dq = x_mx.to_dtype(torch.bfloat16)
    x_dq.backward(gradient=grad)
    torch.testing.assert_close(grad, x.grad, atol=0, rtol=0)


@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_view(elem_dtype):
    x = torch.randn(1, 2, 4)
    block_size = 2
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_2 = x_mx.view(2, 4)  # noqa: F841


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    is_sm_at_least_100(), reason="triton does not work yet on CUDA capability 10.0"
)
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("all_zeros", [False, True])
def test_to_mx_from_mx_compile_numerics(elem_dtype, hp_dtype, all_zeros):
    """
    Verifies that compile does not change numerics of MX casts
    """
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not is_sm_at_least_89():
            # separate ifs because flake8 is outsmarting me
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")

    shape = 4, 8
    if not all_zeros:
        x = torch.randn(*shape, dtype=hp_dtype, device="cuda")
    else:
        x = torch.zeros(*shape, dtype=hp_dtype, device="cuda")
    block_size = 2
    to_mx_c = torch.compile(MXTensor.to_mx, fullgraph=True)

    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_c = to_mx_c(x, elem_dtype, block_size)
    torch.testing.assert_close(
        x_mx._scale_e8m0,
        x_mx_c._scale_e8m0,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(x_mx._data, x_mx_c._data, atol=0, rtol=0)

    to_dtype_c = torch.compile(to_dtype, fullgraph=True)

    use_fp4_custom_triton_dequant_kernel = False
    x_mx_dq = to_dtype(
        x_mx._data,
        x_mx._scale_e8m0,
        x_mx._elem_dtype,
        x_mx._block_size,
        hp_dtype,  # noqa: E501
        use_fp4_custom_triton_dequant_kernel,
    )
    x_mx_c_dq = to_dtype_c(
        x_mx_c._data,
        x_mx_c._scale_e8m0,
        x_mx_c._elem_dtype,
        x_mx_c._block_size,
        hp_dtype,
        use_fp4_custom_triton_dequant_kernel,
    )
    torch.testing.assert_close(x_mx_dq, x_mx_c_dq, atol=0, rtol=0)
