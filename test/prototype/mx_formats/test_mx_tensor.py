# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025, NVIDIA CORPORATION.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    F4_E2M1_MAX,
    SUPPORTED_ELEM_DTYPES,
)
from torchao.prototype.mx_formats.kernels import pack_uint4, pack_uint6
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    ScaleCalculationMode,
    to_dtype,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_89,
    is_sm_at_least_100,
)

torch.manual_seed(2)

if not TORCH_VERSION_AT_LEAST_2_8:
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
        assert_sqnr_gt_threshold(data_hp, data_mx_dq, 13.0)

    # verify that if data.shape is (M, K) then scale.shape is (M, K // block_size)
    prev_dims, K = data_hp.shape[:-1], data_hp.shape[-1]
    if elem_dtype is torch.float4_e2m1fn_x2:
        assert data_mx.qdata.shape == (*prev_dims, K // 2)
    else:
        assert data_mx.qdata.shape == (*prev_dims, K)
    assert data_mx._scale_e8m0.shape == (*prev_dims, K // block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_hello_world(elem_dtype):
    data = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    block_size = 4
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
    block_size = 4
    _test_mx(data, elem_dtype, block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_some_zeros(elem_dtype):
    data = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    data[0, :] = 0.0
    data[:, 2] = 0.0
    block_size = 4
    _test_mx(data, elem_dtype, block_size)


# TODO(future PR): fix and reenable this test
@pytest.mark.skip(reason="does not pass on B200 yet")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_mx_rceil():
    # nan
    # fmt: off
    data_hp = torch.tensor(
        [
        2143289344, 1054459450, 1060527345, 1045656552, 1058239340, 1045057552, 1061158006, 1049626606,
        1052757568, 1032293288, 1056992320, 1064929425, 1061036255, 1047450552, 1057077424, 1055125012,
        1036491424, 1063542041, 1057099838, 1058731224, 1050189482, 1049114228, 1058347802, 1060065968,
        1058846156, 1048878912, 1065109089, 1054494928, 1044803976, 1049117692, 1065222528, 1056965012,
        ],
        dtype=torch.uint32,
    ).view(torch.float32)
    # fmt: on
    ground_truth_scale = torch.tensor([255], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    # fmt: off
    ground_truth_fp8 = torch.tensor(
        [
        127, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        ],
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    # fmt: on
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    assert torch.isnan(data_mx.qdata[0])
    assert torch.all(data_mx.qdata[1:] == 0)
    # fp32 denorm
    # fmt: off
    data_hp = torch.tensor(
        [
        6142315, 5096174, 3345704, 6178415, 5728750, 419002, 1716691, 4335089,
        5785800, 6234845, 1697524, 33075, 3975816, 3714822, 5411407, 3040844,
        7400945, 4474166, 7257182, 1273750, 5872176, 4694081, 2096530, 6273621,
        67028, 7585260, 4532315, 4599275, 6133942, 4542483, 5992199, 6862780,
        ],
        dtype=torch.uint32,
    ).view(torch.float32)
    # fmt: on
    ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ground_truth_fp8 = torch.tensor([0] * 32, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    torch.testing.assert_close(data_mx.qdata, ground_truth_fp8)
    # bf16 denorm
    # fmt: off
    data_hp = torch.tensor(
        [
        101, 3, 47, 54, 36, 19, 70, 79,
        35, 95, 28, 120, 84, 94, 20, 92,
        18, 42, 98, 58, 3, 26, 64, 86,
        60, 86, 52, 23, 61, 70, 59, 74,
        ],
        dtype=torch.uint16,
    ).view(torch.bfloat16)
    # fmt: on
    ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ground_truth_fp8 = torch.tensor([0] * 32, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    torch.testing.assert_close(data_mx.qdata, ground_truth_fp8)
    # fp32 some denorm
    # fmt: off
    data_hp = torch.tensor(
        [
        8388608, 1063716449, 1064039365, 1063568877, 1051091338, 1062185569, 1034449408, 1060813641,
        1054893736, 1034907680, 1036660744, 1023639888, 1058536559, 1050896496, 1049237634, 1064950601,
        1051852994, 1059794063, 1054011102, 1062023602, 1059467900, 1062276774, 1059155029, 1053287574,
        1064378711, 1055768540, 1045266076, 1059575077, 1054928758, 1040468200, 1058061961, 1053066436,
        ],
        dtype=torch.uint32,
    ).view(torch.float32)
    # fmt: on
    ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    # fmt: off
    ground_truth_fp8 = torch.tensor(
        [
        0, 118, 119, 118, 106, 117, 91, 116,
        110, 91, 93, 80, 113, 106, 105, 120,
        107, 115, 109, 117, 114, 117, 114, 108,
        119, 111, 101, 114, 110, 96, 113, 108,
        ],
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    # fmt: on
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    torch.testing.assert_close(data_mx.qdata, ground_truth_fp8)
    # bf16 some denorm
    # fmt: off
    data_hp = torch.tensor(
        [
        128, 16118, 16143, 16074, 16187, 16002, 16193, 16217,
        15680, 16183, 16092, 16158, 16251, 15876, 15896, 16194,
        16135, 16214, 16205, 16110, 16122, 15960, 15824, 16106,
        16220, 16230, 15952, 15896, 16000, 16144, 16232, 16157,
        ],
        dtype=torch.uint16,
    ).view(torch.bfloat16)
    # fmt: on
    ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    # fmt: off
    ground_truth_fp8 = torch.tensor(
        [
        0, 111, 113, 109, 116, 104, 116, 118,
        84, 115, 110, 114, 120, 96, 98, 116,
        112, 117, 117, 111, 112, 102, 93, 111,
        118, 118, 101, 98, 104, 113, 118, 114,
        ],
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    # fmt: on
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    torch.testing.assert_close(data_mx.qdata, ground_truth_fp8)
    # zero
    data_hp = torch.tensor([0] * 32, dtype=torch.uint32).view(torch.float32)
    ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ground_truth_fp8 = torch.tensor([0] * 32, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    torch.testing.assert_close(data_mx.qdata, ground_truth_fp8)
    # fp32 normal
    # fmt: off
    data_hp = torch.tensor(
        [
        1037408064, 1058534842, 1053630662, 1063310394, 994704128, 1057245441, 1060663708, 1058053571,
        1052395648, 1064831570, 1038427336, 1064777688, 1059248393, 1060959028, 1062878286, 1057799482,
        1057854101, 1053562724, 1027482352, 1060498324, 1063238522, 1060472055, 1054346794, 1029092912,
        1056687298, 1059146141, 1037992128, 1064097772, 1056522806, 1059255744, 1064364912, 1060606252,
        ],
        dtype=torch.uint32,
    ).view(torch.float32)
    # fmt: on
    ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    # fmt: off
    ground_truth_fp8 = torch.tensor(
        [
        93, 113, 109, 118, 53, 112, 116, 113,
        108, 120, 94, 119, 114, 116, 118, 113,
        113, 109, 84, 115, 118, 115, 110, 85,
        112, 114, 94, 119, 112, 114, 119, 115,
        ],
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    # fmt: on
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    torch.testing.assert_close(data_mx.qdata, ground_truth_fp8)
    # bf16 normal
    # fmt: off
    data_hp = torch.tensor(
        [
        15752, 16143, 16182, 15896, 16195, 16186, 16048, 16223,
        15988, 16231, 16140, 16088, 16032, 16240, 16228, 16133,
        16210, 16024, 16248, 16187, 16050, 15696, 16060, 15956,
        16131, 16251, 15896, 16014, 15808, 16024, 16159, 16186,
        ],
        dtype=torch.uint16,
    ).view(torch.bfloat16)
    # fmt: on
    ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    # fmt: off
    ground_truth_fp8 = torch.tensor(
        [
        88, 113, 115, 98, 116, 116, 107, 118,
        103, 118, 113, 110, 106, 119, 118, 112,
        117, 106, 120, 116, 107, 85, 108, 101,
        112, 120, 98, 105, 92, 106, 114, 116,
        ],
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    # fmt: on
    data_mx = MXTensor.to_mx(
        data_hp, torch.float8_e4m3fn, 32, ScaleCalculationMode.RCEIL
    )
    torch.testing.assert_close(data_mx._scale_e8m0, ground_truth_scale)
    torch.testing.assert_close(data_mx.qdata, ground_truth_fp8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_exponent_nan_in(elem_dtype):
    """
    If high precision block values has a NaN, the exponent block
    value is set to is NaN
    """
    tensor_hp = torch.tensor(
        [float("nan"), 1, 2, 3, 4, 5, 6, 7], device="cuda", dtype=torch.bfloat16
    )
    block_size = 4
    tensor_mx = MXTensor.to_mx(tensor_hp, elem_dtype, block_size)
    assert torch.all(torch.isnan(tensor_mx._scale_e8m0[0]))
    assert not torch.any(torch.isnan(tensor_mx._scale_e8m0[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("pack_fp6", [False, True])
def test_exponent_nan_out(elem_dtype, pack_fp6):
    """
    If block exponent value is NaN, the MX tensor block value is NaN
    """
    if pack_fp6 and elem_dtype not in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2):
        pytest.skip("invalid configuration")

    scale_e8m0 = torch.tensor(
        [float("nan"), 1.0], dtype=torch.float8_e8m0fnu, device="cuda"
    )

    block_size = 4

    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_bits = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7], dtype=elem_dtype, device="cuda"
        )  # noqa: E501
    elif elem_dtype in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2):
        data_bits = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device="cuda"
        )  # noqa: E501
        if pack_fp6:
            data_bits = data_bits.reshape(-1, block_size)
            data_bits = pack_uint6(data_bits)
    elif elem_dtype == torch.float4_e2m1fn_x2:
        data_bits = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device="cuda"
        )  # noqa: E501
        data_bits = pack_uint4(data_bits)
    else:
        raise AssertionError("unsupported")
    block_size = 4
    use_fp4_custom_triton_dequant_kernel = False
    tensor_mx = MXTensor(
        data_bits,
        scale_e8m0,
        elem_dtype,
        block_size,
        torch.float,
        use_fp4_custom_triton_dequant_kernel,
        MXGemmKernelChoice.EMULATED,
        pack_fp6,
        None,
    )
    tensor_hp = tensor_mx.to_dtype(torch.float)
    assert torch.all(torch.isnan(tensor_hp.flatten()[0:4]))
    assert not torch.any(torch.isnan(tensor_hp.flatten()[4:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_ranks(elem_dtype):
    """
    The reshaping logic works for various ranks
    """
    B = 4
    shapes = ((B * 4,), (B * 4, 4), (B * 4, 4, 4), (B * 4, 4, 4, 4))
    for s in shapes:
        tensor_hp = torch.randn(*s, device="cuda", dtype=torch.bfloat16)
        _test_mx(tensor_hp, elem_dtype, B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("B", [1, 4, 32])
def test_block_sizes(elem_dtype, B):
    """
    Smoke test for various block sizes
    """
    if B == 1 and elem_dtype == torch.float4_e2m1fn_x2:
        pytest.skip("unsupported configuration")
    elif B % 4 != 0 and elem_dtype in [DTYPE_FP6_E2M3, DTYPE_FP6_E3M2]:
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
    if elem_dtype != torch.float4_e2m1fn_x2 and fp4_triton:
        pytest.skip("unsupported configuration")

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
def test_view(elem_dtype):
    x = torch.randn(1, 2, 4, device="cuda")
    block_size = 4
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_2 = x_mx.view(2, 4)  # noqa: F841


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", [DTYPE_FP6_E2M3, DTYPE_FP6_E3M2])
@pytest.mark.parametrize("pack_fp6", [False, True])
def test_fp6_packing(elem_dtype, pack_fp6):
    x = torch.randn(1, 2, 4, device="cuda")
    block_size = 4
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size, pack_fp6=pack_fp6)
    if pack_fp6:
        expected_packed_shape = torch.Size([*x.shape[:-1], 3 * x.shape[-1] // 4])
    else:
        expected_packed_shape = x.shape

    assert x_mx.qdata.shape == expected_packed_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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
    block_size = 4
    to_mx_c = torch.compile(MXTensor.to_mx, fullgraph=True)

    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_c = to_mx_c(x, elem_dtype, block_size)
    torch.testing.assert_close(
        x_mx._scale_e8m0,
        x_mx_c._scale_e8m0,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(x_mx.qdata, x_mx_c.qdata, atol=0, rtol=0)

    to_dtype_c = torch.compile(to_dtype, fullgraph=True)

    use_fp4_custom_triton_dequant_kernel = False
    pack_fp6 = False
    x_mx_dq = to_dtype(
        x_mx.qdata,
        x_mx._scale_e8m0,
        x_mx._elem_dtype,
        x_mx._block_size,
        hp_dtype,  # noqa: E501
        use_fp4_custom_triton_dequant_kernel,
        pack_fp6,
    )
    x_mx_c_dq = to_dtype_c(
        x_mx_c.qdata,
        x_mx_c._scale_e8m0,
        x_mx_c._elem_dtype,
        x_mx_c._block_size,
        hp_dtype,
        use_fp4_custom_triton_dequant_kernel,
        pack_fp6,
    )
    torch.testing.assert_close(x_mx_dq, x_mx_c_dq, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_89(),
    reason="float8 in triton requires CUDA capability 8.9 or greater",
)
def test_to_mx_inductor_single_kernel():
    """
    Verify that inductor can fuse the cast of a high precision tensor to mx
    into a single kernel
    """
    # TODO(future PR): add fp4 and fp6 here
    # TODO(#1773): add swizzled scale format here
    x = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    block_size = 32
    to_mx_c = torch.compile(MXTensor.to_mx, fullgraph=True)
    out, code = run_and_get_code(to_mx_c, x, torch.float8_e4m3fn, block_size)
    FileCheck().check("def call(").check_count(".run(", 1, exactly=True).run(code[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_89(),
    reason="float8 in triton requires CUDA capability 8.9 or greater",
)
def test_cast_to_float8_e4m3fn_saturation_behavior():
    # TODO(#1912): make the saturated cast work in eager mode and remove this
    # test
    max_val = torch.finfo(torch.float8_e4m3fn).max

    # create example data inside the representable range
    data_in_range_bf16 = torch.tensor(
        [
            max_val,
            -1 * max_val,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )

    # create example data outside the representable range
    data_out_of_range_bf16 = torch.tensor(
        [
            max_val * 2,
            -1 * (max_val * 2),
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )

    # verify that in eager mode PyTorch casting to float8 is unsaturated
    data_in_range_f8 = data_in_range_bf16.to(torch.float8_e4m3fn)
    data_out_of_range_f8 = data_out_of_range_bf16.to(torch.float8_e4m3fn)
    assert not torch.any(torch.isnan(data_in_range_f8))
    assert torch.all(torch.isnan(data_out_of_range_f8))

    # verify that in triton, casting to float8 is saturated
    # for simplicity, use torch.compile to generate triton code
    def to_f8(x):
        x = x.to(torch.float8_e4m3fn)
        return x

    to_f8_c = torch.compile(to_f8)
    data_in_range_f8_c = to_f8_c(data_in_range_bf16)
    data_out_of_range_f8_c = to_f8_c(data_out_of_range_bf16)
    assert not torch.any(torch.isnan(data_in_range_f8_c))
    assert not torch.any(torch.isnan(data_out_of_range_f8_c))
    torch.testing.assert_close(
        data_in_range_f8_c, data_out_of_range_f8_c, atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "dtype,shape,use_per_tensor_scale",
    [
        (torch.bfloat16, (32, 64), False),
        (torch.float32, (64, 128), False),
        (torch.bfloat16, (128, 256), False),
        (torch.bfloat16, (64, 128), True),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="torch.compile requires PyTorch 2.8+"
)
def test_nvfp4_reconstruction(dtype, shape, use_per_tensor_scale):
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    x = torch.randn(shape, dtype=dtype, device="cuda")
    if use_per_tensor_scale:
        tensor_amax = torch.max(torch.abs(x))
        scale = per_tensor_amax_to_scale(tensor_amax)
    else:
        scale = None

    x_nvfp4 = NVFP4Tensor.to_nvfp4(x, per_tensor_scale=scale)
    x_reconstructed = x_nvfp4.to_dtype(dtype)

    def assert_sqnr_gt_threshold(orig, new, threshold):
        sqnr = compute_error(orig, new)
        if torch.all(torch.isnan(sqnr)):
            # if both operands are full of zeroes, sqnr is nan and this is ok
            # test for this explicitly
            assert torch.all(orig == 0) and torch.all(new == 0)
        else:
            assert sqnr >= threshold

    reconstructed_amax = x_nvfp4.get_hp_scales().view(shape[0], -1, 1) * F4_E2M1_MAX
    max_abs = torch.amax(
        torch.abs(x.reshape(shape[0], -1, x_nvfp4._block_size)), dim=-1
    ).unsqueeze(-1)

    assert_sqnr_gt_threshold(max_abs, reconstructed_amax, 30.0)
    assert_sqnr_gt_threshold(x, x_reconstructed, 8.0)

    assert x.shape == x_reconstructed.shape, (
        f"Shape mismatch: {x.shape} vs {x_reconstructed.shape}"
    )
    assert x.dtype == x_reconstructed.dtype, (
        f"Dtype mismatch: {x.dtype} vs {x_reconstructed.dtype}"
    )

    x_nvfp4_t = x_nvfp4.t()
    x_reconstructed_t = x_nvfp4_t.to_dtype(dtype)
    assert_sqnr_gt_threshold(x.t(), x_reconstructed_t, 8.0)

    assert x.t().shape == x_reconstructed_t.shape, (
        f"Transpose shape mismatch: {x.t().shape} vs {x_reconstructed_t.shape}"
    )
    assert x.t().dtype == x_reconstructed_t.dtype, (
        f"Transpose dtype mismatch: {x.t().dtype} vs {x_reconstructed_t.dtype}"
    )


@pytest.mark.parametrize(
    "shape",
    [
        (128, 4),
        (256, 8),
        (100, 3),
        (4, 4),
        (50, 10),
        (384, 12),
    ],
)
@pytest.mark.parametrize(
    "use_triton_kernel", [False, True] if torch.cuda.is_available() else [False]
)
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="torch.compile requires PyTorch 2.8+"
)
def test_to_blocked_from_blocked_roundtrip(shape, use_triton_kernel: bool):
    from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

    rows, cols = shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    original = torch.randint(0, 255, (rows, cols), device=device, dtype=torch.uint8)

    blocked = to_blocked(original, use_triton_kernel=use_triton_kernel)
    reconstructed = from_blocked(blocked, rows, cols)

    torch.testing.assert_close(
        original,
        reconstructed,
        atol=0.0,
        rtol=0.0,
        msg=f"Roundtrip failed for shape {shape} with use_triton_kernel={use_triton_kernel}",
    )


@pytest.mark.parametrize("is_swizzled_scales", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 64),
        (16, 32),
        (64, 128),
        (384, 128),
    ],
)
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nvfp4_swizzled_scales_construction(is_swizzled_scales, shape):
    """
    Test that NVFP4Tensor can be constructed with swizzled scales and
    that the _is_swizzled_scales flag is set correctly.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    M, K = shape
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=is_swizzled_scales)
    assert tensor._is_swizzled_scales == is_swizzled_scales
    reconstructed = tensor.to_dtype(torch.bfloat16)
    assert reconstructed.shape == data.shape


@pytest.mark.parametrize(
    "slice_dim,slice_spec",
    [
        # Row slicing - must align with 128-row boundaries
        pytest.param(0, slice(0, 128), id="slice_rows[0:128]"),
        pytest.param(0, slice(128, 256), id="slice_rows[128:256]"),
        # Column slicing - must align with 64-column boundaries (4 scale columns * 16 block_size)
        pytest.param(1, slice(0, 64), id="slice_cols[0:64]"),
        pytest.param(1, slice(64, 128), id="slice_cols[64:128]"),
        pytest.param(1, slice(0, 128), id="slice_cols[0:128]_full_width"),
        # Test tensor parallelism patterns (half splits)
        pytest.param(1, slice(0, 2048), id="slice_cols[0:2048]_tp_first_half"),
        pytest.param(1, slice(2048, 4096), id="slice_cols[2048:4096]_tp_second_half"),
        # Test quarter splits
        pytest.param(1, slice(0, 1024), id="slice_cols[0:1024]_quarter"),
        pytest.param(1, slice(1024, 2048), id="slice_cols[1024:2048]_quarter"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_slicing(slice_dim, slice_spec):
    """
    Test that slicing works correctly with swizzled scales and maintains
    the swizzled state in the output tensor.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    # Use larger tensor sizes that align with swizzled requirements
    if slice_dim == 0:
        # For row slicing, need at least 256 rows to test 128-row boundaries
        M, K = 256, 4096
    else:
        # For column slicing, need multiples of 64 columns for alignment
        M, K = 128, 4096

    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)
    assert tensor._is_swizzled_scales == True

    if slice_dim == 0:
        sliced_tensor = tensor[slice_spec, :]
    else:
        sliced_tensor = tensor[:, slice_spec]

    # Verify sliced tensor maintains swizzled state
    assert sliced_tensor._is_swizzled_scales == True

    # Verify sliced tensor can be dequantized
    sliced_reconstructed = sliced_tensor.to_dtype(torch.bfloat16)

    # Compare with direct slicing of original data
    original_reconstructed = tensor.to_dtype(torch.bfloat16)
    if slice_dim == 0:
        expected = original_reconstructed[slice_spec, :]
    else:
        expected = original_reconstructed[:, slice_spec]

    torch.testing.assert_close(sliced_reconstructed, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "slice_dim,slice_spec,expected_error",
    [
        # Row slicing with misaligned boundaries
        pytest.param(
            0,
            slice(0, 100),
            "Row slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_row_end",
        ),
        pytest.param(
            0,
            slice(50, 150),
            "Row slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_row_start",
        ),
        # Column slicing with misaligned boundaries
        pytest.param(
            1,
            slice(0, 32),
            "Column slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_col_32",
        ),
        pytest.param(
            1,
            slice(16, 80),
            "Column slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_col_start",
        ),
        pytest.param(
            1,
            slice(0, 100),
            "Column slicing of NVFP4Tensor with swizzled scales requires",
            id="misaligned_col_end",
        ),
        # Odd column boundaries (FP4 packing requirement)
        pytest.param(
            1,
            slice(1, 65),
            "start index to be a multiple of 64, got 1",
            id="odd_start",
        ),
        pytest.param(
            1,
            slice(0, 65),
            " multiple of 64 or equal to tensor size 4096, got 65",
            id="odd_end",
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_slicing_errors(slice_dim, slice_spec, expected_error):
    """
    Test that slicing raises appropriate errors for misaligned boundaries.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    M, K = 256, 4096
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    with pytest.raises(RuntimeError, match=expected_error):
        if slice_dim == 0:
            _ = tensor[slice_spec, :]
        else:
            _ = tensor[:, slice_spec]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_view_semantics():
    """
    Test that slicing maintains proper view semantics where possible.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    M, K = 256, 4096
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    # Test row slicing (should maintain views)
    sliced_tensor = tensor[0:128, :]

    # Test that the sliced tensor shares storage with original for data
    # (Note: scales might not share storage due to swizzled layout complexity)
    assert sliced_tensor.qdata.data_ptr() == tensor.qdata.data_ptr()

    # Test full-width column slicing (should maintain views)
    full_width_slice = tensor[:, 0:K]
    assert full_width_slice._scale_e4m3.data_ptr() == tensor._scale_e4m3.data_ptr()
    assert full_width_slice.qdata.data_ptr() == tensor.qdata.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_serialization():
    """
    Test that tensor flatten/unflatten preserves the swizzled scales state.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    M, K = 32, 64
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Create tensor with swizzled scales
    original_tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    # Test serialization
    tensor_list, ctx = original_tensor.__tensor_flatten__()

    # Verify swizzled flag is preserved in context
    assert NVFP4Tensor.tensor_attribute_names[2] == "_is_swizzled_scales"
    assert ctx[2] == True

    # Test deserialization
    inner_tensors = {}
    for name in tensor_list:
        inner_tensors[name] = getattr(original_tensor, name)

    reconstructed_tensor = NVFP4Tensor.__tensor_unflatten__(
        inner_tensors, ctx, None, None
    )

    # Verify the swizzled state is preserved
    assert reconstructed_tensor._is_swizzled_scales == True

    # Verify functionality is preserved
    original_dq = original_tensor.to_dtype(torch.bfloat16)
    reconstructed_dq = reconstructed_tensor.to_dtype(torch.bfloat16)

    torch.testing.assert_close(original_dq, reconstructed_dq, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="NVFP4 requires PyTorch 2.8+"
)
def test_nvfp4_swizzled_scales_get_scales_method():
    """
    Test that the get_scales() method correctly unswizzles scales when needed.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    M, K = 32, 64
    data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Create tensors with both storage methods
    regular_tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=False)
    swizzled_tensor = NVFP4Tensor.to_nvfp4(data, is_swizzled_scales=True)

    # Get scales from both tensors and verify they are equal
    regular_scales = regular_tensor.get_hp_scales()
    swizzled_scales = swizzled_tensor.get_hp_scales()
    torch.testing.assert_close(regular_scales, swizzled_scales, atol=0.0, rtol=0.0)

    # Verify scales have the expected shape
    expected_shape = (M, K // 16)
    assert regular_scales.shape == expected_shape
    assert swizzled_scales.shape == expected_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "M", [128, 256, 512, 1024, 100, 200, 384], ids=lambda m: f"M{m}"
)
@pytest.mark.parametrize("N", [64, 128, 256, 512, 32, 96, 160], ids=lambda n: f"N{n}")
@pytest.mark.parametrize(
    "use_per_tensor_scale", [False, True], ids=["block_scale", "tensor_scale"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="requires sm100+ for raw intrinsics"
)
@torch.no_grad()
def test_triton_nvfp4_quantize_equivalence(M, N, use_per_tensor_scale, dtype):
    """Test that Triton and PyTorch NVFP4 quantization produce equivalent results."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
        unpack_uint4,
    )

    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=dtype, device="cuda")

    per_tensor_scale = None
    if use_per_tensor_scale:
        per_tensor_scale = per_tensor_amax_to_scale(torch.amax(torch.abs(x)))

    nvfp4_pt = NVFP4Tensor.to_nvfp4(
        x.clone(),
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )

    nvfp4_triton = NVFP4Tensor.to_nvfp4(
        x.clone(),
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=True,
    )

    torch.testing.assert_close(
        nvfp4_pt._scale_e4m3.flatten(), nvfp4_triton._scale_e4m3.flatten()
    )
    pt_unpacked = unpack_uint4(nvfp4_pt.qdata)
    triton_unpacked = unpack_uint4(nvfp4_triton.qdata)
    torch.testing.assert_close(
        pt_unpacked,
        triton_unpacked,
        atol=0,
        rtol=0,
    )

    x_pt_dequant = nvfp4_pt.to_dtype(dtype)
    x_triton_dequant = nvfp4_triton.to_dtype(dtype)

    sqnr = compute_error(x_pt_dequant, x_triton_dequant)
    SQNR_THRESHOLD = 40.0

    assert sqnr >= SQNR_THRESHOLD, (
        f"SQNR {sqnr:.2f} < {SQNR_THRESHOLD} for M={M}, N={N}, "
        f"use_per_tensor_scale={use_per_tensor_scale}, dtype={dtype}"
    )
