# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

import torchao.prototype.mx_formats.config as config
from torch.utils._triton import has_triton
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    F4_E2M1_EXP_BIAS,
    F6_E2M3_EXP_BIAS,
    F6_E3M2_EXP_BIAS,
)

from torchao.prototype.mx_formats.custom_cast import (
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    f4_unpacked_to_f32,
    f6_e2m3_unpacked_to_f32,
    f6_e3m2_unpacked_to_f32,
    get_bits,
    pack_uint4,
    triton_f4_to_bf16,
    unpack_uint4,
)

from torchao.prototype.mx_formats.fp_format_spec import (
    _assert_equals,
    dtype_to_interesting_values,
    float4_e2m1_interesting_values,
    float6_e2m3_interesting_values,
    float6_e3m2_interesting_values,
    get_sem_bits,
    sem_bits_to_sem_vals,
    sem_vals_to_f32,
)

from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4


torch.manual_seed(0)


@pytest.mark.skip(
    reason="TODO debug CI failure, low pri since this is not used in the MX code"  # noqa: E501
)
def test_fp32():
    dtype = torch.float
    interesting_values = dtype_to_interesting_values[dtype]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)


@pytest.mark.skip(
    reason="TODO debug CI failure, low pri since this is not used in the MX code"  # noqa: E501
)
def test_bf16():
    dtype = torch.bfloat16
    interesting_values = dtype_to_interesting_values[dtype]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)


def test_fp16():
    dtype = torch.float16
    interesting_values = dtype_to_interesting_values[dtype]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)


def test_float8_e4m3fn():
    dtype = torch.float8_e4m3fn
    interesting_values = dtype_to_interesting_values[dtype]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)


def test_float8_e5m2():
    dtype = torch.float8_e5m2
    interesting_values = dtype_to_interesting_values[dtype]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)


def _sem_enc_to_fp32_val(s_enc, e_enc, m_enc, is_zero, is_denorm, exp_bias):
    s_i = 1.0 if s_enc == "0" else -1.0
    if is_zero:
        e_i = 0
        m_f = 0.0
    elif is_denorm:
        e_i = int(e_enc, 2) - exp_bias + 1
        m_f = 0.0
        cur_pow_of_two = -1
        for m_bit in m_enc:
            m_f += int(m_bit, 2) * pow(2, cur_pow_of_two)
            cur_pow_of_two -= 1
    else:
        e_i = int(e_enc, 2) - exp_bias
        m_f = 1.0
        cur_pow_of_two = -1
        for m_bit in m_enc:
            m_f += int(m_bit, 2) * pow(2, cur_pow_of_two)
            cur_pow_of_two -= 1
    fp32 = s_i * (2**e_i) * m_f
    return fp32


def test_float4_e2m1_table():
    for (
        fp32_ref,
        _formula,
        s_enc,
        e_enc,
        m_enc,
        _label,
    ) in float4_e2m1_interesting_values:
        is_zero = e_enc == "00" and m_enc == "0"
        # normal vs denormal
        is_denorm = e_enc == "00" and m_enc == "1"
        # get exponent and mantissa
        exp_bias = F4_E2M1_EXP_BIAS
        fp32 = _sem_enc_to_fp32_val(
            s_enc, e_enc, m_enc, is_zero, is_denorm, exp_bias
        )  # noqa: E501
        assert abs(fp32_ref - fp32) < 1e-12


def test_float6_e3m2_table():
    for (
        fp32_ref,
        _formula,
        s_enc,
        e_enc,
        m_enc,
        _label,
    ) in float6_e3m2_interesting_values:
        is_zero = e_enc == "000" and m_enc == "00"
        # normal vs denormal
        is_denorm = e_enc == "000" and m_enc != "00"
        # get exponent and mantissa
        exp_bias = F6_E3M2_EXP_BIAS
        fp32 = _sem_enc_to_fp32_val(
            s_enc, e_enc, m_enc, is_zero, is_denorm, exp_bias
        )  # noqa: E501
        assert abs(fp32_ref - fp32) < 1e-12


def test_float6_e2m3_table():
    for (
        fp32_ref,
        _formula,
        s_enc,
        e_enc,
        m_enc,
        _label,
    ) in float6_e2m3_interesting_values:
        is_zero = e_enc == "00" and m_enc == "000"
        # normal vs denormal
        is_denorm = e_enc == "00" and m_enc != "000"
        # get exponent and mantissa
        exp_bias = F6_E2M3_EXP_BIAS
        fp32 = _sem_enc_to_fp32_val(
            s_enc, e_enc, m_enc, is_zero, is_denorm, exp_bias
        )  # noqa: E501
        assert abs(fp32_ref - fp32) < 1e-12


# positive float4 vals, in increasing order:
# 0: 0
# 1: 0.5
# 2: 1.0
# 3: 1.5
# 4: 2.0
# 5: 3.0
# 6: 4.0
# 7: 6.0
# below we test pos and neg versions of all of these


def _test_fp4_case(f32_val, f32_val_ref, f4_enc_ref):
    # 1. verify that a fp32 value gets quantized to correct fp4 encoding
    # TODO test on cuda
    f4_unpacked = f32_to_f4_unpacked(torch.tensor(f32_val))
    s_enc, e_enc, m_enc = get_sem_bits(f4_unpacked, bitwidth=4)
    assert s_enc + e_enc + m_enc == f4_enc_ref

    # 2. verify that fp4 value gets dequantized to correct fp32 value
    f32_dequantized = f4_unpacked_to_f32(f4_unpacked)
    assert f32_val_ref == f32_dequantized.item()


def _test_fp4_cases(cases):
    # test the exp and mantissa with both values of the sign bit
    for s_enc in "0", "1":
        s_i = 1.0 if s_enc == "0" else -1.0
        for val, val_ref, em_enc in cases:
            _test_fp4_case(s_i * val, s_i * val_ref, s_enc + em_enc)


# note: below are written as individual test cases for easy command line
# filtering with pytest, i.e. "-k fp4_0_0"

# Explanation of tie-to-even test cases:
# 1. read https://stackoverflow.com/q/8981913/
#    From above, tie-to-even rule: if GRS == 100, round up if bit before is a 1,  # noqa:  E501
#    and round down if it's a 0
#
# 2. assume 1.mm...m for normals and 0.mm...m for denormals. Since
#    fp4 has only one mantissa bit we are always rounding after that bit. So,
#    G == 0 for fp4 denormal range, and G == 1 for fp4 normal range.
#
# 3. Therefore, when we have a tie (GRS == 100), we round down for fp4 denormals  # noqa: E501
#    and round up for fp4 normals:
#    0.25 -> 0.0 (the only denormal case)
#    0.75 -> 1.0
#    1.25 -> 1.0
#    1.75 -> 2.0
#    2.5 -> 2.0
#    3.5 -> 4.0
#    5.0 -> 4.0


def test_fp4_0_0():
    cases = [
        (0.25, 0.0, "000"),  # tie to even
        (0.1, 0.0, "000"),
        (0.0, 0.0, "000"),
        # note: -0.1 is tested in the negative zero test
    ]
    _test_fp4_cases(cases)


def test_fp4_0_5():
    cases = [
        (0.6, 0.5, "001"),
        (0.5, 0.5, "001"),
        (0.4, 0.5, "001"),
    ]
    _test_fp4_cases(cases)


def test_fp4_1_0():
    cases = [
        (1.25, 1.0, "010"),  # tie to even
        (1.1, 1.0, "010"),
        (1.0, 1.0, "010"),
        (0.9, 1.0, "010"),
        (0.75, 1.0, "010"),  # tie to even
    ]
    _test_fp4_cases(cases)


def test_fp4_1_5():
    cases = [
        (1.6, 1.5, "011"),
        (1.5, 1.5, "011"),
        (1.4, 1.5, "011"),
    ]
    _test_fp4_cases(cases)


def test_fp4_2_0():
    cases = [
        (2.5, 2.0, "100"),  # tie to even
        (2.1, 2.0, "100"),
        (2.0, 2.0, "100"),
        (1.9, 2.0, "100"),
        (1.75, 2.0, "100"),  # tie to even
    ]
    _test_fp4_cases(cases)


def test_fp4_3_0():
    cases = [
        (3.1, 3.0, "101"),
        (3.0, 3.0, "101"),
        (2.9, 3.0, "101"),
    ]
    _test_fp4_cases(cases)


def test_fp4_4_0():
    cases = [
        (5.0, 4.0, "110"),  # tie to even
        (4.1, 4.0, "110"),
        (4.0, 4.0, "110"),
        (3.9, 4.0, "110"),
        (3.5, 4.0, "110"),  # tie to even
    ]
    _test_fp4_cases(cases)


def test_fp4_6_0():
    cases = [
        (6.1, 6.0, "111"),
        (6.0, 6.0, "111"),
        (5.9, 6.0, "111"),
    ]
    _test_fp4_cases(cases)


def test_fp4_pack_unpack():
    orig_vals = torch.Tensor([[0.0, 0.5, 4.0, -0.0], [-0.0, 1.0, -6.0, 3.0]])
    orig_vals_f4_unpacked = f32_to_f4_unpacked(orig_vals)
    orig_vals_f4_packed = pack_uint4(orig_vals_f4_unpacked)
    assert orig_vals_f4_packed.numel() == (orig_vals.numel() / 2)
    orig_vals_f4_packed_unpacked = unpack_uint4(orig_vals_f4_packed)
    orig_vals_dq = f4_unpacked_to_f32(orig_vals_f4_packed_unpacked)
    assert torch.all(orig_vals_dq == orig_vals)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_4, reason="requires PyTorch >= 2.4")
def test_fp4_triton_unscaled_cast():
    packed_vals = torch.arange(0, 255, dtype=torch.uint8, device="cuda")
    f32_ref = f4_unpacked_to_f32(unpack_uint4(packed_vals))
    f32_triton = triton_f4_to_bf16(packed_vals).to(torch.float)
    assert torch.all(torch.eq(f32_ref, f32_triton))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_4, reason="requires PyTorch >= 2.4")
def test_fp4_triton_scaled_cast():
    size = (256,)
    orig_vals = torch.randn(size, dtype=torch.float, device="cuda") * 100
    mxtensor = MXTensor.to_mx(orig_vals, block_size=32, elem_dtype=DTYPE_FP4)

    f32_ref = mxtensor.to_dtype(torch.float)
    config.use_fp4_custom_triton_dequant_kernel = True
    f32_triton = mxtensor.to_dtype(torch.float)
    config.use_fp4_custom_triton_dequant_kernel = False
    assert torch.all(torch.eq(f32_ref, f32_triton))


@pytest.mark.parametrize("dtype_name", (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2))
def test_fp6_values(dtype_name):
    """
    The fp6 dtypes have 2**6 = 64 unique values each. The test
    below tests the f32 -> f6 and f6 -> f32 cast for each value.

    TODO(future PR): also verify rounding tie-to-even works properly.
    """

    for i in range(2**6):
        t = torch.tensor(i, dtype=torch.uint8)
        bits = get_bits(t.to(torch.int8))

        # go from bits to f32 ref
        if dtype_name == DTYPE_FP6_E2M3:
            s_enc, e_enc, m_enc = bits[2], bits[3:5], bits[5:]
        elif dtype_name == DTYPE_FP6_E3M2:
            s_enc, e_enc, m_enc = bits[2], bits[3:6], bits[6:]
        else:
            raise AssertionError("unsupported")
        s_i, e_i, m_f, special_value = sem_bits_to_sem_vals(
            s_enc, e_enc, m_enc, dtype_name
        )
        f32_ref = torch.tensor(sem_vals_to_f32(s_i, e_i, m_f, special_value))

        # test cast to f6
        if dtype_name == DTYPE_FP6_E2M3:
            f6 = f32_to_f6_e2m3_unpacked(f32_ref)
        elif dtype_name == DTYPE_FP6_E3M2:
            f6 = f32_to_f6_e3m2_unpacked(f32_ref)
        else:
            raise AssertionError("unsupported")
        # test that the bits are equivalent to our starting point
        torch.testing.assert_close(f6, t, rtol=0, atol=0)

        # test cast back to f32
        if dtype_name == DTYPE_FP6_E2M3:
            f32 = f6_e2m3_unpacked_to_f32(f6)
        elif dtype_name == DTYPE_FP6_E3M2:
            f32 = f6_e3m2_unpacked_to_f32(f6)
        else:
            raise AssertionError("unsupported")
        torch.testing.assert_close(f32, f32_ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    ]
)
@pytest.mark.parametrize(
    "f32_val,f6_e3m2_enc",
    [
        (29.0,   0b011111),  # normal round down
        (26.0,   0b011110),  # normal round to nearest even
        (0.1251, 0b000010),  # subnormal round down
        (0.0314, 0b000001),  # subnormal round up
        (0.03,   0b000000),  # underflow
    ]
)
def test_fp6_e3m2_rounding(f32_val, f6_e3m2_enc, device):
    f6_e3m2_unpacked = f32_to_f6_e3m2_unpacked(torch.tensor(f32_val, device=device))
    assert f6_e3m2_unpacked.item() == f6_e3m2_enc

    f6_e3m2_unpacked = f32_to_f6_e3m2_unpacked(torch.tensor(-f32_val, device=device))
    assert f6_e3m2_unpacked.item() == (f6_e3m2_enc | 0b100000)
