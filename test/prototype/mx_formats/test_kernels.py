# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils._triton import has_triton

import torchao.prototype.mx_formats.kernels as mx_kernels
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    F4_E2M1_EXP_BIAS,
    F6_E2M3_EXP_BIAS,
    F6_E3M2_EXP_BIAS,
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
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f6_e2m3_unpacked_to_f32,
    f6_e3m2_unpacked_to_f32,
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    get_bits,
    mxfp8_quantize_cuda,
    pack_uint4,
    triton_mxfp8_dequant_dim0,
    triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
    triton_to_mxfp8_dim0,
    triton_to_mxfp8_dim1,
    triton_to_mxfp8_dim1_reference,
    unpack_uint4,
)
from torchao.prototype.mx_formats.mx_tensor import (
    ScaleCalculationMode,
    _e8m0_scale_to_reciprocal_fp32,
    _f32_to_e8m0_rceil,
    to_dtype,
    to_mx,
)
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked
from torchao.quantization.utils import compute_error
from torchao.testing._mxfp8_test_utils import (
    assert_mxfp8_semantics,
    make_f32_to_e8m0_rceil_cases,
    make_mxfp8_semantic_cases,
)
from torchao.utils import (
    is_cuda_version_at_least,
    is_MI350,
    is_sm_at_least_100,
)

torch.manual_seed(0)


if hasattr(mx_kernels, "_triton_f32_to_e8m0_rceil"):
    import triton
    import triton.language as tl

    @triton.jit
    def _test_triton_f32_to_e8m0_rceil_kernel(
        input_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        values = tl.load(input_ptr + offsets, mask=offsets < n_elements)
        result = mx_kernels._triton_f32_to_e8m0_rceil(values)
        tl.store(output_ptr + offsets, result, mask=offsets < n_elements)


@pytest.mark.skipif(
    not hasattr(mx_kernels, "_triton_f32_to_e8m0_rceil"),
    reason="MXFP8 Triton kernels are unavailable",
)
def test_triton_f32_to_e8m0_rceil_fallback():
    values, expected = make_f32_to_e8m0_rceil_cases(device="cuda")
    actual = torch.empty_like(values, dtype=torch.uint8)
    _test_triton_f32_to_e8m0_rceil_kernel[(1,)](
        values, actual, n_elements=values.numel(), BLOCK_SIZE=32
    )
    assert torch.equal(actual.cpu(), expected)


# TODO: shared utils file for benchmarking and testing
def to_mx_dim1_reference(x_hp, block_size, scaling_mode):
    x_hp = x_hp.t().contiguous()
    scale_d1, data_d1 = to_mx(
        x_hp, torch.float8_e4m3fn, block_size, scaling_mode=scaling_mode
    )
    return data_d1.t(), scale_d1


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


def test_float8_e4m3fnuz():
    dtype = torch.float8_e4m3fnuz
    interesting_values = dtype_to_interesting_values[dtype]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)


def test_float8_e5m2fnuz():
    dtype = torch.float8_e5m2fnuz
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
        fp32 = _sem_enc_to_fp32_val(s_enc, e_enc, m_enc, is_zero, is_denorm, exp_bias)  # noqa: E501
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
        fp32 = _sem_enc_to_fp32_val(s_enc, e_enc, m_enc, is_zero, is_denorm, exp_bias)  # noqa: E501
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
        fp32 = _sem_enc_to_fp32_val(s_enc, e_enc, m_enc, is_zero, is_denorm, exp_bias)  # noqa: E501
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

    # ensure packing is
    #
    #   7654:3210
    #   val1:val0
    expected_f4_packed = torch.tensor(
        [
            [
                0b00010000,
                0b10000110,
            ],
            [
                0b00101000,
                0b01011111,
            ],
        ],
        dtype=torch.uint8,
    )

    assert torch.all(orig_vals_f4_packed == expected_f4_packed)
    assert orig_vals_f4_packed.numel() == (orig_vals.numel() / 2)
    orig_vals_f4_packed_unpacked = unpack_uint4(orig_vals_f4_packed)
    orig_vals_dq = f4_unpacked_to_f32(orig_vals_f4_packed_unpacked)
    assert torch.all(orig_vals_dq == orig_vals)


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
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "f32_val,f6_e3m2_enc",
    [
        (29.0, 0b011111),  # normal round down
        (26.0, 0b011110),  # normal round to nearest even
        (0.1251, 0b000010),  # subnormal round down
        (0.0314, 0b000001),  # subnormal round up
        (0.03, 0b000000),  # underflow
    ],
)
def test_fp6_e3m2_rounding(f32_val, f6_e3m2_enc, device):
    f6_e3m2_unpacked = f32_to_f6_e3m2_unpacked(torch.tensor(f32_val, device=device))
    assert f6_e3m2_unpacked.item() == f6_e3m2_enc

    f6_e3m2_unpacked = f32_to_f6_e3m2_unpacked(torch.tensor(-f32_val, device=device))
    assert f6_e3m2_unpacked.item() == (f6_e3m2_enc | 0b100000)


def triton_to_mxfp8_dim0_reference(
    x_hp: torch.Tensor,
    block_size,
    scaling_mode=ScaleCalculationMode.FLOOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A reference version of `triton_to_mxfp8_dim0` for rowwise quantization.
    """
    from torchao.prototype.mx_formats.mx_tensor import to_mx

    # cast across dim0 (rowwise) - no transpose needed
    scale_e8m0_dim0, x_hp_d0_normalized = to_mx(
        x_hp, torch.float8_e4m3fn, block_size, scaling_mode=scaling_mode
    )
    scale_e8m0_dim0 = scale_e8m0_dim0.view(torch.float8_e8m0fnu)
    return (
        x_hp_d0_normalized,
        scale_e8m0_dim0,
    )


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.parametrize("M", (128, 256))
@pytest.mark.parametrize("K", (128, 256))
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_triton_mxfp8_dim1_randn(M, K, scaling_mode):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    x_mx_ref, x_s_ref = triton_to_mxfp8_dim1_reference(
        x, block_size=32, scaling_mode=scaling_mode
    )
    x_mx_t, x_s_t = triton_to_mxfp8_dim1(
        x, inner_block_size=32, scaling_mode=scaling_mode.value.lower()
    )
    torch.testing.assert_close(x_mx_t, x_mx_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_s_t, x_s_ref, rtol=0, atol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.parametrize("M", (128, 256))
@pytest.mark.parametrize("K", (128, 256))
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_triton_mxfp8_dim0_randn(M, K, scaling_mode):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    x_mx_ref, x_s_ref = triton_to_mxfp8_dim0_reference(
        x, block_size=32, scaling_mode=scaling_mode
    )
    x_mx_t, x_s_t = triton_to_mxfp8_dim0(
        x,
        inner_block_size=32,
        scaling_mode=scaling_mode.value.lower(),
    )
    torch.testing.assert_close(x_mx_t, x_mx_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_s_t, x_s_ref, rtol=0, atol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_triton_mxfp8_dim0_zeros(scaling_mode):
    x = torch.zeros(128, 256, dtype=torch.bfloat16, device="cuda")
    x_mx_ref, x_s_ref = triton_to_mxfp8_dim0_reference(
        x, block_size=32, scaling_mode=scaling_mode
    )
    x_mx_t, x_s_t = triton_to_mxfp8_dim0(
        x,
        inner_block_size=32,
        scaling_mode=scaling_mode.value.lower(),
    )
    assert not x_mx_t.isnan().any(), "quantized tensor should not contain NaNs"
    torch.testing.assert_close(x_mx_t, x_mx_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_s_t, x_s_ref, rtol=0, atol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.parametrize("M", (128, 256))
@pytest.mark.parametrize("K", (128, 256))
@pytest.mark.parametrize("orig_dtype", (torch.float32, torch.bfloat16))
def test_triton_mxfp8_dequant_dim0(M, K, orig_dtype):
    x = torch.zeros(M, K, dtype=orig_dtype, device="cuda")
    block_size = 32
    x_data, x_scales = triton_to_mxfp8_dim0_reference(x, block_size=32)
    hp_ref = to_dtype(
        x_data,
        x_scales,
        torch.float8_e4m3fn,
        block_size,
        orig_dtype,
    )
    hp_t = triton_mxfp8_dequant_dim0(x_data, x_scales, orig_dtype, block_size)
    torch.testing.assert_close(hp_t, hp_ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "shape",
    [
        (63, 1023),
        (128, 4),
        (128, 8),
        (256, 8),
        (300, 9),
        (133, 512),
        (528, 512),
        (128, 1),
    ],
)
def test_rearrange(shape):
    scales = torch.randint(256, size=shape, device="cuda", dtype=torch.uint8)
    eager = to_blocked(scales, False)
    triton = to_blocked(scales, True)
    torch.testing.assert_close(eager, triton, atol=0, rtol=0)


@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="MXFP8 requires CUDA capability 10.0 or greater",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
def test_cuda_mx_quantize_on_a_fresh_thread():
    """The kernel must work when it is the first CUDA call on its thread.

    It encodes TMA descriptors through the driver API, which needs a current
    context. A thread that has not used CUDA yet does not have one, and a
    DeviceGuard does not bind it when the device already matches. Autograd runs
    backward on worker threads, so this is the common case for an MXFP8
    backward. Without the fix the kernel raises a cudaErrorIllegalInstruction.
    """
    import threading

    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    failure = []

    def run():
        try:
            mxfp8_quantize_cuda(x, rowwise=False, colwise=True, scaling_mode="rceil")
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            failure.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert not failure, f"quantize failed on a fresh thread: {failure[0]}"


@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="MXFP8 requires CUDA capability 10.0 or greater",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize("M", (32, 256))
@pytest.mark.parametrize("K", (32, 256))
@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
@pytest.mark.parametrize(
    "rowwise,colwise", ((True, False), (False, True), (True, True))
)
def test_cuda_mx_numerics(M, K, input_dtype, scaling_mode, rowwise, colwise):
    scaling_mode_str = (
        "floor" if scaling_mode == ScaleCalculationMode.FLOOR else "rceil"
    )
    block_size = 32

    # Use distinct incrementing values from 0 to M*K-1 to make debugging easier.
    x = (
        torch.arange(0, M * K, dtype=input_dtype, device="cuda")
        .reshape(M, K)
        .contiguous()
    )

    if rowwise:
        s_d0_ref, y_d0_ref = to_mx(
            x,
            torch.float8_e4m3fn,
            block_size,
            scaling_mode=scaling_mode,
        )
    if colwise:
        y_d1_ref, s_d1_ref = to_mx_dim1_reference(
            x,
            block_size=block_size,
            scaling_mode=scaling_mode,
        )

    y_d0, y_d1, s_d0, s_d1 = mxfp8_quantize_cuda(
        x,
        rowwise=rowwise,
        colwise=colwise,
        scaling_mode=scaling_mode_str,
    )

    if rowwise:
        # Rowwise uses independent 1x32 groups, including when colwise is also
        # requested from the fused kernel.
        torch.testing.assert_close(s_d0, s_d0_ref, rtol=0, atol=0)
        torch.testing.assert_close(y_d0, y_d0_ref, rtol=0, atol=0)
        assert y_d0.stride() == y_d0_ref.stride()
    else:
        assert y_d0.numel() == 0
        assert s_d0.numel() == 0

    if colwise:
        # Colwise uses independent 32x1 groups, including when rowwise is also
        # requested from the fused kernel.
        torch.testing.assert_close(s_d1, s_d1_ref, rtol=0, atol=0)
        torch.testing.assert_close(y_d1, y_d1_ref, rtol=0, atol=0)
        assert y_d1.stride() == y_d1_ref.stride()
    else:
        assert y_d1.numel() == 0
        assert s_d1.numel() == 0


@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="MXFP8 requires CUDA capability 10.0 or greater",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize(
    "rowwise,colwise,scale_dim_x,scale_dim_y,error",
    (
        (True, False, 1, 1, "rowwise output requires scale_dim_x == 32"),
        (False, True, 1, 1, "colwise output requires scale_dim_y == 32"),
    ),
)
def test_cuda_mx_rejects_invalid_scale_dimensions(
    rowwise, colwise, scale_dim_x, scale_dim_y, error
):
    M, K = 64, 64
    x = (
        torch.arange(0, M * K, dtype=torch.bfloat16, device="cuda")
        .reshape(M, K)
        .contiguous()
    )
    with pytest.raises(RuntimeError, match=error):
        torch.ops.torchao.mxfp8_quantize.default(
            x,
            rowwise,
            colwise,
            scale_dim_x,
            scale_dim_y,
            "e4m3",
            "rceil",
        )


@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="MXFP8 requires CUDA capability 10.0 or greater",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize(
    "rowwise,colwise", ((True, False), (False, True), (True, True))
)
def test_cuda_mx_fake_shapes_and_strides(rowwise, colwise):
    rows, cols = 64, 96
    block_size = 32
    output_rowwise, output_colwise, scales_rowwise, scales_colwise = (
        torch.ops.torchao.mxfp8_quantize.default(
            torch.empty((rows, cols), device="meta"),
            rowwise,
            colwise,
            block_size if rowwise else 1,
            block_size if colwise else 1,
            "e4m3",
            "rceil",
        )
    )

    if rowwise:
        torch._assert_tensor_metadata(
            output_rowwise,
            size=(rows, cols),
            stride=(cols, 1),
        )
        torch._assert_tensor_metadata(
            scales_rowwise,
            size=(rows, cols // block_size),
            stride=(cols // block_size, 1),
        )
    else:
        torch._assert_tensor_metadata(output_rowwise, size=(0,), stride=(1,))
        torch._assert_tensor_metadata(scales_rowwise, size=(0,), stride=(1,))

    if colwise:
        torch._assert_tensor_metadata(
            output_colwise,
            size=(rows, cols),
            stride=(1, rows),
        )
        torch._assert_tensor_metadata(
            scales_colwise,
            size=(cols, rows // block_size),
            stride=(1, cols),
        )
    else:
        torch._assert_tensor_metadata(output_colwise, size=(0,), stride=(1,))
        torch._assert_tensor_metadata(scales_colwise, size=(0,), stride=(1,))


@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="MXFP8 requires CUDA capability 10.0 or greater",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize(
    "rowwise,colwise", ((True, False), (False, True), (True, True))
)
@pytest.mark.parametrize(
    "shard_dims",
    ((0,), (1,), (0, 1), (1, 0)),
    ids=("S0", "S1", "S0_S1", "S1_S0"),
)
def test_cuda_mx_dtensor_sharding_commutes_with_quantization(
    rowwise, colwise, shard_dims
):
    import torch.distributed as dist
    from torch.distributed._local_tensor import LocalTensorMode
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.debug import CommDebugMode

    mesh_shape = (2,) * len(shard_dims)
    world_size = 2 ** len(shard_dims)
    placements = [Shard(dim) for dim in shard_dims]
    replicated_placements = [Replicate()] * len(shard_dims)

    assert not dist.is_initialized()
    dist.init_process_group(
        "fake",
        store=dist.HashStore(),
        rank=0,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cuda", mesh_shape)
        full_input = torch.randn(
            128, 128, device="cuda", dtype=torch.bfloat16
        ).contiguous()

        with LocalTensorMode(world_size):
            sharded_input = distribute_tensor(full_input, mesh, placements)
            replicated_input = sharded_input.redistribute(
                placements=replicated_placements
            )

            with CommDebugMode() as replicated_quantize_comm:
                expected_outputs = mxfp8_quantize_cuda(
                    replicated_input,
                    rowwise=rowwise,
                    colwise=colwise,
                )

            with CommDebugMode() as sharded_quantize_comm:
                sharded_outputs = mxfp8_quantize_cuda(
                    sharded_input,
                    rowwise=rowwise,
                    colwise=colwise,
                )

            actual_outputs = tuple(
                output.redistribute(placements=replicated_placements)
                for output in sharded_outputs
            )

            assert replicated_quantize_comm.get_total_counts() == 0
            assert sharded_quantize_comm.get_total_counts() == 0
            for expected, actual in zip(expected_outputs, actual_outputs):
                assert expected.shape == actual.shape
                torch.testing.assert_close(
                    expected.to_local(), actual.to_local(), rtol=0, atol=0
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="MXFP8 requires CUDA capability 10.0 or greater",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("scaling_mode", ("floor", "rceil"))
@pytest.mark.parametrize("orientation", ("rowwise", "colwise"))
def test_cuda_mxfp8_special_value_semantics(input_dtype, scaling_mode, orientation):
    cases = make_mxfp8_semantic_cases(input_dtype, scaling_mode, device="cuda")
    num_cases = len(cases.names)
    oriented = torch.zeros(32, 32, dtype=input_dtype, device="cuda")
    oriented[:num_cases] = cases.inputs

    x = oriented if orientation == "rowwise" else oriented.t().contiguous()

    outputs = mxfp8_quantize_cuda(
        x,
        rowwise=orientation == "rowwise",
        colwise=orientation == "colwise",
        scaling_mode=scaling_mode,
    )
    data = outputs[0] if orientation == "rowwise" else outputs[1].t()
    scales = outputs[2] if orientation == "rowwise" else outputs[3]
    assert_mxfp8_semantics(data[:num_cases, :32], scales[:num_cases, :1], cases)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_triton_mxfp8_dim0_special_values(scaling_mode: ScaleCalculationMode):
    block_size = 32
    cases = make_mxfp8_semantic_cases(torch.bfloat16, scaling_mode, device="cuda")

    x_mx_t, x_s_t = triton_to_mxfp8_dim0(
        cases.inputs,
        inner_block_size=block_size,
        scaling_mode=scaling_mode.value.lower(),
    )

    assert_mxfp8_semantics(x_mx_t, x_s_t, cases)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize("scaling_mode", (ScaleCalculationMode.RCEIL,))
def test_triton_mxfp8_dim0_overflow_underflow(scaling_mode):
    """Test with values near overflow and underflow thresholds."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_subnormal_min = 2e-9  # smallest positive subnormal for e4m3: https://www.emergentmind.com/topics/mxfp8-e4m3-floating-point-format
    block_size = 32

    test_vals = torch.zeros(4, block_size, dtype=torch.bfloat16, device="cuda")

    # Row 0: elem 0 is near max, elems 1-3 are above max
    test_vals[0, :4] = torch.tensor(
        [fp8_max * 0.9, fp8_max * 1.1, fp8_max * 2.0, fp8_max * 10.0],
        dtype=torch.bfloat16,
    )

    # Row 1: elem 0 is near min, elems 1-3 are below min
    test_vals[1, :4] = torch.tensor(
        [-fp8_max * 0.9, -fp8_max * 1.1, -fp8_max * 2.0, -fp8_max * 10.0],
        dtype=torch.bfloat16,
    )

    # Row 2: elem 0-1 are below positive subnormal min representable in e4m3, should underflow to zero if scaled down
    test_vals[2, :3] = torch.tensor(
        [
            fp8_subnormal_min * 0.1,
            fp8_subnormal_min * 0.5,
            fp8_max
            * 0.9,  # include a large value to result in scale that would underflow the subnormals
        ],
        dtype=torch.bfloat16,
    )
    # Row 3: elem 0-1 are above below negative subnormal min, should underflow to zero
    test_vals[3, :3] = torch.tensor(
        [
            -fp8_subnormal_min * 0.1,
            -fp8_subnormal_min * 0.5,
            fp8_max
            * 0.9,  # include a large value to result in scale that would underflow the subnormals
        ],
        dtype=torch.bfloat16,
    )

    x_mx_ref, x_s_ref = triton_to_mxfp8_dim0_reference(
        test_vals, block_size=block_size, scaling_mode=scaling_mode
    )
    x_mx_t, x_s_t = triton_to_mxfp8_dim0(
        test_vals,
        inner_block_size=block_size,
        scaling_mode=scaling_mode.value.lower(),
    )

    # Test 1: Verify triton matches reference
    assert not x_mx_t.isnan().any(), "quantized tensor should not contain NaNs"
    assert not x_s_t.isnan().any(), "scales should not contain NaNs"
    torch.testing.assert_close(x_mx_t, x_mx_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_s_t, x_s_ref, rtol=0, atol=0)

    dequantized = to_dtype(
        x_mx_t,
        x_s_t.view(torch.float8_e8m0fnu),
        torch.float8_e4m3fn,
        block_size,
        torch.bfloat16,
    )

    # Verify quantization preserves sign
    original_signbits = torch.signbit(test_vals)
    dequant_signbits = torch.signbit(dequantized)
    assert torch.equal(original_signbits, dequant_signbits), (
        "Sign bit mismatch between original and dequantized values"
    )

    # Verify underflow behavior
    # Check rows 2 and 3 which contain underflow test cases
    for row_idx in [2, 3]:
        # The first two elements should be scaled below the min representable subnormal in e4m3, and thus underflow to zero
        assert torch.all(dequantized[row_idx, :2] == 0.0), (
            f"Row {row_idx}: should underflow to zero"
        )
        # Normal val shouldn't underflow
        assert torch.all(dequantized[row_idx, 2] != 0.0), (
            f"Row {row_idx}: should not underflow to zero"
        )


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.RCEIL, ScaleCalculationMode.FLOOR)
)
def test_triton_mxfp8_dim0_large_tensor_offset_no_overflow(scaling_mode):
    """Test with large tensor whose offsets exceeds the max int32 value."""
    x = torch.randn((184320, 14336), dtype=torch.bfloat16, device="cuda")
    block_size = 32
    x_mx_ref, x_s_ref = triton_to_mxfp8_dim0_reference(
        x, block_size=block_size, scaling_mode=scaling_mode
    )
    x_mx_t, x_s_t = triton_to_mxfp8_dim0(
        x,
        inner_block_size=block_size,
        scaling_mode=scaling_mode.value.lower(),
    )

    assert not x_mx_t.isnan().any(), "quantized tensor should not contain NaNs"
    assert not x_s_t.isnan().any(), "scales should not contain NaNs"
    torch.testing.assert_close(x_mx_t, x_mx_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_s_t, x_s_ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Pure-PyTorch numerical reference for the standalone mxfp8 32x32 cast, ported from the
# quant_cast_gold recipe `mxfp8_32x32_f` (eager RCEIL bit-math branch). Kept local so the test
# depends only on torchao, and pins the standalone 32x32 mxfp8 cast bit-for-bit to this definition.
# ---------------------------------------------------------------------------
def _ref_mxfp8_32x32(x):
    """mxfp8 with square 32x32 blocks (one e8m0 scale per block). Returns
    (qdata float8_e4m3fn (M, N), scale float8_e8m0fnu (M//32, N//32)).

    Uses torchao's `_f32_to_e8m0_rceil` (RCEIL amax->e8m0) and
    `_e8m0_scale_to_reciprocal_fp32` (e8m0->fp32 reciprocal) so the reference tracks torchao's
    canonical mxfp8 numerics rather than a hand-rolled copy."""
    *lead, d1, d2 = x.shape
    n1, n2 = d1 // 32, d2 // 32
    x_b = (
        x.reshape(*lead, n1, 32, n2, 32)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 32 * 32)
    )
    amax = x_b.abs().amax(dim=-1, keepdim=True)  # (..., n1, n2, 1)
    descale = amax.to(torch.float32) * (
        1.0 / torch.finfo(torch.float8_e4m3fn).max
    )  # /448
    scale_biased = _f32_to_e8m0_rceil(descale)  # uint8 e8m0 biased exponent
    qdata_b = (x_b.to(torch.float32) * _e8m0_scale_to_reciprocal_fp32(scale_biased)).to(
        torch.float8_e4m3fn
    )
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 32, 32)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, d1, d2)
    )
    return qdata, scale_biased.view(torch.float8_e8m0fnu).squeeze(-1)


def _ref_e8m0_to_fp32(scale):
    """Inverse of the e8m0 cast: e8m0 biased exponent -> fp32 pow2 factor (used by dequant)."""
    biased_i32 = scale.contiguous().view(torch.uint8).to(torch.int32)
    scale_fp32 = (biased_i32 << 23).view(torch.float32)
    return torch.clamp(scale_fp32, min=2.0**-126)


def _ref_mxfp8_32x32_dq(q, scale):
    """Dequant for the 32x32 mxfp8 cast (ported from quant_cast_gold `mxfp8_32x32_dq_f`):
    un-block the e8m0 scale over the 32x32 grid and multiply."""
    M, N = q.shape
    n1, n2 = M // 32, N // 32
    s = _ref_e8m0_to_fp32(scale).reshape(n1, 1, n2, 1)
    return (q.float().reshape(n1, 32, n2, 32) * s).reshape(M, N)


# ---------------------------------------------------------------------------
# Swizzled 32x32 variants: same square-block quant as `_ref_mxfp8_32x32`, but the per-block
# e8m0 scale is expanded over the block's 32 rows (dim0) or 32 cols (dim1, transposed) and
# emitted in NVIDIA's blocked/swizzled layout -- reusing torchao's `to_blocked` / `from_blocked`
# (a 4D block grid `.reshape(-1)` equals `to_blocked`'s flat buffer).
# ---------------------------------------------------------------------------
def _ref_mxfp8_32x32_swizzle_dim0(x):
    """Reference dim0 swizzle: (qdata (M,N) fp8, swizzled scale as a flat uint8 buffer)."""
    q_ref, scale = _ref_mxfp8_32x32(x)  # (M,N), (M//32, N//32)
    scale_exp = scale.view(torch.uint8).repeat_interleave(32, dim=0)  # (M, N//32)
    return q_ref, to_blocked(scale_exp).view(torch.uint8)


def _ref_mxfp8_32x32_swizzle_dim1(x):
    """Reference dim1 (transposed) swizzle: (qdata (N,M) fp8, swizzled scale flat uint8)."""
    q_ref, scale = _ref_mxfp8_32x32(x)  # (M,N), (M//32, N//32)
    scale_exp = (
        scale.view(torch.uint8).repeat_interleave(32, dim=1).t().contiguous()
    )  # (N, M//32)
    return q_ref.t().contiguous(), to_blocked(scale_exp).view(torch.uint8)


def _swizzle_dequant_sqnr_dim0(x, q_t, s_t):
    """Un-swizzle the dim0 scale back to per-block, dequant, and return SQNR vs x."""
    M, N = x.shape
    scale_unswz = from_blocked(
        s_t.view(torch.uint8).reshape(-1), M, N // 32
    )  # (M, N//32)
    scale_blocks = scale_unswz[::32].contiguous()  # (M//32, N//32)
    return compute_error(x.float(), _ref_mxfp8_32x32_dq(q_t, scale_blocks).float())


def _swizzle_dequant_sqnr_dim1(x, q_t, s_t):
    """Un-swizzle the dim1 (transposed) scale, dequant in the (N,M) frame, SQNR vs x."""
    M, N = x.shape
    scale_unswz = from_blocked(
        s_t.view(torch.uint8).reshape(-1), N, M // 32
    )  # (N, M//32)
    scale_blocks = scale_unswz[::32].contiguous()  # (N//32, M//32)
    x_hat_t = _ref_mxfp8_32x32_dq(q_t, scale_blocks)  # (N, M)
    return compute_error(x.float(), x_hat_t.t().float())


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(
    not is_sm_at_least_100() and not is_MI350(),
    reason="mxfp8 requires CUDA capability 10.0 or greater or ROCm gfx950 or greater.",
)
@pytest.mark.parametrize("M", (64, 128, 256))
@pytest.mark.parametrize("K", (64, 128, 256))
def test_triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(M, K):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    qk, sk, sm = triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(x)
    qk_ref, sk_ref = _ref_mxfp8_32x32_swizzle_dim0(x)
    _, sm_ref = _ref_mxfp8_32x32_swizzle_dim1(x)
    # dim-K qdata + both swizzled scales bit-exact (there is no dim-M qdata output).
    assert torch.equal(qk.view(torch.uint8), qk_ref.view(torch.uint8))
    assert torch.equal(sk.view(torch.uint8).reshape(-1), sk_ref)
    assert torch.equal(sm.view(torch.uint8).reshape(-1), sm_ref)
    # dim-K frame dequants directly; dim-M frame reuses the shared qdata transposed.
    sqnr_k = _swizzle_dequant_sqnr_dim0(x, qk, sk)
    sqnr_m = _swizzle_dequant_sqnr_dim1(x, qk.t().contiguous(), sm)
    assert sqnr_k > 15.0, (
        f"swizzle_dim0_qdata_dim01_scale (dim-k): "
        f"sqnr={sqnr_k.item():.2f} dB below 15 dB"
    )
    assert sqnr_m > 15.0, (
        f"swizzle_dim0_qdata_dim01_scale (dim-m): "
        f"sqnr={sqnr_m.item():.2f} dB below 15 dB"
    )
