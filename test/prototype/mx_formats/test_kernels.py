# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.custom_fp_utils import RoundingMode
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
    triton_to_mxfp8_dim0,
    triton_to_mxfp8_dim1,
    triton_to_mxfp8_dim1_reference,
    unpack_uint4,
)
from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_dtype, to_mx
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import (
    is_cuda_version_at_least,
    is_MI350,
    is_sm_at_least_100,
    torch_version_at_least,
)

torch.manual_seed(0)

if not torch_version_at_least("2.8.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

if has_triton() and torch.cuda.is_available() and is_sm_at_least_100():
    import triton
    import triton.language as tl

    from torchao.prototype.mx_formats.kernels import (
        convert_fp32_to_fp4_packed,
        convert_fp32_to_fp4_packed_rs,
    )

    @triton.jit
    def _triton_f4_pack_kernel(
        x_ptr,
        out_ptr,
        N,
        seed_ptr,
        ROUNDING_MODE: tl.constexpr,
    ):
        """Thin wrapper to test convert_fp32_to_fp4_packed{,_rs} in isolation."""
        pid = tl.program_id(0)
        offs = pid * 64 + tl.arange(0, 64)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        x_pairs = x.reshape(32, 2).split()
        if ROUNDING_MODE == 0:
            x_fp4x2 = convert_fp32_to_fp4_packed(x_pairs)
        else:
            out_offs = pid * 32 + tl.arange(0, 32)
            seed = tl.load(seed_ptr)
            rbits = tl.randint(seed, out_offs)
            x_fp4x2 = convert_fp32_to_fp4_packed_rs(x_pairs, rbits)
        out_offs = pid * 32 + tl.arange(0, 32)
        tl.store(out_ptr + out_offs, x_fp4x2, mask=out_offs < N // 2)

    def triton_f4_pack(x, rounding_mode=RoundingMode.RN):
        """Pack FP32 values to FP4 using Triton convert_fp32_to_fp4_packed{,_rs}."""
        N = x.numel()
        out = torch.empty(N // 2, dtype=torch.uint8, device=x.device)
        seed = torch.randint(0, 2**31, (1,), dtype=torch.int32, device=x.device)
        grid = (triton.cdiv(N, 64),)
        _triton_f4_pack_kernel[grid](
            x,
            out,
            N,
            seed,
            ROUNDING_MODE=rounding_mode.value,
        )
        return out


FP4_RN_EXPECTED = [(5.2, 6.0), (-5.2, -6.0)]

_triton_kernel_params = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not (has_triton() and torch.cuda.is_available() and is_sm_at_least_100()),
            reason="Triton FP4 kernel requires CUDA capability 10.0 or greater",
        ),
    ),
]


def _f4_quantize(x, rounding_mode, use_triton):
    """Quantize FP32 to FP4 and dequantize, using either PyTorch or Triton kernel."""
    if rounding_mode not in RoundingMode:
        raise ValueError(
            f"Unknown rounding_mode: {rounding_mode}. "
            f"Expected RoundingMode.RN or RoundingMode.RS."
        )
    if use_triton:
        xq = triton_f4_pack(x.flatten(), rounding_mode=rounding_mode)
        return f4_unpacked_to_f32(unpack_uint4(xq))
    else:
        rand_bits = (
            torch.randint(0, 2**31, x.shape, dtype=torch.int32, device=x.device)
            if rounding_mode == RoundingMode.RS
            else None
        )
        return f4_unpacked_to_f32(
            f32_to_f4_unpacked(x, rounding_mode=rounding_mode, rand_bits=rand_bits)
        )


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
@pytest.mark.parametrize("M", (32, 256))
@pytest.mark.parametrize("K", (32, 256))
@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize(
    "scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_cuda_mx_dim1_numerics(M, K, input_dtype, scaling_mode):
    scaling_mode_str = (
        "floor" if scaling_mode == ScaleCalculationMode.FLOOR else "rceil"
    )
    block_size = 32

    # Use disinct incrementing values from 0 to M*K-1 to make debugging easier.
    x = (
        torch.arange(0, M * K, dtype=input_dtype, device="cuda")
        .reshape(M, K)
        .contiguous()
    )

    y_d1_ref, s_d1_ref = to_mx_dim1_reference(
        x,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )

    _, y_d1, _, s_d1 = mxfp8_quantize_cuda(
        x,
        rowwise=False,
        colwise=True,
        scaling_mode=scaling_mode_str,
    )

    # check scales
    torch.testing.assert_close(s_d1, s_d1_ref, rtol=0, atol=0)

    # check quantized values
    torch.testing.assert_close(y_d1, y_d1_ref, rtol=0, atol=0)
    assert y_d1.stride() == y_d1_ref.stride(), "quantized tensor strides do not match"


@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="MXFP8 requires CUDA capability 10.0 or greater",
)
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8),
    reason="CUDA version >= 12.8 required for MXFP8 CUDA kernels",
)
def test_cuda_mx_dim0_not_supported():
    M, K = 64, 64
    x = (
        torch.arange(0, M * K, dtype=torch.bfloat16, device="cuda")
        .reshape(M, K)
        .contiguous()
    )
    with pytest.raises(RuntimeError):
        _, y_d1, _, s_d1 = mxfp8_quantize_cuda(
            x,
            rowwise=True,
            colwise=False,
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
@pytest.mark.parametrize("scaling_mode", (ScaleCalculationMode.RCEIL,))
def test_triton_mxfp8_dim0_special_values(scaling_mode: ScaleCalculationMode):
    # Create tensor with special values - make it compatible with block_size=32
    block_size = 32
    special_vals = torch.zeros(2, block_size, dtype=torch.bfloat16, device="cuda")

    # Fill first few elements of each row with special values
    special_vals[0, :4] = torch.tensor(
        [float("inf"), -float("inf"), float("nan"), 0.0], dtype=torch.bfloat16
    )
    special_vals[1, :4] = torch.tensor(
        [
            torch.finfo(torch.float32).max,
            torch.finfo(torch.float32).min,
            torch.finfo(torch.float32).tiny,
            -torch.finfo(torch.float32).tiny,
        ],
        dtype=torch.bfloat16,
    )

    x_mx_ref, x_s_ref = triton_to_mxfp8_dim0_reference(
        special_vals, block_size=block_size, scaling_mode=scaling_mode
    )
    x_mx_t, x_s_t = triton_to_mxfp8_dim0(
        special_vals,
        inner_block_size=block_size,
        scaling_mode=scaling_mode.value.lower(),
    )
    x_mx_t = x_mx_t.to(torch.float32)
    x_mx_ref = x_mx_ref.to(torch.float32)

    # Check NaN behavior: if any value in a block is NaN, scale and entire block become NaN
    for row_idx in range(special_vals.shape[0]):
        input_block_has_nan = special_vals[row_idx].isnan().any()

        if input_block_has_nan:
            # If any value in block is NaN, scale should be NaN
            assert torch.isnan(x_s_t[row_idx].to(torch.float32)), (
                f"Row {row_idx}: Block with any NaN should have NaN scale"
            )
            # And entire quantized block should be NaN
            assert torch.all(torch.isnan(x_mx_t[row_idx])), (
                f"Row {row_idx}: Block with any NaN should have all NaN quantized values"
            )
        else:
            # If no NaN in input block, scale and data should not be NaN
            assert not torch.isnan(x_s_t[row_idx].to(torch.float32)), (
                f"Row {row_idx}: Block without NaN should not have NaN scale"
            )

    # Use NaN-aware comparison to handle nan != nan case properly
    # Check NaN patterns match
    nan_ref = torch.isnan(x_mx_ref)
    nan_triton = torch.isnan(x_mx_t)
    assert torch.equal(nan_ref, nan_triton), (
        "NaN pattern mismatch between reference and triton"
    )

    # Check finite values
    finite_mask = torch.isfinite(x_mx_ref) & torch.isfinite(x_mx_t)
    if finite_mask.any():
        assert torch.equal(x_mx_ref[finite_mask], x_mx_t[finite_mask]), (
            "Finite values mismatch"
        )

    # Check infinity patterns
    inf_ref = torch.isinf(x_mx_ref)
    inf_triton = torch.isinf(x_mx_t)
    assert torch.equal(inf_ref, inf_triton), (
        "Infinity pattern mismatch between reference and triton"
    )
    if inf_ref.any():
        assert torch.equal(x_mx_ref[inf_ref], x_mx_t[inf_ref]), (
            "Infinity values mismatch"
        )

    # Check scales using exact comparison
    x_s_ref_uint8 = x_s_ref.to(torch.uint8)
    x_s_t_uint8 = x_s_t.to(torch.uint8)
    assert torch.equal(x_s_t_uint8, x_s_ref_uint8), (
        "Scale values mismatch between reference and triton"
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_cuda_version_at_least(12, 8), reason="CUDA version >= 12.8 required"
)
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="CUDA capability 10.0 or greater required"
)
@pytest.mark.parametrize("scaling_mode", (ScaleCalculationMode.RCEIL,))
def test_all_nan_block_scale_behavior(scaling_mode):
    """
    Test that PyTorch and Triton implementations align on NaN scale behavior:
    - Any NaN in block: scale = NaN, entire quantized block becomes NaN
    """
    from torchao.prototype.mx_formats.mx_tensor import to_mx

    block_size = 32

    # Create test case with both mixed and all-NaN blocks
    # First 32 elements: mixed NaN + real values
    # Second 32 elements: all NaN values
    # Third 32 elements: normal values for reference
    test_vals = torch.zeros(3 * block_size, dtype=torch.bfloat16, device="cuda")

    # Block 1: Mixed NaN + real values [NaN, 1.0, NaN, 5.0, NaN, 3.0, ...]
    test_vals[:block_size:3] = float("nan")  # Every 3rd element is NaN
    test_vals[1:block_size:3] = 1.0  # Some real values
    test_vals[2:block_size:3] = 5.0

    # Block 2: All NaN values
    test_vals[block_size : 2 * block_size] = float("nan")

    # Block 3: Normal values for reference
    test_vals[2 * block_size :] = torch.linspace(1.0, 10.0, block_size)

    # Test PyTorch implementation through to_mx
    scale_pytorch, data_pytorch = to_mx(
        test_vals, torch.float8_e4m3fn, block_size, scaling_mode
    )

    # Convert to regular tensor for easier inspection
    scale_pytorch_vals = scale_pytorch.to(torch.float32)
    data_pytorch_vals = data_pytorch.to(torch.float32)

    # Test expectations: If any value in a block is NaN, scale = NaN and entire block becomes NaN

    # Block 0 (mixed NaN + real): Should have NaN scale and all NaN data
    assert torch.isnan(scale_pytorch_vals[0]), (
        "Block with any NaN should have NaN scale"
    )
    assert torch.all(torch.isnan(data_pytorch_vals[:block_size])), (
        "Block with any NaN should have all NaN quantized values"
    )

    # Block 1 (all NaN): Should have NaN scale and all NaN data
    assert torch.isnan(scale_pytorch_vals[1]), "All-NaN block should have NaN scale"
    assert torch.all(torch.isnan(data_pytorch_vals[block_size : 2 * block_size])), (
        "All-NaN block should have all NaN quantized values"
    )

    # Block 2 (normal): Should have real scale and finite data
    assert not torch.isnan(scale_pytorch_vals[2]), "Normal block should have real scale"
    assert torch.all(torch.isfinite(data_pytorch_vals[2 * block_size :])), (
        "Normal block should have finite quantized values"
    )

    # Also test the Triton implementation to ensure consistency
    test_vals_2d = test_vals.reshape(3, block_size)
    x_mx_t, x_s_t = triton_to_mxfp8_dim0(
        test_vals_2d,
        inner_block_size=block_size,
        scaling_mode=scaling_mode.value.lower(),
    )

    # Convert for comparison
    x_s_t_vals = x_s_t.to(torch.float32)
    x_mx_t_vals = x_mx_t.to(torch.float32)

    # Test Triton implementation matches PyTorch behavior
    # Block 0 (mixed NaN + real): Should have NaN scale and all NaN data
    assert torch.isnan(x_s_t_vals[0]), (
        "Triton: Block with any NaN should have NaN scale"
    )
    assert torch.all(torch.isnan(x_mx_t_vals[0])), (
        "Triton: Block with any NaN should have all NaN quantized values"
    )

    # Block 1 (all NaN): Should have NaN scale and all NaN data
    assert torch.isnan(x_s_t_vals[1]), "Triton: All-NaN block should have NaN scale"
    assert torch.all(torch.isnan(x_mx_t_vals[1])), (
        "Triton: All-NaN block should have all NaN quantized values"
    )

    # Block 2 (normal): Should have real scale and finite data
    assert not torch.isnan(x_s_t_vals[2]), "Triton: Normal block should have real scale"
    assert torch.all(torch.isfinite(x_mx_t_vals[2])), (
        "Triton: Normal block should have finite quantized values"
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_triton", _triton_kernel_params)
@pytest.mark.parametrize("rounding_mode", [RoundingMode.RN, RoundingMode.RS, 99])
@pytest.mark.parametrize("shape", [(1024, 128)])
@pytest.mark.parametrize("value,rn_expected", FP4_RN_EXPECTED)
def test_f4_rounding(value, rn_expected, shape, rounding_mode, use_triton):
    """Test FP4 rounding: RN is biased, RS is unbiased, invalid raises."""
    x = torch.ones(*shape, device="cuda", dtype=torch.bfloat16) * value

    if rounding_mode not in RoundingMode:
        with pytest.raises(ValueError, match="Unknown rounding_mode"):
            _f4_quantize(x.float(), rounding_mode, use_triton)
        return

    rtol = 1e-2
    r1 = _f4_quantize(x.float(), rounding_mode, use_triton)

    # Check rounding behavior via mean
    r1_mean = torch.mean(r1)
    if rounding_mode == RoundingMode.RN:
        torch.testing.assert_close(r1_mean.item(), rn_expected, rtol=rtol, atol=rtol)
    else:
        input_mean = torch.mean(x.float())
        torch.testing.assert_close(r1_mean, input_mean, rtol=rtol, atol=rtol)

    # Check torch.manual_seed() determinism for RS
    if rounding_mode == RoundingMode.RS:
        torch.manual_seed(42)
        r_a = _f4_quantize(x.float(), rounding_mode, use_triton)
        torch.manual_seed(42)
        r_b = _f4_quantize(x.float(), rounding_mode, use_triton)
        torch.testing.assert_close(r_a, r_b, atol=0, rtol=0)
