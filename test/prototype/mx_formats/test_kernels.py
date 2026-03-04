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
        x_ptr, out_ptr, N, seed, ROUNDING_MODE: tl.constexpr,
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
            rbits = tl.randint(seed, out_offs)
            x_fp4x2 = convert_fp32_to_fp4_packed_rs(x_pairs, rbits)
        out_offs = pid * 32 + tl.arange(0, 32)
        tl.store(out_ptr + out_offs, x_fp4x2, mask=out_offs < N // 2)

    def triton_f4_pack(x, rounding_mode=RoundingMode.RN, seed=0):
        """Pack FP32 values to FP4 using Triton convert_fp32_to_fp4_packed{,_rs}."""
        N = x.numel()
        out = torch.empty(N // 2, dtype=torch.uint8, device=x.device)
        grid = (triton.cdiv(N, 64),)
        _triton_f4_pack_kernel[grid](
            x, out, N, seed, ROUNDING_MODE=rounding_mode.value,
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


def _f4_quantize(x, rounding_mode, use_triton, seed=None):
    """Quantize FP32 to FP4 and dequantize, using either PyTorch or Triton kernel."""
    if rounding_mode not in RoundingMode:
        raise ValueError(
            f"Unknown rounding_mode: {rounding_mode}. "
            f"Expected RoundingMode.RN or RoundingMode.RS."
        )
    if seed is not None and not use_triton:
        torch.manual_seed(seed)
    if use_triton:
        if seed is None:
            seed = torch.randint(2**31, (1,)).item()
        xq = triton_f4_pack(x.flatten(), rounding_mode=rounding_mode, seed=seed)
        return f4_unpacked_to_f32(unpack_uint4(xq))
    else:
        return f4_unpacked_to_f32(f32_to_f4_unpacked(x, rounding_mode=rounding_mode))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_triton", _triton_kernel_params)
@pytest.mark.parametrize("rounding_mode", [RoundingMode.RN, RoundingMode.RS, 99])
@pytest.mark.parametrize("seed_a,seed_b", [(42, 42), (42, 123)])
@pytest.mark.parametrize("shape", [(1024, 128)])
@pytest.mark.parametrize("value,rn_expected", FP4_RN_EXPECTED)
def test_f4_rounding(value, rn_expected, shape, seed_a, seed_b, rounding_mode, use_triton):
    """Test FP4 rounding: RN is biased, RS is unbiased, RS respects seed, invalid raises."""
    x = torch.ones(*shape, device="cuda", dtype=torch.bfloat16) * value

    if rounding_mode not in RoundingMode:
        with pytest.raises(ValueError, match="Unknown rounding_mode"):
            _f4_quantize(x.float(), rounding_mode, use_triton, seed=seed_a)
        return

    rtol = 1e-2
    r1 = _f4_quantize(x.float(), rounding_mode, use_triton, seed=seed_a)
    r2 = _f4_quantize(x.float(), rounding_mode, use_triton, seed=seed_b)

    # Check rounding behavior via mean
    r1_mean = torch.mean(r1)
    if rounding_mode == RoundingMode.RN:
        torch.testing.assert_close(r1_mean.item(), rn_expected, rtol=rtol, atol=rtol)
    else:
        input_mean = torch.mean(x.float())
        torch.testing.assert_close(r1_mean, input_mean, rtol=rtol, atol=rtol)

    # Check seed determinism
    if seed_a == seed_b:
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)
    elif rounding_mode == RoundingMode.RS:
        assert not torch.allclose(r1, r2)

