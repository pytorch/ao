# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A helper script to summarize the key numerical values of various floating
point formats relevant to the MX spec.
"""

import math
from typing import Tuple

import tabulate
import torch

from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
)
from torchao.prototype.mx_formats.kernels import get_bits

dtype_to_bitwidth = {
    torch.float: 32,
    torch.bfloat16: 16,
    torch.float16: 16,
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 8,
    DTYPE_FP6_E3M2: 6,
    DTYPE_FP6_E2M3: 6,
}
dtype_to_sem_len = {
    torch.float: (1, 8, 23),
    torch.bfloat16: (1, 8, 7),
    torch.float16: (1, 5, 10),
    torch.float8_e4m3fn: (1, 4, 3),
    torch.float8_e5m2: (1, 5, 2),
    # the line below is currently representing fp4 with bits 0:3 empty and
    # bits 4:7 containing the fp4 encoding
    # TODO(future): clean this up
    torch.uint8: (1, 2, 1),
}
# bias = 2 ** (exp_bitwidth - 1) - 1
dtype_to_exp_bias = {
    torch.float: 127,
    torch.bfloat16: 127,
    torch.float16: 15,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 15,
    DTYPE_FP6_E2M3: 1,
    DTYPE_FP6_E3M2: 3,
}
dtype_to_int_dtype = {
    torch.float: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float8_e4m3fn: torch.int8,
    torch.float8_e5m2: torch.int8,
    # for fp4
    # TODO(future): clean it up
    torch.uint8: torch.uint8,
}

# format:
# {
#   dtype: [
#     [
#       ref_f32_value, sign_encoding, exp_encoding, mantissa_encoding,
#       description,
#     ],
#     ...,
#   ],
#   ...,
# }
dtype_to_interesting_values = {
    torch.float: [
        # zero and neg zero
        (0.0, "0", "0" * 8, "0" * 23, "zero"),
        (-0.0, "1", "0" * 8, "0" * 23, "zero_neg"),
        # special values
        (float("nan"), "0", "1" * 8, "1" + "0" * 22, "nan"),
        (float("inf"), "0", "1" * 8, "0" * 23, "inf"),
        (float("-inf"), "1", "1" * 8, "0" * 23, "inf_neg"),
        # values below verified with from https://www.h-schmidt.net/FloatConverter/IEEE754.html  # noqa: E501
        # largest normal
        (
            3.402823466385288598117042e38,
            "0",
            "1" * 7 + "0",
            "1" * 23,
            "largest_norm",
        ),  # noqa: E501
        (
            -3.402823466385288598117042e38,
            "1",
            "1" * 7 + "0",
            "1" * 23,
            "largest_norm_neg",
        ),
        # smallest normal
        (
            1.175494350822287507968737e-38,
            "0",
            "0" * 7 + "1",
            "0" * 23,
            "smallest_norm",
        ),  # noqa: E501
        (
            -1.175494350822287507968737e-38,
            "1",
            "0" * 7 + "1",
            "0" * 23,
            "smallest_norm_neg",
        ),
        # largest denormal
        (
            1.175494210692441075487029e-38,
            "0",
            "0" * 8,
            "1" * 23,
            "largest_denorm",
        ),  # noqa: E501
        (
            -1.175494210692441075487029e-38,
            "1",
            "0" * 8,
            "1" * 23,
            "largest_denorm_neg",
        ),  # noqa: E501
        # smallest denormal
        (
            1.401298464324817070923730e-45,
            "0",
            "0" * 8,
            "0" * 22 + "1",
            "smallest_denorm",
        ),
        (
            -1.401298464324817070923730e-45,
            "1",
            "0" * 8,
            "0" * 22 + "1",
            "smallest_denorm_neg",
        ),
        # positive and negative value
        (30.0, "0", "10000011", "1" * 3 + "0" * 20, "random_pos"),
        (-24.0, "1", "10000011", "1" + "0" * 22, "random_neg"),
    ],
    torch.bfloat16: [
        # zero and neg zero
        (0.0, "0", "0" * 8, "0" * 7, "zero"),
        (-0.0, "1", "0" * 8, "0" * 7, "zero_neg"),
        # special values
        (float("nan"), "0", "1" * 8, "1" + "0" * 6, "nan"),
        (float("inf"), "0", "1" * 8, "0" * 7, "inf"),
        (float("-inf"), "1", "1" * 8, "0" * 7, "inf_neg"),
        # values below checked with TODO
        # largest normal
        (3.38953e38, "0", "1" * 7 + "0", "1" * 7, "largest_norm"),
        (-3.38953e38, "1", "1" * 7 + "0", "1" * 7, "largest_norm_neg"),
        # smallest normal
        (1.17549e-38, "0", "0" * 7 + "1", "0" * 7, "smallest_norm"),
        (-1.17549e-38, "1", "0" * 7 + "1", "0" * 7, "smallest_norm_neg"),
        # largest denormal
        (1.16631e-38, "0", "0" * 8, "1" * 7, "largest_denorm"),
        (-1.16631e-38, "1", "0" * 8, "1" * 7, "largest_denorm_neg"),
        # smallest denormal
        (9.18355e-41, "0", "0" * 8, "0" * 6 + "1", "smallest_denorm"),
        (-9.18355e-41, "1", "0" * 8, "0" * 6 + "1", "smallest_denorm_neg"),
        # positive and negative value
        (30.0, "0", "10000011", "1" * 3 + "0" * 4, "random_pos"),
        (-24.0, "1", "10000011", "1" + "0" * 6, "random_neg"),
    ],
    torch.float16: [
        # zero and neg zero
        (0.0, "0", "0" * 5, "0" * 10, "zero"),
        (-0.0, "1", "0" * 5, "0" * 10, "zero_neg"),
        # special values
        (float("nan"), "0", "1" * 5, "1" + "0" * 9, "nan"),
        (float("inf"), "0", "1" * 5, "0" * 10, "inf"),
        (float("-inf"), "1", "1" * 5, "0" * 10, "inf_neg"),
        # values below checked with https://en.wikipedia.org/wiki/Half-precision_floating-point_format  # noqa: E501
        # largest normal
        (65504, "0", "1" * 4 + "0", "1" * 10, "largest_normal"),
        (-65504, "1", "1" * 4 + "0", "1" * 10, "largest_normal_neg"),
        # smallest normal
        (0.00006103515625, "0", "0" * 4 + "1", "0" * 10, "smallest_normal"),
        (
            -0.00006103515625,
            "1",
            "0" * 4 + "1",
            "0" * 10,
            "smallest_normal_neg",
        ),  # noqa: E501
        # largest denormal
        (0.000060975552, "0", "0" * 5, "1" * 10, "largest_denorm"),
        (-0.000060975552, "1", "0" * 5, "1" * 10, "largest_denorm_neg"),
        # smallest denormal
        (0.000000059604645, "0", "0" * 5, "0" * 9 + "1", "smallest_denorm"),
        (
            -0.000000059604645,
            "1",
            "0" * 5,
            "0" * 9 + "1",
            "smallest_denorm_neg",
        ),  # noqa: E501
        # positive and negative value
        (30.0, "0", "10011", "1" * 3 + "0" * 7, "random_pos"),
        (-24.0, "1", "10011", "1" + "0" * 9, "random_neg"),
    ],
    torch.float8_e4m3fn: [
        # zero and neg zero
        (0.0, "0", "0000", "000", "zero"),
        (-0.0, "1", "0000", "000", "zero_neg"),
        # special values
        # note: no pos or neg inf
        (float("nan"), "0", "1111", "111", "nan"),
        # values below checked with https://arxiv.org/pdf/2209.05433.pdf, Table 1  # noqa: E501
        # largest normal
        (448.0, "0", "1111", "110", "largest_normal"),
        (-448.0, "1", "1111", "110", "largest_normal_neg"),
        # smallest normal
        (2**-6, "0", "0001", "000", "smallest_normal"),
        (-(2**-6), "1", "0001", "000", "smallest_normal_neg"),
        # largest denormal
        (0.875 * 2**-6, "0", "0000", "111", "largest_denormal"),
        (-0.875 * 2**-6, "1", "0000", "111", "largest_denormal_neg"),
        # smallest denormal
        (2**-9, "0", "0000", "001", "smallest_denormal"),
        (-(2**-9), "1", "0000", "001", "smallest_denormal_neg"),
        # positive and negative value
        (30.0, "0", "1011", "111", "random_pos"),
        (-24.0, "1", "1011", "100", "random_neg"),
    ],
    torch.float8_e5m2: [
        # zero and neg zero
        (0.0, "0", "00000", "00", "zero"),
        (-0.0, "1", "00000", "00", "zero_neg"),
        # special values
        (float("nan"), "0", "11111", "11", "nan"),
        (float("inf"), "0", "11111", "00", "inf"),
        (float("-inf"), "1", "11111", "00", "inf_neg"),
        # values below checked with https://arxiv.org/pdf/2209.05433.pdf, Table 1  # noqa: E501
        # largest normal
        (57344.0, "0", "11110", "11", "largest_normal"),
        (-57344.0, "1", "11110", "11", "largest_normal_neg"),
        # smallest normal
        (2**-14, "0", "00001", "00", "smallest_normal"),
        (-(2**-14), "1", "00001", "00", "smallest_normal_neg"),
        # largest denormal
        (0.75 * 2**-14, "0", "00000", "11", "largest_denormal"),
        (-0.75 * 2**-14, "1", "00000", "11", "largest_denormal_neg"),
        # smallest denormal
        (2**-16, "0", "00000", "01", "smallest_denormal"),
        (-(2**-16), "1", "00000", "01", "smallest_denormal_neg"),
        # positive and negative value
        (32.0, "0", "10100", "00", "random_pos"),
        (-24.0, "1", "10011", "10", "random_neg"),
    ],
}

# values for fp4_e2m1, as defined in the OCP spec for MXFP4
# other than the sign, there are only 8 values, so just create
# the table by hand
# formula norm: sign * (2 ** (exp - 1)) * 1.x
# formula denorm: sign * (2 ** (exp - 1 + 1)) * 0.x
# format: val, formula, s, e, m, val, note
float4_e2m1_interesting_values = [
    (0, "1.0 * 2^0 * 0.0", "0", "00", "0", "zero"),
    # same as largest denormal, there is only one
    (
        0.5,
        "1.0 * 2^0 * 0.5",
        "0",
        "00",
        "1",
        "smallest_denormal",
    ),  # 2**0 * 0.5  # noqa: E501
    (1.0, "1.0 * 2^0 * 1.0", "0", "01", "0", "smallest_normal"),  # 2**0 * 1.0
    (1.5, "1.0 * 2^0 * 1.5", "0", "01", "1", "val3"),  # 2**0 * 1.5
    (2.0, "1.0 * 2^1 * 1.0", "0", "10", "0", "val4"),  # 2**1 * 1.0
    (3.0, "1.0 * 2^1 * 1.5", "0", "10", "1", "val5"),  # 2**1 * 1.5
    (4.0, "1.0 * 2^2 * 1.0", "0", "11", "0", "val6"),  # 2**2 * 1.0
    (6.0, "1.0 * 2^2 * 1.5", "0", "11", "1", "largest_normal"),  # 2**2 * 1.5
]
float4_e2m1_neg = []
for fp32_ref, formula, _s, e, m, label in float4_e2m1_interesting_values:
    float4_e2m1_neg.append([-1 * fp32_ref, "-" + formula, "1", e, m, label + "_neg"])  # noqa: E501
float4_e2m1_interesting_values.extend(float4_e2m1_neg)
del float4_e2m1_neg

# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf, section 5.3.2  # noqa: E501
float6_e3m2_interesting_values = [
    (0, "1.0 * 2^-2 * 0.0", "0", "000", "00", "zero"),
    (0.0625, "1.0 * 2^-2 * 0.25", "0", "000", "01", "smallest_denormal"),
    (0.1875, "1.0 * 2^-2 * 0.75", "0", "000", "11", "largest_denormal"),
    (0.25, "1.0 * 2^-2 * 1.0", "0", "001", "00", "smallest_normal"),
    (28.0, "1.0 * 2^4 * 1.75", "0", "111", "11", "largest_normal"),
]
float6_e3m2_neg = []
for fp32_ref, formula, _s, e, m, label in float6_e3m2_interesting_values:
    float6_e3m2_neg.append([-1 * fp32_ref, "-" + formula, "1", e, m, label + "_neg"])  # noqa: E501
float6_e3m2_interesting_values.extend(float6_e3m2_neg)
del float6_e3m2_neg

# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf, section 5.3.2  # noqa: E501
float6_e2m3_interesting_values = [
    (0, "1.0 * 2^0 * 0.0", "0", "00", "000", "zero"),
    (0.125, "1.0 * 2^0 * 0.125", "0", "00", "001", "smallest_denormal"),
    (0.875, "1.0 * 2^0 * 0.875", "0", "00", "111", "largest_denormal"),
    (1.0, "1.0 * 2^0 * 1.0", "0", "01", "000", "smallest_normal"),
    (7.5, "1.0 * 2^2 * 1.875", "0", "11", "111", "largest_normal"),
]
float6_e2m3_neg = []
for fp32_ref, formula, _s, e, m, label in float6_e2m3_interesting_values:
    float6_e2m3_neg.append(
        [
            -1 * fp32_ref,
            "-" + formula,
            "1",
            e,
            m,
            label + "_neg",
        ]
    )
float6_e2m3_interesting_values.extend(float6_e2m3_neg)
del float6_e2m3_neg


def _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype):
    # test going from float to encoding
    x = torch.tensor(fp_ref, dtype=dtype)
    bitwidth = dtype_to_bitwidth[dtype]
    s_enc, e_enc, m_enc = get_sem_bits(x, bitwidth=bitwidth)
    assert s_enc_ref == s_enc
    assert e_enc_ref == e_enc, f"{e_enc_ref} != {e_enc}"
    assert m_enc_ref == m_enc, f"{m_enc_ref} != {m_enc}"

    # test going from encoding to float
    s_i, e_i, m_f, special_value = sem_bits_to_sem_vals(
        s_enc,
        e_enc,
        m_enc,
        dtype,
    )
    fp = sem_vals_to_f32(s_i, e_i, m_f, special_value)
    assert_same(fp_ref, fp)


def get_sem_bits(x: torch.Tensor, bitwidth: int) -> Tuple[str, str, str]:
    """
    Input: a tensor with a single element of the target element dtype
    - for PT core dtypes, that dtype (fp32, fp16, fp8_e4m3, etc)
    - for fp4_e2m1, fp6_e3m2, fp6_e2m3, not supported in this function
    Output: bit strings for sign, exponent, mantissa encodings of the input
    """
    assert x.numel() == 1
    s_len, e_len, m_len = dtype_to_sem_len[x.dtype]

    new_dtype = dtype_to_int_dtype[x.dtype]
    x = x.view(new_dtype)
    np_res = get_bits(x)
    if bitwidth == 4:
        # TODO(future): clean up this fp4 codepath
        offset = 4
        s, e, m = (
            np_res[offset],
            np_res[offset + s_len : (offset + s_len + e_len)],  # noqa: E203
            np_res[(offset + s_len + e_len) :],  # noqa: E203
        )
    else:
        s, e, m = (
            np_res[0],
            np_res[s_len : (s_len + e_len)],  # noqa: E203
            np_res[(s_len + e_len) :],  # noqa: E203
        )
    assert len(s) == s_len
    assert len(e) == e_len
    assert len(m) == m_len
    return s, e, m


def exp_encoding_to_exp(exp_bit_str: str, dtype):
    """
    Input: bit string of exponent for dtype
    Output: integer representation of exponent
    """
    exp_biased = int(exp_bit_str, 2)
    exp_bias = dtype_to_exp_bias[dtype]
    exp_unbiased = exp_biased - exp_bias

    # for denormalized values, increment exponent back
    # up by one
    if all(b == "0" for b in exp_bit_str):
        exp_unbiased += 1

    return exp_unbiased


def sem_bits_to_sem_vals(s_enc, e_enc, m_enc, dtype):
    """
    Input: encodings of sign, exponent, mantissa for dtype
    Output: integer sign, integer exponent, float32 mantissa, special value

    Supported dtypes: PT core dtypes and fp6_e3m2 and fp6_e2m3
    Not supported dtypes: fp4

    If special value is filled out, sem are none
    If sem are filled out, special value is none
    """
    sign = 1 if s_enc == "0" else -1

    # handle special values
    if all(bit == "1" for bit in e_enc):
        dtypes = (
            torch.float32,
            torch.bfloat16,
            torch.float16,
            torch.float8_e5m2,
        )
        if dtype in dtypes:
            if all(bit == "0" for bit in m_enc):
                if s_enc == "0":
                    return None, None, None, float("inf")
                else:
                    return None, None, None, float("-inf")
            else:
                return None, None, None, float("nan")
        elif dtype in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2):
            # no special values in f6 dtypes
            pass
        else:
            assert dtype is torch.float8_e4m3fn
            # 1. float8_e4m3fn does not have infinity
            # 2. float8_e4m3fn only sets {s}.{1111}.{111} for nan
            if all(b == "1" for b in e_enc + m_enc):
                return None, None, None, float("nan")

    exponent = exp_encoding_to_exp(e_enc, dtype)

    is_zero = all(b == "0" for b in e_enc + m_enc)
    is_denormal = (not is_zero) and all(b == "0" for b in e_enc)
    is_normal = not is_zero and not is_denormal

    if is_zero:
        return sign, exponent, 0.0, None

    mantissa = 1.0 if is_normal else 0.0
    cur_pow_2 = -1
    for m_bit in m_enc:
        mantissa += int(m_bit) * pow(2, cur_pow_2)
        cur_pow_2 -= 1
    return sign, exponent, mantissa, None


def sem_vals_to_f32(s_i, e_i, m_f, special_value):
    """
    Input: integer sign, integer exponent, float32 mantissa, special value
    Output: float32 value
    """
    if special_value is not None:
        return special_value
    f = s_i * pow(2, e_i) * m_f
    return f


def sem_vals_to_formula(s_i, e_i, m_f, special_value):
    """
    Input: integer sign, integer exponent, float32 mantissa, special value
    Output: formula to get the float32 value
    """
    if special_value is not None:
        return special_value
    return f"{s_i} * 2^{e_i} * {m_f}"


def assert_same(fp1, fp2):
    if math.isnan(fp1):
        assert math.isnan(fp2)
    elif math.isinf(fp1):
        if fp1 > 0:
            assert math.isinf(fp2) and fp2 > 0
        else:
            assert math.isinf(fp2) and fp2 < 0
    else:
        assert (abs(fp2 - fp1) / (fp1 + 1e-20)) - 1 < 1e-12, f"{fp2} != {fp1}"


def run(dtype):
    print("dtype", dtype)

    headers = ["orig_val", "formula", "s_enc", "e_enc", "m_enc", "note"]
    results = []

    if dtype == torch.float4_e2m1fn_x2:
        results = float4_e2m1_interesting_values
    elif dtype == DTYPE_FP6_E3M2:
        results = float6_e3m2_interesting_values
    elif dtype == DTYPE_FP6_E2M3:
        results = float6_e2m3_interesting_values
    else:
        interesting_values = dtype_to_interesting_values[dtype]
        for row in interesting_values:
            fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, notes = row

            # test that things still work
            _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)

            # create the formula
            s_i, e_i, m_f, special_value = sem_bits_to_sem_vals(
                s_enc_ref, e_enc_ref, m_enc_ref, dtype
            )
            formula = sem_vals_to_formula(s_i, e_i, m_f, special_value)

            # create the table row
            results.append(
                [
                    fp_ref,
                    formula,
                    s_enc_ref,
                    e_enc_ref,
                    m_enc_ref,
                    notes,
                ]
            )

    print(tabulate.tabulate(results, headers=headers))
    print("\n")


if __name__ == "__main__":
    for dtype in (
        torch.float,
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        DTYPE_FP6_E3M2,
        DTYPE_FP6_E2M3,
        torch.float4_e2m1fn_x2,
    ):
        run(dtype)
