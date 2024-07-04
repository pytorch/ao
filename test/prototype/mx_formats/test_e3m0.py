# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from torchao.prototype.mx_formats.constants import (
    F32_MIN_NORMAL,
    F4_E3M0_EXP_BIAS,
    F4_E3M0_MAX,
    F4_E3M0_MIN_NORMAL,
)
from torchao.prototype.mx_formats.custom_cast import (
    EBITS_F4_E3M0,
    f32_to_f4_e3m0_unpacked,
    MBITS_F4_E3M0,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_mx


torch.manual_seed(2)


@pytest.mark.parametrize("hp_dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("use_stochastic_rounding", [False, True])
def test_overflow_cast(hp_dtype, device, sign, use_stochastic_rounding):
    data_min = sign * F4_E3M0_MAX
    data_max = sign * F4_E3M0_MAX * F4_E3M0_MAX
    data = (
        torch.rand(1024, 1024, dtype=hp_dtype, device=device) * (data_max - data_min)
        + data_min
    )

    data_lp = f32_to_f4_e3m0_unpacked(data, use_stochastic_rounding)
    if sign == 1:
        target_lp = torch.full_like(data, 2**EBITS_F4_E3M0 - 1, dtype=torch.uint8)
    else:
        target_lp = torch.full_like(
            data, 2 ** (EBITS_F4_E3M0 + 1) - 1, dtype=torch.uint8
        )

    torch.testing.assert_close(
        data_lp,
        target_lp,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("hp_dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_underflow_cast(hp_dtype, device):
    data_min = -F4_E3M0_MIN_NORMAL
    data_max = F4_E3M0_MIN_NORMAL
    data = (
        torch.rand(1024, 1024, dtype=hp_dtype, device=device) * (data_max - data_min)
        + data_min
    )

    data_lp = f32_to_f4_e3m0_unpacked(data, use_stochastic_rounding=False)
    target_lp = torch.where((data >= 0) & (data <= F4_E3M0_MIN_NORMAL / 2), 0, 1).to(
        torch.uint8
    )
    target_lp = torch.where(
        data < -F4_E3M0_MIN_NORMAL / 2, 1 + 2**EBITS_F4_E3M0, data_lp
    )

    torch.testing.assert_close(
        data_lp,
        target_lp,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("hp_dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_underflow_cast_use_stochastic_rounding(hp_dtype, device):
    data_min = -F4_E3M0_MIN_NORMAL
    data_max = F4_E3M0_MIN_NORMAL
    data = (
        torch.rand(1024, 1024, dtype=hp_dtype, device=device) * (data_max - data_min)
        + data_min
    )

    data_lp = f32_to_f4_e3m0_unpacked(data, use_stochastic_rounding=True)
    target_lp = torch.where((data >= 0) & (data <= F4_E3M0_MIN_NORMAL / 2), 0, 1).to(
        torch.uint8
    )
    target_lp = torch.where(
        data < -F4_E3M0_MIN_NORMAL / 2, 1 + 2**EBITS_F4_E3M0, data_lp
    )


    torch.testing.assert_close(
        data_lp,
        target_lp,
        atol=1,
        rtol=0,
    )

    zeros_in_data_lp = (data_lp == 0).sum().item()
    zeros_in_target_lp = (target_lp == 0).sum().item()

    assert (
        zeros_in_data_lp >= zeros_in_target_lp
    ), f"stochastic rounding should have more non-zero values {zeros_in_data_lp} >= {zeros_in_target_lp}"


@pytest.mark.parametrize("exp_range", list(range(-2, 4)))
@pytest.mark.parametrize("hp_dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("use_stochastic_rounding", [False, True])
def test_normal_cast(exp_range, hp_dtype, device, sign, use_stochastic_rounding):
    if sign == 1:
        data_min = pow(2, exp_range)
        data_max = pow(2, exp_range + 1)
    else:
        data_min = - pow(2, exp_range + 1)
        data_max = - pow(2, exp_range)
    
    data = (
        torch.rand(1024, 1024, dtype=hp_dtype, device=device) * (data_max - data_min)
        + data_min
    )

    data_lp = f32_to_f4_e3m0_unpacked(data, use_stochastic_rounding).to(torch.float32)
    if sign == 1:
        data_lp = torch.pow(2, data_lp - F4_E3M0_EXP_BIAS)
    else:
        data_lp = -torch.pow(2, data_lp - F4_E3M0_EXP_BIAS - 8)

    torch.testing.assert_close(
        data_lp,
        data,
        atol=data_max - data_min,
        rtol=0,
    )


@pytest.mark.parametrize("data_range", [1, 0.75, 0.5, 0.25, 0.125])
@pytest.mark.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("use_stochastic_rounding", [False, True])
def test_mx_qdq(data_range, hp_dtype, block_size, device, use_stochastic_rounding):
    data_min = -data_range
    data_max = data_range
    data = (
        torch.rand(1024, 1024, dtype=hp_dtype, device=device) * (data_max - data_min)
        + data_min
    )
    scale_e8m0_biased, data_lp = to_mx(
        data, "fp4_e3m0", block_size, use_stochastic_rounding
    )
    mx_args = MXTensor(scale_e8m0_biased, data_lp, "fp4_e3m0", block_size, data.dtype)
    data_qdq = mx_args.to_dtype(mx_args._orig_dtype)

    scale_e8m0_unbiased = scale_e8m0_biased - 127
    scale_fp = torch.pow(
        torch.full(scale_e8m0_unbiased.size(), 2.0, device=data.device),
        scale_e8m0_unbiased,
    )
    scale_fp = torch.clamp(scale_fp, min=F32_MIN_NORMAL)

    data_lp = data.reshape(-1, block_size) / scale_fp.unsqueeze(1)
    data_lp = data_lp.reshape(data.shape)

    # exclude overflow values whose error is unbounded
    saturate_mask = data_lp >= F4_E3M0_MAX
    data_qdq = torch.where(saturate_mask, data, data_qdq)

    # the largest error equals to max_scale_value * max_exp_range
    max_scale_value = torch.max(scale_fp)
    largest_error = max_scale_value * (2**4 - 2**3)

    torch.testing.assert_close(
        data_qdq,
        data,
        atol=largest_error,
        rtol=0,
    )
