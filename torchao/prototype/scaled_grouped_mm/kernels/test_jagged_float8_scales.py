# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.prototype.scaled_grouped_mm.kernels.jagged_float8_scales import (
    triton_fp8_col_major_jagged_colwise_scales,
    triton_fp8_row_major_jagged_rowwise_scales,
)
from torchao.prototype.scaled_grouped_mm.scaled_grouped_mm import (
    _to_2d_jagged_float8_tensor_colwise,
    _to_2d_jagged_float8_tensor_rowwise,
)
from torchao.prototype.scaled_grouped_mm.utils import _is_column_major


@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_row_major_with_jagged_rowwise_scales(round_scales_to_power_of_2: bool):
    # tests case where rowwise scales are computed for multiple distinct subtensors,
    # with end boundary of each group is determine by their end column indexes (offsets).
    device = "cuda"
    m, k, n_groups = 256, 256, 4
    x = torch.randn(m, k * n_groups, device=device)
    colwise_offs = torch.arange(k, k * n_groups + 1, k, device=device)

    # compute reference with torch impl
    ref_fp8_data, ref_scales = _to_2d_jagged_float8_tensor_rowwise(
        x,
        colwise_offs,
        target_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    kernel_fp8_data, kernel_scales = triton_fp8_row_major_jagged_rowwise_scales(
        x,
        colwise_offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert not _is_column_major(kernel_fp8_data), "fp8 data is not row major"


@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_column_major_with_jagged_colwise_scales(round_scales_to_power_of_2: bool):
    # tests case where colwise scales are computed for multiple distinct subtensors,
    # with end boundary of each group is determine by their end row indexes (offsets).
    device = "cuda"
    m, k, n_groups = 256, 256, 4
    x = torch.randn(m * n_groups, k, device=device).t().contiguous().t()
    rowwise_offs = torch.arange(m, m * n_groups + 1, m, device=device)

    # compute reference with torch impl
    ref_fp8_data, ref_scales = _to_2d_jagged_float8_tensor_colwise(
        x,
        rowwise_offs,
        target_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    kernel_fp8_data, kernel_scales = triton_fp8_col_major_jagged_colwise_scales(
        x,
        rowwise_offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    assert torch.eq(ref_fp8_data, kernel_fp8_data).all(), "fp8 data not equal"
    assert torch.eq(ref_scales, kernel_scales).all(), "scales not equal"
    assert _is_column_major(kernel_fp8_data), "fp8 data is not column major"
