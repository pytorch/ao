# NOTE: these unit tests are based on the pytorch core unit tests here:
# https://github.com/pytorch/pytorch/blob/6eb3c2e2822c50d8a87b43938a9cf7ef0561ede2/test/test_matmul_cuda.py#L1204
from typing import Optional

import pytest
import torch

from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName
from torchao.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig
from torchao.prototype.grouped_mm import grouped_mm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_fast_accum", [True, False])
@pytest.mark.parametrize("strided", [True, False])
def test_grouped_gemm_2d_3d(use_fast_accum, strided):
    float8_recipe_name = Float8LinearRecipeName.ROWWISE
    out_dtype = torch.bfloat16
    device = "cuda"
    s_int = int(strided)
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(m * n_groups, k * (1 + s_int), device=device)[:, :k]
    b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device)[
        :: (1 + s_int), :, :k
    ]
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
    result = grouped_mm(
        a,
        b,
        offs=offs,
        float8_recipe=float8_recipe_name,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )

    # Validate result.
    validate_grouped_mm(
        result,
        a,
        b,
        n_groups,
        out_dtype,
        use_fast_accum,
        float8_recipe_name,
        offs=offs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_fast_accum", [True, False])
@pytest.mark.parametrize("strided", [True, False])
def test_grouped_gemm_3d_3d(use_fast_accum, strided):
    float8_recipe_name = Float8LinearRecipeName.ROWWISE
    out_dtype = torch.bfloat16
    device = "cuda"
    s_int = int(strided)
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device)[
        :: (1 + s_int), :, :k
    ]
    b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device)[
        :: (1 + s_int), :, :k
    ]
    result = grouped_mm(
        a,
        b,
        float8_recipe=float8_recipe_name,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )

    # Validate result.
    validate_grouped_mm(
        result,
        a,
        b,
        n_groups,
        out_dtype,
        use_fast_accum,
        float8_recipe_name,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensorwise_scaling_not_supported():
    device = "cuda"
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(m * n_groups, k, device=device)[:, :k]
    b = torch.randn(n_groups, n, k, device=device)[::1, :, :k]
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError):
        result = grouped_mm(
            a,
            b,
            offs=offs,
            float8_recipe=Float8LinearRecipeName.TENSORWISE,
            out_dtype=torch.bfloat16,
        )


def validate_grouped_mm(
    result: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    n_groups: int,
    out_dtype: torch.dtype,
    use_fast_accum: bool,
    float8_recipe_name: Float8LinearRecipeName,
    offs: Optional[torch.Tensor] = None,
):
    assert isinstance(result, torch.Tensor)
    assert result.dtype == out_dtype

    # Validate output by comparing the partition of the grouped scaled mm output
    # with a corresponding individual scaled mm.
    # We need the scales used for A and B to do this, so convert to Float8Tensors here.
    float8_config = Float8LinearConfig.from_recipe_name(float8_recipe_name)
    A_fp8 = hp_tensor_to_float8_dynamic(
        A,
        float8_dtype=float8_config.cast_config_input.target_dtype,
        linear_mm_config=LinearMMConfig(),
        gemm_input_role=GemmInputRole.INPUT,
        scaling_granularity=float8_config.cast_config_input.scaling_granularity,
        axiswise_dim=get_maybe_axiswise_dim(
            -1, float8_config.cast_config_input.scaling_granularity
        ),
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )

    B_fp8 = hp_tensor_to_float8_dynamic(
        B,
        float8_config.cast_config_input.target_dtype,
        linear_mm_config=LinearMMConfig(),
        gemm_input_role=GemmInputRole.WEIGHT,
        scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
        axiswise_dim=get_maybe_axiswise_dim(
            -1, float8_config.cast_config_input.scaling_granularity
        ),
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    B_fp8_t = B_fp8.transpose(-2, -1)

    # grouped_scaled_mm doesn't support empty dims
    scale_A = A_fp8._scale.squeeze()
    scale_B = B_fp8_t._scale.squeeze()

    A_list, B_list, A_scale_list, B_scale_list, result_list = [], [], [], [], []
    start = 0

    if A.ndim == 2 and offs is not None:
        offs_cpu = offs.cpu()
        for i in range(n_groups):
            A_list.append(A_fp8._data[start : offs_cpu[i]])
            A_scale_list.append(scale_A[start : offs_cpu[i]])
            result_list.append(result[start : offs_cpu[i]])
            start = offs_cpu[i]
    else:
        A_list = A_fp8._data
        B_list = B_fp8_t._data

    A_scale_list = scale_A
    B_scale_list = scale_B

    # Validate result of each part of the grouped mm is equal to the corresponding indiviudal scaled mm.
    for a, b, ascale, bscale, result in zip(
        A_list, B_list, A_scale_list, B_scale_list, result_list
    ):
        result_ref = torch._scaled_mm(
            a,
            b.t(),
            ascale.view(-1, 1),
            bscale.view(1, -1),
            out_dtype=torch.bfloat16,
            use_fast_accum=use_fast_accum,
        )
        assert result == result_ref
