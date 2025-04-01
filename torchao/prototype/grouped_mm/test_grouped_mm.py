from re import I
from typing import List, Tuple
import pytest
import torch
from torch import nn

from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName, ScalingGranularity
from torchao.float8.float8_scaling_utils import (
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig
from torchao.prototype.grouped_mm import _grouped_scaled_mm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensorwise_scaling_not_supported():
    device = "cuda"
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(m * n_groups, k, device=device)[:, :k]
    b = torch.randn(n_groups, n, k, device=device)[::1, :, :k]
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError):
        _grouped_scaled_mm(
            a,
            b.transpose(-2, -1),
            offs=offs,
            float8_recipe=Float8LinearRecipeName.TENSORWISE,
            out_dtype=torch.bfloat16,
        )

# NOTE: this unit test is based on the pytorch core unit tests here:
# https://github.com/pytorch/pytorch/blob/6eb3c2e2822c50d8a87b43938a9cf7ef0561ede2/test/test_matmul_cuda.py#L1204
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_fast_accum", [True, False])
@pytest.mark.parametrize("strided", [True, False])
def test_grouped_gemm_2d_3d(use_fast_accum, strided):
    float8_recipe_name = Float8LinearRecipeName.ROWWISE
    out_dtype = torch.bfloat16
    device = "cuda"
    s_int = int(strided)
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(
        m * n_groups,
        k * (1 + s_int),
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )[:, :k]
    b = torch.randn(
        n_groups * (1 + s_int),
        n,
        k * (1 + s_int),
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )[:: (1 + s_int), :, :k]
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # Compute output.
    out = _grouped_scaled_mm(
        a,
        b.transpose(-2, -1),
        offs=offs,
        float8_recipe=float8_recipe_name,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )

    # Validate result.
    ref_a = a.detach().clone().requires_grad_(True)
    ref_b = b.detach().clone().requires_grad_(True)
    ref_out = compute_reference_forward(
        out,
        ref_a,
        ref_b,
        n_groups,
        out_dtype,
        use_fast_accum,
        float8_recipe_name,
        offs,
    )

    # Run backward pass.
    out.sum().backward()
    ref_out.sum().backward()

    # Validate gradients.
    assert torch.equal(a.grad, ref_a.grad)
    assert torch.equal(b.grad, ref_b.grad)


def compute_reference_forward(
    result: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    n_groups: int,
    out_dtype: torch.dtype,
    use_fast_accum: bool,
    float8_recipe_name: Float8LinearRecipeName,
    offs: torch.Tensor,
):
    assert result.dtype == out_dtype

    # Convert A to fp8.
    float8_config = Float8LinearConfig.from_recipe_name(float8_recipe_name)
    A_scales = tensor_to_scale(
        A,
        float8_config.cast_config_input.target_dtype, 
        scaling_granularity=float8_config.cast_config_input.scaling_granularity, 
        axiswise_dim=-1,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    A_scaled = A.to(torch.float32) * A_scales
    A_fp8 = to_fp8_saturated(A_scaled, float8_config.cast_config_input.target_dtype)

    # Convert B^t to fp8.
    B_t = B.transpose(-2, -1)
    B_t_scales = tensor_to_scale(
        B_t,
        float8_config.cast_config_weight.target_dtype, 
        scaling_granularity=float8_config.cast_config_input.scaling_granularity, 
        axiswise_dim=-2,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    B_t_scaled = B_t.to(torch.float32) * B_t_scales
    B_t_fp8 = to_fp8_saturated(B_t_scaled, float8_config.cast_config_weight.target_dtype)

    # Split A and result into chunks, one for each group.
    offs_cpu = offs.cpu()
    A_list, A_scale_list, result_list = [], [], []
    start = 0
    for i in range(n_groups):
        A_list.append(A_fp8[start : offs_cpu[i]])
        A_scale_list.append(A_scales[start : offs_cpu[i]])
        result_list.append(result[start : offs_cpu[i]])
        start = offs_cpu[i]

    # Validate result of each part of the grouped mm is equal to the separate scaled mm.
    outputs = []
    for a, b, ascale, bscale, group_result in zip(
        A_list, B_t_fp8, A_scale_list, B_t_scales, result_list
    ):
        ref_group_result = torch._scaled_mm(
            a,
            b,
            ascale,
            bscale,
            out_dtype=torch.bfloat16,
            use_fast_accum=use_fast_accum,
        )

        # Verify group result is accurate.
        assert torch.equal(group_result, ref_group_result)
        outputs.append(ref_group_result)

    # Concatenate the outputs and verify the full result is correct.
    output_ref = torch.cat(outputs, dim=0)
    assert torch.equal(result, output_ref)
    return output_ref
