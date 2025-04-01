import pytest
import torch

from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName
from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
from torchao.float8.float8_tensor import LinearMMConfig
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_grouped_gemm_2d_3d():
    float8_recipe_name = Float8LinearRecipeName.ROWWISE
    out_dtype = torch.bfloat16
    device = "cuda"
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(
        m * n_groups,
        k,
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )
    b = torch.randn(
        n_groups,
        n,
        k,
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # Compute output.
    out = _grouped_scaled_mm(
        a,
        b.transpose(-2, -1),
        offs=offs,
        float8_recipe=float8_recipe_name,
        out_dtype=out_dtype,
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
        float8_recipe_name,
        offs,
    )
    assert torch.equal(out, ref_out)

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
    B_t_fp8 = to_fp8_saturated(
        B_t_scaled, float8_config.cast_config_weight.target_dtype
    )

    # Split A and result into chunks, one for each group.
    offs_cpu = offs.cpu()
    A_list, A_list_fp8, A_scale_list, result_list = [], [], [], []
    start = 0
    for i in range(n_groups):
        A_list.append(A[start : offs_cpu[i]])
        A_list_fp8.append(A_fp8[start : offs_cpu[i]])
        A_scale_list.append(A_scales[start : offs_cpu[i]])
        result_list.append(result[start : offs_cpu[i]])
        start = offs_cpu[i]

    # Validate each actual result group from the _scaled_grouped_mm is equal to:
    # 1. A manual _scaled_mm for the group.
    # 2. A matmul_with_hp_or_float8_args for the group (which is differentiable, and thus used to validate gradients).
    outputs = []
    list1 = list(zip(A_list_fp8, B_t_fp8, A_scale_list, B_t_scales, result_list))
    list2 = list(zip(A_list, B_t, result_list))
    for i in range(len(list1)):
        a1, b1, a1scale, b1scale, result1 = list1[i]
        ref_group_result1 = torch._scaled_mm(
            a1,
            b1,
            a1scale.reciprocal(),
            b1scale.reciprocal(),
            out_dtype=torch.bfloat16,
            bias=None,
            use_fast_accum=True,
        )
        a2, b2, result2 = list2[i]
        ref_group_result2 = matmul_with_hp_or_float8_args.apply(
            a2,
            b2,
            LinearMMConfig(),
            float8_config,
        )
        assert torch.equal(result1, ref_group_result1)
        assert torch.equal(result2, ref_group_result2)
        outputs.append(ref_group_result2)

    # Concatenate the outputs and verify the full result is correct.
    output_ref = torch.cat(outputs, dim=0)
    return output_ref
