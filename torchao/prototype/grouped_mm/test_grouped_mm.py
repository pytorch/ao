from typing import List
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
    validate_grouped_mm(
        out,
        a,
        b,
        n_groups,
        out_dtype,
        use_fast_accum,
        float8_recipe_name,
        offs,
    )

    # Run backward pass.
    out.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradients():
    # define a fake W1 for a MoE layer
    device = "cuda"
    m, k, n, n_groups = 16, 16, 32, 4

    x = torch.randn(m * n_groups, k, device=device, dtype=torch.bfloat16)
    params = nn.Parameter(torch.randn(n_groups, k, n, device=device, dtype=torch.bfloat16))
    offs = torch.arange(m, m * n_groups + 1, m, device="cuda", dtype=torch.int32)

    # clone the inputs and params for computing the reference gradients
    ref_x, ref_params = x.clone(), params.clone()

    # compute the output
    out = _grouped_scaled_mm(
        x,
        params,
        offs=offs,
        float8_recipe=Float8LinearRecipeName.ROWWISE,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )

    # compute the gradients
    out.sum().backward()

    # check the gradients exist
    assert params.data.grad is not None

    # check the gradients are not all nan
    assert not params.data.grad.isnan().any()

    # compare with reference gradients
    ref_grad = compute_reference_gradients(ref_x, ref_params, offs)
    assert torch.allclose(params.grad, ref_grad, atol=1e-2, rtol=1e-2)


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

def compute_reference_gradients(x: torch.Tensor, params: torch.Tensor, offs: torch.Tensor):
    assert len(offs) == params.size(0), "len(offs) != params.size(0); expected same number of offs/groups as params"
    assert x.ndim == 2, "expected 2D input"
    assert params.ndim == 3, "expected 3D params"

    # convert params to column-major memory layout
    params = params.transpose(-2, -1).contiguous().transpose(-2, -1)

    # determine group sizes based on offsets
    group_sizes = offs_to_group_sizes(offs.tolist())
    
    # use group sizes to split x into groups
    x_groups = torch.split(x, group_sizes, dim=0)

    # compute a separate _scaled_mm for each group, with the weight (i.e., expert) it's assigned to
    outputs = []
    for group_idx, group in enumerate(x_groups):
        group_output = matmul_with_hp_or_float8_args.apply(
            group,
            params[group_idx],
            LinearMMConfig(),
            Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE),
        )
        outputs.append(group_output)
        
    # compute the param gradients and return them
    output = torch.cat(outputs, dim=0)
    assert output.shape[0] == x.shape[0] and output.shape[-1] == params.shape[-1], "invalid output shape"

    # perform backward pass
    output.sum().backward()
    assert params.data.grad is not None, "params don't have gradients"

    # return reference gradient
    ref_grad = params.data.grad
    return ref_grad, output


def offs_to_group_sizes(offs: List[int]) -> List[int]:
    group_sizes = []
    prev_off = 0
    for off in offs:
        group_sizes.append(off - prev_off)
        prev_off = off
    return group_sizes


def validate_grouped_mm(
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
        axiswise_dim=-1,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )

    B_t_fp8 = hp_tensor_to_float8_dynamic(
        B.transpose(-2, -1),
        float8_config.cast_config_input.target_dtype,
        linear_mm_config=LinearMMConfig(),
        gemm_input_role=GemmInputRole.WEIGHT,
        scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
        axiswise_dim=-2,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )

    # grouped_scaled_mm doesn't support empty dims
    scale_A = A_fp8._scale.squeeze()
    scale_B = B_t_fp8._scale.squeeze()

    A_list, B_list, A_scale_list, B_scale_list, result_list = [], [], [], [], []
    start = 0

    # Since A 2D, we need to split it into parts based on offs, so we can perform
    # separate _scaled_mm calls for each part.
    offs_cpu = offs.cpu()
    for i in range(n_groups):
        A_list.append(A_fp8._data[start : offs_cpu[i]])
        A_scale_list.append(scale_A[start : offs_cpu[i]])
        result_list.append(result[start : offs_cpu[i]])
        start = offs_cpu[i]

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
