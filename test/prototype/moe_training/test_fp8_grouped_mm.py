# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_sm_version, torch_version_at_least

# We need to skip before doing any imports which would use triton, since
# triton won't be available on CPU builds and torch < 2.5
if not (
    torch_version_at_least("2.7.0")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.config import (
    Float8LinearConfig,
    Float8LinearRecipeName,
)
from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
from torchao.float8.float8_training_tensor import LinearMMConfig
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.config import (
    FP8GroupedMMConfig,
    FP8GroupedMMRecipe,
)
from torchao.prototype.moe_training.fp8_grouped_mm import (
    _to_fp8_rowwise_then_scaled_grouped_mm,
)
from torchao.testing.utils import skip_if_rocm
from torchao.utils import is_MI300, is_MI350, is_ROCM

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@pytest.mark.skipif(
    True,
    reason="Skipping FP8 rowwise test pending fix for https://github.com/pytorch/ao/issues/3957",
)
@pytest.mark.parametrize("m", [4096])
@pytest.mark.parametrize("n", [8192])
@pytest.mark.parametrize("k", [5120])
@pytest.mark.parametrize("n_groups", [1, 2, 4, 8])
def test_fp8_rowwise_scaled_grouped_mm(m, n, k, n_groups):
    if is_ROCM():
        if not (is_MI300() or is_MI350()):
            pytest.skip("FP8 rowwise test requires MI300 or MI350 on ROCm")
    else:
        if not is_sm_version(9, 0):
            pytest.skip("FP8 rowwise test requires SM 9.0 on CUDA")

    out_dtype = torch.bfloat16
    device = "cuda"
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
        dtype=torch.bfloat16,
    )
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # b must be transposed and in column major format.
    b_t = b.contiguous().transpose(-2, -1).requires_grad_(True)

    # Compute output.
    config = FP8GroupedMMConfig.from_recipe(FP8GroupedMMRecipe.FP8_ROWWISE)
    out = _to_fp8_rowwise_then_scaled_grouped_mm(
        a,
        b_t,
        offs=offs,
        out_dtype=config.out_dtype,
        float8_dtyep=config.float8_dtype,
    )

    # Validate result.
    ref_a = a.detach().clone().requires_grad_(True)
    ref_b_t = b_t.detach().clone().requires_grad_(True)
    ref_out = _compute_reference_forward(
        out,
        ref_a,
        ref_b_t,
        n_groups,
        out_dtype,
        offs,
    )

    # Run backward pass.
    out.sum().backward()
    ref_out.sum().backward()

    # Validate gradients.
    if is_ROCM():
        # ROCm: reference vs tested path use different backends:
        # - `torch._scaled_mm` uses hipBLASLt
        # - `_to_fp8_rowwise_then_scaled_grouped_mm` uses CK
        # Different backends can use different kernel implementations / accumulation order, so the
        # outputs can differ slightly and we need tolerance.
        # On MI300/MI325 we need rtol=atol=1e-2 for this FP8 test to pass.
        assert torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2)
        assert torch.allclose(a.grad, ref_a.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(b_t.grad, ref_b_t.grad, rtol=1e-2, atol=1e-2)
    else:
        assert torch.equal(out, ref_out)
        assert torch.equal(a.grad, ref_a.grad)
        assert torch.equal(b_t.grad, ref_b_t.grad)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("m", [16, 17])
@pytest.mark.parametrize("k", [16, 18])
@pytest.mark.parametrize("n", [32, 33])
def test_K_or_N_dim_not_multiple_of_16(m, n, k):
    # - Leading dim of A doesn't have to be divisible by 16, since it will be
    # divided up into groups based on offset anyway.
    # - Trailing dim of A must be divisible by 16.
    # - Leading dim of B (n_groups) doesn't need to be divisible by 16.
    # - Last 2 dims of B must be divisible by 16.
    if n % 16 == 0 and k % 16 == 0:
        return
    device = "cuda"
    n_groups = 4
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

    # b must be transposed and in column major format.
    b_t = b.transpose(-2, -1)
    b_t = b_t.transpose(-2, -1).contiguous().transpose(-2, -1)

    config = FP8GroupedMMConfig.from_recipe(FP8GroupedMMRecipe.FP8_ROWWISE)
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # Compute output.
    with pytest.raises(AssertionError):
        _to_fp8_rowwise_then_scaled_grouped_mm(
            a,
            b_t,
            offs=offs,
            out_dtype=config.out_dtype,
            float8_dtype=config.float8_dtype,
        )


def _compute_reference_forward(
    result: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    n_groups: int,
    out_dtype: torch.dtype,
    offs: torch.Tensor,
):
    assert result.dtype == out_dtype

    # Use official rowwise recipe as reference to ensure implementation is correct.
    float8_config = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE)

    # Convert A to fp8.
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
    B_t_scales = tensor_to_scale(
        B_t,
        float8_config.cast_config_weight.target_dtype,
        scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
        axiswise_dim=-2,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    B_t_scaled = B_t.to(torch.float32) * B_t_scales
    B_t_fp8 = to_fp8_saturated(
        B_t_scaled,
        float8_config.cast_config_input.target_dtype,
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

    # Validate each actual result group from the _to_fp8_rowwise_then_scaled_grouped_mm is equal to:
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
            out_dtype=out_dtype,
            bias=None,
            use_fast_accum=float8_config.gemm_config_output.use_fast_accum,
        )
        a2, b2, result2 = list2[i]
        ref_group_result2 = matmul_with_hp_or_float8_args.apply(
            a2,
            b2,
            LinearMMConfig(),
            float8_config,
        )
        if is_ROCM():
            assert torch.allclose(result1, ref_group_result1, rtol=1e-2, atol=1e-2)
            assert torch.allclose(result2, ref_group_result2, rtol=1e-2, atol=1e-2)
        else:
            assert torch.equal(result1, ref_group_result1)
            assert torch.equal(result2, ref_group_result2)
        outputs.append(ref_group_result2)

    # Concatenate the outputs and verify the full result is correct.
    output_ref = torch.cat(outputs, dim=0)
    return output_ref
