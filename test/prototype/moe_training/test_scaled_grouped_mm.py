# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

# We need to skip before doing any imports which would use triton, since
# triton won't be available on CPU builds and torch < 2.5
if not (
    TORCH_VERSION_AT_LEAST_2_5
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


from torchao.float8.config import (
    Float8LinearConfig,
    Float8LinearRecipeName,
)
from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
from torchao.float8.float8_training_tensor import LinearMMConfig
from torchao.float8.float8_utils import compute_error, tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _scaled_grouped_mm,
    emulated_mxfp8_scaled_grouped_mm,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.testing.utils import skip_if_rocm


@skip_if_rocm("ROCm enablement in progress")
def test_valid_scaled_grouped_mm_2d_3d():
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
        dtype=torch.bfloat16,
    )
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # b must be transposed and in column major format.
    b_t = b.contiguous().transpose(-2, -1).requires_grad_(True)

    # Compute output.
    out = _scaled_grouped_mm(
        a,
        b_t,
        offs=offs,
        out_dtype=out_dtype,
    )

    # Validate result.
    ref_a = a.detach().clone().requires_grad_(True)
    ref_b_t = b_t.detach().clone().requires_grad_(True)
    ref_out = compute_reference_forward(
        out,
        ref_a,
        ref_b_t,
        n_groups,
        out_dtype,
        offs,
    )
    assert torch.equal(out, ref_out)

    # Run backward pass.
    out.sum().backward()
    ref_out.sum().backward()

    # Validate gradients.
    assert torch.equal(a.grad, ref_a.grad)
    assert torch.equal(b_t.grad, ref_b_t.grad)


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
    out_dtype = torch.bfloat16
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

    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # Compute output.
    with pytest.raises(AssertionError):
        _scaled_grouped_mm(
            a,
            b_t,
            offs=offs,
            out_dtype=out_dtype,
        )


def compute_reference_forward(
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
    A_fp8 = to_fp8_saturated(A_scaled, torch.float8_e4m3fn)

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
        torch.float8_e4m3fn,
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
        assert torch.equal(result1, ref_group_result1)
        assert torch.equal(result2, ref_group_result2)
        outputs.append(ref_group_result2)

    # Concatenate the outputs and verify the full result is correct.
    output_ref = torch.cat(outputs, dim=0)
    return output_ref


@pytest.mark.parametrize("M", (1024, 4096))
@pytest.mark.parametrize("K", (1024, 4096))
@pytest.mark.parametrize("N", (1024, 4096))
@pytest.mark.parametrize("num_experts", (1, 8, 16))
def test_emulate_mxfp8_grouped_gemm(M, K, N, num_experts):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_t = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)
    x_ref, w_t_ref, offs_ref = x.clone(), w_t.clone(), offs.clone()

    # Quantize inputs to mxpf8 for emulated mxfp8 scaled grouped mm
    block_size = 32
    x_scale, x_mx = to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

    # To cast B_t per-expert to mxfp8 across dim1, we transpose the experts, cast along dim -1, then untranspose.
    w_scale, w_mx = to_mx(
        w_t.transpose(-2, -1).contiguous(),
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
    )
    w_t_scale, w_t_mx = w_scale.transpose(-2, -1), w_mx.transpose(-2, -1)

    ref_out = torch._grouped_mm(x_ref, w_t_ref, offs=offs_ref, out_dtype=torch.bfloat16)
    out = emulated_mxfp8_scaled_grouped_mm(
        x_mx, x_scale, w_t_mx, w_t_scale, offs=offs, out_dtype=torch.bfloat16
    )

    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"
