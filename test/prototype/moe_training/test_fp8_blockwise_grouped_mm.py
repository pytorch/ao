# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from torchao.utils import is_MI300, is_MI350, is_sm_at_least_90

if not (
    torch.cuda.is_available() and (is_sm_at_least_90() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8_training.grouped_kernels import (
    triton_fp8_blockwise_weight_quant_grouped_rhs,
    triton_fp8_blockwise_weight_quant_grouped_transposed_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
)
from torchao.prototype.moe_training.blockwise_fp8_grouped_mm import (
    _to_fp8_blockwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.config import Float8BlockwiseTrainingOpConfig
from torchao.prototype.moe_training.tensor import Float8TrainingWeightWrapperTensor
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error
from torchao.testing.utils import skip_if_rocm

torch._dynamo.config.cache_size_limit = 1000


def _make_column_major_weight_t(E: int, N: int, K: int) -> torch.Tensor:
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    return weight.contiguous().transpose(-2, -1)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("E,K,N", [(1, 128, 128), (3, 256, 384)])
def test_grouped_weight_quant_transposed_rhs_matches_dense(E, K, N):
    weight_t = _make_column_major_weight_t(E, N, K)
    q_grouped, scale_grouped = triton_fp8_blockwise_weight_quant_grouped_transposed_rhs(
        weight_t
    )

    assert q_grouped.shape == (E, K, N)
    assert scale_grouped.shape == (E, K // 128, N // 128)
    assert q_grouped.stride() == (K * N, 1, K)
    assert scale_grouped.stride() == (
        (K // 128) * (N // 128),
        1,
        K // 128,
    )

    for expert_idx in range(E):
        q_ref, scale_ref = triton_fp8_blockwise_weight_quant_transposed_rhs(
            weight_t[expert_idx].transpose(-2, -1).contiguous()
        )
        torch.testing.assert_close(
            q_grouped[expert_idx].to(torch.float32),
            q_ref.to(torch.float32),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(scale_grouped[expert_idx], scale_ref)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("E,K,N", [(1, 128, 128), (3, 256, 384)])
def test_grouped_weight_quant_rhs_matches_dense(E, K, N):
    weight_t = _make_column_major_weight_t(E, N, K)
    q_grouped, scale_grouped = triton_fp8_blockwise_weight_quant_grouped_rhs(weight_t)

    assert q_grouped.shape == (E, N, K)
    assert scale_grouped.shape == (E, N // 128, K // 128)
    assert q_grouped.stride() == (N * K, 1, N)
    assert scale_grouped.stride() == (
        (N // 128) * (K // 128),
        1,
        N // 128,
    )

    for expert_idx in range(E):
        q_ref, scale_ref = triton_fp8_blockwise_weight_quant_rhs(
            weight_t[expert_idx].transpose(-2, -1).contiguous()
        )
        torch.testing.assert_close(
            q_grouped[expert_idx].to(torch.float32),
            q_ref.to(torch.float32),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(scale_grouped[expert_idx], scale_ref)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize(
    "offs,pad_token_groups_for_grouped_mm",
    [
        (torch.tensor([256, 512], dtype=torch.int32), False),
        (torch.tensor([129, 384, 500], dtype=torch.int32), True),
    ],
)
def test_fp8_blockwise_grouped_mm_fwd_bwd(offs, pad_token_groups_for_grouped_mm):
    torch.manual_seed(0)
    offs = offs.cuda()
    E = offs.numel()
    M = int(offs[-1].item())
    K, N = 256, 256
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = _make_column_major_weight_t(E, N, K).requires_grad_(True)

    A_ref = A.detach().clone().requires_grad_(True)
    B_t_ref = B_t.detach().clone().requires_grad_(True)

    out = _to_fp8_blockwise_then_scaled_grouped_mm(
        A,
        B_t,
        offs,
        pad_token_groups_for_grouped_mm=pad_token_groups_for_grouped_mm,
    )
    ref = torch._grouped_mm(A_ref, B_t_ref, offs=offs, out_dtype=torch.bfloat16)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    assert compute_error(ref, out) >= 27.0

    out.float().square().mean().backward()
    ref.float().square().mean().backward()

    assert compute_error(A_ref.grad, A.grad) >= 26.0
    assert compute_error(B_t_ref.grad, B_t.grad) >= 26.0


@skip_if_rocm("ROCm not supported")
def test_fp8_blockwise_grouped_mm_compile_aligned_groups():
    E, M, K, N = 2, 256, 128, 128
    A = torch.randn(E * M, K, dtype=torch.bfloat16, device="cuda")
    B_t = _make_column_major_weight_t(E, N, K)
    offs = torch.arange(M, (E + 1) * M, M, device="cuda", dtype=torch.int32)

    compiled = torch.compile(_to_fp8_blockwise_then_scaled_grouped_mm, fullgraph=True)
    out = compiled(A, B_t, offs, pad_token_groups_for_grouped_mm=False)

    assert out.shape == (E * M, N)
    assert out.dtype == torch.bfloat16


class _GroupedMMModule(nn.Module):
    def __init__(self, weight_t: torch.Tensor):
        super().__init__()
        self.weight_t = nn.Parameter(weight_t)

    def forward(self, A: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        return torch._grouped_mm(A, self.weight_t, offs=offs, out_dtype=torch.bfloat16)


@skip_if_rocm("ROCm not supported")
def test_fp8_blockwise_training_tensor_and_quantize_dispatch():
    E, M, K, N = 2, 256, 128, 128
    A = torch.randn(E * M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight_t = _make_column_major_weight_t(E, N, K)
    offs = torch.arange(M, (E + 1) * M, M, device="cuda", dtype=torch.int32)
    config = Float8BlockwiseTrainingOpConfig(pad_token_groups_for_grouped_mm=False)

    wrapped_weight = Float8TrainingWeightWrapperTensor(weight_t, config)
    out_wrapped = torch._grouped_mm(
        A,
        wrapped_weight,
        offs=offs,
        out_dtype=torch.bfloat16,
    )
    out_direct = _to_fp8_blockwise_then_scaled_grouped_mm(
        A,
        weight_t,
        offs,
        pad_token_groups_for_grouped_mm=False,
    )
    torch.testing.assert_close(out_wrapped, out_direct, atol=0, rtol=0)

    module = _GroupedMMModule(weight_t.detach().clone())
    quantize_(module, config, filter_fn=lambda _mod, _fqn: True)
    assert isinstance(module.weight_t, Float8TrainingWeightWrapperTensor)
    out_quantized = module(A.detach(), offs)
    torch.testing.assert_close(out_quantized, out_direct.detach(), atol=0, rtol=0)
