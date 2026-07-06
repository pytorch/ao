# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_MI300, is_MI350, is_sm_at_least_90

if not (
    torch.cuda.is_available() and (is_sm_at_least_90() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.moe_training.blockwise_fp8.grouped_mm import (
    _to_fp8_blockwise_then_emulated_scaled_grouped_mm,
    fp8_blockwise_grouped_mm,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.testing.utils import skip_if_rocm

torch._dynamo.config.cache_size_limit = 1000


def _make_column_major_weight_t(E: int, N: int, K: int) -> torch.Tensor:
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    return weight.contiguous().transpose(-2, -1)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize(
    "offs,pad_token_groups_for_grouped_mm",
    [
        (torch.tensor([256, 512], dtype=torch.int32), False),
        (torch.tensor([129, 384, 500], dtype=torch.int32), True),
    ],
)
def test_fp8_blockwise_emulated_grouped_mm_fwd_bwd(
    offs, pad_token_groups_for_grouped_mm
):
    torch.manual_seed(0)
    offs = offs.cuda()
    E = offs.numel()
    M = int(offs[-1].item())
    K, N = 256, 256
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = _make_column_major_weight_t(E, N, K).requires_grad_(True)

    A_ref = A.detach().clone().requires_grad_(True)
    B_t_ref = B_t.detach().clone().requires_grad_(True)

    out = _to_fp8_blockwise_then_emulated_scaled_grouped_mm(
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
def test_fp8_blockwise_grouped_mm_public_function_emulated_backend():
    torch.manual_seed(0)
    E, tokens_per_expert, K, N = 2, 256, 256, 256
    M = E * tokens_per_expert
    offs = torch.arange(
        tokens_per_expert,
        M + 1,
        tokens_per_expert,
        device="cuda",
        dtype=torch.int32,
    )
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = _make_column_major_weight_t(E, N, K).requires_grad_(True)

    A_ref = A.detach().clone().requires_grad_(True)
    B_t_ref = B_t.detach().clone().requires_grad_(True)

    out = fp8_blockwise_grouped_mm(
        A,
        B_t,
        offs,
        pad_token_groups_for_grouped_mm=False,
        kernel_preference=KernelPreference.EMULATED,
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
def test_fp8_blockwise_emulated_grouped_mm_compile_aligned_groups():
    E, M, K, N = 2, 256, 128, 128
    A = torch.randn(E * M, K, dtype=torch.bfloat16, device="cuda")
    B_t = _make_column_major_weight_t(E, N, K)
    offs = torch.arange(M, (E + 1) * M, M, device="cuda", dtype=torch.int32)

    compiled = torch.compile(
        _to_fp8_blockwise_then_emulated_scaled_grouped_mm, fullgraph=True
    )
    out = compiled(A, B_t, offs, pad_token_groups_for_grouped_mm=False)

    assert out.shape == (E * M, N)
    assert out.dtype == torch.bfloat16
