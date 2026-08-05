# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import pytest
import torch

import torchao
from torchao.utils import is_MI300, is_MI350, is_sm_at_least_90

if not (
    torch.cuda.is_available() and (is_sm_at_least_90() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8_training.cutedsl_grouped_gemm import (
    _HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED,
    _cutedsl_runtime_available,
    _load_cutedsl_hopper_gemm_module,
    cutedsl_fp8_blockwise_scaled_grouped_mm_2d_2d,
    cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d,
)
from torchao.prototype.blockwise_fp8_training.grouped_kernels import (
    emulated_blockwise_scaled_grouped_mm,
)
from torchao.prototype.blockwise_fp8_training.grouped_weight_quant import (
    triton_fp8_blockwise_weight_quant_grouped_forward_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
)
from torchao.prototype.moe_training.blockwise_fp8.grouped_mm import (
    _to_fp8_blockwise_then_emulated_scaled_grouped_mm,
    _to_fp8_blockwise_then_scaled_grouped_mm,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.testing.utils import skip_if_rocm

torch._dynamo.config.cache_size_limit = 1000


@pytest.mark.skipif(
    not _cutedsl_runtime_available(),
    reason="CuTeDSL runtime packages are not available",
)
def test_cutedsl_hopper_support_is_packaged():
    original_sys_path = sys.path.copy()
    module = _load_cutedsl_hopper_gemm_module()

    assert module is not None
    assert Path(module.__file__).is_relative_to(Path(torchao.__file__).parent)
    assert sys.path == original_sys_path


def _make_column_major_weight_t(E: int, N: int, K: int) -> torch.Tensor:
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    return weight.contiguous().transpose(-2, -1)


def _quantize_column_major_weight_t(
    weight_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, scale = triton_fp8_blockwise_weight_quant_grouped_forward_rhs(weight_t)
    return q.transpose(-2, -1), scale.transpose(-2, -1)


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


@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(
    not _cutedsl_runtime_available(),
    reason="CuTeDSL runtime packages are not available",
)
@pytest.mark.parametrize(
    ("K", "N", "tile_shape_mn"),
    [
        pytest.param(7168, 2048, (128, 192), id="fwd"),
        pytest.param(2048, 7168, (128, 256), id="dgrad"),
    ],
)
def test_cutedsl_fused_2d_3d_correctness_and_cache(monkeypatch, K, N, tile_shape_mn):
    monkeypatch.setenv("TORCHAO_ENABLE_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM", "1")
    torch.manual_seed(0)
    E, M_per_group = 4, 128
    M = E * M_per_group
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_t = _make_column_major_weight_t(E, N, K)
    offs = torch.arange(
        M_per_group,
        M + 1,
        M_per_group,
        device="cuda",
        dtype=torch.int32,
    )
    A_fp8, A_scale = triton_fp8_blockwise_act_quant_lhs(A)
    B_t_fp8, B_t_scale = _quantize_column_major_weight_t(B_t)

    out = cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d(
        A_fp8,
        B_t_fp8,
        A_scale,
        B_t_scale,
        offs,
        torch.bfloat16,
    )
    ref = emulated_blockwise_scaled_grouped_mm(
        A_fp8,
        B_t_fp8,
        A_scale,
        _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        B_t_scale,
        _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
        offs,
        torch.bfloat16,
    )
    key = (
        "blockwise_scaled_persistent",
        M_per_group,
        N,
        E,
        K,
        128,
        K // 128,
        tile_shape_mn,
        (K // 128, M),
        (M, 1),
    )
    compiled = _HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED[key]

    out_again = cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d(
        A_fp8,
        B_t_fp8,
        A_scale,
        B_t_scale,
        offs,
        torch.bfloat16,
    )

    assert _HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED[key] is compiled
    torch.testing.assert_close(out, ref, atol=4.0, rtol=0.0)
    torch.testing.assert_close(out_again, out, atol=0.0, rtol=0.0)


@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(
    not _cutedsl_runtime_available(),
    reason="CuTeDSL runtime packages are not available",
)
@pytest.mark.parametrize("group_blocks", [16, 17])
def test_cutedsl_fused_wgrad_correctness_and_cache(monkeypatch, group_blocks):
    monkeypatch.setenv("TORCHAO_ENABLE_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM", "1")
    torch.manual_seed(0)
    E, N, K = 4, 256, 256
    M_per_group = group_blocks * 128
    M = E * M_per_group
    offs = torch.arange(
        M_per_group,
        M + 1,
        M_per_group,
        device="cuda",
        dtype=torch.int32,
    )
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    activation = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    grad_output_t_fp8, grad_output_t_scale = (
        triton_fp8_blockwise_act_quant_transposed_lhs(grad_output.contiguous())
    )
    activation_rhs_fp8, activation_rhs_scale = triton_fp8_blockwise_act_quant_rhs(
        activation.contiguous()
    )

    out = cutedsl_fp8_blockwise_scaled_grouped_mm_2d_2d(
        grad_output_t_fp8,
        activation_rhs_fp8,
        grad_output_t_scale,
        activation_rhs_scale,
        offs,
        torch.bfloat16,
    )
    ref = emulated_blockwise_scaled_grouped_mm(
        grad_output_t_fp8,
        activation_rhs_fp8,
        grad_output_t_scale,
        _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        activation_rhs_scale,
        _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        offs,
        torch.bfloat16,
    )
    key = (
        "wgrad_blockwise_scaled_persistent",
        M_per_group,
        N,
        K,
        E,
        128,
        group_blocks,
        (128, 128),
    )

    assert key in _HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED
    torch.testing.assert_close(out, ref, atol=1.0, rtol=0.0)


@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(
    not _cutedsl_runtime_available(),
    reason="CuTeDSL runtime packages are not available",
)
def test_cutedsl_auto_fused_fwd_bwd_never_emulates(monkeypatch):
    from torchao.prototype.blockwise_fp8_training import grouped_kernels
    from torchao.prototype.moe_training.blockwise_fp8 import grouped_mm_backend

    monkeypatch.setenv("TORCHAO_ENABLE_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM", "1")
    monkeypatch.setattr(
        grouped_mm_backend,
        "can_use_deepgemm_grouped_training",
        lambda *args, **kwargs: pytest.fail("AUTO should select CuTeDSL first"),
    )
    monkeypatch.setattr(
        grouped_kernels,
        "emulated_blockwise_scaled_grouped_mm",
        lambda *args, **kwargs: pytest.fail("fused CuTeDSL path must not emulate"),
    )

    torch.manual_seed(0)
    E, M_per_group, K, N = 4, 128, 7168, 2048
    M = E * M_per_group
    offs = torch.arange(
        M_per_group,
        M + 1,
        M_per_group,
        device="cuda",
        dtype=torch.int32,
    )
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = _make_column_major_weight_t(E, N, K).requires_grad_(True)

    out = _to_fp8_blockwise_then_scaled_grouped_mm(
        A,
        B_t,
        offs,
        pad_token_groups_for_grouped_mm=False,
        kernel_preference=KernelPreference.AUTO,
    )
    out.float().square().mean().backward()

    assert out.shape == (M, N)
    assert A.grad is not None and A.grad.shape == A.shape
    assert B_t.grad is not None and B_t.grad.shape == B_t.shape
