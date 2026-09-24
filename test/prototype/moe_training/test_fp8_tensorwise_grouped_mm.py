# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import (
    is_MI300,
    is_MI350,
    is_sm_at_least_90,
    torch_version_at_least,
)

if not (
    torch_version_at_least("2.7.0")
    and torch.cuda.is_available()
    and (is_sm_at_least_90() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

import triton

from torchao.prototype.moe_training.config import (  # noqa: E402
    Float8TrainingOpConfig,
)
from torchao.prototype.moe_training.fp8_tensorwise_grouped_mm import (  # noqa: E402
    _to_fp8_tensorwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.kernels.fp8_tensorwise_2d import (  # noqa: E402
    EPS,
    FP8_DTYPE_MAP,
    _fp8_tensorwise_2d_amax_kernel,
    _fp8_tensorwise_2d_dual_layout_quantize_kernel,
    _fp8_tensorwise_2d_quantize_kernel,
    triton_fp8_tensorwise_amax,
    triton_fp8_tensorwise_quantize_2d,
    triton_fp8_tensorwise_quantize_2d_dual_layout,
)


def _atomic_amax(tensor: torch.Tensor) -> torch.Tensor:
    numel = tensor.numel()
    amax_buf = torch.zeros(1, dtype=torch.float32, device=tensor.device)
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _fp8_tensorwise_2d_amax_kernel[grid](
        tensor,
        amax_buf,
        numel,
        INPUT_DTYPE_MAX=torch.finfo(tensor.dtype).max,
    )
    return amax_buf


def _atomic_quantize_2d(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = tensor.shape
    numel = tensor.numel()
    fp8_out = torch.empty(m, k, dtype=output_dtype, device=tensor.device)
    inv_scale = torch.empty(m, dtype=torch.float32, device=tensor.device)
    amax_buf = _atomic_amax(tensor)
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _fp8_tensorwise_2d_quantize_kernel[grid](
        tensor,
        fp8_out,
        amax_buf,
        inv_scale,
        k,
        numel,
        fp8_max=torch.finfo(output_dtype).max,
        fp8_min=torch.finfo(output_dtype).min,
        INPUT_DTYPE_MAX=torch.finfo(tensor.dtype).max,
        EPS=EPS,
        ROUND_POW2=True,
        OUTPUT_DTYPE=FP8_DTYPE_MAP[output_dtype],
    )
    return fp8_out, inv_scale


def _atomic_quantize_2d_dual_layout(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = tensor.shape
    fp8_row = torch.empty(m, k, dtype=output_dtype, device=tensor.device)
    fp8_col = torch.empty(k, m, dtype=output_dtype, device=tensor.device).t()
    inv_scale = torch.empty(m, dtype=torch.float32, device=tensor.device)
    amax_buf = _atomic_amax(tensor)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(k, meta["BLOCK_K"]),
    )
    _fp8_tensorwise_2d_dual_layout_quantize_kernel[grid](
        tensor,
        fp8_row,
        fp8_col,
        amax_buf,
        inv_scale,
        m,
        k,
        fp8_col.stride(0),
        fp8_col.stride(1),
        fp8_max=torch.finfo(output_dtype).max,
        fp8_min=torch.finfo(output_dtype).min,
        INPUT_DTYPE_MAX=torch.finfo(tensor.dtype).max,
        EPS=EPS,
        ROUND_POW2=True,
        OUTPUT_DTYPE=FP8_DTYPE_MAP[output_dtype],
    )
    return fp8_row, fp8_col, inv_scale


@pytest.mark.parametrize("shape", [(128, 256), (129, 512)])
def test_tensorwise_staged_amax_matches_atomic_amax(shape):
    torch.manual_seed(0)
    tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    tensor[0, 0] = float("nan")
    tensor[1, 0] = float("inf")
    tensor[2, 0] = float("-inf")

    staged_amax = triton_fp8_tensorwise_amax(tensor)
    atomic_amax = _atomic_amax(tensor)

    torch.cuda.synchronize()
    assert torch.equal(staged_amax, atomic_amax)


@pytest.mark.parametrize("shape", [(128, 256), (129, 512)])
def test_tensorwise_2d_quantize_matches_atomic_amax_baseline(shape):
    torch.manual_seed(1)
    config = Float8TrainingOpConfig(float8_recipe="tensorwise")
    tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    atomic_fp8, atomic_inv_scale = _atomic_quantize_2d(tensor, config.float8_dtype)
    staged_fp8, staged_inv_scale = triton_fp8_tensorwise_quantize_2d(
        tensor,
        config.float8_dtype,
    )

    torch.cuda.synchronize()
    assert torch.equal(staged_fp8, atomic_fp8)
    assert torch.equal(staged_inv_scale, atomic_inv_scale)


@pytest.mark.parametrize("shape", [(128, 256), (129, 512)])
def test_tensorwise_dual_layout_quantize_matches_atomic_amax_baseline(shape):
    torch.manual_seed(2)
    config = Float8TrainingOpConfig(float8_recipe="tensorwise")
    tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    atomic_row, atomic_col, atomic_inv_scale = _atomic_quantize_2d_dual_layout(
        tensor,
        config.float8_dtype,
    )
    staged_row, staged_col, staged_inv_scale = (
        triton_fp8_tensorwise_quantize_2d_dual_layout(
            tensor,
            config.float8_dtype,
        )
    )

    torch.cuda.synchronize()
    assert torch.equal(staged_row, atomic_row)
    assert torch.equal(staged_col, atomic_col)
    assert staged_col.stride() == atomic_col.stride()
    assert torch.equal(staged_inv_scale, atomic_inv_scale)


def test_tensorwise_grouped_mm_forward_backward_nonuniform_offsets():
    torch.manual_seed(3)
    config = Float8TrainingOpConfig(float8_recipe="tensorwise")
    device = "cuda"
    experts, k, n = 3, 64, 80
    group_sizes = torch.tensor([16, 32, 48], device=device, dtype=torch.int32)
    offs = torch.cumsum(group_sizes, dim=0).to(torch.int32)
    m = int(offs[-1].item())

    a = torch.randn(m, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(
        experts,
        n,
        k,
        device=device,
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    b_t = b.transpose(-2, -1)

    out = _to_fp8_tensorwise_then_scaled_grouped_mm(
        a,
        b_t,
        offs,
        config.out_dtype,
        config.float8_dtype,
    )
    loss = out.float().square().mean()
    loss.backward()

    torch.cuda.synchronize()
    assert out.shape == (m, n)
    assert torch.isfinite(out).all()
    assert a.grad is not None
    assert b.grad is not None
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
