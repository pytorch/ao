# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for partial final tiles in jagged FP8 scale stores."""

import pytest
import torch

from torchao.prototype.moe_training.kernels import jagged_float8_scales as kernels

triton = pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FP8 = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8)
GUARD = -12345.0


def _pin_config(monkeypatch, kernel, configs, *, max_group_size=None):
    config = next(
        config
        for config in configs
        if config.kwargs.get("BLOCK_SIZE") == 32
        and config.kwargs.get("BLOCK_SIZE_ITER", 64) == 64
        and config.kwargs.get("MAX_GROUP_SIZE") == max_group_size
        and config.num_warps == (8 if max_group_size is not None else 4)
        and config.num_stages == (1 if max_group_size is not None else 2)
    )
    monkeypatch.setattr(kernel, "configs", [config])


def _buffers(rows):
    torch.manual_seed(1234)
    n = 100
    hp = torch.randn(rows, n, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([rows], dtype=torch.int64, device="cuda")
    output = torch.empty_like(hp, dtype=FP8).as_strided(hp.size(), (1, rows))
    scales = torch.full((n + 32,), GUARD, dtype=torch.float32, device="cuda")
    return hp, offsets, output, scales


def _assert_scale_and_guard(hp, scales):
    reference = (FP8_INFO.max / hp.abs().amax(dim=0).double().clamp(min=1e-12)).float()
    torch.testing.assert_close(scales[:100], reference, rtol=1e-5, atol=0)
    assert torch.all(scales[100:] == GUARD), "last block wrote past scale group"


def test_unfused_colwise_last_block_preserves_scale_guard(monkeypatch):
    hp, offsets, output, scales = _buffers(32)
    kernel = kernels._triton_fp8_per_group_colwise_scales_kernel
    _pin_config(monkeypatch, kernel, kernels.kernel_configs_2D)
    kernel[(triton.cdiv(100, 32), 1)](
        hp,
        offsets,
        output,
        scales,
        32,
        100,
        1,
        hp.stride(0),
        output.stride(1),
        FP8_INFO.min,
        FP8_INFO.max,
        kernels.FP8_DTYPE_MAP[hp.dtype],
        kernels.FP8_DTYPE_MAP[FP8],
        False,
        EPS=1e-12,
        STRIDE_OUTPUT_ROW=1,
        STRIDE_INPUT_COL=hp.stride(1),
    )
    torch.cuda.synchronize()
    _assert_scale_and_guard(hp, scales)


def test_fused_colwise_last_block_preserves_scale_guard(monkeypatch):
    hp, offsets, output, scales = _buffers(256)
    kernel = kernels._triton_fp8_per_group_colwise_scales_fused_kernel
    _pin_config(monkeypatch, kernel, kernels.kernel_configs_fused, max_group_size=256)
    kernel[(triton.cdiv(100, 32), 1)](
        hp,
        offsets,
        output,
        scales,
        256,
        100,
        hp.stride(0),
        output.stride(1),
        hp.numel(),
        FP8_INFO.min,
        FP8_INFO.max,
        kernels.FP8_DTYPE_MAP[hp.dtype],
        kernels.FP8_DTYPE_MAP[FP8],
        False,
        EPS=1e-12,
        STRIDE_OUTPUT_ROW=1,
        STRIDE_INPUT_COL=hp.stride(1),
    )
    torch.cuda.synchronize()
    _assert_scale_and_guard(hp, scales)


def test_single_block_colwise_wrapper_control(monkeypatch):
    hp = torch.randn(32, 16, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([16, 32], dtype=torch.int64, device="cuda")
    kernel = kernels._triton_fp8_per_group_colwise_scales_kernel
    _pin_config(monkeypatch, kernel, kernels.kernel_configs_2D)
    _, scales = kernels.triton_fp8_per_group_colwise_scales(hp, offsets)
    torch.cuda.synchronize()
    reference = torch.cat(
        [
            (
                FP8_INFO.max / hp[start:end].abs().amax(dim=0).double().clamp(min=1e-12)
            ).float()
            for start, end in ((0, 16), (16, 32))
        ]
    )
    torch.testing.assert_close(scales, reference, rtol=1e-5, atol=0)
