# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.prototype.mx_formats.cutedsl import (
    _mxfp4_rht_cutedsl_kernels_available,
)

pytestmark = pytest.mark.skipif(
    not _mxfp4_rht_cutedsl_kernels_available,
    reason="cutedsl nvfp4 unavailable (SM10.x, CUDA>=12.8, nvidia-cutlass-dsl)",
)


def _eager_nvfp4_e4m3_block_scales(
    x: torch.Tensor, per_tensor_scale: torch.Tensor
) -> torch.Tensor:
    """torchao eager NVFP4 per-16-block E4M3 scale bytes, plain (unswizzled).

    Mirrors the two-level path of
    ``torchao.prototype.mx_formats.nvfp4_tensor.nvfp4_quantize``:

        block_scale          = amax / F4_E2M1_MAX            # amax / 6.0, fp32
        scaled_block_scales  = block_scale / per_tensor_scale
        e4m3                 = clamp(scaled, E4M3_EPS, 448).to(float8_e4m3fn)
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

    scales, _ = nvfp4_quantize(x, block_size=16, per_tensor_scale=per_tensor_scale)
    return scales


class TestNvfp4E4M3Scale:
    def test_scale_bytes_match_eager(self):
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            compute_block_scale_e4m3_nvfp4,
        )
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            per_tensor_amax_to_scale,
        )

        torch.manual_seed(0)
        x = torch.randn(64, 16, dtype=torch.bfloat16, device="cuda") * 7.0

        # global_scale per torchao's convention. torchao stores it as a divisor
        # (``per_tensor_scale = amax_global / (448 * 6)``); our host helper takes
        # the multiplicative ``global_scale = 1 / per_tensor_scale``.
        amax_global = torch.max(torch.abs(x))
        per_tensor_scale = per_tensor_amax_to_scale(amax_global)
        global_scale = (1.0 / per_tensor_scale).item()

        s_ref = _eager_nvfp4_e4m3_block_scales(x, per_tensor_scale)
        assert s_ref.shape == (64, 1)
        assert s_ref.dtype == torch.float8_e4m3fn

        s_ours = compute_block_scale_e4m3_nvfp4(x, global_scale)
        assert s_ours.shape == (64, 1)
        assert s_ours.dtype == torch.float8_e4m3fn

        torch.testing.assert_close(
            s_ours.view(torch.uint8).flatten(),
            s_ref.view(torch.uint8).flatten(),
            rtol=0,
            atol=0,
        )

    def test_scale_bytes_match_eager_wide(self):
        # Wider K (multiple 16-blocks per row) + a separate seed/scale.
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            compute_block_scale_e4m3_nvfp4,
        )
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            per_tensor_amax_to_scale,
        )

        torch.manual_seed(7)
        x = torch.randn(33, 128, dtype=torch.bfloat16, device="cuda") * 3.0

        amax_global = torch.max(torch.abs(x))
        per_tensor_scale = per_tensor_amax_to_scale(amax_global)
        global_scale = (1.0 / per_tensor_scale).item()

        s_ref = _eager_nvfp4_e4m3_block_scales(x, per_tensor_scale)
        s_ours = compute_block_scale_e4m3_nvfp4(x, global_scale)
        assert s_ref.shape == (33, 8)
        assert s_ours.shape == (33, 8)

        torch.testing.assert_close(
            s_ours.view(torch.uint8).flatten(),
            s_ref.view(torch.uint8).flatten(),
            rtol=0,
            atol=0,
        )
