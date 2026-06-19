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


class TestNvfp4RhtSmoke:
    """Shape / dtype / stride + non-zero-scale smoke for the fused kernel.

    Exercises BOTH the plain NVFP4 cast (no RHT) and the fused FWHT(16) + NVFP4
    RHT cast through the public gated wrapper. Bit-exact correctness vs eager is
    a separate task; here we only check the kernel compiles, launches, and emits
    well-formed outputs.
    """

    def _global_scale(self, x: torch.Tensor) -> float:
        # 2688 == F8E4M3_MAX (448) * F4_E2M1_MAX (6); global_scale is the
        # multiplicative reciprocal of torchao's per_tensor_scale.
        return 2688.0 / x.abs().max().item()

    def _check_outputs(self, q, s, M, K, block_size=16):
        assert q.shape == (M, K // 2)
        assert q.dtype == torch.uint8
        assert q.stride() == (K // 2, 1)
        assert s.dtype == torch.float8_e4m3fn
        # scales must be non-zero (a degenerate all-zero scale tensor would mean
        # the consumer never wrote anything).
        s_u8 = s.view(torch.uint8)
        assert int((s_u8 != 0).sum().item()) > 0, "scales are all zero"

    def test_plain_nvfp4_cast(self):
        from torchao.prototype.mx_formats.cutedsl import (
            nvfp4_rht_quantize_cutedsl_2d,
        )

        torch.manual_seed(0)
        M, K = 128, 256
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 5.0
        global_scale = self._global_scale(x)

        # sign_vector=None -> plain NVFP4 cast (no RHT).
        q, s = nvfp4_rht_quantize_cutedsl_2d(
            x, global_scale, sign_vector=None, is_swizzled_scales=True
        )
        self._check_outputs(q, s, M, K)

        # Empty list is the same plain-cast path.
        q2, s2 = nvfp4_rht_quantize_cutedsl_2d(
            x, global_scale, sign_vector=[], is_swizzled_scales=True
        )
        self._check_outputs(q2, s2, M, K)

    def test_rht_nvfp4_cast(self):
        from torchao.prototype.mx_formats.cutedsl import (
            nvfp4_rht_quantize_cutedsl_2d,
        )

        torch.manual_seed(0)
        M, K = 128, 256
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 5.0
        global_scale = self._global_scale(x)

        sign_vector = [1, -1] * 8  # len 16
        q, s = nvfp4_rht_quantize_cutedsl_2d(
            x, global_scale, sign_vector=sign_vector, is_swizzled_scales=True
        )
        self._check_outputs(q, s, M, K)

    def test_plain_nvfp4_cast_unswizzled(self):
        from torchao.prototype.mx_formats.cutedsl import (
            nvfp4_rht_quantize_cutedsl_2d,
        )

        torch.manual_seed(1)
        M, K = 128, 256
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 5.0
        global_scale = self._global_scale(x)

        q, s = nvfp4_rht_quantize_cutedsl_2d(
            x, global_scale, sign_vector=None, is_swizzled_scales=False
        )
        self._check_outputs(q, s, M, K)
        assert s.shape == (M, K // 16)
