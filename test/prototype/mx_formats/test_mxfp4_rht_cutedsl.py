# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.prototype.mx_formats.cutedsl import _mxfp4_rht_cutedsl_kernels_available

pytestmark = pytest.mark.skipif(
    not _mxfp4_rht_cutedsl_kernels_available,
    reason="mxfp4 rht cutedsl unavailable (needs SM10.x, CUDA>=12.8, nvidia-cutlass-dsl)",
)


class TestE2M1Packing:
    def test_pack_bitexact_vs_reference(self):
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            pack32_e2m1_to_bytes,
        )
        from torchao.prototype.mx_formats.kernels import (
            f32_to_f4_unpacked,
            pack_uint4,
        )

        torch.manual_seed(0)
        # len 32, spans the E2M1 grid + saturation + both parities/signs
        x = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.5,
                -1.0,
                -6.0,
                7.0,
                -7.0,
                0.25,
                5.0,
                -2.5,
            ]
            * 2,
            dtype=torch.float32,
            device="cuda",
        )
        ours = pack32_e2m1_to_bytes(x)  # (16,) uint8
        ref = (
            pack_uint4(f32_to_f4_unpacked(x.reshape(1, 32))).view(torch.uint8).flatten()
        )  # (16,)
        torch.testing.assert_close(ours, ref, rtol=0, atol=0)


class TestFp4Scale:
    @pytest.mark.parametrize("mode", ["floor", "rceil"])
    def test_scale_bytes_match_eager(self, mode):
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            compute_block_scale_e8m0_fp4,
        )
        from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx

        torch.manual_seed(0)
        x = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda") * 7.0
        m = {
            "floor": ScaleCalculationMode.FLOOR,
            "rceil": ScaleCalculationMode.RCEIL,
        }[mode]
        # plain (unswizzled) e8m0 scales, shape (64, 1)
        s_ref, _ = to_mx(x, torch.float4_e2m1fn_x2, block_size=32, scaling_mode=m)
        s_ours = compute_block_scale_e8m0_fp4(x, mode)  # (64, 1) float8_e8m0fnu
        torch.testing.assert_close(
            s_ours.view(torch.uint8).flatten(),
            s_ref.view(torch.uint8).flatten(),
            rtol=0,
            atol=0,
        )
