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
