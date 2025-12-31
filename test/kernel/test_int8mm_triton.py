# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal import common_utils

import torchao.kernel.int8mm_triton  # noqa: F401
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@common_utils.instantiate_parametrized_tests
class TestInt8TritonKernel(TorchAOIntegrationTestCase):
    @common_utils.parametrize("M", [128, 256])
    @common_utils.parametrize("N", [128, 512])
    @common_utils.parametrize("K", [128, 256])
    def test_int8_scaled_matmul_vs_reference(self, M, N, K):
        """Compare Triton kernel against simple PyTorch reference"""
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
        B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="cuda")
        scale_a = torch.randn(M, dtype=torch.float32, device="cuda") * 0.01
        scale_b = torch.randn(N, dtype=torch.float32, device="cuda") * 0.01

        # Reference: simple PyTorch ops (C = A @ B * scale_a * scale_b)
        ref_output = (
            (A.to(torch.float32) @ B.to(torch.float32))
            * scale_a[:, None]
            * scale_b[None, :]
        )
        ref_output = ref_output.to(torch.float16)

        # Triton kernel via torch.ops
        triton_output = torch.ops.torchao.int8_scaled_matmul(A, B, scale_a, scale_b)

        # Compare results using SQNR (higher = better match)
        sqnr = compute_error(ref_output, triton_output)
        self.assertGreater(sqnr, 35, f"SQNR too low: {sqnr:.2f} dB")


if __name__ == "__main__":
    common_utils.run_tests()
