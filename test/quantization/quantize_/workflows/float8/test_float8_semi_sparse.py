# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.float8.inference import Float8MMConfig
from torchao.quantization.quantize_.workflows.float8.float8_semi_sparse_tensor import (
    Float8SemiSparseTensor,
)
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.testing.utils import skip_if_rocm
from torchao.utils import is_sm_at_least_90


@unittest.skipIf(not is_sm_at_least_90(), "Need H100+ to run")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
class TestFloat8SemiSparseTensor(TestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @skip_if_rocm("ROCm enablement in progress")
    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 512, 128),
            ((2, 32, 128), 256, 128),
        ],
    )
    def test_sparse_vs_dense_fp8(self, sizes):
        dtype = torch.bfloat16
        device = "cuda"

        M, N, K = sizes
        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)

        apply_fake_sparsity(linear)

        mm_config = Float8MMConfig(use_fast_accum=True)
        input_fp8 = Float8Tensor.from_hp(
            input, float8_dtype=torch.float8_e4m3fn, mm_config=mm_config
        )

        weight_fp8 = Float8Tensor.from_hp(
            linear.weight.data, float8_dtype=torch.float8_e4m3fn, mm_config=mm_config
        )
        dense_output = torch.nn.functional.linear(input_fp8, weight_fp8, linear.bias)

        weight_sparse_fp8 = Float8SemiSparseTensor.from_hp(linear.weight.data, [1, K])
        sparse_output = torch.nn.functional.linear(
            input_fp8, weight_sparse_fp8, linear.bias
        )

        torch.testing.assert_close(dense_output, sparse_output, atol=3e-1, rtol=3e-1)


instantiate_parametrized_tests(TestFloat8SemiSparseTensor)


if __name__ == "__main__":
    run_tests()
