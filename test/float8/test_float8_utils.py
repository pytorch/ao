# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import pytest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.float8.float8_utils import _round_scale_down_to_power_of_2
from torchao.testing.utils import skip_if_rocm
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise unittest.SkipTest("Unsupported PyTorch version")


class TestFloat8Utils(TestCase):
    # source for notable single-precision cases:
    # https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @parametrize(
        "test_case",
        [
            # ("test_case_name", input, expected result)
            ("one", 1.0, 1.0),
            ("inf", float("inf"), float("inf")),
            ("nan", float("nan"), float("nan")),
            ("smallest positive subnormal number", 2**-126 * 2**-23, 2**-126 * 2**-23),
            ("largest normal number", 2**127 * (2 - 2**-23), float("inf")),
            ("smallest positive normal number", 2**-126, 2**-126),
            ("largest number less than one", 1.0 - 2**-24, 0.5),
            ("smallest number larger than one", 1.0 + 2**-23, 1.0),
            # TODO(danielvegamyhre): debug why creating a tensor with largest
            # subnormal value in CI env for pytorch 2.5.1 truncates the value to 0.
            # ("largest subnormal number", [2**-126 * (1 - 2**-23), 1.1754943508222875e-38]),
        ],
    )
    @skip_if_rocm("ROCm enablement in progress")
    def test_round_scale_down_to_power_of_2_valid_inputs(
        self,
        test_case: dict,
    ):
        test_case_name, input, expected_result = test_case
        input_tensor, expected_tensor = (
            torch.tensor(input, dtype=torch.float32).cuda(),
            torch.tensor(expected_result, dtype=torch.float32).cuda(),
        )
        result = _round_scale_down_to_power_of_2(input_tensor)

        assert torch.equal(result, expected_tensor) or (
            result.isnan() and expected_tensor.isnan()
        ), (
            f"test: {test_case_name}, input: {input_tensor}, expected {expected_tensor}, but got {result}"
        )

    @parametrize(
        "invalid_dtype",
        [
            torch.bfloat16,
            torch.float16,
            torch.float64,
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.uint32,
            torch.int64,
        ],
    )
    def test_non_float32_input(self, invalid_dtype: torch.dtype):
        non_float32_tensor = torch.tensor([3.0], dtype=invalid_dtype)
        with pytest.raises(AssertionError, match="scale must be float32 tensor"):
            _round_scale_down_to_power_of_2(non_float32_tensor)


instantiate_parametrized_tests(TestFloat8Utils)

if __name__ == "__main__":
    run_tests()
