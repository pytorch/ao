import unittest

import pytest
import torch

from torchao.float8.float8_utils import _round_scale_down_to_power_of_2


# source for notable single-precision cases:
# https://en.wikipedia.org/wiki/Single-precision_floating-point_format
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@pytest.mark.parametrize(
    "test_case",
    [
        # "test_case_name": [input, expected result]
        ("one", [1.0, 1.0]),
        ("inf", [float("inf"), float("inf")]),
        ("smallest positive subnormal number", [2**-126 * 2**-23, 2**-126 * 2**-23]),
        ("largest subnormal number", [2**-126 * (1 - 2**-23), 1.1754943508222875e-38]),
        ("largest normal number", [2**127 * (2 - 2**-23), float("inf")]),
        ("smallest positive normal number", [2**-126, 2**-126]),
        ("largest number less than one", [1.0 - 2**-24, 0.5]),
        ("smallest number larger than one", [1.0 + 2**-23, 1.0]),
    ],
)
def test_round_scale_down_to_power_of_2_valid_inputs(
    test_case: dict,
):
    test_case_name, (input, expected_result) = test_case
    input_tensor, expected_tensor = (
        torch.tensor(input).cuda(),
        torch.tensor(expected_result).cuda(),
    )
    result = _round_scale_down_to_power_of_2(input_tensor)
    assert torch.equal(
        result, expected_tensor
    ), f"test: {test_case_name}, input: {input_tensor}, expected {expected_tensor}, but got {result}"


@pytest.mark.parametrize(
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
def test_non_float32_input(invalid_dtype: torch.dtype):
    non_float32_tensor = torch.tensor([3.0], dtype=invalid_dtype)
    with pytest.raises(AssertionError, match="scale must be float32 tensor"):
        _round_scale_down_to_power_of_2(non_float32_tensor)
