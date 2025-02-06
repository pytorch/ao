import pytest
import torch

from torchao.float8.float8_utils import _round_scale_down_to_power_of_2


# source for notable single-precision cases:
# https://en.wikipedia.org/wiki/Single-precision_floating-point_format
#
# TODO(danielvegamyhre):
# 1. add case for largest normal fp32 value: 2**127 * (2 - 2**-23).
#    need to investigate why exp2(floor(log2(x)))=inf, but bitshift returns real value.
# 2. add case for "nan"
#    need to investigate why exp2(floor(log2(nan)))=nan, but bitshift returns inf.
# 3. adjust cases for subnormal values so we aren't clamping the expected results
#    into the normal range.
#    preliminary investigation shows it may not be possible to support all subnormals
#    with bitshifting, so we will need to debug/improve performance of exp2(floor(log2(x)))
#    approach.
@pytest.mark.parametrize(
    "input",
    [
        1.0,
        float("inf"),
        # smallest positive subnormal number
        2**-126 * 2**-23,
        # largest subnormal number
        2**-126 * (1 - 2**-23),
        # smallest positive normal number
        2**-126,
        # largest number less than one
        1.0 - 2**-24,
        # smallest number larger than one
        1.0 + 2**-23,
    ],
)
def test_round_scale_down_to_power_of_2_valid_inputs(input: float):
    input_tensor = torch.tensor(input, dtype=torch.float32)
    result = _round_scale_down_to_power_of_2(input_tensor)

    # get expected value for comparison
    # TODO(danielvegamyhre): support subnormal values
    expected_result = torch.exp2(torch.floor(torch.log2(input_tensor)))
    smallest_normal_fp32_value = torch.tensor(2**-126, dtype=torch.float32)
    expected_result = torch.max(expected_result, smallest_normal_fp32_value)

    assert torch.equal(
        result, expected_result
    ), f"input: {input_tensor}, expected {expected_result}, but got {result}"


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
