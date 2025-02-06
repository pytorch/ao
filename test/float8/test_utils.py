import pytest
import torch

from torchao.float8.float8_utils import _round_down_to_power_of_2


@pytest.mark.parametrize(
    "input_shape",
    [
        (1,),
        (2, 3),
        (8, 2048, 4, 1024),
    ],
)
@pytest.mark.parametrize(
    "multiplier",
    [
        1.0,
        2.5,
        10.0,
    ],
)
def test_round_down_to_power_of_2(input_shape: tuple[int], multiplier: int):
    input_tensor = torch.rand(*input_shape, dtype=torch.float32) * multiplier
    expected_output = torch.exp2(torch.floor(torch.log2(input_tensor)))
    result = _round_down_to_power_of_2(input_tensor)
    assert torch.equal(
        result, expected_output
    ), f"expected {expected_output}, but got {result}"


def test_non_float32_input():
    non_float32_tensor = torch.tensor([3.0], dtype=torch.float64)
    with pytest.raises(AssertionError, match="input must be float32 tensor"):
        _round_down_to_power_of_2(non_float32_tensor)
