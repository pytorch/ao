import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.dtypes.fp6 import to_fp6


_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestFp6(TestCase):

    @parametrize("device", _DEVICES)
    @parametrize("dtype", _DTYPES)
    @parametrize(
        "input_output",
        [
            (0.0,    0b000000),  # exact values
            (1.0,    0b001100),  # normal numbers
            (1.25,   0b001101),
            (28.0,   0b011111),  # max
            (0.1875, 0b000011),  # subnormal number
            (0.0625, 0b000001),  # min
            (29.0,   0b011111),  # normal round down
            (26.0,   0b011110),  # normal round to nearest even
            (0.1251, 0b000010),  # subnormal round down
            (0.0314, 0b000001),  # subnormal round up
            (0.03,   0b000000),  # underflow
        ],
    )
    def test_no_bit_packing_correctness(self, device, dtype, input_output):
        input, output = input_output
        input = torch.tensor(input, device=device, dtype=dtype)
        assert to_fp6(input, no_bit_packing=True).item() == output

    @parametrize("device", _DEVICES)
    @parametrize("dtype", _DTYPES)
    def test_bit_packing_correctness(self, device, dtype):
        x = torch.randn(128, 128, device=device, dtype=dtype)
        results_unpacked = to_fp6(x, no_bit_packing=True)
        results_packed = to_fp6(x)

        val0, val1, val2, val3 = results_unpacked.unflatten(-1, (-1, 4)).unbind(-1)
        bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
        bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
        bits2 = (val2 << 6) | (val3);      # 2233 3333

        expected_packed = torch.stack([bits0, bits1, bits2], dim=-1).flatten(-2)
        assert (results_packed == expected_packed).all()

    @parametrize("device", _DEVICES)
    @parametrize("shape", [(), (0,), (10,), (20, 20)])
    def test_no_bit_packing_shape(self, device, shape):
        x = torch.randn(shape, device=device)
        result = to_fp6(x, no_bit_packing=True)
        assert result.shape == shape

    @parametrize("device", _DEVICES)
    @parametrize("shape", [(4,), (20, 20)])
    def test_bit_packing_shape(self, device, shape):
        x = torch.randn(shape, device=device)
        result = to_fp6(x)
        assert result.shape == shape[:-1] + (shape[-1] // 4 * 3,)

    @parametrize("device", _DEVICES)
    @parametrize("dtype", _DTYPES)
    @parametrize("no_bit_packing", [False, True])
    def test_compile(self, device, dtype, no_bit_packing):
        x = torch.randn(20, 20, device=device, dtype=dtype)
        to_fp6_compiled = torch.compile(to_fp6)  # will hit cache_size_limit if fullgraph=True

        actual = to_fp6_compiled(x, no_bit_packing=no_bit_packing)
        expected = to_fp6(x, no_bit_packing=no_bit_packing)
        torch.testing.assert_close(actual, expected)


instantiate_parametrized_tests(TestFp6)


if __name__ == "__main__":
    run_tests()
