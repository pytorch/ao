import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.dtypes.float6_e3m2 import to_float6_e3m2, from_float6_e3m2
from torchao.quantization.fp6_llm import to_tc_float6_e3m2, from_tc_float6_e3m2
from torchao.ops import prepack_fp6_weight


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestTcFloat6E3M2(TestCase):
    @parametrize("device", _DEVICES)
    def test_to_tc_float6_e3m2_correctness(self, device):
        x = torch.randn(256, 64, device=device)

        expected = prepack_fp6_weight(to_float6_e3m2(x.cpu()).view(torch.int32)).view(torch.uint8)
        actual = to_tc_float6_e3m2(x)
        torch.testing.assert_close(actual.view(-1).cpu(), expected.view(-1))

    @parametrize("device", _DEVICES)
    def test_to_tc_float6_e3m2_compile(self, device):
        x = torch.randn(256, 64, device=device)

        expected = to_tc_float6_e3m2(x)
        actual = torch.compile(to_tc_float6_e3m2)(x)
        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    def test_from_tc_float6_e3m2_correctness(self, device):
        x = torch.randn(256, 64, device=device)
        x = from_float6_e3m2(to_float6_e3m2(x))  # quantize and dequantize so that the values are exactly representable in FP6

        actual = from_tc_float6_e3m2(to_tc_float6_e3m2(x), *x.shape)
        torch.testing.assert_close(actual, x)

    @parametrize("device", _DEVICES)
    def test_from_tc_float6_e3m2_compile(self, device):
        M, N = 256, 64
        x = torch.randint(256, size=(M * N * 3 // 4,), dtype=torch.uint8, device=device)

        expected = from_tc_float6_e3m2(x, M, N)
        actual = torch.compile(from_tc_float6_e3m2)(x, M, N)
        torch.testing.assert_close(actual, expected)


instantiate_parametrized_tests(TestTcFloat6E3M2)


if __name__ == "__main__":
    run_tests()
