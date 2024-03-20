# Test 1 bit 
from torchao.quantization.onebit import BitLinear
import torch

import unittest

class Test1Bit(unittest.TestCase):
    def test_1bit():
        model = BitLinear(10, 10)
        assert model is not None
        pass

    def test_compile_1bit():
        model = BitLinear(10, 10)
        model.compile()
        model(torch.randn(10, 10))
        pass

if __name__ == "__main__":
    unittest.main()
