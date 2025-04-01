import copy
import unittest
from typing import Union

import torch
import torch.nn as nn

from torchao.prototype.quantization.module_swap import (
    CodeBookQuantizer,
    QuantizedLinear,
)
from torchao.prototype.quantization.module_swap.algorithms import kmeans_codebook


class SimpleTestNetwork(nn.Module):
    def __init__(self, weight_group_size: Union[int, str] = "per_channel") -> None:
        super().__init__()
        if weight_group_size == "per_channel":
            weight_group_size = 8
        assert isinstance(weight_group_size, int)
        weight_quantizer = CodeBookQuantizer(
            n_bits=2,
            features=16,
            codebook_dim=2,
        )

        self.linear = QuantizedLinear(
            in_features=16,
            out_features=8,
            bias=False,
            weight_quantizer=weight_quantizer,
            activation_bits=8,
            input_quantization=False,
            output_quantization=False,
            weight_quantization=True,
            activation_quantization=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestKmeansCodebook(unittest.TestCase):
    @unittest.skip("No module named 'faiss'")
    def test_kmeans_codebook(self) -> None:
        model = SimpleTestNetwork()
        codebook_before = copy.deepcopy(model.linear.weight_quantizer.codebook)
        kmeans_codebook(model)
        assert not torch.allclose(
            codebook_before,
            model.linear.weight_quantizer.codebook,
        )


if __name__ == "__main__":
    unittest.main()
