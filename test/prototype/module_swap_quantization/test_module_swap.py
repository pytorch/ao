import unittest

import torch
import torch.nn as nn

from torchao.prototype.quantization.module_swap import (
    QuantizationRecipe,
    quantize_module_swap,
)


class SimpleEmbeddingTestNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class TestEmbeddingSwap(unittest.TestCase):
    def test_embedding_swap(self) -> None:
        model = SimpleEmbeddingTestNetwork()
        recipe = QuantizationRecipe()
        recipe.embedding_bits = 4
        recipe.embedding_quantization = True
        model = quantize_module_swap(model, recipe)
        x = torch.randint(0, 10, (10, 64))
        model(x)
        assert model.embedding.weight_quantizer.num_bits == 4
        assert model.embedding.weight_quantizer.group_size == 32


if __name__ == "__main__":
    unittest.main()
