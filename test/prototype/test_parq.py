import unittest

import torch

from torchao.prototype.parq.optim import (
    ProxHardQuant,
    ProxPARQ,
    QuantOptimizer,
)
from torchao.prototype.parq.quant import LSBQuantizer, UnifQuantizer

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_param_groups(model):
    params_no_quant, params_quant = [], []
    for p in model.parameters():
        if p.dim() > 1:
            params_quant.append(p)
        else:
            params_no_quant.append(p)
    return params_no_quant, params_quant


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 256)
        self.linear1 = torch.nn.Linear(256, 128)
        self.linear2 = torch.nn.Linear(128, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        for module in (self.linear1, self.linear2):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def example_inputs(self):
        return torch.randint(1, 10, (1, 256))

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class TestPARQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = M().to(_DEVICE)
        self.params_no_quant, self.params_quant = split_param_groups(self.model)

    def test_2bit_unif_quantizer_hard_prox(self):
        self.model.reset_parameters()
        param_groups = [
            {"params": self.params_no_quant},
            {"params": self.params_quant, "quant_bits": 2},
        ]
        base_optimizer = torch.optim.AdamW(param_groups)
        quantizer = UnifQuantizer()
        prox_map = ProxHardQuant()
        optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map)

        x = self.model.example_inputs().to(_DEVICE)
        out = self.model(x)
        out.sum().backward()
        optimizer.step()

        for child in self.model.children():
            if isinstance(child, torch.nn.Linear):
                self.assertEqual(child.weight.unique().numel(), 4)

    def test_ternarybit_lsbq_parq_prox(self):
        self.model.reset_parameters()
        param_groups = [
            {"params": self.params_no_quant},
            {"params": self.params_quant, "quant_bits": 0},
        ]
        base_optimizer = torch.optim.AdamW(param_groups)
        quantizer = LSBQuantizer()
        prox_map = ProxPARQ(anneal_start=0, anneal_end=2)
        optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map)

        for _ in range(3):
            x = self.model.example_inputs().to(_DEVICE)
            out = self.model(x)
            out.sum().backward()
            optimizer.step()

        for child in self.model.children():
            if isinstance(child, torch.nn.Linear):
                self.assertEqual(child.weight.unique().numel(), 3)


if __name__ == "__main__":
    unittest.main()
