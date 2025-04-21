# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import unittest

import torch
from torch.testing._internal.common_utils import TestCase

from torchao import quantize_
from torchao.prototype.parq.optim import (
    ProxHardQuant,
    ProxPARQ,
    QuantOptimizer,
)
from torchao.prototype.parq.quant import (
    LSBQuantizer,
    UnifQuantizer,
    UnifTorchaoQuantizer,
)
from torchao.quantization.quant_api import int4_weight_only
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_param_groups(model):
    params_no_quant, params_quant = [], []
    for p in model.parameters():
        if p.dim() > 1:
            params_quant.append(p)
        else:
            params_no_quant.append(p)
    return params_no_quant, params_quant


class M(torch.nn.Module):
    def __init__(self, m=256, n=128, k=16, bias=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, m)
        self.linear1 = torch.nn.Linear(m, n, bias=bias)
        self.linear2 = torch.nn.Linear(n, k, bias=bias)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        for module in (self.linear1, self.linear2):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
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


class TestPARQuantization(TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = M(bias=True).to(_DEVICE)
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


class TestUnifTorchaoQuantizer(TestCase):
    def setUp(self, group_size=32):
        torch.manual_seed(123)
        self.model = M(n=1024, k=1024).to(torch.bfloat16).to(_DEVICE)
        self.group_size = group_size

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    def test_int4_weight_only(self):
        self.model.reset_parameters()
        m_copy = copy.deepcopy(self.model)
        quantize_(m_copy, int4_weight_only(group_size=self.group_size))

        # copied from torchao.quantization.quant_api._int4_weight_only_transform
        b = 4
        quantizer = UnifTorchaoQuantizer(
            symmetric=False,
            target_dtype=torch.int32,
            quant_min=0,
            quant_max=2**b - 1,
            eps=1e-6,
            preserve_zero=False,
        )
        self.assertTrue(
            quantizer.get_quant_size(b) == quantizer.quant_max - quantizer.quant_min + 1
        )

        for n, module in self.model.named_children():
            if not isinstance(module, torch.nn.Linear):
                continue

            # simulate grouping from QuantOptimizer.step
            p = module.weight
            original_shape = p.shape
            p = p.view(-1, self.group_size)

            q, Q = quantizer.quantize(p, b=b, dim=-1)
            q = q.view(original_shape)

            # compare to AffineQuantizedTensor instance
            ref = getattr(m_copy, n).weight.dequantize()
            self.assertTrue(q.equal(ref))


if __name__ == "__main__":
    unittest.main()
