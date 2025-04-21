# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import unittest

import torch
from torch import nn
from torch.testing._internal import common_utils

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
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    IntxWeightOnlyConfig,
    _is_linear,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    ZeroPointDomain,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4, TORCH_VERSION_AT_LEAST_2_6

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_param_groups(model):
    params_no_quant, params_quant = [], []
    for p in model.parameters():
        if p.dim() > 1:
            params_quant.append(p)
        else:
            params_no_quant.append(p)
    return params_no_quant, params_quant


class M(nn.Module):
    def __init__(self, m=256, n=128, k=16, bias=False):
        super().__init__()
        self.embedding = nn.Embedding(10, m)
        self.linear1 = nn.Linear(m, n, bias=bias)
        self.linear2 = nn.Linear(n, k, bias=bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for module in (self.linear1, self.linear2):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def example_inputs(self):
        return torch.randint(1, 10, (1, 256))

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class TestPARQuantization(common_utils.TestCase):
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
            if isinstance(child, nn.Linear):
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
            if isinstance(child, nn.Linear):
                self.assertEqual(child.weight.unique().numel(), 3)


class TestUnifTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.group_size = 32

    def compare_quantized_models(
        self,
        model: nn.Module,
        m_ref: nn.Module,
        quantizer: UnifTorchaoQuantizer,
        b: int,
    ):
        for n, module in model.named_children():
            if not _is_linear(module):
                continue

            # simulate grouping from QuantOptimizer.step
            p = module.weight
            original_shape = p.shape
            p = p.view(-1, self.group_size)

            q, Q = quantizer.quantize(p, b=b, dim=-1)
            q = q.view(original_shape)

            # compare to AffineQuantizedTensor instance
            ref = getattr(m_ref, n).weight.dequantize()
            self.assertTrue(q.equal(ref))

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    def test_int4_weight_only(self):
        model = M(n=1024, k=1024).to(torch.bfloat16).to(_DEVICE)
        model.reset_parameters()
        m_ref = copy.deepcopy(model)

        config = Int4WeightOnlyConfig(group_size=self.group_size)
        quantize_(m_ref, config, device=_DEVICE)

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
        self.compare_quantized_models(model, m_ref, quantizer, b)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_6, "Test only enabled for 2.4+")
    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    def test_intx_weight_only(self):
        model = M(n=512, k=512).to(_DEVICE)
        model.reset_parameters()
        m_ref = copy.deepcopy(model)

        config = IntxWeightOnlyConfig(granularity=PerGroup(self.group_size))
        quantize_(m_ref, config, device=_DEVICE)
        b = 8
        q_dtype = torch.int8
        quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[q_dtype]
        quantizer = UnifTorchaoQuantizer(
            symmetric=True,
            target_dtype=q_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=torch.finfo(torch.float32).eps,
            preserve_zero=True,
            zero_point_domain=ZeroPointDomain.INT,
        )
        self.assertTrue(
            quantizer.get_quant_size(b) == max(abs(quant_min), quant_max) + 1
        )
        self.compare_quantized_models(model, m_ref, quantizer, b)


if __name__ == "__main__":
    unittest.main()
