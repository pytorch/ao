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
from torchao.dtypes import Int4CPULayout
from torchao.prototype.parq.optim import (
    ProxHardQuant,
    ProxPARQ,
    QuantOptimizer,
)
from torchao.prototype.parq.quant import (
    Int4UnifTorchaoQuantizer,
    LSBQuantizer,
    UnifQuantizer,
    UnifTorchaoQuantizer,
)
from torchao.prototype.parq.quant.uniform_torchao import _BIT_WIDTH_TO_DTYPE
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import (
    IntxWeightOnlyConfig,
    _is_linear,
    int4_weight_only,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_6,
    check_cpu_version,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_param_groups(model):
    params_quant, params_no_quant = [], []

    def get_param_groups(model):
        for module in model.children():
            is_linear = isinstance(module, nn.Linear)
            for n, p in module.named_parameters(recurse=False):
                if is_linear and n == "weight":
                    params_quant.append(p)
                else:
                    params_no_quant.append(p)
            get_param_groups(module)

    get_param_groups(model)
    return params_quant, params_no_quant


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

    def example_inputs(self, device=None):
        return torch.randint(1, 10, (1, 256), device=device)

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
        self.params_quant, self.params_no_quant = split_param_groups(self.model)

    def train_loop(self, optimizer, steps=1):
        for _ in range(steps):
            x = self.model.example_inputs(device=_DEVICE)
            out = self.model(x)
            out.sum().backward()
            optimizer.step()

    def test_2bit_unif_quantizer_hard_prox(self):
        b = 2
        self.model.reset_parameters()
        param_groups = [
            {"params": self.params_quant, "quant_bits": b},
            {"params": self.params_no_quant},
        ]
        base_optimizer = torch.optim.AdamW(param_groups)
        quantizer = UnifQuantizer()
        optimizer = QuantOptimizer(base_optimizer, quantizer, ProxHardQuant())
        self.train_loop(optimizer)

        for child in self.model.children():
            if isinstance(child, nn.Linear):
                self.assertEqual(
                    child.weight.unique().numel(), quantizer.get_quant_size(b)
                )

    def test_ternarybit_lsbq_parq_prox(self):
        b = 0
        self.model.reset_parameters()
        param_groups = [
            {"params": self.params_quant, "quant_bits": b},
            {"params": self.params_no_quant},
        ]
        base_optimizer = torch.optim.AdamW(param_groups)
        quantizer = LSBQuantizer()
        optimizer = QuantOptimizer(
            base_optimizer, quantizer, ProxPARQ(anneal_start=0, anneal_end=2)
        )
        self.train_loop(optimizer, steps=3)

        for child in self.model.children():
            if isinstance(child, nn.Linear):
                self.assertEqual(
                    child.weight.unique().numel(), quantizer.get_quant_size(b)
                )


class TestUnifTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def compare_quantized_models(
        self,
        model: nn.Module,
        m_ref: nn.Module,
        quantizer: UnifTorchaoQuantizer,
        b: int,
        group_size: int,
    ):
        for n, module in model.named_children():
            if not _is_linear(module):
                continue

            # simulate grouping from QuantOptimizer.step
            p = module.weight
            original_shape = p.shape
            p = p.view(-1, group_size)

            q, Q = quantizer.quantize(p, b=b, dim=-1)
            q = q.view(original_shape)

            # compare to AffineQuantizedTensor instance
            ref = getattr(m_ref, n).weight.dequantize()
            self.assertTrue(q.equal(ref))

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @common_utils.parametrize("group_size", [32, 256])
    def test_int4_weight_only(self, group_size: int = 32):
        model = M(m=512, n=512).to(torch.bfloat16).to(_DEVICE)
        model.reset_parameters()

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = int4_weight_only(group_size=group_size)
        if check_cpu_version(_DEVICE):
            config.layout = Int4CPULayout()
        quantize_(m_ref, config)

        b = 4
        self.compare_quantized_models(
            model, m_ref, Int4UnifTorchaoQuantizer(), b, group_size
        )

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_6, "Test only enabled for 2.6+")
    @common_utils.parametrize("b", [2, 3, 4, 8])
    @common_utils.parametrize("group_size", [32, 512])
    def test_intx_weight_only(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE)
        model.reset_parameters()

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        quantize_(
            m_ref,
            IntxWeightOnlyConfig(
                weight_dtype=_BIT_WIDTH_TO_DTYPE[b], granularity=PerGroup(group_size)
            ),
        )

        quantizer = UnifTorchaoQuantizer(symmetric=True)
        self.compare_quantized_models(model, m_ref, quantizer, b, group_size)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    def test_int4_weight_only_e2e(self, group_size: int = 32):
        model = M(m=512, n=512).to(torch.bfloat16).to(_DEVICE)
        model.reset_parameters()

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = int4_weight_only(group_size=group_size)
        if check_cpu_version(_DEVICE):
            config.layout = Int4CPULayout()
        quantize_(m_ref, config)

        b = 4
        params_quant, params_no_quant = split_param_groups(model)
        param_groups = [
            {"params": params_quant, "quant_bits": b, "quant_block_size": group_size},
            {"params": params_no_quant},
        ]
        base_optimizer = torch.optim.AdamW(param_groups)

        optimizer = QuantOptimizer(
            base_optimizer,
            Int4UnifTorchaoQuantizer(),
            ProxHardQuant(),
            quant_per_channel=True,
        )

        # do not update model weights, just quantize
        optimizer.zero_grad()
        optimizer.step()

        orig_model = copy.deepcopy(model)  # save copy of PARQ quantized model

        # equivalent to torchao's convert step
        model.eval()
        optimizer.torchao_quantize_(model, config)

        for n, module in model.named_modules():
            if not _is_linear(module):
                continue

            p_orig = getattr(orig_model, n).weight  # PARQ weight
            p = module.weight.dequantize()  # PARQ weight after quantize_
            p_ref = getattr(m_ref, n).weight.dequantize()  # natively quantize_

            self.assertTrue(p_orig.equal(p_ref))
            self.assertTrue(p.equal(p_ref))


common_utils.instantiate_parametrized_tests(TestPARQuantization)
common_utils.instantiate_parametrized_tests(TestUnifTorchaoQuantizer)


if __name__ == "__main__":
    unittest.main()
