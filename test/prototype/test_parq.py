# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import unittest
from typing import Optional

import torch
from torch import nn
from torch.testing._internal import common_utils

from torchao.core.config import AOBaseConfig
from torchao.dtypes import Int4CPULayout
from torchao.prototype.parq.optim import (
    ProxHardQuant,
    ProxPARQ,
    QuantOptimizer,
)
from torchao.prototype.parq.quant import (
    Int4UnifTorchaoQuantizer,
    LSBQuantizer,
    Quantizer,
    StretchedUnifTorchaoQuantizer,
    TernaryUnifQuantizer,
    UnifQuantizer,
    UnifTorchaoQuantizer,
)
from torchao.prototype.parq.quant.quant_api import StretchedIntxWeightOnlyConfig
from torchao.prototype.parq.quant.uniform_torchao import _BIT_WIDTH_TO_DTYPE
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    _is_linear,
    int4_weight_only,
    quantize_,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.utils import check_cpu_version

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_param_groups(model):
    params_quant, params_no_quant = [], []

    def get_param_groups(model):
        for module in model.children():
            is_linear = _is_linear(module)
            for n, p in module.named_parameters():
                if is_linear and n == "weight":
                    params_quant.append(p)
                else:
                    params_no_quant.append(p)

    get_param_groups(model)
    return params_quant, params_no_quant


def build_param_groups(
    model,
    b: int = 2,
    group_size: Optional[int] = None,
    quantizer: Optional[Quantizer] = None,
):
    params_quant, params_no_quant = split_param_groups(model)
    quant_kwargs = {}
    if group_size:
        quant_kwargs["quant_block_size"] = group_size
    if quantizer is not None:
        quant_kwargs["quantizer"] = quantizer
    return [
        {"params": params_quant, "quant_bits": b, **quant_kwargs},
        {"params": params_no_quant},
    ]


def compare_quantized_models(
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

        # compare to AffineQuantizedTensor instance
        q = q.view(original_shape)
        ref = getattr(m_ref, n).weight.dequantize()
        torch.testing.assert_close(q, ref, atol=0, rtol=0)


def compare_parq_convert(
    model: nn.Module,
    m_ref: nn.Module,
    optimizer: QuantOptimizer,
    config: AOBaseConfig,
):
    # do not update model weights, just quantize
    optimizer.zero_grad()
    optimizer.step()

    orig_model = copy.deepcopy(model)  # save copy of PARQ quantized model

    # equivalent to torchao's convert step
    model.eval()
    optimizer.restore_latent_params()
    quantize_(model, config, filter_fn=optimizer.get_filter_fn(model))

    for n, module in model.named_modules():
        if not _is_linear(module):
            continue

        p_orig = getattr(orig_model, n).weight  # PARQ weight
        p = module.weight.dequantize()  # PARQ weight after quantize_
        p_ref = getattr(m_ref, n).weight.dequantize()  # native quantize_

        torch.testing.assert_true(p_orig, p_ref, atol=0, rtol=0)
        torch.testing.assert_true(p, p_ref, atol=0, rtol=0)


class M(nn.Module):
    def __init__(self, m=256, n=128, k=16, bias=False, embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(10, m) if embedding else nn.Identity()
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
        return (
            torch.randint(1, 10, (1, self.linear1.in_features), device=device)
            if isinstance(self.embedding, nn.Embedding)
            else torch.randn(1, self.linear1.in_features, device=device)
        )

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

    @common_utils.parametrize("b", [0, 1, 2, 4])
    @common_utils.parametrize("unif_quant", [True, False])
    @common_utils.parametrize("hard_prox", [True, False])
    @common_utils.parametrize("per_group_quantizer", [True, False])
    def test_parq_train_loop(
        self, b: int = 2, unif_quant=True, hard_prox=True, per_group_quantizer=False
    ):
        self.model.reset_parameters()
        if unif_quant:
            quantizer = TernaryUnifQuantizer() if b == 0 else UnifQuantizer()
        else:
            quantizer = LSBQuantizer()
        param_groups = build_param_groups(
            self.model, b, quantizer=quantizer if per_group_quantizer else None
        )
        base_optimizer = torch.optim.AdamW(param_groups)

        prox_map = (
            ProxHardQuant() if hard_prox else ProxPARQ(anneal_start=0, anneal_end=2)
        )
        optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map)
        for _ in range(3):
            x = self.model.example_inputs(device=_DEVICE)
            out = self.model(x)
            out.sum().backward()
            optimizer.step()

        for child in self.model.children():
            if isinstance(child, nn.Linear):
                self.assertEqual(
                    child.weight.unique().numel(), quantizer.get_quant_size(b)
                )


class TestUnifTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @common_utils.parametrize("group_size", [32, 256])
    def test_int4_weight_only(self, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE, dtype=torch.bfloat16)
        model.reset_parameters()

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = int4_weight_only(group_size=group_size, version=1)
        if check_cpu_version(_DEVICE):
            config.layout = Int4CPULayout()
        quantize_(m_ref, config)

        b = 4
        compare_quantized_models(
            model, m_ref, Int4UnifTorchaoQuantizer(), b, group_size
        )

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

        quantizer = UnifTorchaoQuantizer()
        compare_quantized_models(model, m_ref, quantizer, b, group_size)

    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    def test_int4_weight_only_e2e(self, group_size: int = 32):
        model = M(m=512, n=512).to(torch.bfloat16).to(_DEVICE)
        model.reset_parameters()

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = int4_weight_only(group_size=group_size, version=1)
        if check_cpu_version(_DEVICE):
            config.layout = Int4CPULayout()
        quantize_(m_ref, config)

        b = 4
        base_optimizer = torch.optim.AdamW(build_param_groups(model, b, group_size))
        optimizer = QuantOptimizer(
            base_optimizer,
            Int4UnifTorchaoQuantizer(),
            ProxHardQuant(),
            quant_per_channel=True,
        )
        compare_parq_convert(model, m_ref, optimizer, config)

    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    @common_utils.parametrize("b", [2, 3, 4, 8])
    def test_intx_weight_only_e2e(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE)
        model.reset_parameters()

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = IntxWeightOnlyConfig(
            weight_dtype=_BIT_WIDTH_TO_DTYPE[b], granularity=PerGroup(group_size)
        )
        quantize_(m_ref, config)

        base_optimizer = torch.optim.AdamW(build_param_groups(model, b, group_size))
        optimizer = QuantOptimizer(
            base_optimizer,
            UnifTorchaoQuantizer(),
            ProxHardQuant(),
            quant_per_channel=True,
        )
        compare_parq_convert(model, m_ref, optimizer, config)


class TestStretchedUnifTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @common_utils.parametrize("b", [2, 3])
    @common_utils.parametrize("group_size", [32, 256])
    def test_intx_weight_only_parq_equivalent(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE)
        model.reset_parameters()

        quantizer_ref = UnifQuantizer()
        quantizer = StretchedUnifTorchaoQuantizer(b)

        for module in model.children():
            if not _is_linear(module):
                continue

            # simulate grouping from QuantOptimizer.step
            p = module.weight
            p = p.view(-1, group_size)

            q_ref, Q_ref = quantizer_ref.quantize(p, b=b, dim=-1)
            q, Q = quantizer.quantize(p, b=b, dim=-1)

            torch.testing.assert_close(q, q_ref, atol=0, rtol=0)
            torch.testing.assert_close(Q, Q_ref, atol=0, rtol=0)

    @common_utils.parametrize("b", [2, 3])
    @common_utils.parametrize("group_size", [32, 512])
    def test_intx_weight_only(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE)
        model.reset_parameters()

        quantizer = StretchedUnifTorchaoQuantizer(b)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        quantize_(
            m_ref,
            StretchedIntxWeightOnlyConfig(
                b=b,
                quant_min=quantizer.quant_min,
                quant_max=quantizer.quant_max,
                granularity=PerGroup(group_size),
            ),
        )

        compare_quantized_models(model, m_ref, quantizer, b, group_size)

    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    @common_utils.parametrize("b", [2, 3])
    def test_intx_weight_only_e2e(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE)
        model.reset_parameters()

        quantizer = StretchedUnifTorchaoQuantizer(b)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = StretchedIntxWeightOnlyConfig(
            b=b,
            quant_min=quantizer.quant_min,
            quant_max=quantizer.quant_max,
            granularity=PerGroup(group_size),
        )
        quantize_(m_ref, config)

        base_optimizer = torch.optim.AdamW(build_param_groups(model, b, group_size))
        optimizer = QuantOptimizer(
            base_optimizer,
            quantizer,
            ProxHardQuant(),
            quant_per_channel=True,
        )
        compare_parq_convert(model, m_ref, optimizer, config)


class TestInt8DynamicActivationTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @common_utils.parametrize("b", [2, 3, 4, 8])
    @common_utils.parametrize("model_dtype", [torch.float16, torch.float32])
    @common_utils.parametrize("group_size", [32, 128])
    def test_int8_dynamic_activation_intx_e2e(
        self,
        b: int = 2,
        model_dtype: torch.dtype = torch.float32,
        group_size: int = 32,
    ):
        model = M(embedding=False).to(_DEVICE, dtype=model_dtype)
        x = model.example_inputs(device=_DEVICE).to(model_dtype)

        # reference model using native quantization
        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        quantizer = UnifTorchaoQuantizer()
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=_BIT_WIDTH_TO_DTYPE[b],
            weight_granularity=PerGroup(group_size),
            weight_mapping_type=quantizer.mapping_type,
            act_mapping_type=MappingType.ASYMMETRIC,
        )
        quantize_(m_ref, config)
        ref_out = m_ref(x)

        # quantize weights with PARQ
        base_optimizer = torch.optim.SGD(build_param_groups(model, b, group_size))
        optimizer = QuantOptimizer(
            base_optimizer, quantizer, ProxHardQuant(), quant_per_channel=True
        )
        optimizer.zero_grad()
        optimizer.step()

        # apply torchao quantized activations on top
        activation_config = IntxFakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=False
        )
        qat_config = QATConfig(activation_config=activation_config, step="prepare")
        filter_fn = optimizer.get_filter_fn(model)
        quantize_(model, qat_config, filter_fn=filter_fn)
        out = model(x)
        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)

        # equivalent to torchao's convert step
        model.eval()
        optimizer.restore_latent_params()
        quantize_(model, QATConfig(config, step="convert"), filter_fn=filter_fn)
        converted_out = model(x)
        torch.testing.assert_close(converted_out, ref_out, atol=0, rtol=0)


common_utils.instantiate_parametrized_tests(TestPARQuantization)
common_utils.instantiate_parametrized_tests(TestUnifTorchaoQuantizer)
common_utils.instantiate_parametrized_tests(TestInt8DynamicActivationTorchaoQuantizer)


if __name__ == "__main__":
    unittest.main()
