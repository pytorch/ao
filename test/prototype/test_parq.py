# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import tempfile
import unittest
from typing import Optional

import torch
from torch import nn
from torch.testing._internal import common_utils

from torchao.dtypes import Int4CPULayout
from torchao.prototype.parq.optim import (
    ProxHardQuant,
    ProxPARQ,
    QuantOptimizer,
)
from torchao.prototype.parq.quant import (
    Int4UnifTorchaoQuantizer,
    LSBQuantizer,
    StretchedIntxWeightConfig,
    StretchedUnifTorchaoQuantizer,
    TernaryUnifQuantizer,
    UnifQuantizer,
    UnifTorchaoQuantizer,
)
from torchao.prototype.parq.quant.config_torchao import (
    TRANSFORMERS_AVAIL,
    _attach_hf_quantization_config,
    _is_hf_model,
)
from torchao.prototype.parq.quant.uniform_torchao import _BIT_WIDTH_TO_DTYPE
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    _is_linear,
    quantize_,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows import IntxUnpackedToInt8Tensor
from torchao.utils import (
    _is_fbgemm_gpu_genai_available,
    check_cpu_version,
    is_sm_at_least_90,
    torch_version_at_least,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class M(nn.Module):
    _tied_weights_keys: list[str] = []

    def __init__(
        self, m=256, n=128, k=16, bias=False, embedding=True, tied_weights=False
    ):
        nn.Module.__init__(self)
        self.embed_tokens = nn.Embedding(k, m) if embedding else nn.Identity()
        self.linear1 = nn.Linear(m, n, bias=bias)
        self.linear2 = nn.Linear(n, k, bias=bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if embedding and tied_weights:
            assert self.embed_tokens.weight.shape == self.linear2.weight.shape
            self.tie_weights()
            self._tied_weights_keys.append("linear2.weight")

    def tie_weights(self):
        self.linear2.weight = self.embed_tokens.weight

    def example_inputs(self, device=None):
        if isinstance(self.embed_tokens, nn.Identity):
            inputs = torch.randn(1, self.linear1.in_features, device=device)
        else:
            k = self.embed_tokens.num_embeddings
            inputs = torch.randint(1, k, (1, self.linear1.in_features), device=device)
        return inputs

    def forward(self, x):
        x = self.embed_tokens(x)
        x = self.relu(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x


if TRANSFORMERS_AVAIL:
    from transformers import PretrainedConfig, PreTrainedModel, TorchAoConfig

    class MConfig(PretrainedConfig):
        def __init__(
            self,
            m=256,
            n=128,
            k=16,
            bias=False,
            embedding=True,
            tied_weights=False,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.m = m
            self.n = n
            self.k = k
            self.bias = bias
            self.embedding = embedding
            self.tied_weights = tied_weights

    class PreTrainedM(M, PreTrainedModel):
        base_model_prefix = "base"
        config_class = MConfig

        def __init__(self, config: MConfig):
            PreTrainedModel.__init__(self, config)
            M.__init__(
                self,
                m=config.m,
                n=config.n,
                k=config.k,
                bias=config.bias,
                embedding=config.embedding,
                tied_weights=config.tied_weights,
            )

        def get_input_embeddings(self) -> nn.Module:
            return self.embed_tokens


def split_param_groups(model) -> tuple[list, list, list]:
    params_quant, params_embed, params_no_quant = [], [], []

    def get_param_groups(model):
        seen_data_ptrs = set()  # avoid duplicates in case of tied weights
        for module in model.children():
            is_linear = _is_linear(module)
            for n, p in module.named_parameters():
                if n == "weight":
                    data_ptr = p.data_ptr()
                    if data_ptr in seen_data_ptrs:
                        continue
                    seen_data_ptrs.add(data_ptr)

                if is_linear and n == "weight":
                    params_quant.append(p)
                elif isinstance(module, nn.Embedding) and n == "weight":
                    params_embed.append(p)
                else:
                    params_no_quant.append(p)

    get_param_groups(model)
    return params_quant, params_embed, params_no_quant


def build_param_groups(
    model,
    b: int = 2,
    group_size: Optional[int] = None,
    embed_b: int = 4,
):
    params_quant, params_embed, params_no_quant = split_param_groups(model)
    quant_kwargs = {}
    if group_size:
        quant_kwargs["quant_block_size"] = group_size
    param_groups = [
        {"params": params_quant, "quant_bits": b, **quant_kwargs},
        {"params": params_no_quant},
    ]
    if params_embed:
        param_groups.append({"params": params_embed, "quant_bits": embed_b})
    return param_groups


def get_optim_kwargs(
    model, base_optimizer, embedding=True, quant_cls=UnifTorchaoQuantizer
):
    optim_kwargs = {}
    if embedding:
        embed_data_ptrs = set(
            (
                m.weight.data_ptr()
                for m in model.modules()
                if isinstance(m, nn.Embedding)
            )
        )
        group_idx = -1
        for i, group in enumerate(base_optimizer.param_groups):
            if all(p.data_ptr() in embed_data_ptrs for p in group["params"]):
                group_idx = i
                break
        assert group_idx > -1
        optim_kwargs["group_quantizer_map"] = {group_idx: quant_cls()}
    return optim_kwargs


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
    weight_only: bool = False,
):
    # do not update model weights, just quantize
    optimizer.zero_grad()
    optimizer.step()

    orig_model = copy.deepcopy(model)  # save copy of PARQ quantized model

    # equivalent to torchao's convert step
    optimizer.torchao_convert(model, weight_only=weight_only, embed_weight_only=True)

    inputs = model.example_inputs(device=_DEVICE)
    torch.testing.assert_close(model(inputs), orig_model(inputs))

    for n, module in model.named_modules():
        if not _is_linear(module):
            continue

        p_orig = getattr(orig_model, n).weight  # PARQ weight
        p_ref = getattr(m_ref, n).weight.dequantize()  # native quantize_
        torch.testing.assert_close(p_orig, p_ref, atol=0, rtol=0)

        p = module.weight.dequantize()  # PARQ weight after quantize_
        torch.testing.assert_close(p, p_ref, atol=0, rtol=0)


def check_torchao_tensor_subclass(
    test_case: common_utils.TestCase, model: nn.Module, weight_only: bool = False
):
    for name, module in model.named_modules():
        if not hasattr(module, "weight") or f"{name}.weight" in getattr(
            model, "_tied_weights_keys", []
        ):
            continue

        if not weight_only and _is_linear(module):
            test_case.assertTrue(isinstance(module.weight, IntxUnpackedToInt8Tensor))
            test_case.assertTrue(
                module.weight.activation_quantization == "int8_asym_per_token"
            )
        elif weight_only and _is_linear(module) or isinstance(module, nn.Embedding):
            test_case.assertTrue(isinstance(module.weight, IntxUnpackedToInt8Tensor))
            test_case.assertTrue(module.weight.activation_quantization is None)


def apply_activation_quantization(
    model: nn.Module, optimizer: torch.optim.Optimizer, model_dtype: torch.dtype
):
    # apply torchao quantized activations on top
    activation_config = IntxFakeQuantizeConfig(
        torch.int8, "per_token", is_symmetric=False, scale_precision=model_dtype
    )
    qat_config = QATConfig(activation_config=activation_config, step="prepare")
    for filter_fn in optimizer.get_filter_fns(model):
        try:
            quantize_(model, qat_config, filter_fn=filter_fn)
        except ValueError as e:
            if str(e) == "Activation fake quantization is not supported for embedding":
                pass


class TestPARQuantization(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @common_utils.parametrize("b", [0, 1, 2, 4])
    @common_utils.parametrize("unif_quant", [True, False])
    @common_utils.parametrize("hard_prox", [True, False])
    @common_utils.parametrize("per_group_quantizer", [True, False])
    def test_parq_train_loop(
        self, b: int = 2, unif_quant=True, hard_prox=True, per_group_quantizer=False
    ):
        model = M(bias=True).to(_DEVICE)
        if unif_quant:
            quantizer = TernaryUnifQuantizer() if b == 0 else UnifQuantizer()
        else:
            quantizer = LSBQuantizer()
        param_groups = build_param_groups(model, b, embed_b=b)
        base_optimizer = torch.optim.AdamW(param_groups)

        prox_map = (
            ProxHardQuant() if hard_prox else ProxPARQ(anneal_start=0, anneal_end=2)
        )
        optim_kwargs = get_optim_kwargs(
            model, base_optimizer, quant_cls=type(quantizer), embedding=False
        )
        optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map, **optim_kwargs)
        for _ in range(3):
            x = model.example_inputs(device=_DEVICE)
            out = model(x)
            out.sum().backward()
            optimizer.step()

        for child in model.children():
            if isinstance(child, nn.Linear):
                self.assertEqual(
                    child.weight.unique().numel(), quantizer.get_quant_size(b)
                )


class TestUnifTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @unittest.skipIf(not torch_version_at_least("2.8.0"), "Need pytorch >= 2.8.0")
    @unittest.skipIf(not is_sm_at_least_90(), "Need sm >= 90")
    @unittest.skipIf(
        not _is_fbgemm_gpu_genai_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
    )
    @common_utils.parametrize("group_size", [32, 256])
    def test_int4_weight_only(self, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE, dtype=torch.bfloat16)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = Int4WeightOnlyConfig(group_size=group_size)
        if check_cpu_version(_DEVICE):
            config.layout = Int4CPULayout()
            config.version = 1
        quantize_(m_ref, config)

        b = 4
        compare_quantized_models(
            model, m_ref, Int4UnifTorchaoQuantizer(), b, group_size
        )

    @common_utils.parametrize("b", [2, 3, 4, 8])
    @common_utils.parametrize("group_size", [32, 512])
    def test_intx_weight_only(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        quantize_(
            m_ref,
            IntxWeightOnlyConfig(
                weight_dtype=_BIT_WIDTH_TO_DTYPE[b], granularity=PerGroup(group_size)
            ),
        )

        quantizer = UnifTorchaoQuantizer()
        compare_quantized_models(model, m_ref, quantizer, b, group_size)

    @unittest.skipIf(not torch_version_at_least("2.8.0"), "Need pytorch >= 2.8.0")
    @unittest.skipIf(not is_sm_at_least_90(), "Need sm >= 90")
    @unittest.skipIf(
        not _is_fbgemm_gpu_genai_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
    )
    def test_int4_weight_only_e2e(self, group_size: int = 32):
        model = M(m=512, n=512, embedding=False).to(torch.bfloat16).to(_DEVICE)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = Int4WeightOnlyConfig(group_size=group_size)
        quantize_(m_ref, config)

        b = 4
        base_optimizer = torch.optim.AdamW(build_param_groups(model, b, group_size))
        optim_kwargs = get_optim_kwargs(model, base_optimizer, embedding=False)
        optimizer = QuantOptimizer(
            base_optimizer,
            Int4UnifTorchaoQuantizer(),
            ProxHardQuant(),
            quant_per_channel=True,
            **optim_kwargs,
        )
        compare_parq_convert(model, m_ref, optimizer, weight_only=True)

    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    @common_utils.parametrize("b", [2, 3, 4, 8])
    def test_intx_weight_only_e2e(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512, embedding=False).to(_DEVICE)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = IntxWeightOnlyConfig(
            weight_dtype=_BIT_WIDTH_TO_DTYPE[b], granularity=PerGroup(group_size)
        )
        quantize_(m_ref, config)

        base_optimizer = torch.optim.AdamW(build_param_groups(model, b, group_size))
        optim_kwargs = get_optim_kwargs(model, base_optimizer, embedding=False)
        optimizer = QuantOptimizer(
            base_optimizer,
            UnifTorchaoQuantizer(),
            ProxHardQuant(),
            quant_per_channel=True,
            **optim_kwargs,
        )
        compare_parq_convert(model, m_ref, optimizer, weight_only=True)
        check_torchao_tensor_subclass(self, model, weight_only=True)


class TestStretchedUnifTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @common_utils.parametrize("b", [2, 3])
    @common_utils.parametrize("group_size", [32, 256])
    def test_intx_weight_only_parq_equivalent(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512).to(_DEVICE)

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

        quantizer = StretchedUnifTorchaoQuantizer(b)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        quantize_(
            m_ref,
            StretchedIntxWeightConfig(
                b=b,
                quant_min=quantizer.quant_min,
                quant_max=quantizer.quant_max,
                granularity=PerGroup(group_size),
                activation_quantization=None,
            ),
        )

        compare_quantized_models(model, m_ref, quantizer, b, group_size)

    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    @common_utils.parametrize("b", [2, 3])
    def test_intx_weight_only_e2e(self, b: int = 2, group_size: int = 32):
        model = M(m=512, n=512, embedding=False).to(_DEVICE)

        quantizer = StretchedUnifTorchaoQuantizer(b)

        m_ref = copy.deepcopy(model).eval().to(_DEVICE)
        config = StretchedIntxWeightConfig(
            b=b,
            quant_min=quantizer.quant_min,
            quant_max=quantizer.quant_max,
            granularity=PerGroup(group_size),
            activation_quantization=None,
        )
        quantize_(m_ref, config, filter_fn=_is_linear)

        base_optimizer = torch.optim.AdamW(build_param_groups(model, b, group_size))
        optim_kwargs = get_optim_kwargs(model, base_optimizer, embedding=False)
        optimizer = QuantOptimizer(
            base_optimizer,
            quantizer,
            ProxHardQuant(),
            quant_per_channel=True,
            **optim_kwargs,
        )
        compare_parq_convert(model, m_ref, optimizer, weight_only=True)
        check_torchao_tensor_subclass(self, model, weight_only=True)

    @common_utils.parametrize("b", [2, 3])
    @common_utils.parametrize(
        "model_dtype", [torch.float16, torch.float32, torch.bfloat16]
    )
    def test_intx_weight_only_tied_embed_linear(
        self, b: int = 2, model_dtype: torch.dtype = torch.float32
    ):
        model = M(m=256, n=256, tied_weights=True).to(_DEVICE)

        quantizer = StretchedUnifTorchaoQuantizer(b)
        base_optimizer = torch.optim.SGD(build_param_groups(model, b))
        optim_kwargs = get_optim_kwargs(model, base_optimizer)
        optimizer = QuantOptimizer(
            base_optimizer,
            quantizer,
            ProxHardQuant(),
            quant_per_channel=True,
            **optim_kwargs,
        )
        optimizer.zero_grad()
        optimizer.step()

        apply_activation_quantization(model, optimizer, model_dtype)
        optimizer.torchao_convert(model, embed_weight_only=True)
        check_torchao_tensor_subclass(self, model)
        self.assertTrue(
            torch.equal(model.embed_tokens.weight.qdata, model.linear2.weight.qdata)
        )


class TestInt8DynamicActivationTorchaoQuantizer(common_utils.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @unittest.skipIf(_DEVICE == "cpu", "Need GPU available")
    @unittest.skipIf(not TRANSFORMERS_AVAIL, "Need transformers")
    @common_utils.parametrize("b", [2, 3, 4, 8])
    @common_utils.parametrize(
        "model_dtype", [torch.float16, torch.float32, torch.bfloat16]
    )
    @common_utils.parametrize("group_size", [32, 128])
    def test_int8_dynamic_activation_intx_e2e(
        self,
        b: int = 2,
        model_dtype: torch.dtype = torch.float32,
        group_size: int = 32,
    ):
        config = MConfig(embedding=False, bias=True)
        model = PreTrainedM(config).to(_DEVICE, dtype=model_dtype)
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
        optim_kwargs = get_optim_kwargs(model, base_optimizer, embedding=False)
        optimizer = QuantOptimizer(
            base_optimizer,
            quantizer,
            ProxHardQuant(),
            quant_per_channel=True,
            **optim_kwargs,
        )

        optimizer.zero_grad()
        optimizer.step()

        apply_activation_quantization(model, optimizer, model_dtype)

        out = model(x)
        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)

        attach_hf_config = False
        if TRANSFORMERS_AVAIL:
            attach_hf_config = _is_hf_model(model)
            self.assertTrue(attach_hf_config)

        optimizer.torchao_convert(model)
        converted_out = model(x)
        torch.testing.assert_close(converted_out, ref_out)
        check_torchao_tensor_subclass(self, model)

        if attach_hf_config:
            reg_param_names = {
                n for n, m in model.named_modules() if isinstance(m, nn.Embedding)
            }
            reg_param_names.add("_default")
            module_fqn_to_config = (
                model.config.quantization_config.quant_type.module_fqn_to_config
            )
            self.assertEqual(set(module_fqn_to_config.keys()), reg_param_names)
            for torchao_config in module_fqn_to_config.values():
                self.assertTrue(isinstance(torchao_config, config.__class__))


class TestTorchAoConfigIntegration(common_utils.TestCase):
    @unittest.skipIf(torch.backends.mps.is_available(), "MPS not supported")
    @unittest.skipIf(not TRANSFORMERS_AVAIL, "Need transformers")
    def test_tied_weights_quantization(self, b: int = 4):
        config = MConfig(m=128, n=128, tied_weights=True)
        model = PreTrainedM(config).to(_DEVICE)

        quantizer = StretchedUnifTorchaoQuantizer(b)
        linear_config = StretchedIntxWeightConfig(
            b=b,
            quant_min=quantizer.quant_min,
            quant_max=quantizer.quant_max,
            granularity=PerAxis(0),
        )
        embed_config = IntxWeightOnlyConfig(
            weight_dtype=_BIT_WIDTH_TO_DTYPE[b], granularity=PerGroup(32)
        )
        module_to_config = {"_default": linear_config}
        configs = [embed_config]
        filter_fns = [lambda m: isinstance(m, nn.Embedding)]
        _attach_hf_quantization_config(model, filter_fns, configs, module_to_config)

        quantization_config = getattr(model.config, "quantization_config", None)
        self.assertTrue(isinstance(quantization_config, TorchAoConfig))
        self.assertTrue(quantization_config.modules_to_not_convert == ["linear2"])

        # Let HF apply quantize_ given quantization_config
        del model.config.quantization_config
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=False)
            model = PreTrainedM.from_pretrained(
                tmp_dir, quantization_config=quantization_config
            )

        check_torchao_tensor_subclass(self, model.linear1)
        check_torchao_tensor_subclass(self, model.linear2, weight_only=True)
        check_torchao_tensor_subclass(self, model.embed_tokens, weight_only=True)

        self.assertTrue(
            model.linear2.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
        )


common_utils.instantiate_parametrized_tests(TestPARQuantization)
common_utils.instantiate_parametrized_tests(TestUnifTorchaoQuantizer)
common_utils.instantiate_parametrized_tests(TestInt8DynamicActivationTorchaoQuantizer)


if __name__ == "__main__":
    unittest.main()
