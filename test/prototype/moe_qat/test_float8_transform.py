import copy

import pytest
import torch
import torch.nn.functional as F

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.qat import QATStep
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_api import quantize_

from .reference_moe import MoE
from .testing_utils import _moe_input, _expert_weight_filter, _set_seed, device, moe_model, use_grouped_mm, weight_config


class TestFloat8MoEQAT:
    """Tests for FP8 row-wise MoE QAT — model transform."""

    def _prepare(self, model, config, activation_config=None):
        qat_config = MoEQATConfig(
            activation_config=activation_config,
            weight_config=config,
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        return model

    def _convert(self, model):
        qat_config = MoEQATConfig(step="convert")
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        return model

    # =========================================================================
    # Prepare / convert lifecycle
    # =========================================================================

    def test_prepare_wraps_expert_weights(self, moe_model, weight_config):
        model = self._prepare(moe_model, weight_config)

        wrapped_count = 0
        for name, param in model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped_count += 1
        assert wrapped_count == 3, f"Expected 3 wrapped params, got {wrapped_count}"

    def test_prepare_skips_non_expert_params(self, moe_model, weight_config):
        model = self._prepare(moe_model, weight_config)

        wrapped = 0
        for name, param in model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped += 1
                assert param.ndim == 3, f"Wrapped param {name} should be 3D, got {param.ndim}D"
        assert wrapped == 3, f"All 3D expert params should be wrapped, got {wrapped}"
        assert not isinstance(
            model.router.gate.weight.data, Float8FakeQuantizedWeightWrapperTensor
        ), "router.gate.weight should not be wrapped"

    def test_convert_unwraps(self, moe_model, weight_config):
        self._prepare(moe_model, weight_config)
        self._convert(moe_model)

        wrapped_count = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, FakeQuantizedWeightWrapperBaseTensor):
                wrapped_count += 1
        assert wrapped_count == 0, f"{wrapped_count} parameters should not be wrapped after convert"

    def test_convert_round_trip(self, moe_model, weight_config):
        """Prepare → Convert should produce identical weights to the original model."""
        original = copy.deepcopy(moe_model)
        self._prepare(moe_model, weight_config)
        self._convert(moe_model)
        for (name, param), (orig_name, orig_param) in zip(
            moe_model.named_parameters(), original.named_parameters()
        ):
            assert torch.equal(param, orig_param), f"{name} should match after convert"

    def test_bias_bypass(self, weight_config):
        """Wrapped bias in addmm / F.linear is unconditionally bypassed."""
        w = torch.randn(64, 128)
        w_wrapped = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        A = torch.randn(16, 64)

        bias = torch.randn(128)
        bias_wrapped = Float8FakeQuantizedWeightWrapperTensor(bias, weight_config=weight_config)
        out_wrapped = torch.addmm(bias_wrapped, A, w_wrapped)
        out_ref = torch.addmm(bias, A, w_wrapped)
        assert torch.equal(out_wrapped, out_ref), "addmm bias should not be fake-quantized"

        w2 = torch.randn(128, 64)
        w2_wrapped = Float8FakeQuantizedWeightWrapperTensor(w2, weight_config=weight_config)
        bias2 = torch.randn(128)
        bias2_wrapped = Float8FakeQuantizedWeightWrapperTensor(bias2, weight_config=weight_config)
        out_wrapped2 = F.linear(A, w2_wrapped, bias2_wrapped)
        out_ref2 = F.linear(A, w2_wrapped, bias2)
        assert torch.equal(out_wrapped2, out_ref2), "linear bias should not be fake-quantized"

    def test_meta_weights(self, weight_config):
        """Prepare on meta device → copy_ real weights → forward works without extra code."""
        with torch.device("meta"):
            w = torch.randn(64, 128)
            wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        real_w = torch.randn(64, 128)
        wrapper.copy_(real_w)
        A = torch.randn(16, 64)
        out = torch.mm(A, wrapper)
        assert out.shape == (16, 128)

    def test_wrapper_repr(self, weight_config):
        w = torch.randn(64, 128)
        base = FakeQuantizedWeightWrapperBaseTensor(w, weight_config=weight_config)
        r = repr(base)
        assert "FakeQuantizedWeightWrapperBaseTensor" in r
        assert "weight_config" in r
        f8 = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        r = repr(f8)
        assert "Float8FakeQuantizedWeightWrapperTensor" in r
        assert "weight_config" in r

    # =========================================================================
    # Config validation
    # =========================================================================

    def test_config_step_validation(self):
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
        MoEQATConfig(weight_config=weight_config, step="prepare")
        MoEQATConfig(step="convert")
        with pytest.raises(ValueError, match="`step` must be one of"):
            MoEQATConfig(weight_config=weight_config, step="blah")

    def test_config_requires_weight_config(self):
        with pytest.raises(ValueError, match="Must specify"):
            MoEQATConfig(step="prepare")

    def test_config_rejects_base_config_in_convert(self):
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
        MoEQATConfig(weight_config=weight_config, step="prepare")
        base_config = Float8DynamicActivationFloat8WeightConfig()
        with pytest.raises(NotImplementedError, match="Applying PTQ"):
            MoEQATConfig(base_config=base_config, step="convert")

    def test_config_infer_from_base_config(self):
        """MoEQATConfig can infer fake quantize configs from a PTQ base_config."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        base_config = Float8DynamicActivationFloat8WeightConfig()
        qat_config = MoEQATConfig(base_config=base_config, step="prepare")
        assert isinstance(qat_config.weight_config, Float8FakeQuantizeConfig)
        assert isinstance(qat_config.activation_config, Float8FakeQuantizeConfig)
        assert qat_config.base_config is None

    def test_config_prepare_with_base_config(self, moe_model):
        """Model can be prepared using base_config instead of explicit weight_config."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        base_config = Float8DynamicActivationFloat8WeightConfig()
        qat_config = MoEQATConfig(
            base_config=base_config,
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        )
        model = copy.deepcopy(moe_model)
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        wrapped = sum(
            1 for _, p in model.named_parameters()
            if isinstance(p.data, Float8FakeQuantizedWeightWrapperTensor)
        )
        assert wrapped == 3, f"Expected 3 wrapped params, got {wrapped}"

    def test_config_rejects_invalid_base_config(self):
        """Only Float8DynamicActivationFloat8WeightConfig is accepted as base_config."""
        from torchao.quantization import Int4WeightOnlyConfig
        base_config = Int4WeightOnlyConfig(group_size=32)
        with pytest.raises(ValueError, match="Only `Float8DynamicActivationFloat8WeightConfig`"):
            MoEQATConfig(base_config=base_config, step="prepare")

    def test_config_rejects_non_float8_config(self):
        from torchao.quantization.qat import IntxFakeQuantizeConfig
        intx_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")
        with pytest.raises(ValueError, match="Only `Float8FakeQuantizeConfig`"):
            MoEQATConfig(weight_config=intx_config, step="prepare")

    @pytest.mark.parametrize("granularity", [PerRow(), PerTensor()])
    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_weight_config_variants(self, granularity, dtype):
        """All Float8FakeQuantizeConfig variants should be accepted."""
        config = Float8FakeQuantizeConfig(dtype=dtype, granularity=granularity)
        qat_config = MoEQATConfig(weight_config=config, step="prepare")
        assert qat_config.step == QATStep.PREPARE

    def test_default_params_filter(self, moe_model, weight_config):
        """Default filter (_is_parameter) wraps all parameters including 2D gate."""
        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        wrapped_count = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped_count += 1
        assert wrapped_count == 4, f"Expected 4 wrapped params, got {wrapped_count}"
