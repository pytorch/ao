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
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_

from .reference_moe import MoE
from .testing_utils import _moe_input, _expert_weight_filter, create_moe_model, target_devices

from torchao.prototype.moe_qat.config import _is_expert
from torchao.prototype.moe_qat.quantize import (
    _is_parameter,
    _is_parameter_with_wrapped_data,
    _replace_params_with_custom_fn_if_matches_filter,
)


# =========================================================================
# Test _replace_params_with_custom_fn_if_matches_filter
# =========================================================================


def test_replace_params_filters_and_replaces():
    """_replace_params_with_custom_fn_if_matches_filter: filter selects params, replacement is applied."""

    recorded = []

    def my_filter(param, fqn):
        return "weight" in fqn  # match only weight, not bias

    def my_replacement(module, param_fqn, param, extra_args):
        recorded.append((param_fqn, extra_args))
        return torch.nn.Parameter(param.data.clone() * 2)

    model = torch.nn.Linear(2, 2)  # has "weight" and "bias"
    original_weight = model.weight.data.clone()
    original_bias = model.bias.data.clone()
    _replace_params_with_custom_fn_if_matches_filter(
        model, my_replacement, my_filter, extra_args=(42,)
    )

    # Only weight matched, bias skipped
    assert len(recorded) == 1, f"Expected 1 match, got {len(recorded)}"
    assert recorded[0][0] == "weight"
    assert recorded[0][1] == (42,)

    # Weight was replaced with 2× original value, bias was unchanged
    assert torch.equal(model.weight.data, original_weight * 2)
    assert torch.equal(model.bias.data, original_bias)


def test_replace_params_default_filter():
    """Default filter (None) uses _is_parameter, which wraps all nn.Parameters."""

    called = []

    def replacement(module, param_fqn, param, extra_args):
        called.append(param_fqn)
        return param  # no change

    model = torch.nn.Linear(2, 2)
    _replace_params_with_custom_fn_if_matches_filter(model, replacement, None)

    assert len(called) == 2, f"Expected weight + bias, got {len(called)}"
    assert any("weight" in f for f in called)
    assert any("bias" in f for f in called)


def test_replace_params_recursive():
    """_replace_params_with_custom_fn_if_matches_filter recurses into submodules."""

    called = []

    def my_filter(param, fqn):
        return True

    def my_replacement(module, param_fqn, param, extra_args):
        called.append(param_fqn)
        return param

    inner = torch.nn.Linear(2, 2)
    outer = torch.nn.Sequential(inner)
    _replace_params_with_custom_fn_if_matches_filter(outer, my_replacement, my_filter)

    # inner.weight, inner.bias — cur_fqn should include the outer prefix
    assert any("0.weight" in f for f in called)
    assert any("0.bias" in f for f in called)



# =========================================================================
# Prepare / convert lifecycle
# =========================================================================

@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("weight_config, wrapper_cls", [
    (Float8FakeQuantizeConfig(), Float8FakeQuantizedWeightWrapperTensor),
])
def test_prepare_wraps_expert_weights(device, weight_config, wrapper_cls):
    """Prepare wraps expert weights with the configured tensor subclass."""
    # use_grouped_mm only affects the forward computation path — no forward run here.
    moe_model = create_moe_model(device, use_grouped_mm=True)
    orig_values = {name: param.data.clone() for name, param in moe_model.named_parameters()}

    qat_config = MoEQATConfig(
        weight_config=weight_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped_count = 0
    for name, param in moe_model.named_parameters():
        if isinstance(param.data, wrapper_cls):
            wrapped_count += 1
            assert torch.equal(param.data.to_tensor(), orig_values[name]), (
                f"Values of {name} should match after being wrapped in the prepare step."
            )
        else:
            assert torch.equal(param.data, orig_values[name]), (
                f"values of {name} should not be changed after the prepare step."
            )
    assert wrapped_count == 3, f"Expected 3 wrapped params, got {wrapped_count}"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("weight_config, wrapper_cls", [
    (Float8FakeQuantizeConfig(), Float8FakeQuantizedWeightWrapperTensor),
])
def test_prepare_skips_non_expert_params(device, weight_config, wrapper_cls):
    """params_filter_fn excluding 2D params skips router.gate.weight."""
    # use_grouped_mm only affects the forward computation path — no forward run here.
    moe_model = create_moe_model(device, use_grouped_mm=True)
    qat_config = MoEQATConfig(
        weight_config=weight_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped = 0
    for name, param in moe_model.named_parameters():
        if isinstance(param.data, wrapper_cls):
            wrapped += 1
            assert param.ndim == 3, f"Wrapped param {name} should be 3D, got {param.ndim}D"
    assert wrapped == 3, f"All 3D expert params should be wrapped, got {wrapped}"
    # base class as wildcard — any wrapper type on non-expert params is a bug
    assert not isinstance(
        moe_model.router.gate.weight.data, FakeQuantizedWeightWrapperBaseTensor
    ), "router.gate.weight should not be wrapped"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("weight_config, wrapper_cls", [
    (Float8FakeQuantizeConfig(), Float8FakeQuantizedWeightWrapperTensor),
])
def test_convert_unwraps(device, weight_config, wrapper_cls):
    """Convert unwraps all parameters and restores original weight values."""

    # use_grouped_mm only affects the forward computation path, which is
    # never triggered here — only prepare/convert lifecycle is tested.
    moe_model = create_moe_model(device, use_grouped_mm=True)
    model = copy.deepcopy(moe_model)

    qat_config = MoEQATConfig(
        weight_config=weight_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
    
    wrapped = sum(1 for _, p in model.named_parameters() if isinstance(p.data, wrapper_cls))
    assert wrapped == 3, f"Only {wrapped} nn.Parameters are wrapped, 3 expected."
    
    for (name, param), (orig_name, orig_param) in zip(
        model.named_parameters(), moe_model.named_parameters()
    ):
        assert torch.equal(param, orig_param), f"Values of {name} should match after prepare"


    qat_config = MoEQATConfig(step="convert")
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    # base class as wildcard — no wrapper of any type should survive convert
    wrapped = sum(1 for _, p in model.named_parameters() if isinstance(p.data, FakeQuantizedWeightWrapperBaseTensor))
    assert wrapped == 0, f"{wrapped} parameters should not be wrapped after convert"

    for (name, param), (orig_name, orig_param) in zip(
        model.named_parameters(), moe_model.named_parameters()
    ):
        assert torch.equal(param, orig_param), f"Values of {name} should match after convert"



@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("base_config, wrapper_cls", [
    (Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), Float8FakeQuantizedWeightWrapperTensor),
])
def test_config_prepare_with_base_config(device, base_config, wrapper_cls):
    """Model can be prepared using base_config instead of explicit weight_config."""

    # use_grouped_mm only affects the forward computation path — no forward run here.
    moe_model = create_moe_model(device, use_grouped_mm=True)
    qat_config = MoEQATConfig(
        base_config=base_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
    wrapped = sum(1 for _, p in moe_model.named_parameters() if isinstance(p.data, wrapper_cls))
    assert wrapped == 3, f"Expected 3 wrapped params, got {wrapped}"





# =========================================================================
# Test filter functions
# =========================================================================


def test_is_expert_filter():
    """_is_expert returns True for module FQNs ending with 'experts' or 'shared_experts'."""
    class DummyModule(torch.nn.Module):
        pass

    assert _is_expert(DummyModule(), "model.layers.0.experts")
    assert _is_expert(DummyModule(), "shared_experts")
    assert not _is_expert(DummyModule(), "model.layers.0.router")
    assert not _is_expert(DummyModule(), "model.layers.0.attention")


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("weight_config, wrapper_cls", [
    (Float8FakeQuantizeConfig(), Float8FakeQuantizedWeightWrapperTensor),
])
def test_is_expert_integration(device, weight_config, wrapper_cls):
    """_is_expert as filter_fn: only expert submodules are transformed, router skipped."""
    
    # use_grouped_mm only affects the forward computation path — no forward run here.
    moe_model = create_moe_model(device, use_grouped_mm=True)
    qat_config = MoEQATConfig(
        weight_config=weight_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=_is_expert)

    wrapped = sum(1 for _, p in moe_model.named_parameters() if isinstance(p.data, wrapper_cls))
    assert wrapped == 3, f"Expected 3 wrapped params, got {wrapped}"
    # base class as wildcard — any wrapper type on non-expert params is a bug
    assert not isinstance(
        moe_model.router.gate.weight.data, FakeQuantizedWeightWrapperBaseTensor
    ), "router.gate.weight should not be wrapped"


def test_is_parameter_filter():
    """_is_parameter returns True for all nn.Parameter instances."""

    param = torch.nn.Parameter(torch.randn(4, 4))
    assert _is_parameter(param, "any.fqn") is True
    assert _is_parameter(torch.randn(4, 4), "any.fqn") is False
    assert _is_parameter(torch.nn.Module(), "any.fqn") is False
    assert _is_parameter(None, "any.fqn") is False


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("weight_config, wrapper_cls", [
    (Float8FakeQuantizeConfig(), Float8FakeQuantizedWeightWrapperTensor),
])
def test_is_parameter_integration(device, weight_config, wrapper_cls):
    """Default filter (_is_parameter) wraps all parameters including 2D gate."""

    # use_grouped_mm only affects the forward computation path — no forward run here.
    moe_model = create_moe_model(device, use_grouped_mm=True)
    qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped_count = 0
    for name, param in moe_model.named_parameters():
        if isinstance(param.data, wrapper_cls):
            wrapped_count += 1
    assert wrapped_count == 7, f"Expected 7 wrapped params, got {wrapped_count}"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("weight_config, wrapper_cls", [
    (Float8FakeQuantizeConfig(), Float8FakeQuantizedWeightWrapperTensor)
])
def test_is_parameter_with_wrapped_data_filter(device, weight_config, wrapper_cls):
    """_is_parameter_with_wrapped_data returns True only for wrapped nn.Parameters."""

    w = torch.randn(64, 128, device=device)
    wrapped = wrapper_cls(w, weight_config=weight_config)
    wrapped_param = torch.nn.Parameter(wrapped)
    plain_param = torch.nn.Parameter(w)

    assert _is_parameter_with_wrapped_data(wrapped_param, "any.fqn") is True
    assert _is_parameter_with_wrapped_data(plain_param, "any.fqn") is False
    assert _is_parameter_with_wrapped_data(wrapped, "any.fqn") is False
    assert _is_parameter_with_wrapped_data(None, "any.fqn") is False


@pytest.mark.parametrize("weight_config, wrapper_cls", [
    (Float8FakeQuantizeConfig(), Float8FakeQuantizedWeightWrapperTensor),
])
@pytest.mark.parametrize("device", target_devices)
def test_is_parameter_with_wrapped_data_integration(device, weight_config, wrapper_cls):
    """_is_parameter_with_wrapped_data as params_filter_fn: prepare then convert unwraps all."""
    
    # use_grouped_mm only affects the forward computation path — no forward run here.
    moe_model = create_moe_model(device, use_grouped_mm=True)
    qat_config = MoEQATConfig(
        weight_config=weight_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    qat_config = MoEQATConfig(step="convert", params_filter_fn=_is_parameter_with_wrapped_data)
    quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    wrapped = sum(1 for _, p in moe_model.named_parameters() if isinstance(p.data, wrapper_cls))
    assert wrapped == 0



