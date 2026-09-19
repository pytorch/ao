import pytest
import torch

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATStep
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig

# =========================================================================
# Test configs in the prepare step
# =========================================================================


def test_config_step_validation():
    """step must be 'prepare' or 'convert'."""
    weight_config = Float8FakeQuantizeConfig(
        dtype=torch.float8_e4m3fn, granularity=PerRow()
    )
    MoEQATConfig(weight_config=weight_config, step="prepare")
    MoEQATConfig(step="convert")
    with pytest.raises(
        ValueError, match=r"^`step` must be one of \['prepare', 'convert'\]$"
    ):
        MoEQATConfig(weight_config=weight_config, step="blah")


def test_config_requires_weight_config():
    """prepare step requires weight_config, activation_config, or base_config."""
    with pytest.raises(
        ValueError,
        match=r"^Must specify `base_config`, `activation_config`, or `weight_config` in the prepare step$",
    ):
        MoEQATConfig(step="prepare")


def test_config_requires_weight_config_in_prepare():
    """MoEQATConfig requires weight_config during prepare, even if activation_config is set."""
    act_config = Float8FakeQuantizeConfig(
        dtype=torch.float8_e4m3fn, granularity=PerRow()
    )
    with pytest.raises(
        ValueError,
        match=r"^`weight_config` is required for the prepare step of MoEQATConfig\.$",
    ):
        MoEQATConfig(activation_config=act_config, step="prepare")


@pytest.mark.parametrize("granularity", [PerRow()])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_float8_weight_config_variants(granularity, dtype):
    """All Float8FakeQuantizeConfig variants should be accepted."""
    config = Float8FakeQuantizeConfig(dtype=dtype, granularity=granularity)
    qat_config = MoEQATConfig(weight_config=config, step="prepare")
    assert qat_config.step == QATStep.PREPARE


@pytest.mark.parametrize(
    "base_config, expected_weight_config, expected_activation_config",
    [
        (
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            Float8FakeQuantizeConfig,
            Float8FakeQuantizeConfig,
        ),
    ],
)
def test_config_infer_from_base_config(
    base_config, expected_weight_config, expected_activation_config
):
    """MoEQATConfig can infer fake quantize configs from a PTQ base_config."""
    qat_config = MoEQATConfig(base_config=base_config, step="prepare")
    assert isinstance(qat_config.weight_config, expected_weight_config)
    assert isinstance(qat_config.activation_config, expected_activation_config)
    assert qat_config.base_config is None


def test_config_rejects_invalid_weight_config():
    """For now only Float8FakeQuantizeConfig is supported for weight config."""
    intx_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")
    with pytest.raises(
        ValueError,
        match=r"^Only `Float8FakeQuantizeConfig` is supported for `weight_config` in MoEQATConfig yet\.$",
    ):
        MoEQATConfig(weight_config=intx_config, step="prepare")


def test_config_rejects_invalid_activation_config():
    """For now only Float8FakeQuantizeConfig is supported for activation config."""
    weight_config = Float8FakeQuantizeConfig(
        dtype=torch.float8_e4m3fn, granularity=PerRow()
    )
    activation_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")
    with pytest.raises(
        ValueError,
        match=r"^Only `Float8FakeQuantizeConfig` is supported for `activation_config` in MoEQATConfig yet\.$",
    ):
        MoEQATConfig(
            weight_config=weight_config,
            activation_config=activation_config,
            step="prepare",
        )


def test_config_rejects_invalid_base_config_in_prepare_step():
    """Only Float8DynamicActivationFloat8WeightConfig is accepted as base_config."""
    from torchao.quantization import Int4WeightOnlyConfig

    base_config = Int4WeightOnlyConfig(group_size=32)
    with pytest.raises(
        ValueError,
        match=r"^Only `Float8DynamicActivationFloat8WeightConfig` is supported for `base_config` in MoEQATConfig yet\.$",
    ):
        MoEQATConfig(base_config=base_config, step="prepare")


# =========================================================================
# Test configs in the convert step
# =========================================================================


def test_config_rejects_base_config_in_convert():
    """base_config is not supported in the convert step."""
    weight_config = Float8FakeQuantizeConfig(
        dtype=torch.float8_e4m3fn, granularity=PerRow()
    )
    MoEQATConfig(weight_config=weight_config, step="prepare")
    base_config = Float8DynamicActivationFloat8WeightConfig()
    with pytest.raises(
        NotImplementedError,
        match=r"^Applying PTQ in the convert step is not implemented yet\.$",
    ):
        MoEQATConfig(base_config=base_config, step="convert")


def test_config_rejects_weight_config_in_convert():
    """weight_config cannot be specified in the convert step."""
    weight_config = Float8FakeQuantizeConfig(
        dtype=torch.float8_e4m3fn, granularity=PerRow()
    )
    with pytest.raises(
        ValueError,
        match=r"^Cannot specify `weight_config` or `activation_config` in the convert step$",
    ):
        MoEQATConfig(weight_config=weight_config, step="convert")


def test_config_rejects_activation_config_in_convert():
    """activation_config cannot be specified in the convert step."""
    act_config = Float8FakeQuantizeConfig(
        dtype=torch.float8_e4m3fn, granularity=PerRow()
    )
    with pytest.raises(
        ValueError,
        match=r"^Cannot specify `weight_config` or `activation_config` in the convert step$",
    ):
        MoEQATConfig(activation_config=act_config, step="convert")


# =========================================================================
# Test params_filter_fn
# =========================================================================


def test_config_default_params_filter_fn():
    """Default params_filter_fn is _is_parameter (wraps all parameters)."""
    qat_config = MoEQATConfig(
        weight_config=Float8FakeQuantizeConfig(
            dtype=torch.float8_e4m3fn, granularity=PerRow()
        ),
        step="prepare",
    )
    from torchao.prototype.moe_qat.transform import _is_parameter

    assert qat_config.params_filter_fn is _is_parameter


def test_config_custom_params_filter_fn():
    """Custom params_filter_fn is stored and forwarded to the handler."""

    def custom_filter(param, fqn):
        return True

    qat_config = MoEQATConfig(
        weight_config=Float8FakeQuantizeConfig(
            dtype=torch.float8_e4m3fn, granularity=PerRow()
        ),
        step="prepare",
        params_filter_fn=custom_filter,
    )
    assert qat_config.params_filter_fn is custom_filter
