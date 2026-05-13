from typing import Callable, Optional

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.moe_qat.quantize import (
    _is_parameter,
    _is_parameter_with_wrapped_data,
    _replace_params_with_custom_fn_if_matches_filter,
    unwrap_param,
)
from torchao.prototype.moe_qat.tensor import (
    _MoE_QAT_PARAMETER_QUANTIZE_CONFIG_HANDLER,
)
from torchao.quantization.qat import QATConfig, QATStep
from torchao.quantization.qat.fake_quantize_config import (
    FakeQuantizeConfigBase,
    Float8FakeQuantizeConfig,
)
from torchao.quantization.transform_module import register_quantize_module_handler


def _is_expert(module: nn.Module, fqn: str) -> bool:
    """
    An example filter to be used with MoEQATConfig, targeting modules whose names end
    with "experts" or "shared_experts"
    """
    return fqn.split(".")[-1] in ("experts", "shared_experts")


class MoEQATConfig(QATConfig):
    """
    Config for applying quantization-aware training (QAT) to MoE (Mixture of Experts) models,
    to be used with :func:`~torchao.quantization.quant_api.quantize_`.

    This config extends :class:`~torchao.quantization.qat.QATConfig` for the MoE setting.
    Unlike dense QAT (which replaces ``torch.nn.Linear`` modules with ``FakeQuantizedLinear``),
    MoE QAT wraps 3D expert weight ``nn.Parameter`` data with a fake-quantized tensor
    subclass (``FakeQuantizedWeightWrapperBaseTensor``). This intercepts ``torch._grouped_mm``
    calls via ``__torch_function__`` and applies fake quantization to the weights.

    The workflow follows the same two-step pattern:

    1. Prepare: wraps 3D expert weight parameters with ``FakeQuantizedWeightWrapperBaseTensor``
    2. Convert: unwraps ``FakeQuantizedWeightWrapperBaseTensor`` back to regular tensors

    Currently only FP8 row-wise weight-only fake quantization is supported.

    Example usage::

        from torchao.quantization import quantize_
        from torchao.prototype.moe_qat import MoEQATConfig
        from torchao.quantization.qat import Float8FakeQuantizeConfig

        weight_config = Float8FakeQuantizeConfig(
            dtype=torch.float8_e4m3fn,
            granularity=PerRow(),
        )
        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(model, qat_config)

    Args:
        base_config: Base PTQ config (not yet supported in convert step).
        activation_config: Fake quantize config for input activations
            (not yet supported).
        weight_config: Fake quantize config for weights. Currently only
            :class:`~torchao.quantization.qat.Float8FakeQuantizeConfig` is supported.

    Keyword args:
        step: "prepare" or "convert" (default "prepare").
        params_filter_fn: Filter for selecting which parameters to wrap within
            a matched module. Receives ``(param, fqn)`` and returns ``bool``.
            Defaults to :func:`_is_parameter`, which accepts all parameters.
    """

    def __init__(
        self,
        base_config: Optional[AOBaseConfig] = None,
        activation_config: Optional[FakeQuantizeConfigBase] = None,
        weight_config: Optional[FakeQuantizeConfigBase] = None,
        *,
        step: QATStep = "prepare",
        params_filter_fn: Callable[[nn.Parameter, str], bool] = _is_parameter,
    ):
        self.params_filter_fn = params_filter_fn
        super().__init__(base_config, activation_config, weight_config, step=step)

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.prototype.moe_qat.MoEQATConfig")
        if self.activation_config is not None:
            raise ValueError(
                "`activation_config` is not supported in MoeQATConfig yet."
            )

        if self.weight_config is not None and not isinstance(
            self.weight_config, Float8FakeQuantizeConfig
        ):
            raise ValueError(
                "Only `Float8FakeQuantizeConfig` is supported for `weight_config` in MoEQATConfig yet."
            )

        super().__post_init__()

        if self.step == QATStep.CONVERT and self.base_config is not None:
            raise NotImplementedError(
                "Applying PTQ in the convert step is not implemented yet."
            )


@register_quantize_module_handler(MoEQATConfig)
def _moe_qat_config_transform(
    module: nn.Module,
    config: MoEQATConfig,
) -> nn.Module:
    """Prepare or convert parameter-level wrapping for MoEQATConfig"""

    if config.step == QATStep.PREPARE:
        params_handler = _MoE_QAT_PARAMETER_QUANTIZE_CONFIG_HANDLER[
            type(config.weight_config)
        ]
        # for each parameter in the module, apply the transform if filtering passes
        _replace_params_with_custom_fn_if_matches_filter(
            module, params_handler, config.params_filter_fn, extra_args=(config,)
        )

    elif config.step == QATStep.CONVERT:
        _replace_params_with_custom_fn_if_matches_filter(
            module, unwrap_param, _is_parameter_with_wrapped_data, extra_args=(config,)
        )

    else:
        raise ValueError(f"Invalid value of config.step: {config.step}")

    return module
