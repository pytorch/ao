# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.quantization.unified import TwoStepQuantizer

from .embedding import FakeQuantizedEmbedding
from .fake_quantize_config import (
    FakeQuantizeConfig,  # noqa: F401, for BC
    FakeQuantizeConfigBase,
    _infer_fake_quantize_configs,
)
from .linear import FakeQuantizedLinear
from .utils import _log_deprecation_warning


class QATStep(str, Enum):
    """
    Enum value for the `step` field in :class:`~torchao.quantization.qat.QATConfig`.
    """

    PREPARE = "prepare"
    CONVERT = "convert"


@dataclass
class QATConfig(AOBaseConfig):
    """
    Config for applying quantization-aware training (QAT) to a `torch.nn.Module`,
    to be used with :func:`~torchao.quantization.quant_api.quantize_`.

    This config has two steps, "prepare" and "convert". The prepare step applies
    "fake" quantization to the model and should be applied before training, while
    the convert step converts the model into an actual quantized model. Fake
    quantization here refers to simulating the quantization numerics (e.g. int4)
    using high precision arithmetic (e.g. bf16), with the goal of reducing
    eventual degradation from quantization.

    There are two ways to use this config. The first involves passing a base
    post-training quantization (PTQ) config, which we will use to automatically
    infer the corresponding fake quantization schemes to use in the prepare phase.
    In the convert phase, we will then apply the base PTQ config to the model.
    This will be the most common use case.

    Example usage::

        from torchao.quantization import (
            quantize_,
            Int8DynamicActivationInt4WeightConfig,
        )
        from torchao.quantization.qat import QATConfig

        base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
        quantize_(model, QATConfig(base_config, step="prepare"))
        train_loop(model)
        quantize_(model, QATConfig(base_config, step="convert"))

    Currently only the following are supported as base configs:

        - :class:`~torchao.quantization.Int8DynamicActivationInt4WeightConfig`
        - :class:`~torchao.quantization.Int4WeightOnlyConfig`

    The second way to use this config involves specifying the fake quantization
    schemes directly. Users will pass in :class:`~torchao.quantization.qat.FakeQuantizeConfigBase`
    for weights and/or activations instead of the base PTQ config. This use case
    is mostly for experimentation, e.g. when the corresponding PTQ config does
    not exist yet.

    Example usage::

        from torchao.quantization import quantize_
        from torchao.quantization.qat import IntxFakeQuantizeConfig

        activation_config = IntxFakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=False,
        )
        weight_config = IntxFakeQuantizeConfig(
            torch.int4, group_size=32, is_symmetric=True,
        )
        qat_config = QATConfig(
            # must specify one of `base_config` or `weight_config`
            activation_config=act_config,
            weight_config=weight_config,
            step="prepare",
        )
        quantize_(model, qat_config)

    Args:
        base_config (Optional[AOBaseConfig]): Base PTQ config to infer the fake
            quantization configs during the prepare phase, and to apply directly
            during the convert phase.
        activation_config (Optional[FakeQuantizeConfigBase]): Custom fake
            quantization config for input activations, always optional.
            Must be None if `base_config` is used.
        weight_config (Optional[FakeQuantizeConfigBase]): Custom fake quantization
            config for weights. Must be None if `base_config` is used.

    Keyword args:
        step (str): One of "prepare" or "convert", determines the QAT phase

    Raises:
        ValueError: If `base_config` and `activation_config` are both specified
        ValueError: If `base_config` and `weight_config` are both specified
        ValueError: If neither `base_config` nor `weight_config` is specified
        ValueError: If `step` is not one of "prepare" or "convert"
        ValueError: If `base_config` is None but `step` is "convert"
        ValueError: If the config is applied on a module that is not a
            `torch.nn.Linear` or `torch.nn.Embedding`, or it is applied on
            `torch.nn.Embedding` with an activation config
    """

    base_config: Optional[AOBaseConfig]
    activation_config: Optional[FakeQuantizeConfigBase]
    weight_config: Optional[FakeQuantizeConfigBase]
    step: QATStep

    # Express `step` as a keyword argument
    # TODO: Use `kw_only=True` instead, added in python 3.10
    def __init__(
        self,
        base_config: Optional[AOBaseConfig] = None,
        activation_config: Optional[FakeQuantizeConfigBase] = None,
        weight_config: Optional[FakeQuantizeConfigBase] = None,
        *,
        step: QATStep = "prepare",
    ):
        self.base_config = base_config
        self.activation_config = activation_config
        self.weight_config = weight_config
        self.step = step
        self.__post_init__()

    def __post_init__(self):
        self.step = self.step.lower()
        all_step_values = [s.value for s in QATStep]
        if self.step not in all_step_values:
            raise ValueError("`step` must be one of %s" % all_step_values)
        if self.base_config is None and self.weight_config is None:
            raise ValueError(
                "One of `base_config` or `weight_config` must be specified"
            )
        if self.base_config is not None and self.activation_config is not None:
            raise ValueError(
                "Cannot specify both `base_config` and `activation_config`"
            )
        if self.base_config is not None and self.weight_config is not None:
            raise ValueError("Cannot specify both `base_config` and `weight_config`")
        if self.base_config is None and self.step == "convert":
            raise ValueError("`base_config` must be specified in the convert step")
        if isinstance(self.base_config, FakeQuantizeConfigBase):
            config_type = self.base_config.__class__.__name__
            raise ValueError(
                "%s was passed as `base_config`. Did you mean to do the following instead?\n"
                "    qat_config = QATConfig(\n"
                "        activation_config=%s(...),\n"
                "        weight_config=%s(...),\n"
                '        step="prepare",\n'
                "    )" % (config_type, config_type, config_type)
            )


@register_quantize_module_handler(QATConfig)
def _qat_config_transform(
    module: torch.nn.Module,
    config: QATConfig,
) -> torch.nn.Module:
    """
    During the prepare step, perform module swap to apply fake quantization.
    If the base PTQ config is specified, derive the fake quantization configs from it.

    During the convert step, first perform module swap to revert all fake quantized
    modules to the corresponding built-in `torch.nn.Module`s, then apply the
    base config directly to quantize the module.
    """
    # Prepare step
    # Swap nn.Linear -> FakeQuantizedLinear
    # Swap nn.Embedding -> FakeQuantizedEmbedding
    base_config = config.base_config
    step = config.step
    if step == QATStep.PREPARE:
        if base_config is not None:
            (act_config, weight_config) = _infer_fake_quantize_configs(base_config)
        else:
            act_config = config.activation_config
            weight_config = config.weight_config
        if isinstance(module, torch.nn.Linear):
            return FakeQuantizedLinear.from_linear(module, act_config, weight_config)
        elif isinstance(module, torch.nn.Embedding):
            if act_config is not None:
                raise ValueError(
                    "Activation fake quantization is not supported for embedding"
                )
            return FakeQuantizedEmbedding.from_embedding(module, weight_config)
        else:
            raise ValueError(
                "Module of type '%s' does not have QAT support" % type(module)
            )
    else:
        # Convert step
        # Swap FakeQuantizedLinear -> nn.Linear
        # Swap FakeQuantizedEmbedding -> nn.Embedding
        # Then apply the base config's transform function to quantize the model
        assert step == QATStep.CONVERT, "unexpected step '%s' in QATConfig" % step
        assert base_config is not None, "expected `base_config` in convert step"
        if isinstance(module, FakeQuantizedLinear):
            module = module.to_linear()
        elif isinstance(module, FakeQuantizedEmbedding):
            module = module.to_embedding()
        else:
            # Unrelated module, ignore
            return module
        return _QUANTIZE_CONFIG_HANDLER[type(base_config)](module, base_config)


@dataclass
class IntXQuantizationAwareTrainingConfig(AOBaseConfig):
    """
    (Deprecated) Please use :class:`~torchao.quantization.qat.QATConfig` instead.

    Config for applying fake quantization to a `torch.nn.Module`.
    to be used with :func:`~torchao.quantization.quant_api.quantize_`.

    Example usage::

        from torchao.quantization import quantize_
        from torchao.quantization.qat import IntxFakeQuantizeConfig
        activation_config = IntxFakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=False,
        )
        weight_config = IntxFakeQuantizeConfig(
            torch.int4, group_size=32, is_symmetric=True,
        )
        quantize_(
            model,
            IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
        )

    Note: If the config is applied on a module that is not
    `torch.nn.Linear` or `torch.nn.Embedding`, or it is applied on
    `torch.nn.Embedding` with an activation config, then we will raise
    ValueError as these are not supported.
    """

    activation_config: Optional[FakeQuantizeConfigBase] = None
    weight_config: Optional[FakeQuantizeConfigBase] = None

    def __post_init__(self):
        _log_deprecation_warning(self)


# for BC
class intx_quantization_aware_training(IntXQuantizationAwareTrainingConfig):
    pass


@register_quantize_module_handler(IntXQuantizationAwareTrainingConfig)
def _intx_quantization_aware_training_transform(
    module: torch.nn.Module,
    config: IntXQuantizationAwareTrainingConfig,
) -> torch.nn.Module:
    mod = module
    activation_config = config.activation_config
    weight_config = config.weight_config

    if isinstance(mod, torch.nn.Linear):
        return FakeQuantizedLinear.from_linear(
            mod,
            activation_config,
            weight_config,
        )
    elif isinstance(mod, torch.nn.Embedding):
        if activation_config is not None:
            raise ValueError(
                "Activation fake quantization is not supported for embedding"
            )
        return FakeQuantizedEmbedding.from_embedding(mod, weight_config)
    else:
        raise ValueError("Module of type '%s' does not have QAT support" % type(mod))


@dataclass
class FromIntXQuantizationAwareTrainingConfig(AOBaseConfig):
    """
    (Deprecated) Please use :class:`~torchao.quantization.qat.QATConfig` instead.

    Config for converting a model with fake quantized modules,
    such as :func:`~torchao.quantization.qat.linear.FakeQuantizedLinear`
    and :func:`~torchao.quantization.qat.linear.FakeQuantizedEmbedding`,
    back to model with the original, corresponding modules without
    fake quantization. This should be used with
    :func:`~torchao.quantization.quant_api.quantize_`.

    Example usage::

        from torchao.quantization import quantize_
        quantize_(
            model_with_fake_quantized_linears,
            FromIntXQuantizationAwareTrainingConfig(),
        )
    """

    def __post_init__(self):
        _log_deprecation_warning(self)


# for BC
class from_intx_quantization_aware_training(FromIntXQuantizationAwareTrainingConfig):
    pass


@register_quantize_module_handler(FromIntXQuantizationAwareTrainingConfig)
def _from_intx_quantization_aware_training_transform(
    mod: torch.nn.Module,
    config: FromIntXQuantizationAwareTrainingConfig,
) -> torch.nn.Module:
    """
    If the given module is a fake quantized module, return the original
    corresponding version of the module without fake quantization.
    """
    if isinstance(mod, FakeQuantizedLinear):
        return mod.to_linear()
    elif isinstance(mod, FakeQuantizedEmbedding):
        return mod.to_embedding()
    else:
        return mod


class ComposableQATQuantizer(TwoStepQuantizer):
    """
    Composable quantizer that users can use to apply multiple QAT quantizers easily.
    Quantizers will be applied in the order they are specified in the constructor.

    Note: the quantizers provided must apply to different modules in the model,
    e.g. nn.Linear and nn.Embedding, otherwise the behavior will be undefined.

    Example usage::

        my_quantizer = ComposableQATQuantizer([
            QATQuantizer1(),
            QATQuantizer2(),
            QATQuantizer3(),
        ])
        model = my_quantizer.prepare(model)
        train(model)
        model = my_quantizer.convert(model)
    """

    def __init__(self, quantizers: List[TwoStepQuantizer]):
        self.quantizers = quantizers

    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        for quantizer in self.quantizers:
            model = quantizer.prepare(model)
        return model

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        for quantizer in self.quantizers:
            model = quantizer.convert(model)
        return model


def initialize_fake_quantizers(
    model: torch.nn.Module,
    example_inputs: Tuple[Any, ...],
) -> None:
    """
    (Prototype) Initialize the scales and zero points on all
    :class:`~torchao.quantization.qat.fake_quantizer.FakeQuantizer`
    in the model based on the provided example inputs.
    """
    # avoid circular dependencies
    from torchao.quantization.qat.fake_quantizer import FakeQuantizer

    def _set_initialized(m: torch.nn.Module):
        if isinstance(m, FakeQuantizer):
            m._initialized = True

    model.apply(_set_initialized)
    model(*example_inputs)
