# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.unified import TwoStepQuantizer

from .fake_quantize_config import (
    FakeQuantizeConfig,  # noqa: F401, for BC
    FakeQuantizeConfigBase,
)


@dataclass
class IntXQuantizationAwareTrainingConfig(AOBaseConfig):
    """
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


# for BC
intx_quantization_aware_training = IntXQuantizationAwareTrainingConfig


@register_quantize_module_handler(IntXQuantizationAwareTrainingConfig)
def _intx_quantization_aware_training_transform(
    module: torch.nn.Module,
    config: IntXQuantizationAwareTrainingConfig,
) -> torch.nn.Module:
    from .embedding import FakeQuantizedEmbedding
    from .linear import FakeQuantizedLinear

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


class FromIntXQuantizationAwareTrainingConfig(AOBaseConfig):
    """
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

    pass


# for BC
from_intx_quantization_aware_training = FromIntXQuantizationAwareTrainingConfig


@register_quantize_module_handler(FromIntXQuantizationAwareTrainingConfig)
def _from_intx_quantization_aware_training_transform(
    mod: torch.nn.Module,
    config: FromIntXQuantizationAwareTrainingConfig,
) -> torch.nn.Module:
    """
    If the given module is a fake quantized module, return the original
    corresponding version of the module without fake quantization.
    """
    from .embedding import FakeQuantizedEmbedding
    from .linear import FakeQuantizedLinear

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
    :class:`~`torchao.quantization.qat.fake_quantizer.FakeQuantizer`
    in the model based on the provided example inputs.
    """
    # avoid circular dependencies
    from torchao.quantization.qat.fake_quantizer import FakeQuantizer

    def _set_initialized(m: torch.nn.Module):
        if isinstance(m, FakeQuantizer):
            m._initialized = True

    model.apply(_set_initialized)
    model(*example_inputs)
