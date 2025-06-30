# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import ClassVar

import torch

__all__ = [
    "QuantizeTensorKwargs",
    "_choose_quant_func_and_quantize_tensor",
]


class QuantizeTensorKwargs(abc.ABC):
    """Base class for keyword argument container for quantized tensor creation.  This is needed to support storing activation construction arguments on the weight tensor while supporting multiple types of activation quantization.

    e.g.

    class Float8Tensor(...)
        @classmethod
        def to_float8(cls, tensor, quant_kwargs: QuantizeTensorKwargs)
            ...
    """

    # Base Version of a config
    VERSION: ClassVar[int] = 1


def _choose_quant_func_and_quantize_tensor(
    tensor: torch.Tensor, quant_kwargs: QuantizeTensorKwargs
) -> torch.Tensor:
    """Given a tensor and a kwargs container, chooses a derived dtype (float8, int8, etc) to quantize tensor to, based on the type of quant_kwargs
    quantizes tensor to the derived dtype chosen in (1)
    This is needed to support flexible quantization of activation and weights to various derived dtypes.
    """
    from torchao.quantization.quantize_.workflows import (
        Float8Tensor,
        QuantizeTensorToFloat8Kwargs,
    )

    if isinstance(quant_kwargs, QuantizeTensorToFloat8Kwargs):
        return Float8Tensor.to_float8(
            tensor,
            quant_kwargs.float8_dtype,
            quant_kwargs.granularity,
            quant_kwargs.mm_config,
            quant_kwargs.hp_value_lb,
            quant_kwargs.hp_value_ub,
            quant_kwargs.kernel_preference,
        )

    raise NotImplementedError(f"Quant kwargs not supported: {quant_kwargs}")
