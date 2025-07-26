# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.quantization.quantize_.common import QuantizeTensorKwargs


def _choose_quant_func_and_quantize_tensor(
    tensor: torch.Tensor, quant_kwargs: QuantizeTensorKwargs
) -> torch.Tensor:
    """Given a tensor and a kwargs container, chooses a derived dtype (float8, int8, etc) to quantize tensor to, based on the type of quant_kwargs
    quantizes tensor to the derived dtype chosen in (1)
    This is needed to support flexible quantization of activations to various derived dtypes.
    """
    from torchao.quantization.quantize_.workflows import (
        Float8Tensor,
        QuantizeTensorToFloat8Kwargs,
    )

    if isinstance(quant_kwargs, QuantizeTensorToFloat8Kwargs):
        return Float8Tensor.to_float8(tensor, quant_kwargs)

    raise NotImplementedError(f"Quant kwargs not supported: {quant_kwargs}")
