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
    """Applies the kwargs to quantize tensor, to get a quantized Tensor."""
    from torchao.quantization.quantize_.workflows import (
        Float8Tensor,
        QuantizeTensorToFloat8Kwargs,
    )

    if isinstance(quant_kwargs, QuantizeTensorToFloat8Kwargs):
        return Float8Tensor.to_float8(tensor, quant_kwargs)

    raise NotImplementedError(f"Quant kwargs not supported: {quant_kwargs}")
