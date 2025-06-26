# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.quantization.coreml_codebook.codebook_quantized_tensor import (
    CodebookQuantizedTensor,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


@dataclass
class CodebookWeightOnlyConfig(AOBaseConfig):
    dtype: torch.dtype
    group_size: int


@register_quantize_module_handler(CodebookWeightOnlyConfig)
def _codebook_weight_only_transform(
    module: torch.nn.Module,
    config: CodebookWeightOnlyConfig,
):
    """
    Applies codebook weight-only quantization to linear layers.

    Args:
        dtype: torch.uint1 to torch.uint8, torch.int32 supported.
    Returns:
        Callable for quantization transformation.
    """
    dtype = config.dtype
    group_size = config.group_size
    weight = module.weight

    if weight.numel() > 2**27:
        return module  # k_means is too numerically unstable

    quantized_weight = CodebookQuantizedTensor.from_float(
        weight,
        dtype,
        group_size,
    )
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    return module
