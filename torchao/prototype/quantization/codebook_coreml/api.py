# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.quantization.codebook_coreml.codebook_quantized_tensor import (
    CodebookQuantizedTensor,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import is_package_at_least


@dataclass
class CodebookWeightOnlyConfig(AOBaseConfig):
    dtype: torch.dtype
    block_size: List[int]


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
    if not is_package_at_least("coremltools", "8.3.0"):
        raise ImportError("Requires coremltools >= 8.3.0")

    dtype = config.dtype
    weight = module.weight

    quantized_weight = CodebookQuantizedTensor.from_float(
        weight,
        dtype,
        config.block_size,
    )
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    return module
