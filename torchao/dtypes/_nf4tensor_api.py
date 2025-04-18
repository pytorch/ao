# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torchao.core.config import AOBaseConfig
from torchao.dtypes.nf4tensor import NF4Tensor
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


class NF4WeightOnlyConfig(AOBaseConfig):
    """
    Note: the file location of this workflow is temporary.
    TODO(future PR): integrate this properly into torchao's directory structure
    """

    block_size: int = 64
    scaler_block_size: int = 256


# for bc
nf4_weight_only = NF4WeightOnlyConfig


@register_quantize_module_handler(NF4WeightOnlyConfig)
def _nf4_weight_only_transform(
    module: torch.nn.Module,
    config: NF4WeightOnlyConfig,
) -> torch.nn.Module:
    block_size = config.block_size
    scaler_block_size = config.scaler_block_size

    new_weight = NF4Tensor.from_tensor(module.weight, block_size, scaler_block_size)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    return module
