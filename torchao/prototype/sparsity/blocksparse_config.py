# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.sparsity.blocksparse import BlockSparseTensor
from torchao.quantization.quant_api import _linear_extra_repr
from torchao.quantization.transform_module import register_quantize_module_handler


@dataclass
class BlockSparseWeightConfig(AOBaseConfig):
    blocksize: int = 64

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.prototype.sparsity.BlockSparseWeightConfig"
        )


# for bc
block_sparse_weight = BlockSparseWeightConfig


@register_quantize_module_handler(BlockSparseWeightConfig)
def _block_sparse_weight_transform(
    module: torch.nn.Module,
    config: BlockSparseWeightConfig,
):
    blocksize = config.blocksize
    new_weight = BlockSparseTensor.from_dense(module.weight, blocksize)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module
