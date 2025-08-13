# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.sparse import to_sparse_semi_structured

from torchao.core.config import AOBaseConfig
from torchao.prototype.sparsity.sparsifier.weight_norm_sparsifier import (
    WeightNormSparsifier,
)
from torchao.quantization.quant_api import (
    _is_linear,
    _linear_extra_repr,
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.sparsity.blocksparse import BlockSparseTensor


# Sparsity helper functions
def apply_fake_sparsity(model, **kwargs):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    """
    filter_fn = kwargs.pop("filter_fn", _is_linear)
    # torch.ao.pruning flow
    sparse_config = []
    for name, mod in model.named_modules():
        if filter_fn(mod, name):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


@dataclass
class BlockSparseWeightConfig(AOBaseConfig):
    blocksize: int = 64

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.sparsity.BlockSparseWeightConfig")


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


class SemiSparseWeightConfig(AOBaseConfig):
    """
    Configuration for converting the weight of linear modules to semi-structured (2:4) sparsity
    """

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.sparsity.SemiSparseWeightConfig")


# for bc
semi_sparse_weight = SemiSparseWeightConfig


@register_quantize_module_handler(SemiSparseWeightConfig)
def _semi_sparse_weight_transform(
    module: torch.nn.Module,
    config: SemiSparseWeightConfig,
) -> torch.nn.Module:
    new_weight = to_sparse_semi_structured(module.weight)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def sparsify_(
    model: torch.nn.Module,
    config: AOBaseConfig,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> torch.nn.Module:
    """Convert the weight of linear modules in the model with `apply_tensor_subclass`.
    This function is essentially the same as quantize, put for sparsity subclasses.

    Currently, we support three options for sparsity:
        - semi-structured (2:4) sparsity with `semi_sparse_weight`
        - int8 dynamic quantization + 2:4 sparsity with `layout=SemiSparseLayout`
        - int4 weight-only quantization + 2:4 sparsity with `layout=SparseMarlinLayout`

    Args:
        model (torch.nn.Module): input model
        config (AOBaseConfig): a workflow configuration object
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): function that takes a nn.Module instance and fully qualified name of the module, returns True if we want to apply the specified workflow to this module.

    **Example:**
    ::
            import torch
            import torch.nn as nn
            from torchao.sparsity import sparsify_

            def filter_fn(module: nn.Module, fqn: str) -> bool:
                return isinstance(module, nn.Linear)

            m = nn.Sequential(nn.Linear(32, 1024), nn.Linear(1024, 32))

            # for 2:4 sparsity
            from torchao.sparse_api import semi_sparse_weight
            m = sparsify_(m, semi_sparse_weight(), filter_fn)

            # for int8 dynamic quantization + 2:4 sparsity
            from torchao.dtypes import SemiSparseLayout
            m = quantize_(m, int8_dynamic_activation_int8_weight(layout=SemiSparseLayout), filter_fn)
    """
    torch._C._log_api_usage_once("torchao.sparsity.sparsify_")
    handler = _QUANTIZE_CONFIG_HANDLER[type(config)]
    _replace_with_custom_fn_if_matches_filter(
        model,
        handler,
        _is_linear if filter_fn is None else filter_fn,
        extra_args=(config,),
    )
