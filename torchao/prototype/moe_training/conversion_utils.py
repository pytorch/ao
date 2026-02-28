# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Callable, Optional

from torch import nn

from torchao.prototype.moe_training.config import (
    TrainingBaseConfig,
)

logger: logging.Logger = logging.getLogger(__name__)


def _swap_params(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    config: Optional[TrainingBaseConfig] = None,
    target_parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Recurses through the nn.Module, recursively swapping the data tensor of
    each nn.Parameter with a MXFP8TrainingTensor. Only applies if the module
    passed the module_filter_fn, if specified.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Parameter` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    from torchao.prototype.moe_training.tensor import MXFP8TrainingTensor

    if isinstance(module, nn.Parameter) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Parameter with children: {module}"
            )
        if not isinstance(module.data, MXFP8TrainingTensor):
            new_data = MXFP8TrainingTensor(module.data, config)
            return nn.Parameter(new_data, requires_grad=module.requires_grad)
        return module

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)
        if module_filter_fn is None or module_filter_fn(module, cur_fqn):
            for param_name, param in module.named_parameters(recurse=False):
                if (
                    target_parameter_name is not None
                    and param_name != target_parameter_name
                ):
                    continue
                if not isinstance(param.data, MXFP8TrainingTensor):
                    new_param = nn.Parameter(
                        MXFP8TrainingTensor(param.data, config),
                        requires_grad=param.requires_grad,
                    )
                    setattr(module, param_name, new_param)
                    logger.info(
                        f"Swapped {cur_fqn}.{param_name} to MXFP8TrainingTensor"
                    )

    post_order_traversal(root_module)
    return root_module
