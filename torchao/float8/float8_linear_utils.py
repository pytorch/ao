# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Callable, Optional

import torch.nn as nn

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_linear import Float8Linear

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def swap_linear_layers(
    module: nn.Module,
    from_float_func: Callable[[nn.Linear], nn.Linear],
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """
    Generic function to swap linear layers in a module with a new type of linear layer.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module: Module to modify.
        from_float_func: Function that accepts a linear layer and returns a new type of linear layer.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if isinstance(module, nn.Linear) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Linear with children: {module}"
            )
        return from_float_func(
            module,
        )

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

        if isinstance(module, nn.Linear) and (
            module_filter_fn is None or module_filter_fn(module, cur_fqn)
        ):
            assert parent_module is not None, (
                f"Linear root module should return early: {module}"
            )
            new_linear_module = from_float_func(module)
            cur_module_name = cur_fqn.split(".")[-1]
            setattr(parent_module, cur_module_name, new_linear_module)

    post_order_traversal(root_module)
    return root_module


def convert_to_float8_training(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    config: Optional[Float8LinearConfig] = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Linear` in `module` with `Float8Linear`.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance and the FQN.
        config (Float8LinearConfig): configuration for conversion to float8

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if config is None:
        config = Float8LinearConfig()

    from_float = lambda m: Float8Linear.from_float(
        m,
        config=config,
    )

    return swap_linear_layers(
        module,
        from_float,
        module_filter_fn=module_filter_fn,
    )
