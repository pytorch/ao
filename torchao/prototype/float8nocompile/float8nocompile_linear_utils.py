# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_linear_utils import swap_linear_layers

from torchao.prototype.float8nocompile.float8nocompile_linear import (
    Float8LinearNoCompile,
)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def convert_to_float8_nocompile_training(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Linear` in `module` with `Float8LinearNoCompile`.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance and the FQN.
        config (Float8LinearConfig): configuration for conversion to float8

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    from_float = lambda m: Float8LinearNoCompile.from_float(m)
    return swap_linear_layers(
        module,
        from_float,
        module_filter_fn=module_filter_fn,
    )
