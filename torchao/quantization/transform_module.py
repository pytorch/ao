# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Callable, Dict

import torch

from torchao.core.config import AOBaseConfig

_QUANTIZE_CONFIG_HANDLER: Dict[
    AOBaseConfig,
    Callable[[torch.nn.Module, AOBaseConfig], torch.nn.Module],
] = {}


def register_quantize_module_handler(config_type):
    """
    A decorator to register a transform function to map from a workflow
    configuration (child of `AOBaseConfig`) to a function that transforms
    a `torch.nn.Module` according to the specified configuration.

    For example::

        # user facing code
        class WorkflowFooConfig(AOBaseConfig): ...
            # configuration for workflow `Foo` is defined here
            bar = 'baz'

        # non user facing code
        @register_quantize_module_handler(WorkflowFooConfig)
        def _transform(
            mod: torch.nn.Module,
            config: WorkflowFooConfig,
        ) -> torch.nn.Module:
            # the transform is implemented here, usually a tensor sublass
            # weight swap or a module swap
            ...

        # then, the user calls `quantize_` with a config, and `_transform` is called
        # under the hood by `quantize_.

    """

    @functools.wraps(config_type)
    def decorator(func):
        _QUANTIZE_CONFIG_HANDLER[config_type] = func

    return decorator
