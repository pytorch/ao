import functools
from typing import Callable, Dict

import torch

from torchao.core.config import AOBaseConfig

_QUANTIZE_CONFIG_HANDLER: Dict[
    AOBaseConfig,
    Callable[[torch.nn.Module, AOBaseConfig], torch.nn.Module],
] = {}


def register_quantize_module_handler(config_type):
    @functools.wraps(config_type)
    def decorator(func):
        _QUANTIZE_CONFIG_HANDLER[config_type] = func

    return decorator
