import torch
from typing import Dict, Callable
from torchao.core.config import AOBaseWorkflowConfig

_QUANTIZE_CONFIG_HANDLER: Dict[
    AOBaseWorkflowConfig,
    Callable[[torch.nn.Module, AOBaseWorkflowConfig], torch.nn.Module],
] = {}


def register_quantize_module_handler(config_type):
    def decorator(func):
        _QUANTIZE_CONFIG_HANDLER[config_type] = func

    return decorator
