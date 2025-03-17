from typing import Dict

import torch.nn as nn

from torchao.prototype.quantization.module_swap.quantized_modules import QuantizedLinear
from torchao.prototype.quantization.module_swap.quantizers import IntQuantizer


def get_layer_by_name(model: nn.Module, query_name: str) -> nn.Module:
    """
    Retrieves a layer from a PyTorch model by its name.

    Args:
        model (nn.Module): The PyTorch model.
        name (str): The name of the layer to retrieve.

    Returns:
        nn.Module: The retrieved layer.
    """
    for name, module in model.named_modules():
        if name == query_name:
            return module
    raise ValueError(f"Layer '{query_name}' not found in model")


def all_quantizers_off(module: nn.Module) -> None:
    if isinstance(module, QuantizedLinear):
        module.weight_quantization = False
        module.activation_quantization = False


def all_quantizers_on(module: nn.Module) -> None:
    if isinstance(module, QuantizedLinear):
        module.weight_quantization = True
        module.activation_quantization = True


def all_activation_quantizers_off(module: nn.Module) -> None:
    if isinstance(module, QuantizedLinear):
        module.activation_quantization = False


def all_activation_quantizers_on(module: nn.Module) -> None:
    if isinstance(module, QuantizedLinear):
        module.activation_quantization = True


def all_weight_quantizers_on(module: nn.Module) -> None:
    if isinstance(module, QuantizedLinear):
        module.weight_quantization = True


def set_bit_widths_by_name(
    model: nn.Module, bit_width_dict: Dict[str, Dict[str, int]]
) -> None:
    for name, bit_width_assignment in bit_width_dict.items():
        this_layer = get_layer_by_name(model, name)
        for quantizer, bit_width in bit_width_assignment.items():
            assert isinstance(this_layer, QuantizedLinear)
            if quantizer == "weight":
                assert isinstance(this_layer.weight_quantizer, IntQuantizer)
                this_layer.weight_quantizer.num_bits = bit_width
            elif quantizer == "activation":
                if this_layer.input_quantizer is not None:
                    this_layer.input_quantizer.num_bits = bit_width
                if this_layer.output_quantizer is not None:
                    this_layer.output_quantizer.num_bits = bit_width
            else:
                raise ValueError(
                    f"Unknown quantizer {quantizer}, should be either 'weight' or 'activation'"
                )
