import logging
from dataclasses import dataclass, field
from typing import List, Union

import torch
import torch.nn as nn

from torchao.prototype.quantization.module_swap.algorithms import (
    kmeans_codebook,
)
from torchao.prototype.quantization.module_swap.quantized_modules import (
    QuantizedEmbedding,
    QuantizedLinear,
)
from torchao.prototype.quantization.module_swap.quantizers import (
    CodeBookQuantizer,
    IntQuantizer,
)
from torchao.prototype.quantization.module_swap.range_setting_methods import (
    set_weight_min_max,
)

logger: logging.Logger = logging.getLogger(__name__)


# TODO: express this using AOBaseConfig
@dataclass
class QuantizationRecipe:
    # weights
    weight_bits: int = 4
    weight_group_size: Union[int, str] = 32
    weight_quantization: bool = True
    dynamic_weights: bool = False

    # weight codebooking settings
    weight_codebook: bool = False  # if we're using weight codebooks
    codebook_dim: int = 1

    # activations
    activation_bits: int = 8
    activation_group_size: Union[int, str] = "per_token"
    activation_quantization: bool = False
    input_quantization: bool = False
    output_quantization: bool = False
    dynamic_activations: bool = True

    # general
    range_learning: bool = False
    embedding_quantization: bool = True
    embedding_bits: int = 4
    embedding_group_size: Union[int, str] = 32
    exclude_layers: List[str] = field(default_factory=lambda: ["lm_head"])


def get_layer_parent_by_name(model: nn.Module, input_name: str) -> nn.Module:
    parent_name = input_name.rsplit(".", 1)[:-1]
    if len(parent_name) == 0:  # parent is model itself
        return model
    else:
        parent_name = parent_name[0]

    for name, module in model.named_modules():
        if parent_name == name:
            return module
    raise ValueError(f"Layer {input_name} not found in model")


# TODO: delete this, use quantize_ instead
def quantize_module_swap(
    model: nn.Module, recipe: QuantizationRecipe, dtype: torch.dtype = torch.float32
) -> nn.Module:
    model = replace_all_linear_with_quantized_linear(model, recipe)
    if recipe.embedding_quantization:
        model = replace_all_embedding_with_quantized(model, recipe)
    initialize_model_parameters(model, recipe, dtype)
    return model


def replace_all_embedding_with_quantized(
    model: nn.Module, recipe: QuantizationRecipe
) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if name in recipe.exclude_layers:
                logger.info(f"skip layer {name} in exclude list")
            else:
                quantized_embedding = QuantizedEmbedding(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    padding_idx=module.padding_idx,
                    max_norm=module.max_norm,
                    norm_type=module.norm_type,
                    scale_grad_by_freq=module.scale_grad_by_freq,
                    sparse=module.sparse,
                    _weight=module.weight,
                    num_bits=recipe.embedding_bits,
                    group_size=recipe.embedding_group_size,
                    quantization_mode="symmetric",
                    range_learning=recipe.range_learning,
                    dynamic_weights=recipe.dynamic_weights,
                )
                attribute_name = name.rsplit(".", 1)[-1]
                parent_of_module = get_layer_parent_by_name(model, name)
                setattr(parent_of_module, attribute_name, quantized_embedding)

                logger.info(f"replaced {name} with quantized embedding")
    return model


def replace_all_linear_with_quantized_linear(
    model: nn.Module, recipe: QuantizationRecipe
) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name in recipe.exclude_layers:
                logger.info(f"skip layer {name} in exclude list")
            else:
                if recipe.weight_codebook:
                    weight_quantizer = CodeBookQuantizer(
                        n_bits=recipe.weight_bits,
                        features=module.out_features,
                        codebook_dim=recipe.codebook_dim,
                    )
                else:
                    weight_quantizer = IntQuantizer(
                        num_bits=recipe.weight_bits,
                        group_size=recipe.weight_group_size,
                        dynamic=recipe.dynamic_weights,
                        quantization_mode="symmetric",
                        range_learning=recipe.range_learning,
                    )
                quantized_linear = QuantizedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    weight_quantizer=weight_quantizer,
                    weight_quantization=recipe.weight_quantization,
                    activation_bits=recipe.activation_bits,
                    activation_group_size=recipe.activation_group_size,
                    activation_quantization=recipe.activation_quantization,
                    input_quantization=recipe.input_quantization,
                    output_quantization=recipe.output_quantization,
                    dynamic_activations=recipe.dynamic_activations,
                    range_learning=recipe.range_learning,
                )
                quantized_linear.weight = module.weight
                quantized_linear.bias = module.bias

                # replace the module with the quantized linear module
                attribute_name = name.rsplit(".", 1)[-1]
                parent_of_module = get_layer_parent_by_name(model, name)
                setattr(parent_of_module, attribute_name, quantized_linear)

                # logger.info(f"replaced {name} with quantized linear")
    return model


def initialize_model_parameters(
    model: nn.Module, recipe: QuantizationRecipe, dtype: torch.dtype = torch.float32
) -> None:
    """
    Initialize the model weights and/or codebook if codebook quantization is used
    """
    if not recipe.dynamic_weights:
        set_weight_min_max(model)
    if recipe.weight_codebook:
        kmeans_codebook(model, dtype=dtype)
