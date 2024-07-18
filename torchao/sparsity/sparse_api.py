from typing import Callable, Optional

import torch
from torch.ao.pruning import WeightNormSparsifier
from torch.sparse import to_sparse_semi_structured
from torchao.quantization.quant_api import (
    _is_linear,
    _replace_with_custom_fn_if_matches_filter,
    _get_linear_subclass_inserter,
    int8_dynamic_activation_int8_semi_sparse_weight,
)

# Sparsity helper functions
def apply_fake_sparsity(model, **kwargs):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    It uses the torch.ao.pruning flow.
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

def semi_sparse_weight():
    """
    Convert the weight of linear moduels to semi-structured (2:4) sparsity
    """
    return _get_linear_subclass_inserter(to_sparse_semi_structured)

def sparsify_(model: torch.nn.Module,
             apply_tensor_subclass: Callable[[torch.Tensor], torch.Tensor],
             filter_fn: Optional[Callable[[torch.nn.Module, str], bool]]=None) -> torch.nn.Module:
    """Convert the weight of linear modules in the model with `apply_tensor_subclass`
    This function is essentially the same as quantize, put for sparsity subclasses.

    Currently, we support two options for sparsity:
        - semi-structured (2:4) sparsity with `semi_sparse_weight`
        - int8 dynamic quantization + 2:4 sparsity with `int8_dynamic_activation_int8_semi_sparse_weight`, which is also available via the quantize API

    Args:
        model (torch.nn.Module): input model
        apply_tensor_subclass (Callable[[torch.Tensor], torch.Tensor]): function that convert a floating point Tensor to a (sparsified) tensor subclass instance (e.g. affine quantized tensor instance)
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): function that takes a nn.Module instance and fully qualified name of the module, returns True if we want to run `apply_tensor_subclass` on
        the weight of the module

    Example::
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
        from torchao.sparsity.prototype import int8_dynamic_activation_int8_semi_sparse_weight
        m = sparsify_(m, int8_dynamic_activation_int8_semi_sparse_weight(), filter_fn)
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        apply_tensor_subclass, 
        _is_linear if filter_fn is None else filter_fn,
    )
