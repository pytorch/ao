from typing import Callable, Optional
import torch
from torch.ao.pruning import WeightNormSparsifier
from torch.sparse import to_sparse_semi_structured
from torchao.quantization.quant_api import (
    _is_linear,
    _replace_with_custom_fn_if_matches_filter,
    _get_linear_subclass_inserter,
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

# to be deprecated
def apply_sparse_semi_structured(model, **kwargs):
    filter_fn = kwargs.pop("filter_fn", _is_linear)

    apply_fake_sparsity(model, filter_fn=filter_fn)
    for name, mod in model.named_modules():
        if filter_fn(mod, name):
            mod.weight = torch.nn.Parameter(to_sparse_semi_structured(mod.weight))


def sparsify_(model: torch.nn.Module, apply_tensor_subclass: Callable[[torch.Tensor], torch.Tensor], filter_fn: Optional[Callable[[torch.nn.Module, str], bool]]=None, prune=False) -> torch.nn.Module:
    """Convert the weight of linear modules in the model with `apply_tensor_subclass`

    This function is essentially the same as quantize_

    Args:
        model (torch.nn.Module): input model
        apply_tensor_subclass (Callable[[torch.Tensor], torch.Tensor]): function that convert a floating point Tensor to a (sparsified) tensor subclass instance (e.g. affine quantized tensor instance)
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): function that takes a nn.Module instance and fully qualified name of the module, returns True if we want to run `apply_tensor_subclass` on
        the weight of the module

    Example::

        import torch
        import torch.nn as nn
        from torchao import quantize

        # 1. quantize with some predefined `apply_tensor_subclass` method that corresponds to
        # optimized execution paths or kernels (e.g. int4 tinygemm kernel)
        # also customizable with arguments
        # currently options are
            to_sparse_semi_structured
            int8_dynamic_activation_int8_2x4_sparse_weight

        from torch.sparse import to_sparse_semi_structured

        # apply to modules under block0 submodule
        def filter_fn(module: nn.Module, fqn: str) -> bool:
            return isinstance(module, nn.Linear)

        m = nn.Sequential(nn.Linear(32, 1024), nn.Linear(1024, 32))
        m = sparsify_(m, to_sparse_semi_structured, filter_fn)

    """
    if prune:
        apply_fake_sparsity(model, filter_fn=filter_fn)

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_linear_subclass_inserter(apply_tensor_subclass),
        _is_linear if filter_fn is None else filter_fn,
    )

    return model
