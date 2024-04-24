import torch
from torch.ao.pruning import WeightNormSparsifier
from torch.sparse import to_sparse_semi_structured
from torchao.quantization.quant_api import _is_linear, Int8DynamicallyQuantizedLinearWeight, apply_dynamic_quant

# Sparsity helper functions
def apply_fake_sparse_semi_structured(model):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    It uses the torch.ao.pruning flow.
    """
    # torch.ao.pruning flow
    sparse_config = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()

def apply_sparse_semi_structured(model, **kwargs):
    filter_fn = kwargs.pop("filter_fn", _is_linear)

    for name, mod in model.named_modules():
        if filter_fn(mod, name):
            if isinstance(mod.weight.data, Int8DynamicallyQuantizedLinearWeight):
                mod.weight.data.int_data = to_sparse_semi_structured(mod.weight.data.int_data.t())
            else:
                mod.weight.data = to_sparse_semi_structured(mod.weight.data)

def apply_dynamic_quant_sparse_semi_structured(model, **kwargs):
    apply_dynamic_quant(model, **kwargs)
    apply_sparse_semi_structured(model, **kwargs)
