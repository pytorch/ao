
import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torchao.sparsity.dynamic_quant_sparse import Int8DynamicallyQuantized24CusparseltLinearWeight, Int8DynamicallyQuantized24CutlassLinearWeight

# Sparsity helper functions
def apply_fake_sparsity(model):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    It uses the torch.ao.pruning flow.
    """
    # torch.ao.pruning flow
    from torch.ao.pruning import WeightNormSparsifier
    sparse_config = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(sparsity_level=1.0,
                                      sparse_block_shape=(1,4),
                                      zeros_per_block=2)
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()

def apply_sparse(model):
    apply_fake_sparsity(model)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight = torch.nn.Parameter(to_sparse_semi_structured(mod.weight))


def change_linear_weights_to_int8_dq_semi_structured_sparsetensors(model, **kwargs):
    filter_fn = kwargs.pop("filter_fn", _is_linear)

    from torch.sparse import SparseSemiStructuredTensor
    if SparseSemiStructuredTensor._FORCE_CUTLASS:
        subclass = Int8DynamicallyQuantized24CutlassLinearWeight
    else:
        subclass = Int8DynamicallyQuantized24CusparseltLinearWeight

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(subclass, **kwargs),
        filter_fn,
    )


