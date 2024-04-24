import torch
from torch.ao.pruning import WeightNormSparsifier
from torch.sparse import to_sparse_semi_structured

from torchao.quantization.quant_api import QuantizedLinearWeightBase, apply_dynamic_quant, Int8DynamicallyQuantizedLinearWeight

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
    # cannot use _is_linear because it will not work for quantized results
    filter_fn = kwargs.get("filter_fn", lambda mod, name: isinstance(mod, torch.nn.Linear))

    for name, mod in model.named_modules():
        if filter_fn(mod, name):
            if isinstance(mod.weight.data, Int8DynamicallyQuantizedLinearWeight):
                temp = mod.weight.data

                if isinstance(temp.int_data, SparseSemiStructuredTensor):
                    int_data = temp.int_data
                else:
                    int_data = to_sparse_semi_structured(temp.int_data.contiguous())

                new = Int8DynamicallyQuantizedLinearWeight(int_data, temp.q_scales, temp.transposed, temp.shape, dtype=temp.dtype)

                mod.weight = torch.nn.Parameter(new)
            else:
                mod.weight = torch.nn.Parameter(to_sparse_semi_structured(mod.weight))

def apply_dynamic_quant_sparse_semi_structured(model, **kwargs):
    apply_dynamic_quant(model, **kwargs)
    apply_sparse_semi_structured(model, **kwargs)

from torch.sparse import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS

def sparse24_pointwise_op(
    func, types, args=(), kwargs=None, allow_sparsify_args_list=()
):

    self = None
    for tensor in args:
        if isinstance(tensor, SparseSemiStructuredTensor):
            self = tensor
    assert self is not None
    args_updated = []
    for i, tensor in enumerate(args):
        args_updated.append(tensor)


    assert isinstance(self, SparseSemiStructuredTensorCUTLASS)
    return SparseSemiStructuredTensorCUTLASS(
        self.shape,
        func(
            *[(x.packed if isinstance(x, SparseSemiStructuredTensor) else x) for x in args_updated]
        ),
        self.meta,
        func(
            *[
                (x.packed_t if isinstance(x, SparseSemiStructuredTensor) else x)
                for x in args_updated
            ]
        ),
        self.meta_t,
        self.threads_masks,
    )


SPARSE24_DISPATCH_CUTLASS = {
    torch.ops.aten.add: sparse24_pointwise_op,
    torch.ops.aten.sub: sparse24_pointwise_op,
    torch.ops.aten.mul: sparse24_pointwise_op,
}

SparseSemiStructuredTensorCUTLASS._load_dispatch_table(SPARSE24_DISPATCH_CUTLASS)
