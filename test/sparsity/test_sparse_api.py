import copy
import logging
import unittest

import torch
from torch import nn

from torchao.sparsity import (
    apply_fake_sparsity,
    sparsify_,
    int8_dynamic_activation_int8_semi_sparse_weight,
    semi_sparse_weight,
)
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    _get_subclass_inserter,
    _is_linear,
    int8_dynamic_activation_int8_weight,
    quantize_,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_3
from torch.testing._internal.common_utils import TestCase

from torch.ao.pruning import WeightNormSparsifier


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

def apply_fake_block_sparsity(model, **kwargs):
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
        sparsity_level=0.5, sparse_block_shape=(64, 64)
    )
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


class TestSemiStructuredSparse(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        dense_result = model(input)

        sparsify_(model, semi_sparse_weight())
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

class TestQuantSemiSparse(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quant_semi_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )
        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)
        quantize_(model_copy, int8_dynamic_activation_int8_weight())
        dense_result = model_copy(input)

        quantize_(model, int8_dynamic_activation_int8_semi_sparse_weight())
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-2, atol=1e-2)



class TestBlockSparseWeight(TestCase):
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_sparse(self):
        input = torch.rand((1024, 1024)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(1024, 2048),
                nn.Linear(2048, 1024),
            )
            .half()
            .cuda()
        )

        from torchao.sparsity.utils import create_block_sparse_tensor
        M, N = model[0].weight.shape
        model[0].weight.data = create_block_sparse_tensor(M, N, 64, 0.5, torch.float16)
        M, N = model[1].weight.shape
        model[1].weight.data = create_block_sparse_tensor(M, N, 64, 0.5, torch.float16)
        dense_result = model(input)

        from torchao.sparsity.prototype.superblock.blocksparse import block_sparse_weight
        sparsify_(model, block_sparse_weight())
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

class TestQuantBlockSparseWeight(TestCase):
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_sparse(self):
        input = torch.rand((128, 128)).to(torch.bfloat16).cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .to(torch.bfloat16)
            .cuda()
        )

        from torchao.sparsity.utils import create_block_sparse_tensor
        M, N = model[0].weight.shape
        model[0].weight.data = create_block_sparse_tensor(M, N, 64, 0.5, torch.bfloat16) * torch.rand(M, N, dtype=torch.bfloat16).cuda()
        print(model[0].weight)
        M, N = model[1].weight.shape
        model[1].weight.data = create_block_sparse_tensor(M, N, 64, 0.5, torch.bfloat16)
        print(model[1].weight)  

        model_copy = copy.deepcopy(model)

        quantize_(model_copy, int8_dynamic_activation_int8_weight())
        reference = model_copy(input)

        from torchao.dtypes.affine_quantized_tensor import BlockSparseLayoutType
        quantize_(model, int8_dynamic_activation_int8_weight(layout_type=BlockSparseLayoutType(), ))
        sparse_result = model(input)

        print(reference)
        print(sparse_result)
        assert torch.allclose(reference, sparse_result, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
