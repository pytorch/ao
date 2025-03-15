# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import unittest

import torch
from torch import nn
from torch.ao.pruning import FakeSparsity
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_pruning import SimpleLinear
from torch.testing._internal.common_utils import TestCase

from torchao.sparsity import WandaSparsifier

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TestWandaSparsifier(TestCase):
    """
    Test Wanda Sparsifier
    """

    def test_prepare(self):
        model = SimpleLinear()
        sparsifier = WandaSparsifier()
        sparsifier.prepare(model, config=None)
        for g in sparsifier.groups:
            module = g["module"]
            # Check mask exists
            assert hasattr(module.parametrizations["weight"][0], "mask")
            # Check parametrization exists and is correct
            assert is_parametrized(module, "weight")
            assert isinstance(
                module.parametrizations.weight[0], FakeSparsity
            ), "FakeSparsity not found"
            # check activation observer is present
            assert hasattr(module, "activation_post_process")

    def test_squash_mask(self):
        # check observers and parameterizations removed
        model = SimpleLinear()
        sparsifier = WandaSparsifier()
        sparsifier.prepare(model, config=None)
        sparsifier.squash_mask()
        for g in sparsifier.groups:
            module = g["module"]
            assert not is_parametrized(module, "weight")
            assert not hasattr(module, "mask")
            assert not hasattr(module, "activation_post_process")

    def test_one_layer_mlp_2x4(self):
        model = nn.Sequential(nn.Linear(8, 1))
        weights = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        model[0].weight.data.copy_(weights.data)
        X = torch.ones(1, 8)

        sparsifier = WandaSparsifier(semi_structured_block_size=4)
        sparsifier.prepare(model, config=None)

        model(X)

        sparsifier.step()
        sparsifier.squash_mask()

        sparsity = (model[0].weight == 0).float().mean()
        assert sparsity == 0.5

        expected_fc = torch.tensor([[0, 0, 3, 4, 0, 0, 7, 8]], dtype=torch.float32)
        assert torch.allclose(model[0].weight.data, expected_fc, rtol=1e-05, atol=1e-07)

    def test_one_layer_mlp_unstructured(self):
        model = nn.Sequential(nn.Linear(4, 1))
        weights = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        model[0].weight.data.copy_(weights.data)
        X = torch.tensor([[100, 10, 1, 0.1]], dtype=torch.float32)

        sparsifier = WandaSparsifier(sparsity_level=0.5)
        sparsifier.prepare(model, config=None)

        model(X)

        sparsifier.step()
        sparsifier.squash_mask()

        sparsity = (model[0].weight == 0).float().mean()
        assert sparsity == 0.5

        expected_fc = torch.tensor([[1, 2, 0, 0]], dtype=torch.float32)
        assert torch.allclose(model[0].weight.data, expected_fc, rtol=1e-05, atol=1e-07)

    def test_two_layer_mlp_unstructured(self):
        model = nn.Sequential(
            nn.Linear(128, 200), nn.ReLU(), nn.Linear(200, 10)
        )  # C_in by C_out
        X1 = torch.randn(100, 128)  # B1 by C_in
        X2 = torch.randn(50, 128)  # B2 by C_in

        sparsifier = WandaSparsifier(sparsity_level=0.5)
        sparsifier.prepare(model, config=None)

        model(X1)
        model(X2)
        sparsifier.step()

        cnt = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                cnt += 1
                sparsity_level = (m.weight == 0).float().mean()
                assert (
                    sparsity_level == 0.5
                ), f"sparsity for linear layer {cnt} should be 0.5"

        sparsifier.squash_mask()

    def test_two_layer_mlp_unstructured_custom_config(self):
        model = nn.Sequential(
            nn.Linear(128, 200), nn.ReLU(), nn.Linear(200, 10)
        )  # C_in by C_out
        X1 = torch.randn(100, 128)  # B1 by C_in
        X2 = torch.randn(50, 128)  # B2 by C_in

        # Define custom config to sparsify only the first Linear layer for testing
        config = [{"tensor_fqn": "0.weight"}]

        sparsifier = WandaSparsifier(sparsity_level=0.5)
        sparsifier.prepare(model, config=config)

        model(X1)
        model(X2)
        sparsifier.step()

        cnt = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                cnt += 1
                sparsity_level = (m.weight == 0).float().mean()
                if cnt == 1:  # First Linear layer should have 50% sparsity
                    assert (
                        sparsity_level == 0.5
                    ), f"sparsity for linear layer {cnt} should be 0.5"
                else:  # Other layers should not be sparsified
                    assert (
                        sparsity_level != 0.5
                    ), f"sparsity for linear layer {cnt} should not be 0.5"

        sparsifier.squash_mask()


if __name__ == "__main__":
    unittest.main()
