# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torch.testing._internal.common_pruning import SimpleLinear
from torch.testing._internal.common_utils import TestCase

from torchao.prototype.sparsity.pruner.wanda_pp import WandaPlusPlusSparsifier


class TestWandaPlusPlusSparsifier(TestCase):
    """Test Wanda++ Sparsifier"""

    def _setup_model_and_sparsifier(self, model, sparsifier, block_configs):
        """Helper to setup model with calibration and forward pass"""
        sparsifier.prepare(model, config=None)

        # Setup calibration for each block
        for block_name, input_shape in block_configs.items():
            for _ in range(5):
                calibration_input = torch.randn(1, *input_shape)
                sparsifier.store_calibration_input(block_name, calibration_input)

    def _verify_sparsity(self, layer, expected, tolerance=0.02):
        """Helper to verify sparsity level"""
        actual = (layer.weight == 0).float().mean()
        assert abs(actual - expected) < tolerance, (
            f"Expected ~{expected} sparsity, got {actual}"
        )

    def test_prepare_and_squash(self):
        """Test preparation and cleanup inherit from Wanda"""
        model = SimpleLinear()
        sparsifier = WandaPlusPlusSparsifier()
        sparsifier.prepare(model, config=None)

        # Should inherit Wanda's preparation
        assert hasattr(sparsifier.groups[0]["module"], "activation_post_process")

        sparsifier.squash_mask()
        assert not hasattr(sparsifier.groups[0]["module"], "activation_post_process")

    def test_one_layer_sparsity(self):
        """Test single layer sparsification"""
        model = nn.Sequential(nn.Linear(4, 1))
        model[0].weight.data = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

        sparsifier = WandaPlusPlusSparsifier(sparsity_level=0.5)
        self._setup_model_and_sparsifier(model, sparsifier, {"layer_0": (4,)})

        model(torch.tensor([[100, 10, 1, 0.1]], dtype=torch.float32))
        sparsifier.set_context(model[0], "layer_0")
        sparsifier.step()
        sparsifier.squash_mask()

        self._verify_sparsity(model[0], 0.5)

    def test_one_layer_mlp_2x4(self):
        """Test 2:4 semi-structured sparsity"""
        model = nn.Sequential(nn.Linear(8, 1))
        model[0].weight.data = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32
        )

        sparsifier = WandaPlusPlusSparsifier(semi_structured_block_size=4)
        self._setup_model_and_sparsifier(model, sparsifier, {"layer_0": (8,)})

        model(torch.ones(1, 8))
        sparsifier.set_context(model[0], "layer_0")
        sparsifier.step()
        sparsifier.squash_mask()

        self._verify_sparsity(model[0], 0.5)

    def test_multi_layer_sparsification(self):
        """Test multi-layer sparsification"""
        model = nn.Sequential(nn.Linear(128, 200), nn.ReLU(), nn.Linear(200, 10))
        sparsifier = WandaPlusPlusSparsifier(sparsity_level=0.5)

        block_configs = {"layer_0": (128,), "layer_2": (200,)}
        self._setup_model_and_sparsifier(model, sparsifier, block_configs)

        model(torch.randn(100, 128))

        # Sparsify each linear layer
        for layer, block_name in [(model[0], "layer_0"), (model[2], "layer_2")]:
            sparsifier.set_context(layer, block_name)
            sparsifier.step()
            self._verify_sparsity(layer, 0.5)

        sparsifier.squash_mask()

    def test_two_layer_mlp_unstructured_custom_config(self):
        """Test custom config for selective sparsification"""
        model = nn.Sequential(nn.Linear(128, 200), nn.ReLU(), nn.Linear(200, 10))
        config = [{"tensor_fqn": "0.weight"}]

        sparsifier = WandaPlusPlusSparsifier(sparsity_level=0.5)
        sparsifier.prepare(model, config=config)

        # Only setup calibration for first layer
        for _ in range(5):
            sparsifier.store_calibration_input("layer_0", torch.randn(1, 128))

        model(torch.randn(100, 128))
        sparsifier.set_context(model[0], "layer_0")
        sparsifier.step()

        self._verify_sparsity(model[0], 0.5)
        self._verify_sparsity(model[2], 0.0)
        sparsifier.squash_mask()


if __name__ == "__main__":
    unittest.main()
