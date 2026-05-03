# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import unittest
import warnings

import torch
from torch import nn
from torch.ao.pruning import FakeSparsity
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_pruning import SimpleLinear
from torch.testing._internal.common_utils import TestCase

from torchao.prototype.sparsity import FisherPruner

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def _run_calibration(model, pruner, n_batches=4, in_features=128):
    """Run calibration forward+backward passes to accumulate FIM stats."""
    criterion = nn.MSELoss()
    for _ in range(n_batches):
        X = torch.randn(16, in_features)
        # Collect output shape from a single forward without hooks active yet
        with torch.no_grad():
            out_shape = model(X).shape
        X = torch.randn(16, in_features)
        target = torch.zeros(out_shape)
        loss = criterion(model(X), target)
        loss.backward()
        pruner.accumulate_fim()
        model.zero_grad()


class TestFisherPruner(TestCase):
    """Tests for FisherPruner."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_invalid_sparsity_level(self):
        with self.assertRaises(ValueError):
            FisherPruner(sparsity_level=1.5)
        with self.assertRaises(ValueError):
            FisherPruner(sparsity_level=-0.1)

    def test_semi_structured_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FisherPruner(semi_structured_block_size=4)
        self.assertTrue(any("semi_structured_block_size" in str(x.message) for x in w))

    # ------------------------------------------------------------------
    # prepare()
    # ------------------------------------------------------------------

    def test_prepare_attaches_fake_sparsity(self):
        model = SimpleLinear()
        pruner = FisherPruner()
        pruner.prepare(model, config=None)
        for g in pruner.groups:
            module = g["module"]
            self.assertTrue(hasattr(module.parametrizations["weight"][0], "mask"))
            self.assertTrue(is_parametrized(module, "weight"))
            self.assertIsInstance(module.parametrizations.weight[0], FakeSparsity)
            self.assertTrue(hasattr(module, "activation_post_process"))

    def test_prepare_custom_config(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        pruner = FisherPruner()
        pruner.prepare(model, config=[{"tensor_fqn": "0.weight"}])
        # Only first layer should be parametrized
        self.assertTrue(is_parametrized(model[0], "weight"))
        self.assertFalse(is_parametrized(model[2], "weight"))

    def test_prepare_raises_on_missing_tensor_fqn(self):
        model = SimpleLinear()
        pruner = FisherPruner()
        with self.assertRaises(ValueError):
            pruner.prepare(model, config=[{"bad_key": "0.weight"}])

    # ------------------------------------------------------------------
    # accumulate_fim()
    # ------------------------------------------------------------------

    def test_accumulate_fim_increments_steps(self):
        model = nn.Sequential(nn.Linear(4, 2))
        pruner = FisherPruner()
        pruner.prepare(model, config=None)
        self.assertEqual(pruner._fim_steps, 0)
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        pruner.accumulate_fim()
        self.assertEqual(pruner._fim_steps, 1)

    def test_accumulate_fim_stores_scores(self):
        model = nn.Sequential(nn.Linear(4, 2))
        pruner = FisherPruner()
        pruner.prepare(model, config=None)
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        pruner.accumulate_fim()
        self.assertEqual(len(pruner._fim_scores), 1)
        key = list(pruner._fim_scores.keys())[0]
        self.assertEqual(pruner._fim_scores[key].shape, model[0].weight.shape)

    def test_accumulate_fim_no_grad_is_safe(self):
        """accumulate_fim should be a no-op for layers with no gradient."""
        model = nn.Sequential(nn.Linear(4, 2))
        pruner = FisherPruner()
        pruner.prepare(model, config=None)
        # Do NOT call backward — gradients are None
        pruner.accumulate_fim()
        self.assertEqual(pruner._fim_steps, 1)
        self.assertEqual(len(pruner._fim_scores), 0)

    # ------------------------------------------------------------------
    # squash_mask()
    # ------------------------------------------------------------------

    def test_squash_mask_cleans_up(self):
        model = SimpleLinear()
        pruner = FisherPruner()
        pruner.prepare(model, config=None)
        pruner.squash_mask()
        for g in pruner.groups:
            module = g["module"]
            self.assertFalse(is_parametrized(module, "weight"))
            self.assertFalse(hasattr(module, "mask"))
            self.assertFalse(hasattr(module, "activation_post_process"))
        self.assertEqual(len(pruner._fim_scores), 0)
        self.assertEqual(pruner._fim_steps, 0)

    # ------------------------------------------------------------------
    # End-to-end sparsity correctness
    # ------------------------------------------------------------------

    def test_unstructured_sparsity_level(self):
        """50% of weights in each layer should be zero after step+squash."""
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        pruner = FisherPruner(sparsity_level=0.5)
        pruner.prepare(model, config=None)
        _run_calibration(model, pruner, in_features=128)
        pruner.step()
        pruner.squash_mask()
        for m in model.modules():
            if isinstance(m, nn.Linear):
                actual = (m.weight == 0).float().mean().item()
                self.assertAlmostEqual(actual, 0.5, places=5)

    def test_known_weights_pruned_correctly(self):
        """Weights with zero gradient (FIM=0) should be pruned at 50% sparsity."""
        # weight = [1, 2, 3, 4]; only position 0 gets a gradient
        # FIM scores after 1 step: [grad^2, 0, 0, 0]
        # At 50% sparsity, 2 weights pruned — the two lowest FIM: positions 1,2 or 1,3 etc.
        # Actually lowest 2 are the three zeros; any 2 of {1,2,3} get pruned.
        model = nn.Sequential(nn.Linear(4, 1, bias=False))
        model[0].weight.data = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

        pruner = FisherPruner(sparsity_level=0.5)
        pruner.prepare(model, config=None)

        # Single forward that only activates input[0]
        X = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        loss = model(X).sum()
        loss.backward()
        pruner.accumulate_fim()
        model.zero_grad()

        pruner.step()
        sparsity = (model[0].weight == 0).float().mean()
        self.assertEqual(sparsity.item(), 0.5)

        # Weight at position 0 has the highest FIM score and MUST be kept
        pruner.squash_mask()
        self.assertNotEqual(model[0].weight.data[0, 0].item(), 0.0)

    def test_semi_structured_2x4_sparsity(self):
        """2:4 sparsity: exactly 2 zeros per block of 4."""
        model = nn.Sequential(nn.Linear(8, 1, bias=False))
        pruner = FisherPruner(semi_structured_block_size=4)
        pruner.prepare(model, config=None)

        X = torch.randn(8, 8)
        loss = model(X).sum()
        loss.backward()
        pruner.accumulate_fim()
        model.zero_grad()

        pruner.step()
        pruner.squash_mask()

        weight = model[0].weight.data.view(-1, 4)
        for row in weight:
            zeros = (row == 0).sum().item()
            self.assertEqual(zeros, 2)

    def test_fallback_to_magnitude_without_calibration(self):
        """Without calibration, FisherPruner warns and falls back to magnitude pruning."""
        model = nn.Sequential(nn.Linear(4, 1, bias=False))
        pruner = FisherPruner(sparsity_level=0.5)
        pruner.prepare(model, config=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pruner.step()
        self.assertTrue(any("no FIM statistics" in str(x.message) for x in w))
        pruner.squash_mask()
        sparsity = (model[0].weight == 0).float().mean().item()
        self.assertAlmostEqual(sparsity, 0.5, places=5)

    def test_custom_config_only_prunes_specified_layer(self):
        """With a per-layer config, only the specified layer should be pruned."""
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        pruner = FisherPruner(sparsity_level=0.5)
        pruner.prepare(model, config=[{"tensor_fqn": "0.weight"}])
        _run_calibration(model, pruner, in_features=128)
        pruner.step()
        pruner.squash_mask()

        sparsity_0 = (model[0].weight == 0).float().mean().item()
        sparsity_2 = (model[2].weight == 0).float().mean().item()
        self.assertAlmostEqual(sparsity_0, 0.5, places=5)
        self.assertEqual(sparsity_2, 0.0)  # untouched


if __name__ == "__main__":
    unittest.main()
