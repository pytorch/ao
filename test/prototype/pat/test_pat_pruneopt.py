# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import torch
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import TwoLayerMLP, optim_step
from torchao.prototype.pat.optim import ProxLasso, PruneOptimizer
from torchao.prototype.pat.utils import get_param_groups


class TestPruneOptimizer(common_utils.TestCase):
    def __init__(self, methodName):
        super(TestPruneOptimizer, self).__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxLasso(self.reg_lambda)

    @common_utils.parametrize("warmup_steps", [0, 5])
    def test_warmup_steps(self, warmup_steps=0, total_steps=10):
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=self.reg_lambda,
            warmup_steps=warmup_steps,
        )

        dummy_input = torch.randn(total_steps, 10)
        label = torch.randint(0, 2, (total_steps,))
        for step in range(total_steps):
            optim_step(model, optimizer, dummy_input, label, step)
            if step < warmup_steps:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        assert "latent" in optimizer.state[p]
            else:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        # check that param magnitude is smaller than latent
                        latent = optimizer.state[p]["latent"]
                        self.assertTrue(
                            torch.linalg.norm(p) < torch.linalg.norm(latent)
                        )

    @common_utils.parametrize("healing_start_step", [3, 5])
    def test_healing_start_step(self, healing_start_step=5, total_steps=10):
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.AdamW(param_groups, lr=0.01),
            reg_lambda=self.reg_lambda,
            warmup_steps=0,
            healing_start_step=healing_start_step,
        )

        dummy_input = torch.randn(total_steps, 10)
        label = torch.randint(0, 2, (total_steps,))
        prev_pruned_p_dct = {}
        for step in range(total_steps):
            optim_step(model, optimizer, dummy_input, label, step)
            if step == healing_start_step:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        if p.count_nonzero().item():
                            prev_pruned_p_dct[p.data_ptr()] = p.clone().detach()
            elif step > healing_start_step:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        prev_pruned_p = prev_pruned_p_dct.get(p.data_ptr())
                        if prev_pruned_p is not None:
                            zero_mask = prev_pruned_p.eq(0)
                            self.assertTrue(p.grad[zero_mask].eq(0).all())
                            self.assertTrue(p[zero_mask].eq(0).all())
        self.assertFalse(prev_pruned_p is None)


common_utils.instantiate_parametrized_tests(TestPruneOptimizer)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
