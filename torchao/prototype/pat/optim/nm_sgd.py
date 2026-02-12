# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim import Optimizer

from .pruneopt import PruneOptimizer


class NMSGDOptimizer(PruneOptimizer):
    """From "A Normal Map-Based Proximal Stochastic Gradient Method": https://arxiv.org/pdf/2305.05828v2
    Other parameters:
        norm_gamma: float, default 0.0
            If > 0, then normalize gamma by parameter group dimension.
            This is the same as the "normalized" option in N:M sparsity.
    """

    def __init__(
        self, base_optimizer: Optimizer, nm_gamma: float = 0.0, **kwargs
    ) -> None:
        super().__init__(base_optimizer=base_optimizer, **kwargs)
        self.nm_gamma = nm_gamma
        for group in self.regularized_param_groups():
            group["gamma"] = self.nm_gamma

    def _set_gamma(self, group):
        pass

    @torch._disable_dynamo
    def restore_latent_params(self) -> None:
        """Restore latent parameters as optimizer parameters"""
        gamma_inv = 1.0 / self.nm_gamma
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    p.grad.add_(self.state[p]["latent"] - p, alpha=gamma_inv)
                    p.copy_(self.state[p]["latent"])
