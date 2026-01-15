# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from torch.optim import Optimizer

from .group_lasso import ProxGroupLasso, ProxGroupLassoReduce  # noqa: F401
from .lasso import ProxLasso  # noqa: F401
from .nm_sgd import NMSGDOptimizer
from .nuclear_norm import ProxNuclearNorm  # noqa: F401
from .proxmap import ProxMap  # noqa: F401
from .pruneopt import PruneOptimizer


def build_prune_optimizer(
    base_optimizer: Optimizer,
    prune_reg_lambda: float,
    prune_warmup_steps: int = 0,
    nm_gamma: float = 0.0,
) -> PruneOptimizer:
    if nm_gamma > 0:
        prune_opt_cls = partial(NMSGDOptimizer, nm_gamma=nm_gamma)
    else:
        prune_opt_cls = PruneOptimizer

    return prune_opt_cls(
        base_optimizer,
        warmup_steps=prune_warmup_steps,
        reg_lambda=prune_reg_lambda,
    )
