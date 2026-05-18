# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Union

import torch
from torch import Tensor

from .proxmap import ProxMap


class _TopKZeroMixin:
    """Zero the ``n_zero`` leading-dim slices of ``p`` with smallest scores."""

    @staticmethod
    def _topk_zero_(p: Tensor, scores: Tensor, n_zero: int) -> tuple[Tensor, Tensor]:
        """Returns ``(zeros_count, group_norm)``; ``zeros_count`` is in
        elements of ``p``, not slices."""
        total = scores.numel()
        if n_zero <= 0:
            zeros = 0
        elif n_zero >= total:
            p.zero_()
            zeros = p.numel()
        else:
            _, idx = torch.topk(scores, k=n_zero, largest=False, sorted=False)
            p[idx] = 0.0
            zeros = n_zero * (p.numel() // total)
        zeros_count = torch.tensor(zeros, device=p.device, dtype=torch.long)
        group_norm = torch.linalg.vector_norm(p)
        return zeros_count, group_norm


class MinSparsityConstraint(ProxMap, _TopKZeroMixin):
    """Drops the smallest-L2-norm ``ceil(min_sparsity * n_groups)`` whole
    groups of a 2-D ``(n_groups, group_size)`` view. ``whole_tensor = True``
    routes the optimizer around ``torch.vmap`` so the prox sees all groups
    at once. ``reg_lambda``, ``gamma``, and ``tau_reweight`` are ignored.

    Pair with ``Dim0Grouper`` / ``Dim1Grouper`` / ``ConvFilterGrouper`` for
    row/column/filter pruning, or with ``KElementGrouper(k=1)`` for global
    magnitude pruning of every element of the tensor (each element becomes
    its own group, so per-row L2 collapses to per-element ``|x|``).
    """

    whole_tensor = True

    def __init__(self, reg_lambda: float, min_sparsity: float) -> None:
        super().__init__(reg_lambda)
        assert 0.0 <= min_sparsity <= 1.0, (
            f"min_sparsity must be in [0, 1], but got {min_sparsity}"
        )
        self.min_sparsity = min_sparsity

    def _get_norm(self, p: Tensor) -> Tensor:
        return torch.linalg.vector_norm(p, dim=1)

    def tau(self, p: Tensor) -> float:
        return 1.0

    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> tuple[Tensor, Tensor]:
        assert p.dim() == 2, (
            f"MinSparsityConstraint expects a 2-D (n_groups, group_size) view, "
            f"got shape {tuple(p.shape)}."
        )
        n_groups = p.size(0)
        n_zero = math.ceil(self.min_sparsity * n_groups)
        scores = self._get_norm(p)
        return self._topk_zero_(p, scores, n_zero)


class NMSparseConstraint(ProxMap, _TopKZeroMixin):
    """Keeps at most ``n_nonzero`` largest-magnitude elements per group.
    Vmapped per group, so ``apply_`` receives one 1-D group at a time.
    ``reg_lambda`` is ignored.
    """

    def __init__(self, reg_lambda: float, n_nonzero: int) -> None:
        super().__init__(reg_lambda)
        assert n_nonzero >= 0, f"n_nonzero must be non-negative, but got {n_nonzero}"
        self.n_nonzero = n_nonzero

    def _get_norm(self, p: Tensor) -> Tensor:
        return p.abs()

    def tau(self, p: Tensor) -> float:
        return 1.0

    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> tuple[Tensor, Tensor]:
        assert self.n_nonzero <= p.numel(), (
            f"n_nonzero ({self.n_nonzero}) must be at most group_size ({p.numel()})"
        )
        n_zero = p.numel() - self.n_nonzero
        scores = self._get_norm(p)
        return self._topk_zero_(p, scores, n_zero)
