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
    at once. ``reg_lambda`` and ``gamma`` are ignored.

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
    ) -> tuple[Tensor, Tensor]:
        assert p.dim() == 2, (
            f"MinSparsityConstraint expects a 2-D (n_groups, group_size) view, "
            f"got shape {tuple(p.shape)}."
        )
        n_groups = p.size(0)
        n_zero = math.ceil(self.min_sparsity * n_groups)
        scores = self._get_norm(p)
        return self._topk_zero_(p, scores, n_zero)


class GlobalMinSparsityConstraint(MinSparsityConstraint):
    """Allocate one structured sparsity budget across a parameter group.

    Unlike ``MinSparsityConstraint``, which applies the target independently to
    each tensor, this constraint ranks groups from every tensor jointly. The
    optimizer collects scores with :meth:`score`, selects the globally smallest
    ``ceil(min_sparsity * total_groups)`` groups, and applies the selection with
    :meth:`zero_groups_`. The inherited ``min_sparsity`` attribute is retained
    for constructor validation and API consistency; ``PruneOptimizer`` owns the
    scheduled global budget computation.

    ``score_type`` controls comparisons across different group sizes:

    - ``"rms"``: L2 norm divided by ``sqrt(group_size)``.
    - ``"l2"``: raw L2 norm.
    - ``"param_cost"``: L2 norm divided by ``group_size``.
    """

    whole_tensor = True

    def __init__(
        self, reg_lambda: float, min_sparsity: float, score_type: str = "rms"
    ) -> None:
        super().__init__(reg_lambda, min_sparsity)
        assert score_type in ("rms", "l2", "param_cost"), (
            f"score_type must be one of rms/l2/param_cost, got {score_type!r}"
        )
        self.score_type = score_type

    def score(self, p: Tensor) -> Tensor:
        """Return one importance score per leading-dimension group."""
        assert p.dim() == 2, (
            "GlobalMinSparsityConstraint.score expects a 2-D "
            f"(n_groups, group_size) view, got shape {tuple(p.shape)}."
        )
        norm = torch.linalg.vector_norm(p, dim=1)
        group_size = p.size(1)
        if self.score_type == "rms":
            return norm / math.sqrt(group_size)
        if self.score_type == "param_cost":
            return norm / group_size
        return norm

    @staticmethod
    def zero_groups_(p: Tensor, zero_idx: Tensor) -> Tensor:
        """Zero selected leading-dimension groups and return element count."""
        assert p.dim() == 2, (
            "GlobalMinSparsityConstraint.zero_groups_ expects a 2-D view, "
            f"got shape {tuple(p.shape)}."
        )
        if zero_idx.numel() > 0:
            p[zero_idx] = 0.0
        zeros = zero_idx.numel() * p.size(1)
        return torch.tensor(zeros, device=p.device, dtype=torch.long)


class MinRankConstraint(ProxMap, _TopKZeroMixin):
    """Zeros the smallest ``ceil(min_sparsity * k)`` singular values of an
    SVD-grouped tensor. Here the shared ``min_sparsity`` key is the fraction of
    singular values zeroed, so each matrix retains
    ``k - ceil(min_sparsity * k)`` singular values. Pair with ``SVDGrouper`` or
    ``PackedSVDGrouper``. ``reg_lambda`` and ``gamma`` are ignored; the count
    is optionally resolved on the cubic schedule by
    ``PruneOptimizer._effective_min_sparsity``.

    ``whole_tensor = True`` routes the optimizer around ``torch.vmap`` so
    ``apply_`` sees the complete singular-value vector for each matrix.
    """

    whole_tensor = True

    def __init__(self, reg_lambda: float, min_sparsity: float) -> None:
        super().__init__(reg_lambda)
        assert 0.0 <= min_sparsity <= 1.0, (
            f"min_sparsity must be in [0, 1], but got {min_sparsity}"
        )
        self.min_sparsity = min_sparsity

    def _get_norm(self, p: Tensor) -> Tensor:
        return p

    def tau(self, p: Tensor) -> float:
        return 1.0

    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
    ) -> tuple[Tensor, Tensor]:
        # SVDGrouper.p is (k,); PackedSVDGrouper.p is (npack, k).
        n_zero = math.ceil(self.min_sparsity * p.shape[-1])
        if p.dim() == 1:
            return self._topk_zero_(p, self._get_norm(p), n_zero)

        zeros_total = torch.zeros((), dtype=torch.long, device=p.device)
        for i in range(p.size(0)):
            zeros, _ = self._topk_zero_(p[i], self._get_norm(p[i]), n_zero)
            zeros_total += zeros
        return zeros_total, torch.linalg.vector_norm(p)


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
    ) -> tuple[Tensor, Tensor]:
        assert self.n_nonzero <= p.numel(), (
            f"n_nonzero ({self.n_nonzero}) must be at most group_size ({p.numel()})"
        )
        n_zero = p.numel() - self.n_nonzero
        scores = self._get_norm(p)
        return self._topk_zero_(p, scores, n_zero)
