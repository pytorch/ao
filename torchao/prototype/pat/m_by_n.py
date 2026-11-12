# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit N:M magnitude masks for pruning-aware training."""

import torch
from torch import Tensor


def _validate_m_by_n(m_by_n: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(m_by_n, (tuple, list)) or len(m_by_n) != 2:
        raise ValueError(f"m_by_n must be a pair (m, n), but got {m_by_n!r}")
    m, n = m_by_n
    if (
        isinstance(m, bool)
        or isinstance(n, bool)
        or not isinstance(m, int)
        or not isinstance(n, int)
    ):
        raise ValueError(f"m_by_n must contain integers, but got {m_by_n!r}")
    if m <= 0 or n <= 0 or m > n:
        raise ValueError(f"m_by_n must satisfy 1 <= m <= n, but got ({m}, {n})")
    return m, n


def _group_keep_mask(grouped: Tensor, keep: int) -> Tensor:
    """Keep the ``keep`` largest magnitudes in each group of the last dim."""
    scores = grouped.abs()
    # Stable descending sort prefers the lower index for tied magnitudes.
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, order[..., :keep], True)
    return mask


def make_m_by_n_mask(
    weight_2d: Tensor,
    m_by_n: tuple[int, int] = (2, 4),
    remainder_policy: str = "pad",
) -> Tensor:
    """Return an explicit N:M magnitude mask for a 2-D weight.

    Groups of ``n`` consecutive elements are formed along the last dimension.
    ``True`` marks positions to keep. Ties in magnitude keep the lower index.

    With ``remainder_policy="pad"``, a final group of size ``r < n`` keeps its
    ``min(m, r)`` largest-magnitude elements. With ``"error"``, the width must
    be divisible by ``n``.
    """
    m, n = _validate_m_by_n(m_by_n)
    if remainder_policy not in ("pad", "error"):
        raise ValueError(
            f"remainder_policy must be 'pad' or 'error', but got {remainder_policy!r}"
        )
    if not isinstance(weight_2d, Tensor):
        raise TypeError(
            f"weight_2d must be a torch.Tensor, but got {type(weight_2d).__name__}"
        )
    if weight_2d.dim() != 2:
        raise ValueError(
            f"weight_2d must be 2-D, but got shape {tuple(weight_2d.shape)}"
        )

    rows, cols = weight_2d.shape
    full_groups, remainder = divmod(cols, n)
    if remainder and remainder_policy == "error":
        raise ValueError(
            f"weight_2d width {cols} is not divisible by n={n}; "
            "use remainder_policy='pad' to allow partial groups"
        )

    mask = torch.zeros_like(weight_2d, dtype=torch.bool)
    full_cols = full_groups * n
    if full_groups:
        grouped = weight_2d[:, :full_cols].reshape(rows, full_groups, n)
        mask[:, :full_cols] = _group_keep_mask(grouped, m).reshape(rows, full_cols)
    if remainder:
        grouped = weight_2d[:, full_cols:]
        mask[:, full_cols:] = _group_keep_mask(grouped, min(m, remainder))
    return mask
