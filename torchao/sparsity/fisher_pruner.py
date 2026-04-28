# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""FisherPruner: Fisher Information Matrix-guided weight pruning.

Prunes weights whose removal causes the smallest expected increase in loss,
estimated via the diagonal of the empirical Fisher Information Matrix (eFIM).
Unlike magnitude pruning (remove small weights) or Wanda (weight * activation
norm), FisherPruner uses squared gradient statistics to measure per-parameter
sensitivity to the loss — i.e. removing a parameter is cheap when its gradient
variance is low.

Reference: Optimal Brain Damage (LeCun et al. 1990), Optimal Brain Surgeon
(Hassibi & Stork 1993); empirical FIM diagonal approximation follows
Singh & Alistarh, "WoodFisher" (NeurIPS 2020).

Usage::

    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    pruner = FisherPruner(sparsity_level=0.5)
    pruner.prepare(model, config=None)

    # Run calibration forward+backward passes to accumulate FIM statistics
    for X, y in calibration_loader:
        loss = criterion(model(X), y)
        loss.backward()        # gradients are accumulated by FIM hooks
        pruner.accumulate_fim()
        model.zero_grad()

    pruner.step()              # apply masks using accumulated FIM scores
    pruner.squash_mask()       # finalize: remove parametrizations
"""

import warnings
from typing import Optional

import torch
from torch import nn
from torch.ao.pruning import BaseSparsifier, get_arg_info_from_tensor_fqn
from torch.ao.quantization import QConfig, default_placeholder_observer
from torch.ao.quantization.quantize import _remove_qconfig

from .utils import PerChannelNormObserver

__all__ = ["FisherPruner"]


class FisherPruner(BaseSparsifier):
    r"""Fisher Information Matrix-guided weight pruner.

    Assigns per-parameter importance scores using the diagonal of the empirical
    Fisher Information Matrix (eFIM).  The diagonal eFIM entry for parameter
    :math:`\theta_i` is approximated as the expected squared gradient:

    .. math::
        F_{ii} \approx \frac{1}{T} \sum_{t=1}^{T} \left(\frac{\partial \ell_t}
        {\partial \theta_i}\right)^2

    Weights with *low* FIM scores (low gradient variance) are pruned first —
    their removal has the smallest expected impact on the loss.

    Args:
        sparsity_level: Fraction of weights to prune in each layer (0–1).
            Default: ``0.5``.
        semi_structured_block_size: When set, forces 2:N semi-structured
            sparsity (N/2 weights kept per block of N), e.g. ``4`` for 2:4
            sparsity.  ``sparsity_level`` is ignored.  Default: ``None``.

    Example::

        pruner = FisherPruner(sparsity_level=0.5)
        pruner.prepare(model, config=None)
        for X, y in calibration_data:
            loss = criterion(model(X), y)
            loss.backward()
            pruner.accumulate_fim()
            model.zero_grad()
        pruner.step()
        pruner.squash_mask()
    """

    def __init__(
        self,
        sparsity_level: float = 0.5,
        semi_structured_block_size: Optional[int] = None,
    ) -> None:
        if not 0.0 <= sparsity_level <= 1.0:
            raise ValueError(
                f"sparsity_level must be in [0, 1], got {sparsity_level}"
            )
        defaults = {
            "sparsity_level": sparsity_level,
            "semi_structured_block_size": semi_structured_block_size,
        }
        if semi_structured_block_size is not None:
            m = semi_structured_block_size
            warnings.warn(
                f"FisherPruner got semi_structured_block_size={m}; "
                f"sparsity_level fixed to 50% ({m // 2}:{m}) sparsity"
            )
        super().__init__(defaults=defaults)
        # Maps module id → accumulated squared-gradient tensor (eFIM diagonal)
        self._fim_scores: dict[int, torch.Tensor] = {}
        # Number of calibration batches accumulated (for normalisation)
        self._fim_steps: int = 0

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare(self, model: nn.Module, config: list[dict] | None) -> None:
        """Prepare the model for Fisher-based pruning.

        Applies a ``PerChannelNormObserver`` qconfig to track activation
        statistics (kept for API parity with WandaSparsifier; the FIM
        computation itself is done via :meth:`accumulate_fim`).  Then calls
        ``super().prepare()`` which attaches ``FakeSparsity`` parametrizations.

        Args:
            model: The model to prune.
            config: Optional list of ``{"tensor_fqn": "..."}`` dicts that
                restrict pruning to specific weight tensors.  Pass ``None``
                to prune all ``nn.Linear`` layers.
        """
        if config is None:
            model.qconfig = QConfig(  # type: ignore[assignment]
                activation=PerChannelNormObserver,
                weight=default_placeholder_observer,
            )
        else:
            for module_config in config:
                tensor_fqn = module_config.get("tensor_fqn")
                if tensor_fqn is None:
                    raise ValueError("Each config entry must contain 'tensor_fqn'.")
                info = get_arg_info_from_tensor_fqn(model, tensor_fqn)
                module = info["module"]
                if module is not None:
                    module.qconfig = QConfig(  # type: ignore[assignment]
                        activation=PerChannelNormObserver,
                        weight=default_placeholder_observer,
                    )
        torch.ao.quantization.prepare(model, inplace=True)
        super().prepare(model, config)

    # ------------------------------------------------------------------
    # FIM accumulation (call after each calibration backward pass)
    # ------------------------------------------------------------------

    def accumulate_fim(self) -> None:
        """Accumulate eFIM diagonal statistics from the current gradients.

        Call this *after* ``loss.backward()`` and *before*
        ``model.zero_grad()`` for each calibration mini-batch.  The squared
        gradients are added to an internal per-parameter accumulator.

        Example::

            for X, y in calibration_loader:
                loss = criterion(model(X), y)
                loss.backward()
                pruner.accumulate_fim()
                model.zero_grad()
        """
        for group in self.groups:
            module: nn.Module = group["module"]
            tensor_name: str = group["tensor_name"]

            # Retrieve the original (pre-mask) parameter tensor
            param = getattr(module.parametrizations, tensor_name).original
            if param.grad is None:
                continue

            grad_sq = param.grad.detach() ** 2
            key = id(module)
            if key not in self._fim_scores:
                self._fim_scores[key] = torch.zeros_like(param.data)
            self._fim_scores[key].add_(grad_sq)

        self._fim_steps += 1

    # ------------------------------------------------------------------
    # Mask update — core pruning logic
    # ------------------------------------------------------------------

    def update_mask(  # type: ignore[override]
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity_level: float,
        **kwargs: object,
    ) -> None:
        """Compute and apply the FIM-guided sparsity mask.

        Weights with the *lowest* eFIM scores are pruned (their removal is
        cheapest in terms of expected loss increase).

        Args:
            module: The module whose weight is being pruned.
            tensor_name: Name of the weight parameter (usually ``"weight"``).
            sparsity_level: Fraction of parameters to prune.
        """
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        key = id(module)
        if key in self._fim_scores and self._fim_steps > 0:
            # Normalise by number of calibration steps
            fim = self._fim_scores[key] / max(self._fim_steps, 1)
        else:
            # No calibration data — fall back to magnitude pruning
            warnings.warn(
                "FisherPruner: no FIM statistics found for a module. "
                "Run calibration forward/backward passes and call "
                "accumulate_fim() before step(). Falling back to "
                "magnitude-based pruning for this layer."
            )
            tensor = getattr(module.parametrizations, tensor_name).original
            fim = torch.abs(tensor)

        block_size = fim.numel()
        semi_block = kwargs.get("semi_structured_block_size")
        if semi_block is not None:
            block_size = int(semi_block)
            num_pruned = block_size // 2
        else:
            num_pruned = int(block_size * sparsity_level)

        # Prune lowest-FIM-score weights (they matter least to the loss)
        prune_inds = fim.view(-1, block_size).argsort(dim=1)[:, :num_pruned]
        mask.data.view(-1, block_size).scatter_(
            1, prune_inds, torch.zeros_like(prune_inds, dtype=mask.dtype)
        )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def squash_mask(
        self,
        params_to_keep: Optional[tuple[str, ...]] = None,
        params_to_keep_per_layer: Optional[dict[str, tuple[str, ...]]] = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Finalise pruning: remove qconfig observers and parametrizations."""
        for group in self.groups:
            _remove_qconfig(group["module"])
        self._fim_scores.clear()
        self._fim_steps = 0
        super().squash_mask(
            params_to_keep=params_to_keep,
            params_to_keep_per_layer=params_to_keep_per_layer,
        )
