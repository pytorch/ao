# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict

from ..distributed_utils import (
    _is_dtensor,
    _is_main_process,
    _maybe_async_aggregate,
    _sum_async_streams,
)
from ..utils import instantiate_module
from .prox_executor import apply_global_prox, apply_prox_to_param


class PruneOptimizer(Optimizer):
    """Wraps a base optimizer to apply proximal updates that induce sparsity
    or low-rank structure during training.

    Arguments:
        base_optimizer: The underlying optimizer (e.g., SGD or AdamW) that
            updates the latent parameters.
        warmup_steps: Number of initial steps to run before applying proximal
            updates, during which the optimizer behaves like the base optimizer.
        healing_start_step: Step at which to start the "healing" phase, where
            pruned parameters are frozen. Must be greater than warmup_steps.
        reg_lambda: Regularization strength for the proximal updates. Can be
            overridden per parameter group.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        warmup_steps: int = 0,
        healing_start_step: int = sys.maxsize,
        reg_lambda: float = 0.0,
    ) -> None:
        # need to reconstruct these objects if loading checkpoint
        self.base_optimizer = base_optimizer

        # need to store these attributes in state_dict for checkpoint
        assert warmup_steps < healing_start_step, (
            f"Invalid {warmup_steps=} >= {healing_start_step=}"
        )
        self.num_steps = 0
        self.warmup_steps = warmup_steps
        self.healing_start_step = healing_start_step

        for group in self.regularized_param_groups():
            group.setdefault("gamma", 0.0)
            group.setdefault("reg_lambda", reg_lambda)
            self._validate_prox_through_heal(group)
            if group.get("min_sparsity_schedule", False):
                assert self.healing_start_step != sys.maxsize, (
                    "min_sparsity_schedule requires a finite healing_start_step; "
                    "the ramp ends when the mask freezes."
                )

        self.relative_sparsity = 0
        self.relative_factored_frac = 0

        # NOTE: Filling state dict here cause Adam(W) error, which assumes
        # empty state[p] at first step() where optimizer states are initialized

    def __getattribute__(self, name: str):
        try:
            attr = super(Optimizer, self).__getattribute__(name)
        except AttributeError:
            attr = self.base_optimizer.__getattribute__(name)
        return attr

    def __repr__(self) -> str:
        base_optimizer = "\n    ".join(self.base_optimizer.__repr__().split("\n"))
        extra_repr = "\n  ".join(("(", base_optimizer))
        return f"{self.__class__.__name__} {extra_repr}\n)"

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.base_optimizer.__setstate__(state)
        for i, group in enumerate(self.regularized_param_groups()):
            group.setdefault("gamma", 0.0)
            group.setdefault("reg_lambda", 0.0)
            if i == 0:
                group.setdefault("num_steps", 0)

    @property
    def state(self) -> defaultdict[Tensor, Any]:  # pyre-ignore[3]
        return self._state if hasattr(self, "_state") else self.base_optimizer.state

    @torch._disable_dynamo
    def state_dict(self) -> StateDict:
        return self.base_optimizer.state_dict()

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        self.base_optimizer.load_state_dict(state_dict)

    @torch._disable_dynamo
    def patch_state_dict(self, state_dict: StateDict) -> None:
        """Fix missing state after calling torch.distributed.checkpoint.load"""
        for i, group in enumerate(self.regularized_param_groups()):
            state_group = state_dict["param_groups"][i]
            for k in ("reg_lambda", "num_steps", "gamma"):
                if k in state_group:
                    group[k] = state_group[k]

    @property
    def num_steps(self) -> int:
        for group in self.regularized_param_groups():
            return group.get("num_steps", 0)

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        for group in self.regularized_param_groups():
            group["num_steps"] = value
            return

    @num_steps.deleter
    def num_steps(self) -> None:
        for group in self.regularized_param_groups():
            group.pop("num_steps", None)
            return

    def regularized_param_groups(self):  # pyre-ignore[3]
        """Yield parameter groups that need to be pruned."""
        for group in self.param_groups:
            if group.get("prox_type"):
                yield group

    def _get_prox_kwargs(self, group: dict[str, Any]) -> dict[str, Any]:
        prox_kwargs = {}
        if group["prox_type"] == "NMSparseConstraint":
            assert "n_nonzero" in group, (
                "NMSparseConstraint requires 'n_nonzero' in prune config"
            )
            prox_kwargs["n_nonzero"] = group["n_nonzero"]
        elif group["prox_type"] in (
            "MinSparsityConstraint",
            "MinRankConstraint",
        ):
            assert "min_sparsity" in group, (
                f"{group['prox_type']} requires 'min_sparsity' in prune config"
            )
            prox_kwargs["min_sparsity"] = self._effective_min_sparsity(group)
        return prox_kwargs

    def _effective_min_sparsity(self, group: dict[str, Any]) -> float:
        """Cubic ramp from 0 to the target before healing freezes the mask.

        When ``min_sparsity_schedule`` is unset, returns the static target. The
        ramp reaches its target at ``healing_start_step - 1``, the last step on
        which a proximal update can materialize the final hard mask.
        """
        target = group["min_sparsity"]
        if not group.get("min_sparsity_schedule", False):
            return target
        n = self.num_steps
        final_prune_step = self.healing_start_step - 1
        if n >= final_prune_step:
            return target
        if n <= self.warmup_steps:
            return 0.0
        t = (n - self.warmup_steps) / (self.healing_start_step - self.warmup_steps)
        return target * (1 - (1 - t) ** 3)

    @staticmethod
    def _get_grouper_kwargs(group: dict[str, Any]) -> dict[str, Any]:
        grouper_kwargs = {}
        if group["group_type"].startswith("AttentionHeadGrouper"):
            grouper_kwargs["num_heads"] = group["num_heads"]
        elif group["group_type"] == "KElementGrouper":
            grouper_kwargs["k"] = group["k"]
        elif group["group_type"] == "PackedSVDGrouper":
            grouper_kwargs["npack"] = group["npack"]
            if "pack_dim" in group:
                grouper_kwargs["pack_dim"] = group["pack_dim"]
        return grouper_kwargs

    def _set_gamma(self, group):
        # AProx in practice: ensure shrinkage coefficient >= 1
        group["gamma"] += group["lr"]

    @staticmethod
    def _get_sv_count(p, state, grouper_kwargs, prox_kwargs):
        if not prox_kwargs["is_svd_grouper"]:
            return None
        if _is_dtensor(p) and p.device_mesh.get_coordinate() is None:
            return None
        npack = grouper_kwargs.get("npack", 1)
        return state.setdefault(
            "sv_count", torch.zeros(npack, dtype=torch.int, device=p.device)
        )

    def _build_group_artifacts(self, group: dict[str, Any]):
        grouper_cls = instantiate_module(
            f"torchao.prototype.pat.group.{group['group_type']}"
        )
        return grouper_cls, self._get_grouper_kwargs(group)

    def _build_global_prox_artifacts(self, group: dict[str, Any]):
        """Build global prox artifacts without resolving the scheduled budget."""
        assert "min_sparsity" in group, (
            "GlobalMinSparsityConstraint requires 'min_sparsity' in prune config"
        )
        prox_map = instantiate_module(
            f"torchao.prototype.pat.optim.{group['prox_type']}"
        )(
            group["reg_lambda"],
            min_sparsity=group["min_sparsity"],
            score_type=group.get("score_type", "rms"),
        )
        grouper_cls, grouper_kwargs = self._build_group_artifacts(group)
        return prox_map, grouper_cls, grouper_kwargs

    def _build_prox_artifacts(self, group: dict[str, Any]):
        """Build the prox and grouper objects shared by pruning and healing."""
        prox_map = instantiate_module(
            f"torchao.prototype.pat.optim.{group['prox_type']}"
        )(group["reg_lambda"], **self._get_prox_kwargs(group))
        grouper_cls, grouper_kwargs = self._build_group_artifacts(group)
        prox_kwargs = {
            "gamma": group["gamma"],
            "gamma_index_slope": group.get("gamma_index_slope", 0.0),
            "disable_vmap": group["group_type"].endswith(
                ("ElemGrouper", "LayerGrouper")
            ),
            "is_svd_grouper": group["group_type"].endswith("SVDGrouper"),
            "zero_elts_are_counts": group["prox_type"]
            in ("NMSparseConstraint", "MinSparsityConstraint", "MinRankConstraint"),
        }
        return prox_map, grouper_cls, grouper_kwargs, prox_kwargs

    def should_prune(self, group: dict[str, Any], step: int) -> bool:
        """Run the group's prox map every ``prox_freq`` steps after warmup."""
        hard_constraints = {
            "GlobalMinSparsityConstraint",
            "MinRankConstraint",
            "MinSparsityConstraint",
            "NMSparseConstraint",
        }
        if (
            step == self.healing_start_step - 1
            and group.get("prox_type") in hard_constraints
        ):
            # Materialize the final hard mask immediately before healing,
            # regardless of prox_freq alignment.
            return True
        freq = group.get("prox_freq", 1)
        if freq <= 1:
            return True
        offset = step - self.warmup_steps
        return offset >= 0 and offset % freq == 0

    @staticmethod
    def _validate_prox_through_heal(group: dict[str, Any]) -> bool:
        is_svd_grouper = str(group.get("group_type", "")).endswith("SVDGrouper")
        if group.get("prox_through_heal", False) and not is_svd_grouper:
            raise ValueError(
                "prox_through_heal=True requires an SVD grouper, but got "
                f"group_type={group.get('group_type')!r}."
            )
        return is_svd_grouper

    def _prox_through_heal(self, group: dict[str, Any]) -> bool:
        """Default to reapplying hard rank constraints during healing."""
        is_svd_grouper = self._validate_prox_through_heal(group)
        if "prox_through_heal" in group:
            return bool(group["prox_through_heal"])
        return is_svd_grouper and group.get("prox_type") == "MinRankConstraint"

    def _init_latent_state(self):
        for group in self.regularized_param_groups():
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None or "latent" in state:
                    continue
                state["latent"] = p.detach().clone()

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # During healing, literal zeros are frozen by gradient masking. Groups
        # opted into through-heal instead reapply their prox map because dense
        # low-rank weights have no literal zeros to mask. Soft SVD maps default
        # to unconstrained healing unless they explicitly opt in.
        healing_masks = {}
        is_healing = self.num_steps >= self.healing_start_step
        if is_healing:
            for group in self.regularized_param_groups():
                if self._prox_through_heal(group):
                    continue
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    mask = p.ne(0)
                    healing_masks[id(p)] = mask
                    if _is_dtensor(p):
                        p.grad.mul_(mask)
                    else:
                        p.grad.masked_fill_(~mask, 0)

        if self.num_steps < self.warmup_steps or is_healing:
            # run base optimizer only during warmup and healing periods
            loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]
            # re-zero pruned params after step
            for group in self.regularized_param_groups():
                for p in group["params"]:
                    mask = healing_masks.get(id(p))
                    if mask is not None:
                        if _is_dtensor(p):
                            p.mul_(mask)
                        else:
                            p.masked_fill_(~mask, 0)
            del healing_masks
            if is_healing:
                self._apply_prox_to_through_heal_groups()
            self._init_latent_state()
            self.num_steps += 1
            return loss

        if self.num_steps == self.warmup_steps:
            # first PAT step: save latent params
            self.save_latent_params()
        else:
            # restore latent params for base optimizer update
            self.restore_latent_params()

        # call base optimizer step() method to update latent parameters
        loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]

        if hasattr(self, "_state"):
            assert self.warmup_steps == 0
            # restore the temporary state to the base optimizer's state
            for p in self._state.keys():
                self.base_optimizer.state[p]["latent"] = self._state[p]["latent"]
            del self._state

        regularized_params = 0
        regularized_unfactored_size = 0
        dist_is_init = torch.distributed.is_initialized()
        if dist_is_init:
            regularized_zeros_buf = []
            regularized_factored_size_buf = []

        regularized_zeros = 0
        regularized_factored_size = 0
        all_groups_ran = True
        for group in self.regularized_param_groups():
            # Advance on every step so gamma tracks cumulative learning rate,
            # including steps skipped by prox_freq.
            self._set_gamma(group)

            if not self.should_prune(group, self.num_steps):
                all_groups_ran = False
                # Keep latent parameters aligned with the base optimizer while
                # retaining cached sparsity and factorization metrics.
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    self.state[p]["latent"].copy_(p)
                continue

            if group["prox_type"] == "GlobalMinSparsityConstraint":
                prox_map, grouper_cls, grouper_kwargs = (
                    self._build_global_prox_artifacts(group)
                )
                params = [p for p in group["params"] if p.requires_grad]
                for p in params:
                    self.state[p]["latent"].copy_(p)
                global_result = apply_global_prox(
                    params,
                    prox_map,
                    grouper_cls,
                    grouper_kwargs,
                    self._effective_min_sparsity(group),
                )
                for param_result in global_result.parameters:
                    state = self.state[param_result.parameter]
                    state["sparsity_frac"] = (
                        param_result.zero_elts / param_result.numel
                        if param_result.numel
                        else 0.0
                    )
                regularized_zeros += global_result.zero_elts
                regularized_params += global_result.numel
                continue

            prox_map, grouper_cls, grouper_kwargs, prox_kwargs = (
                self._build_prox_artifacts(group)
            )
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                state = self.state[p]
                state["latent"].copy_(p)
                result = apply_prox_to_param(
                    p,
                    prox_map,
                    grouper_cls,
                    grouper_kwargs,
                    prox_kwargs,
                    sv_count=self._get_sv_count(p, state, grouper_kwargs, prox_kwargs),
                )
                if result is None:
                    continue

                zero_elts = result.zero_elts
                zeros_are_summed = result.zeros_are_summed
                numel = result.numel

                if zeros_are_summed:
                    state["sparsity_frac"] = zero_elts / numel
                elif dist_is_init:
                    _maybe_async_aggregate(regularized_zeros_buf, zero_elts)

                if torch.is_tensor(zero_elts):
                    zero_elts = zero_elts.item()

                if prox_kwargs["is_svd_grouper"]:
                    assert result.unfactored_size is not None
                    assert result.matrix_rows is not None
                    assert result.matrix_cols is not None
                    unfactored_size = result.unfactored_size
                    n_singular_vals = numel - zero_elts
                    factored_size = (
                        result.matrix_rows + result.matrix_cols
                    ) * n_singular_vals
                    group["factored_frac"] = factored_size / unfactored_size
                    if zeros_are_summed:
                        regularized_factored_size += factored_size
                    else:
                        _maybe_async_aggregate(
                            regularized_factored_size_buf,
                            torch.tensor(
                                factored_size, dtype=torch.int, device=p.device
                            ),
                        )

                    regularized_unfactored_size += unfactored_size
                    regularized_zeros += max(unfactored_size - factored_size, 0)
                    regularized_params += unfactored_size
                else:
                    regularized_zeros += zero_elts
                    regularized_params += numel

        self.num_steps += 1

        if torch.distributed.is_initialized() and _is_main_process():
            regularized_zeros += _sum_async_streams(regularized_zeros_buf)
            regularized_factored_size += _sum_async_streams(
                regularized_factored_size_buf
            )

        if all_groups_ran and (
            regularized_params > 0 or regularized_unfactored_size > 0
        ):
            # DTensor subset-mesh participants compute complete metrics locally;
            # ranks outside the mesh have no processed parameters and retain
            # their previous values instead of publishing zeros.
            self.relative_sparsity = (
                regularized_zeros / regularized_params
                if regularized_params > 0
                else 0.0
            )
            self.relative_factored_frac = (
                regularized_factored_size / regularized_unfactored_size
                if regularized_unfactored_size > 0
                else 0.0
            )

        return loss

    @torch.no_grad()
    def _apply_prox_to_through_heal_groups(self) -> None:
        """Reapply opted-in prox maps during healing without advancing gamma."""
        for group in self.regularized_param_groups():
            if not self._prox_through_heal(group):
                continue
            if not self.should_prune(group, self.num_steps):
                continue
            prox_map, grouper_cls, grouper_kwargs, prox_kwargs = (
                self._build_prox_artifacts(group)
            )
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                apply_prox_to_param(
                    p,
                    prox_map,
                    grouper_cls,
                    grouper_kwargs,
                    prox_kwargs,
                    sv_count=self._get_sv_count(p, state, grouper_kwargs, prox_kwargs),
                )

    @torch._disable_dynamo
    def restore_latent_params(self) -> None:
        """Restore latent parameters as optimizer parameters"""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    p.copy_(self.state[p]["latent"])

    @torch._disable_dynamo
    def save_latent_params(self) -> None:
        """Save updated latent parameters before applying prox-map"""
        if self.warmup_steps == 0:
            assert len(self.state) == 0, "Expected empty state at first step()"
            # Maintain the invariant that `len(self.state) == 0` before first
            # self.base_optimizer.step() call by using a temporary state buffer
            self._state = defaultdict(dict)

        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    try:
                        self.state[p]["latent"].copy_(p)
                    except KeyError:
                        self.state[p]["latent"] = p.detach().clone()
