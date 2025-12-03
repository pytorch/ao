# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from ..utils import HAS_DTENSOR, instantiate_module, is_dtensor, is_main_process
from ..utils.distributed import _maybe_async_aggregate, _sum_async_streams
from ..utils.torch import get_index_linspace

if HAS_DTENSOR:
    from torch.distributed.tensor import distribute_tensor
    from torch.distributed.tensor.experimental import local_map
    from torch.distributed.tensor.placement_types import Partial, Replicate, Shard


class PruneOptimizer(Optimizer):
    """PruneOptimizer assembles functionalities of the following objects:
    a base optimizer (e.g., SGD or AdamW)
        - update the latent variables for QAT
    Other parameters:
        warmup_steps: int >= 0
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        warmup_steps: int = 0,
        reg_lambda: float = 0.0,
    ) -> None:
        # need to reconstruct these objects if loading checkpoint
        self.base_optimizer = base_optimizer

        # need to store these attributes in state_dict for checkpoint
        self.num_steps = 0
        self.warmup_steps = warmup_steps

        self.has_svd = False
        for group in self.regularized_param_groups():
            group["gamma"] = 0.0
            group.setdefault("reg_lambda", reg_lambda)
            if group.get("group_type", None) == "SVDGrouper":
                self.has_svd = True

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

    @property
    def state(self) -> defaultdict[Tensor, Any]:  # pyre-ignore[3]
        return self._state if hasattr(self, "_state") else self.base_optimizer.state

    @torch._disable_dynamo
    def state_dict(self) -> dict[str, Any]:
        return self.base_optimizer.state_dict()

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict)

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

    @staticmethod
    def _get_grouper_kwargs(group) -> dict[str, Any]:
        grouper_kwargs = {}
        if group["group_type"].startswith("AttentionHeadGrouper"):
            grouper_kwargs["num_heads"] = group["num_heads"]
        elif group["group_type"] == "QKGrouper":
            if "qk_pack_dim" in group:
                grouper_kwargs["qk_pack_dim"] = group["qk_pack_dim"]
            if "qk_reg_index" in group:
                grouper_kwargs["qk_reg_index"] = group["qk_reg_index"]
        elif group["group_type"] == "PackedSVDGrouper":
            grouper_kwargs["npack"] = group["npack"]
            if "pack_dim" in group:
                grouper_kwargs["pack_dim"] = group["pack_dim"]
        return grouper_kwargs

    @staticmethod
    def _apply_prox(
        grouper, prox_map, p, sv_count=None, **prox_kwargs
    ) -> tuple[Tensor, bool]:
        """
        Apply `prox_map` to the grouped parameter tensor `p` in place. Update
        `sv_count` if provided. Handles both torch.Tensor and DTensor inputs,
        mirroring `torch.vmap` semantics. Assumes prox_map.apply_ returns an
        integer per group.

        Returns:
            zero_elts: number of zero elements after applying prox map
            zeros_are_summed: whether zero_elts is already globally summed
        """
        gamma = prox_kwargs["gamma"]
        zeros_are_summed = False
        with grouper:
            gamma_in_dims = None
            if prox_kwargs["gamma_index_slope"] > 0:
                # y = slope(2x - 1) + 1
                gamma = gamma * get_index_linspace(
                    prox_kwargs["gamma_index_slope"],
                    grouper.n_groups(),
                    device=p.device,
                )
                gamma_in_dims = 0

            if prox_kwargs["disable_vmap"]:
                # Element- or layer-wise pruning
                zero_elts = prox_map.apply_(grouper.p, gamma)
            else:
                if not prox_kwargs["is_svd_grouper"] and is_dtensor(p):
                    if not torch.is_tensor(gamma):
                        gamma = torch.tensor(gamma, device=p.device)

                    gamma_placements = (Replicate(),)
                    if grouper.in_dims is not None and gamma.dim() > 0:
                        # Shard gamma according to grouper.in_dims
                        gamma_placements = (Shard(grouper.in_dims),)
                        if gamma.dim() <= grouper.in_dims:
                            gamma = gamma.unsqueeze(0)
                    gamma = distribute_tensor(
                        gamma,
                        device_mesh=p.device_mesh,
                        placements=gamma_placements,
                    )

                    # Derive input placements from grouper.p
                    p_in_placements = (
                        Shard(grouper.in_dims)
                        if grouper.in_dims is not None and plc.is_shard()
                        else plc
                        for plc in grouper.p.placements
                    )

                    # Use local_map for DTensor-aware vectorization
                    zero_elts_per_group = local_map(
                        prox_map.apply_,
                        out_placements=[Partial()],
                        in_placements=(
                            p_in_placements,
                            gamma.placements if is_dtensor(gamma) else None,
                        ),
                        redistribute_inputs=True,
                    )(grouper.p, gamma)

                    # Gather counts by calling redistribute implicitly
                    zero_elts = zero_elts_per_group.full_tensor().item()
                else:
                    # torch.Tensor branch - use standard vmap
                    zero_elts_per_group = torch.vmap(
                        prox_map.apply_,
                        in_dims=(grouper.in_dims, gamma_in_dims),
                        out_dims=0,
                    )(grouper.p, gamma)
                    zero_elts = zero_elts_per_group.sum().item()
                zeros_are_summed = True

                # Adjust for group-based pruning
                if not prox_kwargs["is_svd_grouper"]:
                    zero_elts *= grouper.group_size()

            # Record for reconstruction and logging
            if prox_kwargs["is_svd_grouper"]:
                dim = 0 if sv_count.dim() > 1 else None
                sv_count.copy_(
                    (grouper.p != 0).to(torch.uint8).sum(dim=dim)
                    if is_dtensor(p)
                    else torch.count_nonzero(grouper.p, dim=dim)
                )

            return zero_elts, zeros_are_summed

    def _set_gamma(self, group):
        # AProx in practice: ensure shrinkage coefficient >= 1
        group["gamma"] += group["lr"]

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.num_steps < self.warmup_steps:
            # warmup stage: running the base optimizer only
            loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]
            self.num_steps += 1
            return loss

        if self.num_steps == self.warmup_steps:
            # first step of qat, save latent params, instead of restore
            self.save_latent_params()
        else:
            # qat: restore latent params for update by the base optimizer
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
        if torch.distributed.is_initialized():
            regularized_zeros_buf = []
            regularized_factored_size_buf = []

        regularized_zeros = 0
        regularized_factored_size = 0

        for group in self.regularized_param_groups():
            self._set_gamma(group)

            # apply shrinkage to latent parameters in place
            prox_map = instantiate_module(
                f"torchao.prototype.pat.optim.{group['prox_type']}"
            )(group["reg_lambda"])

            # grouper is a context manager that reshapes p if needed
            grouper_cls = instantiate_module(
                f"torchao.prototype.pat.group.{group['group_type']}"
            )
            grouper_kwargs = self._get_grouper_kwargs(group)
            prox_kwargs = {
                "gamma": group["gamma"],
                "gamma_index_slope": group.get("gamma_index_slope", 0.0),
                "disable_vmap": group["group_type"].endswith(
                    ("ElemGrouper", "LayerGrouper")
                ),
                "is_svd_grouper": group["group_type"].endswith("SVDGrouper"),
            }
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                # save latent parameters
                state = self.state[p]
                state["latent"].copy_(p)

                # store the number of non-zero singular values
                if prox_kwargs["is_svd_grouper"]:
                    npack = grouper_kwargs.get("npack", 1)
                    state.setdefault(
                        "sv_count", torch.zeros(npack, dtype=torch.int, device=p.device)
                    )

                # update the full tensor if sharded
                sharded_p = None
                if is_dtensor(p) and prox_kwargs["is_svd_grouper"]:
                    sharded_p = p
                    p = p.full_tensor()

                # only rank 0 of the device mesh should run the grouper
                sv_count = state.get("sv_count")
                if sharded_p is None or sharded_p.device_mesh.get_rank() == 0:
                    grouper = grouper_cls(p, **grouper_kwargs)
                    zero_elts, zeros_are_summed = self._apply_prox(
                        grouper, prox_map, p, sv_count=sv_count, **prox_kwargs
                    )
                    if zeros_are_summed:
                        state["sparsity_frac"] = zero_elts / grouper.p.numel()
                    else:
                        _maybe_async_aggregate(regularized_zeros_buf, zero_elts)

                    if torch.is_tensor(zero_elts):
                        zero_elts = zero_elts.item()

                    if prox_kwargs["is_svd_grouper"]:
                        unfactored_size = grouper.U.size(0) * grouper.Vh.size(1)
                        n_singular_vals = grouper.p.numel() - zero_elts
                        factored_size = (
                            grouper.U.size(0) + grouper.Vh.size(1)
                        ) * n_singular_vals
                        group["factored_frac"] = factored_size / unfactored_size
                        # Only aggregate if not already globally summed
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

                        # Only factor matrices if it reduces params
                        regularized_zeros += max(unfactored_size - factored_size, 0)
                        regularized_params += unfactored_size
                    else:
                        regularized_zeros += zero_elts
                        regularized_params += grouper.p.numel()

                # copy the updated full tensor to the sharded tensor
                if sharded_p is not None:
                    torch.distributed.barrier()
                    if isinstance(sv_count, Tensor):
                        torch.distributed.broadcast(sv_count, src=0)
                    sharded_p.copy_(
                        distribute_tensor(
                            p,
                            device_mesh=sharded_p.device_mesh,
                            placements=sharded_p.placements,
                        )
                    )

        self.num_steps += 1

        if torch.distributed.is_initialized() and is_main_process():
            regularized_zeros += _sum_async_streams(regularized_zeros_buf)
            regularized_factored_size += _sum_async_streams(
                regularized_factored_size_buf
            )

        if is_main_process():
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
                    state = self.state[p]
                    if "latent" not in state:
                        state["latent"] = p.detach().clone()
                    else:
                        state["latent"].copy_(p)
