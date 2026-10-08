# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Partial, Shard

from ..distributed_utils import _is_dtensor
from ..group.dim import Dim0Grouper, Dim1Grouper
from ..group.grouper import ElemGrouper, LayerGrouper
from ..utils import get_index_linspace
from .group_lasso import ProxGroupLasso, ProxGroupLassoVectorized


@dataclass
class ProxResult:
    """Result of applying a prox map to one parameter."""

    zero_elts: int | Tensor
    group_norm: Tensor
    zeros_are_summed: bool
    numel: int
    matrix_rows: int | None = None
    matrix_cols: int | None = None
    unfactored_size: int | None = None


@dataclass(frozen=True)
class ParameterSparsity:
    """Element-level sparsity produced for one globally pruned parameter."""

    parameter: Tensor
    zero_elts: int
    numel: int


@dataclass(frozen=True)
class GlobalProxResult:
    """Result of applying one global budget across a parameter group."""

    parameters: tuple[ParameterSparsity, ...]
    zero_elts: int
    numel: int


@dataclass(frozen=True)
class CoupledProxResult:
    """Final literal zeros and shared channel selection for a coupled cluster."""

    parameters: tuple[ParameterSparsity, ...]
    zero_elts: int
    numel: int
    zero_channels: int
    channels: int


class GlobalDTensorMaterializationPolicy(Enum):
    """Internal policy for full DTensor views used by global selection."""

    CACHE_FULL_TENSORS = "cache_full_tensors"


def _apply_prox_dtensor(grouper, prox_map, p, gamma, gamma_in_dims):
    """Apply ``prox_map`` to a DTensor parameter via ``local_map``."""
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, device=p.device)

    p_in_placements = tuple(
        Shard(grouper.in_dims)
        if grouper.in_dims is not None and placement.is_shard()
        else placement
        for placement in grouper.p.placements
    )
    if grouper.in_dims is not None and gamma.dim() > 0:
        gamma = distribute_tensor(
            gamma.unsqueeze(int(not grouper.in_dims)),
            device_mesh=p.device_mesh,
            placements=p_in_placements,
        )
        gamma_in_dims = grouper.in_dims
    else:
        gamma = distribute_tensor(gamma, device_mesh=p.device_mesh)

    if isinstance(prox_map, ProxGroupLasso):
        prox_map_vec = ProxGroupLassoVectorized(
            prox_map.reg_lambda,
            reduce_dim=int(not grouper.in_dims),
        )
        local_fn = prox_map_vec.apply_
    else:
        local_fn = torch.vmap(
            prox_map.apply_,
            in_dims=(grouper.in_dims, gamma_in_dims),
            out_dims=(0, 0),
        )

    # Redistribute explicitly so the in-place prox mutation lands on a tensor
    # that can be copied back; local_map's redistribution would discard it.
    needs_redistribute = tuple(grouper.p.placements) != p_in_placements
    p_for_prox = (
        grouper.p.redistribute(placements=p_in_placements)
        if needs_redistribute
        else grouper.p
    )

    zero_out_placements = tuple(
        Partial() if placement.is_shard() else placement
        for placement in p_in_placements
    )
    group_norm_out_placements = tuple(
        Shard(0) if placement.is_shard() else placement for placement in p_in_placements
    )

    zero_elts_per_group, group_norm = local_map(
        local_fn,
        out_placements=(
            zero_out_placements,
            group_norm_out_placements,
        ),
        in_placements=(
            p_in_placements,
            gamma.placements if _is_dtensor(gamma) else None,
        ),
        redistribute_inputs=False,
    )(p_for_prox, gamma)

    if needs_redistribute:
        grouper.p.copy_(p_for_prox.redistribute(placements=grouper.p.placements))

    return zero_elts_per_group.full_tensor().sum().item(), group_norm


def apply_prox(
    grouper,
    prox_map,
    p,
    sv_count=None,
    **prox_kwargs,
) -> tuple[int | Tensor, Tensor, bool]:
    """Apply ``prox_map`` to the grouped parameter tensor ``p`` in place.

    The return value mirrors the historical ``PruneOptimizer._apply_prox``
    contract: zero element count, group norm, and whether the zero count is
    already globally summed.
    """
    if _is_dtensor(p) and p.device_mesh.get_coordinate() is None:
        return 0, p.to_local().new_zeros(()), True

    gamma = prox_kwargs["gamma"]
    zeros_are_summed = False
    with grouper:
        gamma_in_dims = None
        if prox_kwargs["gamma_index_slope"] > 0:
            gamma = gamma * get_index_linspace(
                prox_kwargs["gamma_index_slope"],
                grouper.n_groups(),
                device=p.device,
            )
            gamma_in_dims = 0

        if prox_kwargs["disable_vmap"] or prox_map.whole_tensor:
            transpose = getattr(grouper, "in_dims", 0) == 1 and grouper.p.dim() == 2
            if _is_dtensor(grouper.p):
                # Index-mutating and whole-tensor prox maps need a global view.
                full = grouper.p.full_tensor()
                view = full.transpose(0, 1) if transpose else full
                zero_elts, group_norm = prox_map.apply_(view, gamma)
                grouper.p.copy_(
                    distribute_tensor(
                        full,
                        device_mesh=grouper.p.device_mesh,
                        placements=grouper.p.placements,
                    )
                )
            else:
                view = grouper.p.transpose(0, 1) if transpose else grouper.p
                zero_elts, group_norm = prox_map.apply_(view, gamma)
            zeros_are_summed = zero_elts.dim() == 0
        else:
            if not prox_kwargs["is_svd_grouper"] and _is_dtensor(p):
                zero_elts, group_norm = _apply_prox_dtensor(
                    grouper,
                    prox_map,
                    p,
                    gamma,
                    gamma_in_dims,
                )
            else:
                zero_elts_per_group, group_norm = torch.vmap(
                    prox_map.apply_,
                    in_dims=(grouper.in_dims, gamma_in_dims),
                    out_dims=(0, 0),
                )(grouper.p, gamma)
                zero_elts = zero_elts_per_group.sum().item()
            zeros_are_summed = True

            if not prox_kwargs["is_svd_grouper"] and not prox_kwargs.get(
                "zero_elts_are_counts", False
            ):
                zero_elts *= grouper.group_size()

        if prox_kwargs["is_svd_grouper"]:
            dim = -1 if grouper.p.dim() > 1 else None
            sv_count.copy_(
                (grouper.p != 0).to(torch.uint8).sum(dim=dim)
                if _is_dtensor(p)
                else torch.count_nonzero(grouper.p, dim=dim)
            )

        return zero_elts, group_norm, zeros_are_summed


def apply_prox_to_param(
    p,
    prox_map,
    grouper_cls,
    grouper_kwargs: dict[str, Any],
    prox_kwargs: dict[str, Any],
    *,
    sv_count=None,
) -> ProxResult | None:
    """Apply a prox map to one parameter, including SVD synchronization."""
    sharded_p = None
    prox_p = p
    if _is_dtensor(p):
        if p.device_mesh.get_coordinate() is None:
            return None
        if prox_kwargs["is_svd_grouper"]:
            sharded_p = p
            prox_p = p.full_tensor()

    grouper = grouper_cls(prox_p, **grouper_kwargs)
    zero_elts, group_norm, zeros_are_summed = apply_prox(
        grouper,
        prox_map,
        prox_p,
        sv_count=sv_count,
        **prox_kwargs,
    )
    result = ProxResult(
        zero_elts=zero_elts,
        group_norm=group_norm,
        zeros_are_summed=zeros_are_summed,
        numel=grouper.p.numel(),
    )
    if prox_kwargs["is_svd_grouper"]:
        result.matrix_rows = grouper.U.size(-2)
        result.matrix_cols = grouper.Vh.size(-1)
        result.unfactored_size = prox_p.numel()

    if sharded_p is not None:
        # Every mesh participant materializes the same full tensor and applies
        # the deterministic SVD/prox locally. Reshard from each rank's local
        # result to avoid WORLD collectives and global-rank-0 assumptions.
        sharded_p.copy_(
            distribute_tensor(
                prox_p,
                device_mesh=sharded_p.device_mesh,
                placements=sharded_p.placements,
                src_data_rank=None,
            )
        )
    return result


def grouped_view(grouper):
    """Return the dense 2-D view represented by ``grouper``."""
    full = grouper.p.full_tensor() if _is_dtensor(grouper.p) else grouper.p
    if isinstance(grouper, ElemGrouper):
        view = full.reshape(-1, 1)
    elif isinstance(grouper, LayerGrouper):
        view = full.reshape(1, -1)
    elif getattr(grouper, "in_dims", 0) == 1 and full.dim() == 2:
        view = full.transpose(0, 1)
    else:
        view = full
    return view, full


def _check_mesh_compatibility(params: Sequence[Tensor], what: str) -> bool:
    """Return False when this rank sits outside the shared DeviceMesh."""
    dtensor_params = [p for p in params if _is_dtensor(p)]
    if not dtensor_params:
        return True
    if len(dtensor_params) != len(params):
        raise ValueError(f"{what} cannot mix dense tensors and DTensors in one group.")
    first_mesh = dtensor_params[0].device_mesh
    if any(p.device_mesh != first_mesh for p in dtensor_params[1:]):
        raise ValueError(f"{what} requires all DTensors to use the same DeviceMesh.")
    return first_mesh.get_coordinate() is not None


def _validate_coupled_specs(specs) -> bool:
    """Preflight layouts using meta tensors, without gathering or changing weights."""
    params = [p for p, _, _ in specs]
    if not params:
        return True
    if len({id(p) for p in params}) != len(params):
        raise ValueError("CoupledMinSparsityConstraint requires distinct parameters")
    if not _check_mesh_compatibility(params, "CoupledMinSparsityConstraint"):
        return False
    if any(p.device != params[0].device for p in params):
        raise ValueError(
            "CoupledMinSparsityConstraint requires every parameter to live on the same device."
        )
    if any(not p.is_floating_point() for p in params):
        raise ValueError(
            "CoupledMinSparsityConstraint requires real floating-point tensors."
        )
    n_channels = None
    for p, cls, kwargs in specs:
        if cls.__name__.endswith("SVDGrouper"):
            raise ValueError("SVD groupers cannot be composed with coupled pruning")
        surrogate = (
            torch.empty(p.shape, device="meta", dtype=p.dtype)
            if _is_dtensor(p)
            else torch.empty_strided(p.shape, p.stride(), device="meta", dtype=p.dtype)
        )
        try:
            with cls(surrogate, **kwargs) as grouper:
                if hasattr(grouper, "_pad_size"):
                    raise ValueError(
                        "CoupledMinSparsityConstraint does not support padded KElementGrouper groups."
                    )
                if isinstance(grouper, ElemGrouper):
                    shape = (grouper.p.numel(), 1)
                elif isinstance(grouper, LayerGrouper):
                    shape = (1, grouper.p.numel())
                elif getattr(grouper, "in_dims", 0) == 1 and grouper.p.dim() == 2:
                    shape = tuple(reversed(grouper.p.shape))
                else:
                    shape = grouper.p.shape
                if len(shape) != 2:
                    raise ValueError(
                        "CoupledMinSparsityConstraint requires a grouper that produces a 2-D (n_channels, group_size) view."
                    )
                if n_channels is None:
                    n_channels = shape[0]
                elif shape[0] != n_channels:
                    raise ValueError(
                        "CoupledMinSparsityConstraint requires every parameter to expose the same leading dimension."
                    )
        except RuntimeError as error:
            raise ValueError(
                "CoupledMinSparsityConstraint encountered an unsupported tensor layout"
            ) from error
    return True


def _shard_mesh_dim(p: Tensor) -> int | None:
    """Find a single plain Shard(0) mesh axis, with replication elsewhere."""
    found = None
    for mesh_dim, placement in enumerate(p.placements):
        if type(placement) is Shard:
            if placement.dim != 0 or found is not None:
                return None
            found = mesh_dim
        elif not placement.is_replicate():
            return None
    return found


def _coupled_fast_path(specs) -> tuple[Any, int, int] | None:
    """Find eligible mesh-local reader reductions and writer gathers.

    Readers (Dim1) are row-sharded and hold partial norms for every channel.
    Writers (Dim0) own complete channels, requiring evenly sized row blocks.
    Reshaped groupers and other layouts use the materializing reference path.
    """
    if not specs:
        return None
    mesh = None
    mesh_dim = None
    n_channels = None
    defaults = {"start_dim": 1, "end_dim": -1}
    for p, grouper_cls, grouper_kwargs in specs:
        if grouper_cls not in (Dim0Grouper, Dim1Grouper):
            return None
        if any(
            key not in defaults or value != defaults[key]
            for key, value in grouper_kwargs.items()
        ):
            return None
        if not _is_dtensor(p) or p.dim() != 2:
            return None
        p_mesh_dim = _shard_mesh_dim(p)
        if p_mesh_dim is None:
            return None
        if mesh is None:
            mesh, mesh_dim = p.device_mesh, p_mesh_dim
        elif p.device_mesh != mesh or p_mesh_dim != mesh_dim:
            return None
        channels = p.size(1) if grouper_cls is Dim1Grouper else p.size(0)
        if n_channels is None:
            n_channels = channels
        elif channels != n_channels:
            return None
    n_shard = mesh.size(mesh_dim)
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        return None
    if any(cls is Dim0Grouper and p.size(0) % n_shard for p, cls, _ in specs):
        return None
    return mesh.get_group(mesh_dim), coordinate[mesh_dim], n_shard


def _coupled_prox_sharded(
    specs, prox_map, min_sparsity, shard_group, shard_rank, n_shard
) -> CoupledProxResult:
    """Apply a coupled mask with at most three mesh-scoped collectives.

    Aggregate all readers' partials together, gather all writers' channel
    blocks together, then reduce one vector of literal-zero counts. Replica
    mesh dimensions never participate in these collectives.
    """
    reader_sq = None
    writer_blocks = []
    group_size_sum = 0
    for p, cls, _ in specs:
        local = p.to_local()
        if cls is Dim1Grouper:
            part = local.float().square().sum(dim=0)
            reader_sq = part if reader_sq is None else reader_sq + part
            group_size_sum += p.size(0)
        else:
            writer_blocks.append(local.float().square().sum(dim=1))
            group_size_sum += p.size(1)

    if reader_sq is not None:
        dist.all_reduce(reader_sq, group=shard_group)
    writer_sq = None
    if writer_blocks:
        stacked = torch.stack(writer_blocks).contiguous()
        n_writers, local_channels = stacked.shape
        gathered = torch.empty(
            n_shard * n_writers,
            local_channels,
            dtype=stacked.dtype,
            device=stacked.device,
        )
        dist.all_gather_into_tensor(gathered, stacked, group=shard_group)
        writer_sq = (
            gathered.view(n_shard, n_writers, local_channels)
            .permute(1, 0, 2)
            .reshape(n_writers, n_shard * local_channels)
            .sum(dim=0)
        )
    sq_norm_sum = (
        reader_sq
        if writer_sq is None
        else (writer_sq if reader_sq is None else reader_sq + writer_sq)
    )
    scores = prox_map.combine_scores(sq_norm_sum, group_size_sum)
    n_channels = scores.numel()
    n_zero = math.ceil(min_sparsity * n_channels)
    if n_zero <= 0:
        zero_idx = torch.empty(0, dtype=torch.long, device=scores.device)
    elif n_zero >= n_channels:
        zero_idx = torch.arange(n_channels, device=scores.device)
    else:
        _, zero_idx = torch.topk(scores, k=n_zero, largest=False, sorted=False)

    zero_counts = []
    for p, cls, _ in specs:
        local = p.to_local()
        if cls is Dim1Grouper:
            local[:, zero_idx] = 0.0
        else:
            offset = shard_rank * local.size(0)
            local_idx = (
                zero_idx[(zero_idx >= offset) & (zero_idx < offset + local.size(0))]
                - offset
            )
            local[local_idx] = 0.0
        zero_counts.append((local == 0).sum())
    counts = torch.stack(zero_counts).to(torch.long)
    dist.all_reduce(counts, group=shard_group)
    parameters = tuple(
        ParameterSparsity(parameter=p, zero_elts=zeros, numel=p.numel())
        for (p, _, _), zeros in zip(specs, counts.tolist())
    )
    return CoupledProxResult(
        parameters=parameters,
        zero_elts=sum(p.zero_elts for p in parameters),
        numel=sum(p.numel for p in parameters),
        zero_channels=zero_idx.numel(),
        channels=n_channels,
    )


def apply_coupled_prox(
    specs: Sequence[tuple[Tensor, Any, dict[str, Any]]],
    prox_map,
    min_sparsity: float,
    *,
    materialization_policy: GlobalDTensorMaterializationPolicy = (
        GlobalDTensorMaterializationPolicy.CACHE_FULL_TENSORS
    ),
) -> CoupledProxResult:
    """Zero one shared set of channel indices across ``specs`` in place.

    Each spec holds (parameter, grouper class, grouper kwargs). Groupers must
    expose 2-D (channels, group_size) views of the same leading dimension.
    Reported element counts include all final literal zeros, including zeros
    from a prior prox; callers must account each participating parameter once.

    Eligible Shard(0) layouts avoid full tensors. The fallback caches every
    grouped full tensor, so peak dense memory is the sum of clustered tensors.
    Floating-point reduction order can affect selection at tied or near-tied
    score boundaries across layouts; selection is consistent within a mesh.
    """
    if (
        materialization_policy
        is not GlobalDTensorMaterializationPolicy.CACHE_FULL_TENSORS
    ):
        raise NotImplementedError(
            f"Unsupported materialization policy: {materialization_policy}"
        )
    if not 0.0 <= min_sparsity <= 1.0:
        raise ValueError("min_sparsity must be in [0, 1]")
    empty = CoupledProxResult(
        parameters=(), zero_elts=0, numel=0, zero_channels=0, channels=0
    )
    if not specs:
        return empty
    if not _validate_coupled_specs(specs):
        return empty
    fast_path = _coupled_fast_path(specs) if dist.is_initialized() else None
    if fast_path is not None:
        return _coupled_prox_sharded(specs, prox_map, min_sparsity, *fast_path)

    with contextlib.ExitStack() as stack:
        entries = []
        n_channels = None
        sq_norm_sum = None
        group_size_sum = 0
        for p, cls, kwargs in specs:
            # Group the dense view so uneven DTensor reshapes cannot fail before
            # the fallback gets a chance to materialize the parameter.
            dense = p.full_tensor() if _is_dtensor(p) else p
            grouper = stack.enter_context(cls(dense, **kwargs))
            if hasattr(grouper, "_pad_size"):
                raise ValueError(
                    "CoupledMinSparsityConstraint does not support padded KElementGrouper groups."
                )
            view, full = grouped_view(grouper)
            if view.dim() != 2:
                raise ValueError(
                    "CoupledMinSparsityConstraint requires a grouper that produces a 2-D "
                    f"(n_channels, group_size) view; {cls.__name__} produced shape {tuple(view.shape)}."
                )
            if n_channels is None:
                n_channels = view.size(0)
            elif view.size(0) != n_channels:
                raise ValueError(
                    "CoupledMinSparsityConstraint requires every parameter to expose the same leading dimension."
                )
            sq_norm = torch.linalg.vector_norm(view.float(), dim=1).square()
            sq_norm_sum = sq_norm if sq_norm_sum is None else sq_norm_sum + sq_norm
            group_size_sum += view.size(1)
            entries.append((p, grouper, view, full, dense))

        scores = prox_map.combine_scores(sq_norm_sum, group_size_sum)
        n_zero = math.ceil(min_sparsity * n_channels)
        if n_zero <= 0:
            zero_idx = torch.empty(0, dtype=torch.long, device=scores.device)
        elif n_zero >= n_channels:
            zero_idx = torch.arange(n_channels, device=scores.device)
        else:
            _, zero_idx = torch.topk(scores, k=n_zero, largest=False, sorted=False)
        for p, grouper, view, full, dense in entries:
            prox_map.zero_groups_(view, zero_idx)
            if isinstance(grouper, (ElemGrouper, LayerGrouper)):
                # Flattening a noncontiguous tensor may have allocated a copy.
                full.copy_(view.reshape(full.shape))
        # Flush reshaped groupers' write-back before redistributing/counting.
        stack.close()
        zero_counts = []
        for p, _, _, _, dense in entries:
            if _is_dtensor(p):
                p.copy_(
                    distribute_tensor(
                        dense,
                        device_mesh=p.device_mesh,
                        placements=p.placements,
                        src_data_rank=None,
                    )
                )
            zero_counts.append((dense == 0).sum())
        parameters = tuple(
            ParameterSparsity(parameter=p, zero_elts=zeros, numel=dense.numel())
            for (p, _, _, _, dense), zeros in zip(
                entries, torch.stack(zero_counts).to(torch.long).tolist()
            )
        )
        return CoupledProxResult(
            parameters=parameters,
            zero_elts=sum(p.zero_elts for p in parameters),
            numel=sum(p.numel for p in parameters),
            zero_channels=zero_idx.numel(),
            channels=n_channels,
        )


def apply_global_prox(
    params: Sequence[Tensor],
    prox_map,
    grouper_cls,
    grouper_kwargs: dict[str, Any],
    min_sparsity: float,
    *,
    score_group_count_ref: int | None = None,
    materialization_policy: GlobalDTensorMaterializationPolicy = (
        GlobalDTensorMaterializationPolicy.CACHE_FULL_TENSORS
    ),
) -> GlobalProxResult:
    """Apply one global group-count budget across ``params``.

    ``CACHE_FULL_TENSORS`` enters every grouper once and retains its grouped
    view until the shared top-k mask is selected. For DTensors this performs one
    ``full_tensor()`` materialization per parameter instead of gathering twice,
    but peak dense memory on each rank is the sum of all materialized parameters
    in the optimizer group. The policy is explicit so a lower-memory/two-gather
    implementation can be added without changing optimizer orchestration.
    """
    if (
        materialization_policy
        is not GlobalDTensorMaterializationPolicy.CACHE_FULL_TENSORS
    ):
        raise NotImplementedError(
            f"Unsupported materialization policy: {materialization_policy}"
        )

    if score_group_count_ref is not None and (
        not isinstance(score_group_count_ref, int)
        or isinstance(score_group_count_ref, bool)
        or score_group_count_ref <= 0
    ):
        raise ValueError("score_group_count_ref must be a positive integer")

    dtensor_params = [p for p in params if _is_dtensor(p)]
    if dtensor_params:
        if len(dtensor_params) != len(params):
            raise ValueError(
                "GlobalMinSparsityConstraint cannot mix dense tensors and "
                "DTensors in one optimizer parameter group."
            )
        first_mesh = dtensor_params[0].device_mesh
        if any(p.device_mesh != first_mesh for p in dtensor_params[1:]):
            raise ValueError(
                "GlobalMinSparsityConstraint requires all DTensors in an "
                "optimizer parameter group to use the same DeviceMesh."
            )
        if first_mesh.get_coordinate() is None:
            return GlobalProxResult(parameters=(), zero_elts=0, numel=0)

    with contextlib.ExitStack() as stack:
        entries = []
        score_chunks = []
        for p in params:
            grouper = stack.enter_context(grouper_cls(p, **grouper_kwargs))
            if hasattr(grouper, "_pad_size"):
                raise ValueError(
                    "GlobalMinSparsityConstraint does not support padded "
                    "KElementGrouper groups; choose a k that divides each "
                    "grouped dimension."
                )
            view, full = grouped_view(grouper)
            if view.dim() != 2:
                raise ValueError(
                    "GlobalMinSparsityConstraint requires a grouper that produces "
                    "a 2-D (n_groups, group_size) view; "
                    f"{type(grouper).__name__} produced shape {tuple(view.shape)} "
                    f"for parameter shape {tuple(p.shape)}."
                )
            scores = prox_map.score(view).detach()
            if score_group_count_ref is not None:
                scores = scores * math.sqrt(scores.numel() / score_group_count_ref)
            entries.append((p, grouper, view, full, scores, grouper.p.numel()))
            score_chunks.append(scores)

        if not score_chunks:
            return GlobalProxResult(parameters=(), zero_elts=0, numel=0)

        score_device = score_chunks[0].device
        if any(scores.device != score_device for scores in score_chunks[1:]):
            raise ValueError(
                "GlobalMinSparsityConstraint requires every parameter in an "
                "optimizer group to produce scores on the same device."
            )

        # Every rank gathers the same full tensors and performs the same global
        # top-k selection, so all ranks must produce identical masks.
        all_scores = torch.cat([scores.reshape(-1) for scores in score_chunks])
        total_groups = all_scores.numel()
        n_zero = math.ceil(min_sparsity * total_groups)
        if n_zero <= 0:
            zero_mask = torch.zeros(
                total_groups, dtype=torch.bool, device=all_scores.device
            )
        elif n_zero >= total_groups:
            zero_mask = torch.ones(
                total_groups, dtype=torch.bool, device=all_scores.device
            )
        else:
            _, drop_idx = torch.topk(all_scores, k=n_zero, largest=False, sorted=False)
            zero_mask = torch.zeros(
                total_groups, dtype=torch.bool, device=all_scores.device
            )
            zero_mask[drop_idx] = True

        zero_counts = []
        offset = 0
        for p, grouper, view, full, scores, numel in entries:
            n_local = scores.numel()
            local_zero_idx = zero_mask[offset : offset + n_local].nonzero(
                as_tuple=True
            )[0]
            offset += n_local

            zeros = prox_map.zero_groups_(view, local_zero_idx)
            if _is_dtensor(grouper.p):
                grouper.p.copy_(
                    distribute_tensor(
                        full,
                        device_mesh=grouper.p.device_mesh,
                        placements=grouper.p.placements,
                    )
                )
            zero_counts.append(
                zeros
                if torch.is_tensor(zeros)
                else torch.tensor(zeros, device=view.device, dtype=torch.long)
            )

        # One device-to-host synchronization for the whole parameter group.
        counts = torch.stack(zero_counts).to(torch.long).tolist()
        parameter_results = []
        group_zeros = 0
        group_params = 0
        for (p, grouper, view, full, scores, numel), zero_elts in zip(entries, counts):
            parameter_results.append(
                ParameterSparsity(parameter=p, zero_elts=zero_elts, numel=numel)
            )
            group_zeros += zero_elts
            group_params += numel

        return GlobalProxResult(
            parameters=tuple(parameter_results),
            zero_elts=group_zeros,
            numel=group_params,
        )
