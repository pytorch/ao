# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom DTensor strategy for aten._scaled_mm with MX blockwise scales.

The default DTensor _scaled_mm strategy copies the data operand's placement
to its scale (e.g., B_t=Shard(1) -> scale_B=Shard(1)). This fails for
MX block scales which are 1D tensors — Shard(1) is invalid on a 1D tensor.

This module registers a custom strategy that correctly maps:
- Data Shard on non-contracting dim -> scale Shard(0) on 1D
- Data Shard on contracting dim -> filtered out (not supported for blockwise)
- Replicate/Partial -> Replicate
"""

from math import prod

import torch
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy
from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
)
from torch.distributed.tensor.placement_types import Partial

aten = torch.ops.aten


def _is_blockwise_1d(scale_shape):
    """Check if scale is a 1D blockwise scale (not scalar, not 2D)."""
    return len(scale_shape) == 1 and prod(scale_shape) > 1


def _mx_scale_placement_for_data(
    data_spec: DTensorSpec,
    scale_shape,
    operand_label: str,
) -> DTensorSpec | None:
    """
    Compute the DTensorSpec for an MX blockwise scale given its data spec.

    For _scaled_mm(A[M,K], B_t[K,N], scale_A, scale_B), the einsum is "mk,kn->mn":
    - A (mk): dim 0 = m (free/non-contracting), dim 1 = k (contracting)
    - B_t (kn): dim 0 = k (contracting), dim 1 = n (free/non-contracting)

    For 1D blockwise scales (flattened from (rows, cols//block_size)):
    - Non-contracting dim shard -> scale Shard(0) (row blocks split)
    - Contracting dim shard -> NOT supported (column blocks interleave within
      row blocks in the swizzled layout)
    """
    mesh = data_spec.mesh
    is_blockwise = _is_blockwise_1d(scale_shape)

    if prod(scale_shape) == 1:
        # Scalar (tensorwise) scale — always Replicate
        return DTensorSpec(mesh, tuple(Replicate() for _ in data_spec.placements))

    new_placements = []
    for p in data_spec.placements:
        if isinstance(p, Replicate):
            new_placements.append(Replicate())
        elif isinstance(p, Partial):
            # Partial on data means this is an intermediate reduction.
            # Scale should be Replicate.
            new_placements.append(Replicate())
        elif isinstance(p, Shard):
            if not is_blockwise:
                # 2D scale: propagate placement directly (existing behavior)
                new_placements.append(p)
            else:
                # 1D blockwise scale: check if shard is on contracting dim
                # A (mk): contracting = dim 1, non-contracting = dim 0
                # B_t (kn): contracting = dim 0, non-contracting = dim 1
                if operand_label == "A":
                    contracting_dim = 1
                else:  # B_t
                    contracting_dim = 0

                if p.dim == contracting_dim:
                    # Contracting dim shard with blockwise scale: not supported
                    return None
                else:
                    # Non-contracting dim shard: scale rows are split
                    new_placements.append(Shard(0))
        else:
            new_placements.append(p)

    return DTensorSpec(mesh, tuple(new_placements))


def _mx_scaled_mm_strategy(op_schema: OpSchema) -> OpStrategy:
    """Custom _scaled_mm strategy supporting MX blockwise scales."""
    mesh = op_schema.get_mesh_from_args()
    (
        self_strategy,
        mat2_strategy,
        scale_self_strategy,
        scale_mat2_strategy,
        bias_strategy,
        scale_result_strategy,
        *_,
    ) = op_schema.args_schema

    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    assert isinstance(scale_self_strategy, OpStrategy)
    assert isinstance(scale_mat2_strategy, OpStrategy)
    if bias_strategy is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support bias")
    if scale_result_strategy is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support scale_result")

    mm_strategy = gen_einsum_strategies("mk,kn->mn", mesh)
    filtered_strategies = []

    for strtg in mm_strategy.strategies:
        if strtg.input_specs is None:
            continue

        self_spec = strtg.input_specs[0]  # A placement
        mat2_spec = strtg.input_specs[1]  # B_t placement

        scale_self_spec = _mx_scale_placement_for_data(
            self_spec, scale_self_strategy.shape, "A"
        )
        scale_mat2_spec = _mx_scale_placement_for_data(
            mat2_spec, scale_mat2_strategy.shape, "B_t"
        )

        if scale_self_spec is None or scale_mat2_spec is None:
            continue  # Skip strategies with unsupported contracting-dim shards

        strtg.input_specs = list(strtg.input_specs) + [
            scale_self_spec,
            scale_mat2_spec,
        ]

        if (
            is_tensor_shardable(
                self_strategy.shape, self_spec, allow_unbacked_sharding=True
            )
            and is_tensor_shardable(
                mat2_strategy.shape, mat2_spec, allow_unbacked_sharding=True
            )
            and is_tensor_shardable(
                scale_self_strategy.shape,
                scale_self_spec,
                allow_unbacked_sharding=True,
            )
            and is_tensor_shardable(
                scale_mat2_strategy.shape,
                scale_mat2_spec,
                allow_unbacked_sharding=True,
            )
        ):
            strtg.redistribute_cost = [
                generate_redistribute_costs(self_strategy, self_spec),
                generate_redistribute_costs(mat2_strategy, mat2_spec),
                generate_redistribute_costs(scale_self_strategy, scale_self_spec),
                generate_redistribute_costs(scale_mat2_strategy, scale_mat2_spec),
            ]
            filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies
    return mm_strategy


_mx_strategy_registered = False


def ensure_mx_scaled_mm_strategy_registered():
    """Register the custom MX _scaled_mm strategy (idempotent)."""
    global _mx_strategy_registered
    if _mx_strategy_registered:
        return
    DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
        aten._scaled_mm.default,
        _mx_scaled_mm_strategy,
    )
    DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()
    _mx_strategy_registered = True
