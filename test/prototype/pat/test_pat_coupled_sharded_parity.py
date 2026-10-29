# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import ExitStack
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests

from test.prototype.pat.test_pat_subset_mesh import _GlooMultiProcessTestCase
from torchao.prototype.pat.group import (
    AttentionHeadGrouperDim0,
    AttentionHeadGrouperDim1,
    Dim0Grouper,
    Dim1Grouper,
)
from torchao.prototype.pat.optim import CoupledMinSparsityConstraint
from torchao.prototype.pat.optim import prox_executor as executor


def _channel_view(p, grouper_cls, kwargs):
    if grouper_cls in (Dim1Grouper, AttentionHeadGrouperDim1):
        channels = kwargs.get("num_heads", p.size(1))
        return p.reshape(p.size(0), channels, -1).permute(1, 0, 2).reshape(channels, -1)
    return p.reshape(kwargs.get("num_heads", p.size(0)), -1)


def _fixture(groupers):
    # Different tensors dominate different channels, with large score gaps.
    amplitudes = 10.0 ** torch.tensor([3, 0, 6, 1, 4, 2, 7, 5])
    specs = []
    for i, cls in enumerate(groupers):
        width = 5 + 2 * i
        scale = torch.where(torch.arange(8) % len(groupers) == i, 1.0, 0.001)
        view = (amplitudes * scale)[:, None] * torch.arange(1.0, width + 1)[None, :]
        view[-1, -1] = 0.0  # An isolated zero outside the usual selected channels.
        p = view.t().contiguous() if cls is Dim1Grouper else view.clone()
        specs.append((p, cls, {}))
    return specs


def _dense_reference(specs, budget):
    views = [_channel_view(p, cls, kwargs) for p, cls, kwargs in specs]
    squared_scores = sum(view.float().square().sum(1) for view in views)
    chosen = math.ceil(budget * views[0].size(0))
    indices = squared_scores.argsort()[:chosen]
    expected = [(p.clone(), cls, kwargs) for p, cls, kwargs in specs]
    for p, cls, kwargs in expected:
        axis = int(cls in (Dim1Grouper, AttentionHeadGrouperDim1))
        width = p.size(axis) // views[0].size(0)
        packed_indices = (indices[:, None] * width + torch.arange(width)).flatten()
        p.index_fill_(axis, packed_indices, 0.0)
    return expected, chosen


class _CoupledParityMixin:
    def _distribute(self, originals, mesh, placements):
        return [
            (
                distribute_tensor(p.clone(), mesh, placements, src_data_rank=None),
                cls,
                kwargs,
            )
            for p, cls, kwargs in originals
        ]

    def _apply(self, specs, budget, score_type, route):
        prox = CoupledMinSparsityConstraint(0.0, budget, score_type)
        with ExitStack() as stack:
            if route == "fast":
                self.assertIsNotNone(executor._coupled_fast_path(specs))
                sharded = stack.enter_context(
                    mock.patch.object(
                        executor,
                        "_coupled_prox_sharded",
                        wraps=executor._coupled_prox_sharded,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        executor,
                        "grouped_view",
                        side_effect=AssertionError("materialized grouped view"),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        DTensor,
                        "full_tensor",
                        side_effect=AssertionError("materialized DTensor"),
                    )
                )
                collectives = {}
                for name in (
                    "all_reduce",
                    "all_gather_into_tensor",
                    "all_gather",
                    "broadcast",
                    "reduce",
                    "reduce_scatter_tensor",
                    "all_to_all_single",
                ):
                    collectives[name] = stack.enter_context(
                        mock.patch.object(dist, name, wraps=getattr(dist, name))
                    )
            else:
                if route == "forced":
                    stack.enter_context(
                        mock.patch.object(
                            executor, "_coupled_fast_path", return_value=None
                        )
                    )
                else:
                    self.assertIsNone(executor._coupled_fast_path(specs))
                stack.enter_context(
                    mock.patch.object(
                        executor,
                        "_coupled_prox_sharded",
                        side_effect=AssertionError("unexpected fast path"),
                    )
                )
                grouped = stack.enter_context(
                    mock.patch.object(
                        executor, "grouped_view", wraps=executor.grouped_view
                    )
                )
                redistributed = stack.enter_context(
                    mock.patch.object(
                        executor, "distribute_tensor", wraps=executor.distribute_tensor
                    )
                )
            result = executor.apply_coupled_prox(specs, prox, budget)
            if route == "fast":
                self.assertEqual(sharded.call_count, 1)
                self.assertGreater(
                    sum(call.call_count for call in collectives.values()), 0
                )
                self.assertLessEqual(
                    sum(call.call_count for call in collectives.values()), 3
                )
                shard_group = executor._coupled_fast_path(specs)[0]
                for collective in collectives.values():
                    for call in collective.call_args_list:
                        self.assertIs(call.kwargs["group"], shard_group)
                # Only integer counts may use backend-dependent reduction order.
                for call in collectives["all_reduce"].call_args_list:
                    self.assertEqual(call.args[0].dtype, torch.long)
            else:
                self.assertEqual(grouped.call_count, len(specs))
                self.assertEqual(redistributed.call_count, len(specs))
                for call in redistributed.call_args_list:
                    self.assertIn("src_data_rank", call.kwargs)
                    self.assertIsNone(call.kwargs["src_data_rank"])
        return result

    def _check_result(self, result, specs, expected, chosen):
        self.assertEqual(result.zero_channels, chosen)
        self.assertEqual(result.channels, 8)
        self.assertEqual(len(result.parameters), len(specs))
        zeros = []
        sizes = []
        for entry, (p, _, _), (dense, cls, kwargs) in zip(
            result.parameters, specs, expected
        ):
            full = p.full_tensor()
            self.assertEqual(full.eq(0), dense.eq(0))
            self.assertEqual(full, dense, rtol=0, atol=0)
            self.assertIs(entry.parameter, p)
            zeros.append(int(dense.eq(0).sum()))
            sizes.append(dense.numel())
            self.assertEqual(entry.zero_elts, zeros[-1])
            self.assertEqual(entry.numel, sizes[-1])
            self.assertEqual(
                int(_channel_view(full, cls, kwargs).eq(0).all(1).sum()), chosen
            )
        self.assertEqual(result.zero_elts, sum(zeros))
        self.assertEqual(result.numel, sum(sizes))

    def _parity(self, originals, mesh, placements, route="fast", budgets=(0.26,)):
        for score_type in ("rms", "l2", "param_cost"):
            for budget in budgets:
                with self.subTest(
                    score_type=score_type, budget=budget, placements=placements
                ):
                    expected, chosen = _dense_reference(originals, budget)
                    actual = self._distribute(originals, mesh, placements)
                    forced = self._distribute(originals, mesh, placements)
                    a = self._apply(actual, budget, score_type, route)
                    b = self._apply(forced, budget, score_type, "forced")
                    self._check_result(a, actual, expected, chosen)
                    self._check_result(b, forced, expected, chosen)
                    self.assertEqual(
                        (a.zero_elts, a.numel, a.zero_channels, a.channels),
                        (b.zero_elts, b.numel, b.zero_channels, b.channels),
                    )


class TestPATCoupledShardedParity(_CoupledParityMixin, _GlooMultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    def test_mixed_readers_writers_and_replicated_axes(self):
        self._init_process_group()
        try:
            originals = _fixture(
                (Dim1Grouper, Dim0Grouper, Dim1Grouper, Dim0Grouper, Dim0Grouper)
            )
            mesh = DeviceMesh("cpu", [0, 1, 2, 3])
            self._parity(originals, mesh, (Shard(0),), budgets=(0.0, 0.26, 1.0))
            double_specs = [(p.double(), cls, kwargs) for p, cls, kwargs in originals]
            self._parity(double_specs, mesh, (Shard(0),))
            mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
            for placements in ((Shard(0), Replicate()), (Replicate(), Shard(0))):
                self._parity(originals, mesh, placements)
        finally:
            dist.destroy_process_group()

    def test_readers_only_uneven_rows_and_writers_only(self):
        self._init_process_group()
        try:
            mesh = DeviceMesh("cpu", [0, 1, 2, 3])
            for groupers in (
                (Dim1Grouper,) * 3,
                (Dim0Grouper,) * 3,
                (Dim1Grouper, Dim0Grouper),
            ):
                self._parity(_fixture(groupers), mesh, (Shard(0),))
        finally:
            dist.destroy_process_group()

    def test_permuted_mesh_preserves_mixed_reader_writer_channels(self):
        self._init_process_group()
        try:
            for ranks, placements in (
                ([2, 0, 3, 1], (Shard(0),)),
                ([[0, 1], [3, 2]], (Replicate(), Shard(0))),
                ([[0, 3], [1, 2]], (Shard(0), Replicate())),
            ):
                mesh = DeviceMesh("cpu", ranks)
                for groupers in (
                    (Dim1Grouper, Dim0Grouper, Dim1Grouper, Dim0Grouper),
                    (Dim0Grouper,) * 3,
                    (Dim1Grouper,) * 3,
                ):
                    for score_type in ("rms", "l2", "param_cost"):
                        with self.subTest(
                            ranks=ranks, groupers=groupers, score_type=score_type
                        ):
                            originals = _fixture(groupers)
                            expected, chosen = _dense_reference(originals, 0.26)
                            reference = [
                                (p.clone(), cls, kwargs) for p, cls, kwargs in originals
                            ]
                            b = executor.apply_coupled_prox(
                                reference,
                                CoupledMinSparsityConstraint(0.0, 0.26, score_type),
                                0.26,
                            )
                            actual = self._distribute(originals, mesh, placements)
                            a = self._apply(actual, 0.26, score_type, "fast")
                            shard_dim = next(
                                i
                                for i, placement in enumerate(placements)
                                if placement.is_shard()
                            )
                            coordinate = mesh.get_coordinate()[shard_dim]
                            for entry, (p, _, _), (dense, _, _), (oracle, _, _) in zip(
                                a.parameters, actual, reference, expected
                            ):
                                self.assertEqual(dense, oracle, rtol=0, atol=0)
                                # Check local shards directly: full_tensor() in
                                # some PyTorch builds also gathers reordered
                                # meshes without restoring coordinate order.
                                offset = coordinate * math.ceil(
                                    dense.size(0) / mesh.size(shard_dim)
                                )
                                self.assertEqual(
                                    p.to_local(),
                                    dense[offset : offset + p.to_local().size(0)],
                                    rtol=0,
                                    atol=0,
                                )
                                self.assertEqual(
                                    entry.zero_elts, int(dense.eq(0).sum())
                                )
                                self.assertEqual(entry.numel, dense.numel())
                            self.assertEqual(a.zero_channels, chosen)
                            self.assertEqual(
                                (a.zero_elts, a.numel, a.zero_channels, a.channels),
                                (b.zero_elts, b.numel, b.zero_channels, b.channels),
                            )
        finally:
            dist.destroy_process_group()

    def test_unsupported_layouts_and_groupers_materialize(self):
        self._init_process_group()
        try:
            originals = _fixture((Dim1Grouper, Dim0Grouper))
            mesh = DeviceMesh("cpu", [0, 1, 2, 3])
            self._parity(originals, mesh, (Shard(1),), "fallback")
            kwargs_specs = [
                (p, cls, {"start_dim": 0, "end_dim": 1}) for p, cls, _ in originals
            ]
            self._parity(kwargs_specs, mesh, (Shard(0),), "fallback")
            head_specs = [
                (
                    p.repeat_interleave(2, dim=1 if cls is Dim1Grouper else 0),
                    AttentionHeadGrouperDim1
                    if cls is Dim1Grouper
                    else AttentionHeadGrouperDim0,
                    {"num_heads": 8},
                )
                for p, cls, _ in originals
            ]
            self._parity(head_specs, mesh, (Shard(0),), "fallback")
            mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
            self._parity(originals, mesh, (Shard(0), Shard(0)), "fallback")
        finally:
            dist.destroy_process_group()

    def test_tied_and_near_tied_selection_agrees_on_all_ranks(self):
        self._init_process_group()
        try:
            mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
            for placements in ((Shard(0), Replicate()), (Replicate(), Shard(0))):
                for near_tied in (False, True):
                    for route in ("fast", "forced"):
                        with self.subTest(
                            placements=placements, near_tied=near_tied, route=route
                        ):
                            scales = torch.ones(8)
                            if near_tied:
                                scales += (
                                    torch.arange(8) * torch.finfo(torch.float32).eps
                                )
                            originals = [
                                (scales.repeat(8, 1), Dim1Grouper, {}),
                                (scales[:, None].repeat(1, 4), Dim0Grouper, {}),
                            ]
                            specs = self._distribute(originals, mesh, placements)
                            result = self._apply(specs, 0.5, "rms", route)
                            mask = specs[0][0].to_local().eq(0).all(0)
                            self.assertEqual(int(mask.sum()), 4)
                            shard_dim = next(
                                i
                                for i, p in enumerate(placements)
                                if isinstance(p, Shard)
                            )
                            shard_rank = mesh.get_coordinate()[shard_dim]
                            self.assertEqual(
                                specs[1][0].to_local().eq(0).all(1),
                                mask.chunk(2)[shard_rank],
                            )
                            record = (
                                mask.tolist(),
                                result.zero_channels,
                                result.zero_elts,
                                result.numel,
                            )
                            records = [None] * self.world_size
                            dist.all_gather_object(records, record)
                            self.assertEqual(records, [record] * self.world_size)
                            # Dense tie-breaking need not match; shared masks must.
                            expected = [
                                (p.clone(), cls, kwargs) for p, cls, kwargs in originals
                            ]
                            expected[0][0][:, mask] = 0.0
                            expected[1][0][mask] = 0.0
                            self._check_result(result, specs, expected, 4)
        finally:
            dist.destroy_process_group()


class TestPATCoupledReplicaScoreOrder(_CoupledParityMixin, _GlooMultiProcessTestCase):
    @property
    def world_size(self):
        return 8

    def test_reader_scores_agree_across_four_shard_replicas(self):
        self._init_process_group()
        try:
            reader = torch.ones(4, 8)
            reader[0] = torch.tensor(
                [4096, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
            )
            reader[1:, 1] = 0.5
            # Accumulating the small partials first changes the first channel's
            # FP32 score. Replicas must all use the same mesh-coordinate order.
            expected_sq = reader[0].square()
            for row in reader[1:]:
                expected_sq += row.square()
            for ranks, placements in (
                ([[0, 1, 2, 3], [4, 5, 6, 7]], (Replicate(), Shard(0))),
                ([[0, 4], [1, 5], [2, 6], [3, 7]], (Shard(0), Replicate())),
                ([[0, 1, 2, 3], [7, 6, 5, 4]], (Replicate(), Shard(0))),
                ([[0, 7], [1, 6], [2, 5], [3, 4]], (Shard(0), Replicate())),
            ):
                with self.subTest(placements=placements):
                    mesh = DeviceMesh("cpu", ranks)
                    specs = self._distribute(
                        [(reader, Dim1Grouper, {})], mesh, placements
                    )
                    prox = CoupledMinSparsityConstraint(0.0, 0.125, "l2")
                    with (
                        mock.patch.object(
                            prox, "combine_scores", wraps=prox.combine_scores
                        ) as combined,
                        mock.patch.object(
                            dist, "all_reduce", wraps=dist.all_reduce
                        ) as reduced,
                        mock.patch.object(
                            DTensor,
                            "full_tensor",
                            side_effect=AssertionError("materialized DTensor"),
                        ),
                    ):
                        result = executor.apply_coupled_prox(specs, prox, 0.125)
                        self.assertEqual(reduced.call_count, 1)
                        self.assertEqual(reduced.call_args.args[0].dtype, torch.long)
                    actual_sq = combined.call_args.args[0]
                    mask = specs[0][0].to_local().eq(0).all(0)
                    record = (actual_sq.tolist(), mask.tolist(), result.zero_elts)
                    records = [None] * self.world_size
                    dist.all_gather_object(records, record)
                    self.assertEqual(actual_sq, expected_sq, rtol=0, atol=0)
                    self.assertEqual(int(mask.sum()), 1)
                    self.assertEqual(result.zero_elts, 4)
                    self.assertEqual(records, [record] * self.world_size)
        finally:
            dist.destroy_process_group()


class TestPATCoupledUnevenShards(_CoupledParityMixin, _GlooMultiProcessTestCase):
    @property
    def world_size(self):
        return 3

    def test_uneven_writers_fall_back_but_uneven_readers_are_fast(self):
        self._init_process_group()
        try:
            mesh = DeviceMesh("cpu", [0, 1, 2])
            self._parity(
                _fixture((Dim1Grouper, Dim0Grouper, Dim0Grouper)),
                mesh,
                (Shard(0),),
                "fallback",
            )
            self._parity(_fixture((Dim1Grouper,) * 3), mesh, (Shard(0),))
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
