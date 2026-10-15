# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from test.prototype.pat.test_common import make_prox_kwargs
from torchao.prototype.pat.group import Dim0Grouper, Dim1Grouper
from torchao.prototype.pat.optim import (
    CoupledMinSparsityConstraint,
    ProxGroupLasso,
    PruneOptimizer,
)
from torchao.prototype.pat.optim import prox_executor as executor
from torchao.prototype.pat.optim.prox_executor import apply_prox


class _GlooMultiProcessTestCase(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _init_process_group(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=60),
        )


class TestPATSubsetDeviceMesh(_GlooMultiProcessTestCase):
    @property
    def world_size(self):
        return 3

    @staticmethod
    def _step(optimizer, params):
        optimizer.zero_grad()
        for param in params:
            param.grad = torch.zeros_like(param)
        optimizer.step()

    def test_coupled_subset_mesh_materialized_state(self):
        self._init_process_group()
        try:
            mesh = DeviceMesh("cpu", [1, 2])
            participant = mesh.get_coordinate() is not None
            scores = torch.tensor(
                [100.0, 1.0, 10000.0, 10.0, 1000.0, 2.0, 100000.0, 20.0]
            )
            originals = [scores.repeat(5, 1), scores[:, None].repeat(1, 3)]
            originals[0][-1, 0] = 0.0
            originals[1][0, -1] = 0.0
            selected = torch.tensor([1, 3, 5, 7])
            expected = [p.clone() for p in originals]
            expected[0][:, selected] = 0.0
            expected[1][selected] = 0.0
            counts = [int(p.eq(0).sum()) for p in expected]
            total = sum(p.numel() for p in expected)
            groupers = (Dim1Grouper, Dim0Grouper)
            direct = [
                distribute_tensor(p.clone(), mesh, [Shard(0)], src_data_rank=None)
                for p in originals
            ]
            params = [
                torch.nn.Parameter(
                    distribute_tensor(p.clone(), mesh, [Shard(0)], src_data_rank=None)
                )
                for p in originals
            ]
            groups = [
                {
                    "params": [p],
                    "group_type": cls.__name__,
                    "prox_type": "CoupledMinSparsityConstraint",
                    "couple_key": "residual",
                    "min_sparsity": 0.5,
                }
                for p, cls in zip(params, groupers)
            ]
            optimizer = PruneOptimizer(torch.optim.SGD(groups, lr=0.0))
            optimizer.relative_sparsity = 0.375
            optimizer.relative_factored_frac = 0.25
            with mock.patch.object(
                executor, "distribute_tensor", wraps=executor.distribute_tensor
            ) as redistributed:
                result = executor.apply_coupled_prox(
                    [(p, cls, {}) for p, cls in zip(direct, groupers)],
                    CoupledMinSparsityConstraint(0.0, 0.5),
                    0.5,
                )
                for _ in range(2):
                    self._step(optimizer, params)
                    if participant:
                        self.assertEqual(
                            optimizer.relative_sparsity, sum(counts) / total
                        )
                    else:
                        self.assertEqual(optimizer.relative_sparsity, 0.375)
                        self.assertEqual(optimizer.relative_factored_frac, 0.25)
                        for p in params:
                            self.assertNotIn("sparsity_frac", optimizer.state[p])
            self.assertEqual(redistributed.call_count, 6 if participant else 0)
            for call in redistributed.call_args_list:
                self.assertIn("src_data_rank", call.kwargs)
                self.assertIsNone(call.kwargs["src_data_rank"])
            if participant:
                self.assertEqual(result.zero_channels, 4)
                self.assertEqual(result.channels, 8)
                self.assertEqual(result.zero_elts, sum(counts))
                self.assertEqual(result.numel, total)
                self.assertEqual(len(result.parameters), 2)
                for i, (p, q, dense) in enumerate(zip(direct, params, expected)):
                    self.assertIs(result.parameters[i].parameter, p)
                    self.assertEqual(result.parameters[i].zero_elts, counts[i])
                    self.assertEqual(result.parameters[i].numel, dense.numel())
                    self.assertEqual(p.full_tensor(), dense, rtol=0, atol=0)
                    self.assertEqual(q.full_tensor(), dense, rtol=0, atol=0)
                    self.assertEqual(
                        optimizer.state[q]["sparsity_frac"], counts[i] / dense.numel()
                    )
                self.assertEqual(optimizer.relative_factored_frac, 0.0)
                record = (
                    True,
                    result.zero_channels,
                    result.zero_elts,
                    optimizer.relative_sparsity,
                )
            else:
                self.assertEqual(result.parameters, ())
                self.assertEqual(
                    (
                        result.zero_elts,
                        result.numel,
                        result.zero_channels,
                        result.channels,
                    ),
                    (0, 0, 0, 0),
                )
                record = (
                    False,
                    optimizer.relative_sparsity,
                    optimizer.relative_factored_frac,
                )
            records = [None] * self.world_size
            dist.all_gather_object(records, record)
            self.assertEqual(records[0], (False, 0.375, 0.25))
            self.assertEqual(records[1], records[2])
            self.assertEqual(records[1], (True, 4, sum(counts), sum(counts) / total))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_subset_mesh_state_values_and_metrics(self):
        self._init_process_group()
        try:
            torch.manual_seed(0)
            mesh = DeviceMesh("cpu", [1, 2])
            is_participant = mesh.get_coordinate() is not None

            sv_param = torch.nn.Parameter(
                distribute_tensor(torch.randn(8, 8), mesh, [Shard(0)])
            )
            sv_group = {
                "params": [sv_param],
                "group_type": "SVDGrouper",
                "prox_type": "MinRankConstraint",
                "min_sparsity": 0.5,
            }
            sv_optimizer = PruneOptimizer(torch.optim.SGD([sv_group], lr=0.0))
            self._step(sv_optimizer, [sv_param])

            high = torch.full((8, 4), 10.0)
            low = torch.full((8, 4), 0.1)
            global_params = [
                torch.nn.Parameter(distribute_tensor(high, mesh, [Shard(0)])),
                torch.nn.Parameter(distribute_tensor(low, mesh, [Shard(0)])),
            ]
            global_group = {
                "params": global_params,
                "group_type": "Dim0Grouper",
                "prox_type": "GlobalMinSparsityConstraint",
                "min_sparsity": 0.5,
            }
            global_optimizer = PruneOptimizer(torch.optim.SGD([global_group], lr=0.0))
            self._step(global_optimizer, global_params)

            if is_participant:
                sv_count = sv_optimizer.state[sv_param]["sv_count"].item()
                sv_full = sv_param.full_tensor()
                singular_values = torch.linalg.svdvals(sv_full.to(torch.float32))
                effective_rank = int((singular_values > 1e-5).sum().item())

                global_full = [param.full_tensor() for param in global_params]
                zero_rows = sum(
                    sum(row.eq(0).all().item() for row in param)
                    for param in global_full
                )
                record = {
                    "participant": True,
                    "sv_count": sv_count,
                    "effective_rank": effective_rank,
                    "sv_metric": sv_optimizer.relative_factored_frac,
                    "sv_checksum": sv_full.sum().item(),
                    "global_zero_rows": zero_rows,
                    "global_metric": global_optimizer.relative_sparsity,
                    "global_checksums": tuple(
                        param.sum().item() for param in global_full
                    ),
                }
            else:
                self.assertNotIn("sv_count", sv_optimizer.state[sv_param])
                self.assertEqual(sv_optimizer.relative_sparsity, 0)
                self.assertEqual(sv_optimizer.relative_factored_frac, 0)
                for param in global_params:
                    self.assertNotIn("sparsity_frac", global_optimizer.state[param])
                self.assertEqual(global_optimizer.relative_sparsity, 0)
                record = {"participant": False}

            records = [None] * self.world_size
            dist.all_gather_object(records, record)
            self.assertEqual(records[0], {"participant": False})
            self.assertEqual(records[1], records[2])
            self.assertEqual(records[1]["sv_count"], 4)
            self.assertEqual(records[1]["effective_rank"], 4)
            self.assertGreater(records[1]["sv_metric"], 0)
            self.assertEqual(records[1]["global_zero_rows"], 8)
            self.assertEqual(records[1]["global_metric"], 0.5)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


class TestPATReplicatedPlacement(_GlooMultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    def test_replicated_mesh_dimension_does_not_duplicate_counts(self):
        self._init_process_group()
        try:
            mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
            full = torch.cat([torch.full((4, 4), 0.1), torch.full((4, 4), 2.0)])
            dense = full.clone()
            zero_dense, norm_dense, _ = apply_prox(
                Dim0Grouper(dense),
                ProxGroupLasso(reg_lambda=0.5),
                dense,
                **make_prox_kwargs(gamma=1.0),
            )

            dtensor = distribute_tensor(full.clone(), mesh, (Shard(0), Replicate()))
            zero_dtensor, norm_dtensor, _ = apply_prox(
                Dim0Grouper(dtensor),
                ProxGroupLasso(reg_lambda=0.5),
                dtensor,
                **make_prox_kwargs(gamma=1.0),
            )

            self.assertEqual(zero_dense, 16)
            self.assertEqual(zero_dtensor, zero_dense)
            self.assertEqual(norm_dtensor.placements, (Shard(0), Replicate()))
            self.assertEqual(norm_dtensor.full_tensor(), norm_dense)
            self.assertEqual(dtensor.full_tensor(), dense)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
