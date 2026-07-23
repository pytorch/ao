# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from torchao.prototype.pat.optim import PruneOptimizer


class TestPATSubsetDeviceMesh(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 3

    def _init_process_group(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=60),
        )

    @staticmethod
    def _step(optimizer, params):
        optimizer.zero_grad()
        for param in params:
            param.grad = torch.zeros_like(param)
        optimizer.step()

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


if __name__ == "__main__":
    run_tests()
