# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchao.prototype.parq.optim import ProxPARQ, QuantOptimizer
from torchao.prototype.parq.quant import UnifQuantizer
from torchao.prototype.parq.utils import HAS_DTENSOR, channel_bucketize


def make_optimizer(parameter, period=1, block_size=16):
    base = torch.optim.AdamW(
        [{"params": [parameter], "quant_bits": 4, "quant_block_size": block_size}],
        lr=0.01,
    )
    return QuantOptimizer(
        base,
        UnifQuantizer(),
        ProxPARQ(0, 8, 5),
        warmup_steps=1,
        quant_period=period,
        quant_per_channel=True,
    )


class TestChannelBucketize(unittest.TestCase):
    def test_rows_match_torch_bucketize(self):
        for dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]:
            bounds = torch.tensor([[-2, 0, 0, 3], [-4, -1, 2, 4]], dtype=dtype)
            values = torch.tensor(
                [[-5, -2, -1, 0, 1, 3, 5], [-5, -4, -1, 0, 2, 4, 5]], dtype=dtype
            )
            if dtype.is_floating_point:
                values = torch.cat(
                    (
                        values,
                        torch.tensor(
                            [[float("-inf"), float("inf"), float("nan")]] * 2,
                            dtype=dtype,
                        ),
                    ),
                    dim=1,
                )
            for right in [False, True]:
                with self.subTest(dtype=dtype, right=right):
                    expected = torch.stack(
                        [
                            torch.bucketize(v, b, right=right)
                            for v, b in zip(values, bounds)
                        ]
                    )
                    torch.testing.assert_close(
                        channel_bucketize(values, bounds, right),
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_strided_and_empty_inputs(self):
        torch.manual_seed(0)
        for levels in [0, 1, 4, 16, 256]:
            storage = torch.randn(3, 2 * levels).sort(dim=-1).values
            bounds = storage[:, ::2]
            inputs = torch.randn(3, 34)[:, ::2]
            for right in [False, True]:
                expected = torch.stack(
                    [
                        torch.bucketize(v.contiguous(), b.contiguous(), right=right)
                        for v, b in zip(inputs, bounds)
                    ]
                )
                torch.testing.assert_close(
                    channel_bucketize(inputs, bounds, right), expected
                )
        result = channel_bucketize(torch.empty(0, 3), torch.empty(0, 4))
        self.assertEqual(result.shape, (0, 3))
        self.assertEqual(result.dtype, torch.int64)


class TestQuantOptimizerCache(unittest.TestCase):
    def test_target_storage_reused_on_refresh(self):
        for block_size in [None, 16]:
            with self.subTest(block_size=block_size):
                parameter = torch.nn.Parameter(torch.randn(7, 64))
                optimizer = make_optimizer(parameter, block_size=block_size)
                pointer = None
                for _ in range(4):
                    parameter.grad = torch.randn_like(parameter)
                    optimizer.step()
                    if optimizer.num_steps > optimizer.warmup_steps:
                        targets = optimizer.state[parameter]["quants"]
                        if pointer is not None:
                            self.assertEqual(targets.data_ptr(), pointer)
                        pointer = targets.data_ptr()

    def test_checkpoint_cache_policy_and_exact_resume(self):
        for period in [1, 3]:
            with self.subTest(period=period):
                torch.manual_seed(13)
                parameter = torch.nn.Parameter(torch.randn(7, 64))
                optimizer = make_optimizer(parameter, period=period)
                for _ in range(3):
                    parameter.grad = torch.randn_like(parameter)
                    optimizer.step()
                live_targets = optimizer.state[parameter]["quants"]
                checkpoint = copy.deepcopy(optimizer.state_dict())
                saved_state = next(iter(checkpoint["state"].values()))
                self.assertEqual("quants" in saved_state, period != 1)
                self.assertIs(optimizer.state[parameter]["quants"], live_targets)
                restored = torch.nn.Parameter(parameter.detach().clone())
                resumed = make_optimizer(restored, period=period)
                resumed.load_state_dict(checkpoint)
                self.assertEqual(resumed.num_steps, optimizer.num_steps)
                for _ in range(3):
                    gradient = torch.randn_like(parameter)
                    parameter.grad = gradient.clone()
                    restored.grad = gradient.clone()
                    optimizer.step()
                    resumed.step()
                    torch.testing.assert_close(restored, parameter, rtol=0, atol=0)
                    torch.testing.assert_close(
                        resumed.state[restored]["latent"],
                        optimizer.state[parameter]["latent"],
                        rtol=0,
                        atol=0,
                    )


def distributed_worker(rank, store):
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + store, rank=rank, world_size=4
    )
    try:
        mesh = init_device_mesh("cpu", (4,))
        grid = init_device_mesh("cpu", (2, 2))
        subset = DeviceMesh("cpu", torch.tensor([0, 1]))
        cases = [
            (mesh, [Shard(0)], (7, 64), 16),
            (mesh, [Shard(0)], (7, 64), None),
            (mesh, [Shard(0)], (1, 64), 16),
            (mesh, [Shard(1)], (7, 256), 16),
            (mesh, [Replicate()], (7, 64), 16),
            (grid, [Shard(0), Shard(1)], (7, 128), 16),
            (grid, [Replicate(), Shard(0)], (7, 64), 16),
            (grid, [Shard(0), Shard(0)], (7, 64), 16),
            (subset, [Shard(0)], (7, 64), 16),
            (mesh, [Shard(1)], (7, 128), 16),
        ]
        for index, (device_mesh, placements, shape, block_size) in enumerate(cases):
            for period in [1, 3]:
                generator = torch.Generator().manual_seed(100 + index)
                initial = torch.randn(shape, generator=generator)
                reference = torch.nn.Parameter(initial.clone())
                if index == len(cases) - 1:
                    width = shape[1] // 4
                    local = initial[:, rank * width : (rank + 1) * width]
                    assert not local.is_contiguous()
                    sharded = torch.nn.Parameter(
                        DTensor.from_local(
                            local,
                            device_mesh,
                            placements,
                            shape=initial.shape,
                            stride=initial.stride(),
                        )
                    )
                else:
                    sharded = torch.nn.Parameter(
                        distribute_tensor(initial.clone(), device_mesh, placements)
                    )
                expected_optimizer = make_optimizer(reference, period, block_size)
                sharded_optimizer = make_optimizer(sharded, period, block_size)
                for _ in range(4):
                    gradient = torch.randn(shape, generator=generator)
                    reference.grad = gradient.clone()
                    sharded.grad = distribute_tensor(gradient, device_mesh, placements)
                    expected_optimizer.step()
                    # Projection must not scatter a globally allocated target buffer.
                    with mock.patch(
                        "torch.distributed.tensor._collective_utils.mesh_scatter",
                        side_effect=AssertionError("projection scattered targets"),
                    ):
                        sharded_optimizer.step()
                    if device_mesh.get_coordinate() is not None:
                        expected = distribute_tensor(
                            reference.detach().clone(), device_mesh, placements
                        )
                        torch.testing.assert_close(
                            sharded.to_local(), expected.to_local(), rtol=0, atol=0
                        )
                    else:
                        assert "quants" not in sharded_optimizer.state[sharded]
                dist.barrier()
        # An ungrouped per-row calibration cannot span column shards.
        parameter = torch.nn.Parameter(
            distribute_tensor(torch.randn(7, 64), mesh, [Shard(1)])
        )
        optimizer = make_optimizer(parameter, block_size=None)
        parameter.grad = distribute_tensor(torch.ones(7, 64), mesh, [Shard(1)])
        optimizer.step()
        parameter.grad = distribute_tensor(torch.ones(7, 64), mesh, [Shard(1)])
        try:
            optimizer.step()
        except ValueError as error:
            assert "crosses an input-column shard" in str(error)
        else:
            raise AssertionError("accepted a split calibration group")
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    HAS_DTENSOR and dist.is_available(), "requires DTensor and distributed"
)
class TestLocalDistributedProjection(unittest.TestCase):
    def test_complete_groups_on_owning_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                distributed_worker,
                args=(str(Path(directory) / "store"),),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
