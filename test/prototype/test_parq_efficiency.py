# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from itertools import product
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
        dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]
        for dtype, bounds_dtype in [(dtype, dtype) for dtype in dtypes] + [
            (torch.bfloat16, torch.float32)
        ]:
            bounds = torch.tensor([[-2, 0, 0, 3], [-4, -1, 2, 4]], dtype=bounds_dtype)
            if dtype != bounds_dtype:
                # Expose boundary precision lost by a bfloat16 cast.
                bounds.add_(0.001)
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
                with self.subTest(dtype=dtype, bounds_dtype=bounds_dtype, right=right):
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
        for dtype, period, block_size in product(
            [torch.float32, torch.bfloat16], [1, 3], [None, 16]
        ):
            with self.subTest(dtype=dtype, period=period, block_size=block_size):
                parameter = torch.nn.Parameter(torch.randn(7, 64, dtype=dtype))
                optimizer = make_optimizer(parameter, period, block_size)
                targets = None
                for _ in range(5):
                    parameter.grad = torch.randn_like(parameter)
                    optimizer.step()
                    if optimizer.num_steps > optimizer.warmup_steps:
                        current_targets = optimizer.state[parameter]["quants"]
                        self.assertEqual(current_targets.dtype, torch.float32)
                        if targets is not None:
                            self.assertEqual(
                                current_targets.data_ptr(), targets.data_ptr()
                            )
                        targets = current_targets

    def test_checkpoint_cache_policy_and_exact_resume(self):
        for dtype, (saved_period, period, expected_calls) in product(
            [torch.float32, torch.bfloat16],
            [(1, 1, [1, 2, 3]), (3, 3, [0, 1, 1]), (1, 3, [1, 2, 2])],
        ):
            with self.subTest(dtype=dtype, saved_period=saved_period, period=period):
                torch.manual_seed(13)
                parameter = torch.nn.Parameter(torch.randn(7, 64, dtype=dtype))
                optimizer = make_optimizer(parameter, period=saved_period)
                for _ in range(3):
                    parameter.grad = torch.randn_like(parameter)
                    optimizer.step()
                live_targets = optimizer.state[parameter]["quants"]
                if dtype == torch.bfloat16 and saved_period != 1:
                    live_targets.add_(0.001)
                    self.assertFalse(
                        torch.equal(live_targets, live_targets.bfloat16().float())
                    )
                checkpoint = copy.deepcopy(optimizer.state_dict())
                saved_state = next(iter(checkpoint["state"].values()))
                self.assertEqual("quants" in saved_state, saved_period != 1)
                self.assertIs(optimizer.state[parameter]["quants"], live_targets)
                restored = torch.nn.Parameter(parameter.detach().clone())
                resumed = make_optimizer(restored, period=period)
                resumed.load_state_dict(checkpoint)
                self.assertEqual(resumed.num_steps, optimizer.num_steps)
                restored_targets = resumed.state[restored].get("quants")
                if saved_period == 1:
                    self.assertIsNone(restored_targets)
                else:
                    self.assertEqual(restored_targets.dtype, torch.get_default_dtype())
                    torch.testing.assert_close(
                        restored_targets, saved_state["quants"], rtol=0, atol=0
                    )
                    self.assertNotEqual(
                        restored_targets.data_ptr(), saved_state["quants"].data_ptr()
                    )
                if saved_period != period:
                    # Match off-cycle recovery in the live numerical reference.
                    optimizer.quant_period = period
                    optimizer.state[parameter].pop("quants")
                with mock.patch.object(
                    resumed.quantizer, "quantize", wraps=resumed.quantizer.quantize
                ) as quantize:
                    for calls in expected_calls:
                        gradient = torch.randn_like(parameter)
                        parameter.grad = gradient.clone()
                        restored.grad = gradient.clone()
                        optimizer.step()
                        resumed.step()
                        self.assertEqual(quantize.call_count, calls)
                        self.assertEqual(resumed.quant_period, period)
                        torch.testing.assert_close(restored, parameter, rtol=0, atol=0)
                        for name in ["latent", "quants"]:
                            torch.testing.assert_close(
                                resumed.state[restored][name],
                                optimizer.state[parameter][name],
                                rtol=0,
                                atol=0,
                            )

    def test_checkpoint_cache_mapping_across_groups(self):
        parameters = [
            torch.nn.Parameter(torch.zeros(2, 16, dtype=torch.bfloat16))
            for _ in range(3)
        ]
        base = torch.optim.SGD(
            [{"params": parameters[:2]}, {"params": parameters[2:]}], lr=0.01
        )
        for group in base.param_groups:
            group["quant_bits"] = 4
        optimizer = QuantOptimizer(
            base, UnifQuantizer(), ProxPARQ(0, 8, 5), quant_period=3
        )
        checkpoint = optimizer.state_dict()
        for group, ids in zip(checkpoint["param_groups"], [[29, 7], [41]]):
            group["params"] = ids
        checkpoint["state"] = {
            key: {"latent": torch.full((2, 16), float(key))} for key in [41, 7, 29]
        }
        for key in [29, 41]:
            checkpoint["state"][key]["quants"] = torch.full((16,), key + 0.001)
        for remap in ["none", "inplace", "replacement"]:
            with self.subTest(remap=remap):

                def remap_ids(base, state):
                    if remap == "none":
                        return None
                    if remap == "replacement":
                        state = copy.deepcopy(state)
                    first, last = [group["params"] for group in state["param_groups"]]
                    first[0], last[0] = last[0], first[0]
                    return state if remap == "replacement" else None

                with base.register_load_state_dict_pre_hook(remap_ids):
                    optimizer.load_state_dict(copy.deepcopy(checkpoint))
                order = [29, 7, 41] if remap == "none" else [41, 7, 29]
                for parameter, key in zip(parameters, order):
                    state = optimizer.state[parameter]
                    saved = checkpoint["state"][key]
                    self.assertEqual("quants" in state, "quants" in saved)
                    for name, expected in saved.items():
                        actual = state[name]
                        expected = (
                            expected.to(parameter) if name == "latent" else expected
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        if name == "quants":
                            self.assertNotEqual(actual.data_ptr(), expected.data_ptr())

    def test_cache_recovery_is_per_parameter(self):
        for mismatch in ["missing", "shape", "device", "dtype"]:
            with self.subTest(mismatch=mismatch):
                torch.manual_seed(17)
                parameters = [
                    torch.nn.Parameter(torch.randn(7, 64)),
                    torch.nn.Parameter(torch.randn(5, 64)),
                    torch.nn.Parameter(torch.empty(0, 64)),
                ]
                optimizer = make_optimizer(parameters[0], period=3)
                optimizer.param_groups[0]["params"].extend(parameters[1:])
                for _ in range(2):
                    for parameter in parameters:
                        parameter.grad = torch.randn_like(parameter)
                    optimizer.step()
                states = [optimizer.state[parameter] for parameter in parameters]
                stale_targets = states[0].pop("quants")
                if mismatch != "missing":
                    if mismatch == "shape":
                        stale_targets = torch.empty(1, stale_targets.size(-1))
                    elif mismatch == "device":
                        stale_targets = torch.empty_like(stale_targets, device="meta")
                    else:
                        stale_targets = stale_targets.bfloat16()
                    states[0]["quants"] = stale_targets
                other_targets = states[1]["quants"]
                saved_targets = other_targets.clone()
                for updating in [
                    (True, False, False),
                    (False, False, False),
                    (True, True, True),
                ]:
                    for parameter in parameters:
                        parameter.grad = torch.randn_like(parameter)
                    with mock.patch.object(
                        optimizer.prox_map, "apply_", wraps=optimizer.prox_map.apply_
                    ) as projection:
                        optimizer.step()
                    self.assertEqual(
                        tuple(
                            call.args[1] is not None
                            for call in projection.call_args_list
                        ),
                        updating,
                    )
                    if updating[2]:
                        p, q = projection.call_args_list[2].args[:2]
                        self.assertIs(q, p)
                    targets = states[0]["quants"]
                    self.assertEqual(targets.dtype, torch.get_default_dtype())
                    self.assertIsNot(targets, stale_targets)
                    if updating[0]:
                        _, expected = optimizer.quantizer.quantize(
                            states[0]["latent"].view(-1, 16), 4, dim=-1
                        )
                        torch.testing.assert_close(targets, expected, rtol=0, atol=0)
                    self.assertIs(states[1]["quants"], other_targets)
                    if not updating[1]:
                        torch.testing.assert_close(
                            other_targets, saved_targets, rtol=0, atol=0
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
        equivalent_mesh = DeviceMesh("cpu", mesh.mesh.clone())
        cache_cases = {
            0: (mesh, [Replicate()], True),
            1: (None, None, True),
            4: (equivalent_mesh, [Replicate()], False),
            8: (mesh, [Shard(0)], True),
        }
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
                for step in range(4):
                    gradient = torch.randn(shape, generator=generator)
                    reference.grad = gradient.clone()
                    sharded.grad = distribute_tensor(gradient, device_mesh, placements)
                    cache_case = (
                        cache_cases.get(index) if period == 3 and step == 2 else None
                    )
                    if cache_case is not None:
                        # Inject stale state only on the first nonscheduled update.
                        cache_mesh, cache_placements, rebuild = cache_case
                        target_shape = (
                            (shape[0], shape[1] // block_size, 16)
                            if block_size is not None
                            else (shape[0], 16)
                        )
                        dense_state = expected_optimizer.state[reference]
                        old_targets = dense_state["quants"].reshape(target_shape)
                        stale_targets = (
                            distribute_tensor(
                                old_targets.clone(), cache_mesh, cache_placements
                            )
                            if cache_mesh is not None
                            else old_targets.clone()
                        )
                        if device_mesh.get_coordinate() is not None:
                            sharded_optimizer.state[sharded]["quants"] = stale_targets
                        if rebuild:
                            dense_state.pop("quants")
                        else:
                            assert stale_targets.device_mesh is not device_mesh
                            assert stale_targets.device_mesh == device_mesh
                    expected_optimizer.step()
                    # Projection must not scatter a globally allocated target buffer.
                    with mock.patch(
                        "torch.distributed.tensor._collective_utils.mesh_scatter",
                        side_effect=AssertionError("projection scattered targets"),
                    ):
                        sharded_optimizer.step()
                    if device_mesh.get_coordinate() is not None:
                        comparisons = [(sharded, reference.detach())]
                        if cache_case is not None:
                            targets = sharded_optimizer.state[sharded]["quants"]
                            assert isinstance(targets, DTensor)
                            assert targets.shape == target_shape
                            assert targets.placements == tuple(placements)
                            assert targets.device_mesh == device_mesh
                            assert (targets is stale_targets) == (not rebuild)
                            comparisons.append(
                                (targets, dense_state["quants"].reshape(target_shape))
                            )
                        for actual, dense in comparisons:
                            expected = distribute_tensor(
                                dense.clone(), device_mesh, placements
                            )
                            torch.testing.assert_close(
                                actual.to_local(), expected.to_local(), rtol=0, atol=0
                            )
                    else:
                        assert "quants" not in sharded_optimizer.state[sharded]
                if index == 0 and period == 3:
                    targets = sharded_optimizer.state[sharded]["quants"]
                    local_targets = targets.to_local().add_(0.001)
                    assert not local_targets.equal(local_targets.bfloat16().float())
                    checkpoint = copy.deepcopy(sharded_optimizer.state_dict())
                    restored = torch.nn.Parameter(sharded.detach().bfloat16().clone())
                    resumed = make_optimizer(restored, period, block_size)
                    with mock.patch(
                        "torch.distributed.tensor._collective_utils.mesh_scatter",
                        side_effect=AssertionError("checkpoint load scattered targets"),
                    ):
                        resumed.load_state_dict(checkpoint)
                    loaded = resumed.state[restored]["quants"]
                    saved = next(iter(checkpoint["state"].values()))["quants"]
                    assert isinstance(loaded, DTensor)
                    assert loaded.dtype == torch.get_default_dtype()
                    assert loaded.shape == saved.shape
                    assert loaded.placements == saved.placements
                    assert loaded.device_mesh == saved.device_mesh
                    assert loaded.to_local().data_ptr() != saved.to_local().data_ptr()
                    torch.testing.assert_close(
                        loaded.to_local(), saved.to_local(), rtol=0, atol=0
                    )
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
