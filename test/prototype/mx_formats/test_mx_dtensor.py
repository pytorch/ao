# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of manually defined float16 TP vs mxfp4 TP of toy models

Note: for now, this does not run in CI.
TODO(future): make this run in CI
"""

import os

import pytest
import torch

from torchao.utils import torch_version_at_least

if not torch_version_at_least("2.7.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from tqdm import tqdm

from torchao.prototype.mx_formats import MXLinearConfig
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.testing.training.dtensor_utils import (
    _test_lowp_mlp_tensor_parallelism_base,
)

torch.set_float32_matmul_precision("high")


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    device_mesh = init_device_mesh("cuda", (world_size,))
    # seed must be the same in all processes
    torch.manual_seed(1)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    return device_mesh


def _test_dtensor_cast_to_mxfp4(mesh: DeviceMesh, size=4):
    device = mesh.device_type

    x_fp32 = torch.rand(size, size, device=device)
    x_fp4 = MXTensor.to_mx(x_fp32, torch.float4_e2m1fn_x2, block_size=size // 2)

    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])
    dist_x_fp4 = MXTensor.to_mx(
        dist_x_fp32, torch.float4_e2m1fn_x2, block_size=size // 2
    )
    assert isinstance(dist_x_fp4, DTensor)

    # Verify that the result of to_mx with DTensor matches the slice of the
    # result of to_mx without DTensor. This will fail on numeric op mismatches.
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert size % world_size == 0, "unsupported"
    x_fp4_fp32 = x_fp4.dequantize(torch.float32)
    rows_per_slice = size // world_size
    slice_start = local_rank * rows_per_slice
    slice_end = (local_rank + 1) * rows_per_slice
    x_fp4_fp32_slice = x_fp4_fp32[slice_start:slice_end]
    torch.testing.assert_close(
        x_fp4_fp32_slice,
        dist_x_fp4.to_local().dequantize(torch.float32),
        atol=0,
        rtol=0,
    )


def _test_mxfp4_mlp_tensor_parallelism(mesh: DeviceMesh, size=128):
    config = MXLinearConfig.from_recipe_name("mxfp4_emulated")
    config.block_size = 32
    _test_lowp_mlp_tensor_parallelism_base(
        mesh, config, size, compile=False, allgather_in_lowp=False
    )
    _test_lowp_mlp_tensor_parallelism_base(
        mesh, config, size, compile=True, allgather_in_lowp=False
    )


if __name__ == "__main__":
    device_mesh = setup_distributed()
    tests = [
        _test_dtensor_cast_to_mxfp4,
        _test_mxfp4_mlp_tensor_parallelism,
    ]

    for test in tqdm(tests, desc="Running tests"):
        try:
            test(device_mesh)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            raise e

    torch.distributed.destroy_process_group()
