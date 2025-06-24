# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of manually defined float16 TP vs float8 TP of toy models

Note: for now, this does not run in CI.
TODO(future): make this run in CI
"""

import copy
import os

import pytest
import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from tqdm import tqdm

from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_tensor_parallel import (
    Float8ColwiseParallel,
    Float8RowwiseParallel,
)
from torchao.testing.float8.dtensor_utils import ToyModel


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    # https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
    device_mesh = init_device_mesh(
        "cuda",
        (world_size // 2, 2),
        mesh_dim_names=("dp", "tp"),
    )
    # seed must be the same in all processes
    torch.manual_seed(1)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    return device_mesh


def _test_fp8_mlp_tensor_parallelism_base(
    mesh: DeviceMesh, size=16, compile: bool = False
):
    device = mesh.device_type

    config = Float8LinearConfig(
        emulate=True,
        enable_fsdp_float8_all_gather=True,
    )

    toy_model = ToyModel().to(device)

    tp_model = copy.deepcopy(toy_model)
    tp_model = convert_to_float8_training(tp_model, config=config)

    # apply TP
    tp_model = parallelize_module(
        tp_model,
        mesh["tp"],
        {
            "ffn.w1": Float8ColwiseParallel(),
            "ffn.w2": Float8ColwiseParallel(),
            "ffn.out_proj": Float8RowwiseParallel(),
        },
    )

    if compile:
        tp_model = torch.compile(tp_model)

    # apply FSDP
    fsdp_config = {"mesh": mesh["dp"]}
    tp_model = fully_shard(tp_model, **fsdp_config)

    x_fp32 = torch.rand(size, size * 2, size, device=device, requires_grad=False)
    x_fp32_tp_input = x_fp32.clone()

    tp_out = tp_model(x_fp32_tp_input)
    tp_out.sum().backward()
    torch.cuda.synchronize()

    # TODO(future PR): test numerics, and add more cases


def _test_fp8_mlp_tensor_parallelism_eager(mesh: DeviceMesh, size=16):
    _test_fp8_mlp_tensor_parallelism_base(mesh, size, compile=False)


def _test_fp8_mlp_tensor_parallelism_compile(mesh: DeviceMesh, size=16):
    _test_fp8_mlp_tensor_parallelism_base(mesh, size, compile=True)


if __name__ == "__main__":
    # float8 only works on CUDA H100 so we only test cuda and we follow
    # other test files to not use TestCase but instead just add the test
    # cases in the main func.
    device_mesh = setup_distributed()

    tests = [
        _test_fp8_mlp_tensor_parallelism_eager,
        _test_fp8_mlp_tensor_parallelism_compile,
    ]

    for test in tqdm(tests, desc="Running tests"):
        try:
            test(device_mesh)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            raise e

    torch.distributed.destroy_process_group()
