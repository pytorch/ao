# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of manually defined float16 TP vs mxfp8 TP of toy models

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

from torchao.prototype.moe_training.config import (
    MXFP8TrainingOpConfig,
    MXFP8TrainingRecipe,
)
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


def _test_dtensor_cast_to_mxfp8(mesh: DeviceMesh, size=1024):
    device = mesh.device_type

    x_fp32 = torch.rand(size, size, device=device)
    x_fp8 = MXTensor.to_mx(x_fp32, torch.float8_e4m3fn, block_size=32)

    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])
    dist_x_fp8 = MXTensor.to_mx(dist_x_fp32, torch.float8_e4m3fn, block_size=32)

    # With the new wrapping order, MXTensor is the outer wrapper with DTensor
    # inner tensors (MXTensor(DTensor_qdata, DTensor_scale)).
    assert isinstance(dist_x_fp8, MXTensor), (
        f"Expected MXTensor, got {type(dist_x_fp8)}"
    )
    assert isinstance(dist_x_fp8.qdata, DTensor), (
        f"Expected qdata to be DTensor, got {type(dist_x_fp8.qdata)}"
    )
    assert isinstance(dist_x_fp8.scale, DTensor), (
        f"Expected scale to be DTensor, got {type(dist_x_fp8.scale)}"
    )

    # Verify that the result of to_mx with DTensor matches the slice of the
    # result of to_mx without DTensor. This will fail on numeric op mismatches.
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert size % world_size == 0, "unsupported"
    x_fp8_fp32 = x_fp8.dequantize(torch.bfloat16)
    rows_per_slice = size // world_size
    slice_start = local_rank * rows_per_slice
    slice_end = (local_rank + 1) * rows_per_slice
    x_fp8_fp32_slice = x_fp8_fp32[slice_start:slice_end]
    # dequantize handles DTensor inner tensors and returns a DTensor
    dist_x_fp8_dequant = dist_x_fp8.dequantize(torch.bfloat16)
    assert isinstance(dist_x_fp8_dequant, DTensor), (
        f"Expected dequantize result to be DTensor, got {type(dist_x_fp8_dequant)}"
    )
    torch.testing.assert_close(
        x_fp8_fp32_slice,
        dist_x_fp8_dequant.to_local(),
        atol=0,
        rtol=0,
    )


def _test_mxfp8_mlp_tensor_parallelism_emulated(mesh: DeviceMesh, size=64):
    recipe = MXFP8TrainingRecipe("mxfp8_emulated_rceil")
    config = MXFP8TrainingOpConfig.from_recipe(recipe)
    _test_lowp_mlp_tensor_parallelism_base(
        mesh, config, size, compile=False, allgather_in_lowp=False
    )


def _test_mxfp8_mlp_tensor_parallelism_auto(mesh: DeviceMesh, size=64):
    recipe = MXFP8TrainingRecipe("mxfp8_rceil")
    config = MXFP8TrainingOpConfig.from_recipe(recipe)
    _test_lowp_mlp_tensor_parallelism_base(
        mesh, config, size, compile=False, allgather_in_lowp=False
    )


if __name__ == "__main__":
    device_mesh = setup_distributed()
    tests = [
        _test_dtensor_cast_to_mxfp8,
        _test_mxfp8_mlp_tensor_parallelism_emulated,
    ]
    # The auto test requires the mxfp8_quantize CUDA kernel to be available.
    # _mxfp8_cuda_kernels_available checks hardware/driver prerequisites (SM >= 100,
    # CUDA >= 12.8), but we also need the C++ extension to be built, so we
    # verify by actually calling the kernel.
    from torchao.prototype.moe_training.kernels.mxfp8.quant import (
        _mxfp8_cuda_kernels_available,
    )

    if _mxfp8_cuda_kernels_available:
        try:
            from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda

            t = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
            mxfp8_quantize_cuda(t, rowwise=False, colwise=True)
            del t
            tests.append(_test_mxfp8_mlp_tensor_parallelism_auto)
        except Exception:
            print("Skipping auto test: mxfp8_quantize CUDA kernel not available")
    else:
        print("Skipping auto test: requires SM >= 100 and CUDA >= 12.8")

    for test in tqdm(tests, desc="Running tests"):
        try:
            test(device_mesh)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            raise e

    torch.distributed.destroy_process_group()
