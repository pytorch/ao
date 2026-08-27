# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_mx_inference_weight_fsdp2(tmp_path, elem_dtype):
    if dist.is_initialized():
        pytest.skip("Test requires ownership of the default process group")
    init_file = tmp_path / f"dist_init_{elem_dtype}"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(1)
        model = torch.nn.Linear(64, 256, bias=False, dtype=torch.bfloat16)
        config = MXDynamicActivationMXWeightConfig(
            block_size=32,
            activation_dtype=elem_dtype,
            weight_dtype=elem_dtype,
            kernel_preference=KernelPreference.EMULATED,
            scaling_mode=ScaleCalculationMode.FLOOR,
        )
        quantize_(model, config)
        assert isinstance(model.weight, MXTensor)

        torch.manual_seed(2)
        input_tensor = torch.randn(2, 64, dtype=torch.bfloat16)
        expected = model(input_tensor)

        mesh = init_device_mesh("cpu", (1,))
        fully_shard(model, mesh=mesh)
        assert isinstance(model.weight, DTensor)
        assert isinstance(model.weight.to_local(), MXTensor)

        actual_first = model(input_tensor)
        actual_second = model(input_tensor)
        torch.testing.assert_close(actual_first, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_second, expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()
