# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import sys
import warnings

import pytest
import torch

from torchao.prototype.dtypes.uintx.uintx_layout import to_uintx
from torchao.quantization.quant_api import quantize_  # noqa: F401
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.utils import get_current_accelerator_device

dtypes = (
    torch.uint1,
    torch.uint2,
    torch.uint3,
    torch.uint4,
    torch.uint5,
    torch.uint6,
    torch.uint7,
)

group_sizes = [32, 64, 128]
devices = ["cpu"] + (
    [get_current_accelerator_device()] if torch.accelerator.is_available() else []
)


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset()  # reset cache between tests


class Linear16(torch.nn.Module):
    def __init__(self, scale, device):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                scale * 2, scale, bias=False, dtype=torch.float16, device=device
            ),
            torch.nn.Linear(
                scale, scale, bias=False, dtype=torch.float16, device=device
            ),
            torch.nn.Linear(
                scale, scale // 2, bias=False, dtype=torch.float16, device=device
            ),
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("group_size", group_sizes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.skipif(not torch.accelerator.is_available(), reason="GPU not available")
def test_uintx_weight_only_quant(dtype, group_size, device):
    input_float = torch.randn((1, 256), dtype=torch.float16, device=device)
    mapping_type = MappingType.SYMMETRIC
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int32
    block_size = (1, group_size)

    scale, zero_point = choose_qparams_affine(
        input_float,
        mapping_type,
        block_size,
        dtype,
        eps=eps,
        scale_dtype=torch.float32,
        zero_point_dtype=zero_point_dtype,
    )

    aqt = quantize_affine(
        input_float,
        block_size,
        scale,
        zero_point,
        dtype,
    )
    # Note: output will be uint8 tensor for sub byte tensors for now

    q = to_uintx(aqt, dtype, -1)
    assert q is not None, "quantization failed"
    deqaunt = dequantize_affine(q, block_size, scale, zero_point, dtype)
    assert deqaunt is not None, "deqauntization failed"


def test_uintx_api_deprecation():
    """
    Test that deprecated uintx APIs trigger deprecation warnings on import.
    TODO: Remove this test once the deprecated APIs have been removed.
    """
    deprecated_apis = [
        (
            "Int8DynamicActInt4WeightCPULayout",
            "torchao.dtypes.uintx.dyn_int8_act_int4_wei_cpu_layout",
        ),
        ("BlockSparseLayout", "torchao.dtypes.uintx.block_sparse_layout"),
        ("UintxLayout", "torchao.dtypes.uintx.uintx_layout"),
    ]

    for api_name, module_path in deprecated_apis:
        # Clear the cache to force re-importing and trigger the warning again
        modules_to_clear = [module_path, "torchao.dtypes"]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all warnings are captured

            # Dynamically import the deprecated API
            exec(f"from torchao.dtypes import {api_name}")

            assert any(
                issubclass(warning.category, DeprecationWarning)
                and api_name in str(warning.message)
                for warning in w
            ), (
                f"Expected deprecation warning for {api_name}, got: {[str(warning.message) for warning in w]}"
            )
