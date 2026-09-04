# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile

import pytest
import torch
import torch.nn as nn

from torchao.prototype.mx_formats.config import NoSwizzle, Swizzle_32_4_4
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import is_sm_at_least_100


@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="CUDA or XPU not available",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm_at_least_100(),
    reason="needs CUDA capability 10.0+",
)
@pytest.mark.parametrize("recipe_name", ["mxfp8", "nvfp4"])
def test_serialization(recipe_name):
    """
    Ensure that only `import torchao.prototype.mx_formats` is needed to load MX
    and NV checkpoints.
    """
    device = torch.accelerator.current_accelerator().type
    if recipe_name == "nvfp4" and device == "xpu":
        pytest.skip("NVFP4 is not supported on XPU")

    m = nn.Linear(32, 128, bias=False, dtype=torch.bfloat16, device=device)
    fname = None
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        if recipe_name == "mxfp8":
            config = MXDynamicActivationMXWeightConfig(
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                kernel_preference=KernelPreference.EMULATED,
            )
        else:
            assert recipe_name == "nvfp4", "unsupported"
            config = NVFP4DynamicActivationNVFP4WeightConfig(
                use_triton_kernel=False,
                use_dynamic_per_tensor_scale=False,
            )

        quantize_(m, config=config)
        torch.save(m.state_dict(), f.name)
        fname = f.name

    assert fname is not None

    code = f"""
import torch
import torchao.prototype.mx_formats
_ = torch.load('{fname}', weights_only=True)
    """

    subprocess_out = subprocess.run(["python"], input=code, text=True)
    os.remove(fname)
    assert subprocess_out.returncode == 0, "failed weights-only load"


@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="CUDA or XPU not available",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm_at_least_100(),
    reason="needs CUDA capability 10.0+",
)
@pytest.mark.parametrize("old_is_swizzled", [False, True])
def test_setstate_migrates_old_is_swizzled_scales(old_is_swizzled):
    """
    Regression: old checkpoints stored ``is_swizzled_scales: bool`` instead of
    ``swizzle_type``.  We can't produce old-format files from current code, so
    we call ``__setstate__`` directly with simulated old-format data (this is
    the exact code path ``torch.load``).
    """
    device = torch.accelerator.current_accelerator().type
    # NoSwizzle() is used here only to create valid tensor data for the test;
    # the actual backward-compat migration is driven by old_is_swizzled below.
    m = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16, device=device)
    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=KernelPreference.EMULATED,
        swizzle_type=NoSwizzle(),
    )
    quantize_(m, config=config)

    weight = m.weight
    assert isinstance(weight, MXTensor)

    # Build an old-format state dict (as torch.load would pass to __setstate__)
    old_state = {
        "qdata": weight.qdata,
        "scale": weight.scale,
        "elem_dtype": weight.elem_dtype,
        "block_size": weight.block_size,
        "orig_dtype": weight.orig_dtype,
        "kernel_preference": weight.kernel_preference,
        "act_quant_kwargs": weight.act_quant_kwargs,
        "is_swizzled_scales": old_is_swizzled,  # old bool field
    }

    # Create a shell MXTensor and apply __setstate__ (simulates torch.load path)
    shell = torch.Tensor._make_wrapper_subclass(
        MXTensor,
        weight.shape,
        strides=weight.stride(),
        dtype=weight.orig_dtype,
        device=weight.device,
    )
    shell.__setstate__(old_state)

    # Verify top-level migration
    expected = Swizzle_32_4_4 if old_is_swizzled else NoSwizzle
    assert isinstance(shell.swizzle_type, expected), (
        f"Top-level: expected {expected.__name__}, got {type(shell.swizzle_type).__name__}"
    )


@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="CUDA or XPU not available",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm_at_least_100(),
    reason="needs CUDA capability 10.0+",
)
@pytest.mark.parametrize("old_is_swizzled", [False, True])
def test_tensor_unflatten_migrates_old_is_swizzled_scales(old_is_swizzled):
    """
    Regression: old checkpoints stored ``is_swizzled_scales: bool`` instead of
    ``swizzle_type``.  We can't produce old-format files from current code, so
    we call ``__tensor_unflatten__`` directly with simulated old-format metadata
    (this is the exact code path Dynamo tracing takes).
    """
    device = torch.accelerator.current_accelerator().type
    # NoSwizzle() is used here only to create valid tensor data for the test;
    # the actual backward-compat migration is driven by old_is_swizzled below.
    m = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16, device=device)
    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=KernelPreference.EMULATED,
        swizzle_type=NoSwizzle(),
    )
    quantize_(m, config=config)

    weight = m.weight
    assert isinstance(weight, MXTensor)

    tensor_data = {"qdata": weight.qdata, "scale": weight.scale}
    old_attrs = {
        "elem_dtype": weight.elem_dtype,
        "block_size": weight.block_size,
        "orig_dtype": weight.orig_dtype,
        "kernel_preference": weight.kernel_preference,
        "act_quant_kwargs": weight.act_quant_kwargs,
        "is_swizzled_scales": old_is_swizzled,
    }

    expected = Swizzle_32_4_4 if old_is_swizzled else NoSwizzle
    restored = MXTensor.__tensor_unflatten__(
        tensor_data, dict(old_attrs), weight.shape, None
    )
    assert isinstance(restored.swizzle_type, expected)
