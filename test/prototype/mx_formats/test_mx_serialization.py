# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import subprocess
import tempfile

import pytest
import torch
import torch.nn as nn

from torchao.prototype.mx_formats.config import NO_SWIZZLE, SWIZZLE_32_4_4
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
def test_load_old_format_is_swizzled_scales():
    """
    Regression test: load a state_dict saved with the old bool-based
    `is_swizzled_scales` field (both top-level MXTensor attribute and nested
    QuantizeTensorToMXKwargs). Verifies backward compat shims work.
    """
    device = torch.accelerator.current_accelerator().type
    m = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16, device=device)
    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=KernelPreference.EMULATED,
        swizzle_type=NO_SWIZZLE(),
    )
    quantize_(m, config=config)

    # Save state_dict and manually patch to old format
    sd = m.state_dict()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        # Patch the MXTensor to simulate old format:
        # Replace swizzle_type with is_swizzled_scales in tensor_attributes
        for key, val in sd.items():
            if isinstance(val, MXTensor):
                # Patch act_quant_kwargs to old format
                if val.act_quant_kwargs is not None:
                    old_kwargs = copy.copy(val.act_quant_kwargs)
                    old_kwargs.__dict__["is_swizzled_scales"] = isinstance(
                        old_kwargs.swizzle_type, SWIZZLE_32_4_4
                    )
                    del old_kwargs.__dict__["swizzle_type"]
                    val.act_quant_kwargs = old_kwargs
        # Save with pickle (bypass weights_only for patching)
        torch.save(sd, f.name)
        fname = f.name

    # Load in a subprocess to prove weights_only loading works
    code = f"""
import torch
import torchao.prototype.mx_formats
sd = torch.load('{fname}', weights_only=True)
# Verify act_quant_kwargs has swizzle_type after loading
for k, v in sd.items():
    if hasattr(v, 'act_quant_kwargs') and v.act_quant_kwargs is not None:
        assert hasattr(v.act_quant_kwargs, 'swizzle_type'), (
            f"act_quant_kwargs missing swizzle_type after load: {{v.act_quant_kwargs.__dict__}}"
        )
print("OK")
"""
    result = subprocess.run(["python"], input=code, text=True, capture_output=True)
    os.remove(fname)
    assert result.returncode == 0, f"Failed: {result.stderr}"


@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="CUDA or XPU not available",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm_at_least_100(),
    reason="needs CUDA capability 10.0+",
)
def test_load_old_format_top_level_is_swizzled_scales():
    """
    Regression test: MXTensor.__tensor_unflatten__ converts old
    is_swizzled_scales metadata to swizzle_type.
    """
    device = torch.accelerator.current_accelerator().type
    m = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16, device=device)
    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=KernelPreference.EMULATED,
        swizzle_type=NO_SWIZZLE(),
    )
    quantize_(m, config=config)

    # Extract the MXTensor and manually reconstruct it via __tensor_unflatten__
    # using the old metadata format (is_swizzled_scales instead of swizzle_type)
    weight = m.weight
    assert isinstance(weight, MXTensor)

    tensor_data = {"qdata": weight.qdata, "scale": weight.scale}
    old_attrs = {
        "elem_dtype": weight.elem_dtype,
        "block_size": weight.block_size,
        "orig_dtype": weight.orig_dtype,
        "kernel_preference": weight.kernel_preference,
        "act_quant_kwargs": weight.act_quant_kwargs,
        "is_swizzled_scales": False,  # old bool format
    }

    # This exercises the __tensor_unflatten__ backward compat shim
    restored = MXTensor.__tensor_unflatten__(
        tensor_data, dict(old_attrs), weight.shape, None
    )
    assert isinstance(restored.swizzle_type, NO_SWIZZLE)

    # Also test with is_swizzled_scales=True
    old_attrs_swizzled = dict(old_attrs)
    old_attrs_swizzled["is_swizzled_scales"] = True
    restored2 = MXTensor.__tensor_unflatten__(
        tensor_data, old_attrs_swizzled, weight.shape, None
    )
    assert isinstance(restored2.swizzle_type, SWIZZLE_32_4_4)
