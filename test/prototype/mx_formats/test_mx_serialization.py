# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import (
    is_sm_at_least_100,
    torch_version_at_least,
)

if not torch_version_at_least("2.8.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="No accelerator available"
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
    device = torch.accelerator.current_accelerator()

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

    subprocess_out = subprocess.run([sys.executable], input=code, text=True)
    os.remove(fname)
    assert subprocess_out.returncode == 0, "failed weights-only load"
