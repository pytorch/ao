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

from torchao.prototype.mx_formats.config import (
    MXGemmKernelChoice,
)
from torchao.prototype.mx_formats.inference_workflow import (
    MXFPInferenceConfig,
    NVFP4InferenceConfig,
    NVFP4MMConfig,
)
from torchao.quantization import quantize_
from torchao.utils import (
    is_sm_at_least_100,
    torch_version_at_least,
)

if not torch_version_at_least("2.8.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="needs CUDA capability 10.0+")
@pytest.mark.parametrize("recipe_name", ["mxfp8", "nvfp4"])
def test_serialization(recipe_name):
    """
    Ensure that only `import torchao.prototype.mx_formats` is needed to load MX
    and NV checkpoints.
    """

    m = nn.Linear(32, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    fname = None
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        if recipe_name == "mxfp8":
            config = MXFPInferenceConfig(
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
            )
        else:
            assert recipe_name == "nvfp4", "unsupported"
            config = NVFP4InferenceConfig(
                mm_config=NVFP4MMConfig.DYNAMIC,
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
