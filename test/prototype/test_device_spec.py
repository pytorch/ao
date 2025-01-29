import pytest

cuda_driver = pytest.importorskip(
    "triton.runtime.driver", reason="requires triton cuda driver module"
)
import itertools

import torch
from utils import patch_device

from torchao.prototype.profiler.device_spec import (
    _AVAILABLE_GPU_SPECS,
    CUDADeviceSpec,
    get_chip_name,
)

# -------------------- Device Spec Tests ------------------- #
DEVICE_NAMES = ["h100 sxm", "a100", "nvidia geforce rtx 4090"]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
USE_TENSORCORES = [True, False]
DEVICE_CONFIGS = itertools.product(DEVICE_NAMES, DTYPES, USE_TENSORCORES)


@pytest.mark.parametrize(
    "device_name, dtype, use_tensorcores", DEVICE_CONFIGS, ids=lambda x: str(x)
)
def test_device_spec(device_name, dtype, use_tensorcores):
    with patch_device(device_name):
        device_spec = CUDADeviceSpec(dtype=dtype, use_tensorcores=use_tensorcores)
        if dtype == torch.float32 and use_tensorcores:
            dtype = "tfloat32"
        chip_name = get_chip_name(device_name)
        expected_flops = _AVAILABLE_GPU_SPECS[chip_name][dtype]
        assert device_spec.flops_per_s == expected_flops
        assert device_spec.flops_by_dtype[dtype] == expected_flops
        assert (
            device_spec.roofline_balancepoint == expected_flops / device_spec.bandwidth
        )

        with pytest.raises(AssertionError):
            device_spec.flops_per_s = None
            print(device_spec.roofline_balancepoint)
        # Prevent setting attributes not in named fields to guard against user error
        with pytest.raises(AttributeError):
            device_spec.FLOPs = None


def test_empty_device_spec():
    device_name = "fake device"
    with patch_device(device_name):
        with pytest.raises(AssertionError):
            _ = CUDADeviceSpec()

        # Ok to instantiate as long as fields are filled
        _ = CUDADeviceSpec(
            name=device_name,
            flops_per_s=1.0,
            bandwidth=1.0,
            dtype=torch.float32,
            use_tensorcores=True,
        )
    device_name = DEVICE_NAMES[0]

    with patch_device(device_name):
        # All critical fields will be auto-filled except for dtype (and vram, but vram is not used for downstream calcs atm)
        _ = CUDADeviceSpec(dtype=torch.float32)

        # No dtype specified
        with pytest.raises(AssertionError):
            _ = CUDADeviceSpec()
