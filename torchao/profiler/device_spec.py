from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Union

import torch

"""This module contains the device specs for theoretical peak performance calculations.

- Contains a list of available chips and their corresponding theoretical peak FLOPs performance for various torch.dtypes.
- Exposes a DeviceSpec interface and a concrete CUDADeviceSpec implementation for CUDA gpus.  Extendable to other device types.
- Where possible, the CUDADeviceSpec auto-populates its fields by utilizing `torch.cuda` API and `triton.runtime.driver`.

"""
# Copied from https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/utilities/throughput.py
_AVAILABLE_GPU_SPECS: Dict[str, Dict[Union[str, torch.dtype], float]] = {
    # Hopper
    # source: https://resources.nvidia.com/en-us-tensor-core
    "h100 nvl": {
        torch.float64: 67e12,
        torch.float32: 133.8e12,
        "tfloat32": 989.4e12,
        torch.bfloat16: 1978.8e12,
        torch.float16: 1978.8e12,
        torch.int8: 3957.8e12,
    },
    "h100 sxm": {
        torch.float64: 33.5e12,
        torch.float32: 66.9e12,
        "tfloat32": 494.7e12,
        torch.bfloat16: 989.4e12,
        torch.float16: 989.4e12,
        torch.int8: 1978.9e12,
    },
    "h100 pcie": {
        torch.float64: 25.6e12,
        torch.float32: 51.2e12,
        "tfloat32": 378e12,
        torch.bfloat16: 756e12,
        torch.float16: 756e12,
        torch.int8: 1513e12,
    },
    # Ada
    # source: https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    "rtx 4090": {
        torch.float32: 82.6e12,
        "tfloat32": 82.6e12,
        torch.bfloat16: 82.6e12,
        torch.float16: 82.6e12,
        torch.int8: 660.6e12,
        "int4": 1321.2e12,
    },
    "rtx 4080": {
        torch.float32: 48.7e12,
        "tfloat32": 48.7e12,
        torch.bfloat16: 48.7e12,
        torch.float16: 48.7e12,
        torch.int8: 389.9e12,
        "int4": 779.8e12,
    },
    "l4": {
        torch.float32: 30.3e12,
        "tfloat32": 60e12,
        torch.bfloat16: 121e12,
        torch.float16: 121e12,
        torch.int8: 242e12,
        "int4": 484e12,
    },
    "l40": {
        torch.float32: 90.5e12,
        "tfloat32": 90.5e12,
        torch.bfloat16: 181e12,
        torch.float16: 181e12,
        torch.int8: 362e12,
        "int4": 724e12,
    },
    # Ampere
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        torch.float64: 9.7e12,
        torch.float32: 19.5e12,
        "tfloat32": 156e12,
        torch.bfloat16: 312e12,
        torch.float16: 312e12,
        torch.int8: 624e12,
    },
    "a6000": {
        torch.float32: 38.7e12,
        "tfloat32": 77.4e12,
        torch.bfloat16: 38.7e12,
        torch.float16: 38.7e12,
        torch.int8: 309.7e12,
        "int4": 619.3e12,
    },
    "a40": {
        torch.float32: 37.4e12,
        "tfloat32": 74.8e12,
        torch.bfloat16: 37.4e12,
        torch.float16: 37.4e12,
        torch.int8: 299.3e12,
        "int4": 598.7e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10g": {
        torch.float32: 31.2e12,
        "tfloat32": 62.5e12,
        torch.bfloat16: 125e12,
        torch.float16: 125e12,
        torch.int8: 250e12,
        "int4": 500e12,
    },
    "rtx 3090 ti": {
        torch.float32: 40e12,
        "tfloat32": 40e12,
        torch.bfloat16: 40e12,
        torch.float16: 40e12,
        torch.int8: 320e12,
        "int4": 640e12,
    },
    "rtx 3090": {
        torch.float32: 35.6e12,
        "tfloat32": 35.6e12,
        torch.bfloat16: 35.6e12,
        torch.float16: 35.6e12,
        torch.int8: 284e12,
        "int4": 568e12,
    },
    "rtx 3080 ti": {
        torch.float32: 34.1e12,
        "tfloat32": 34.1e12,
        torch.bfloat16: 34.1e12,
        torch.float16: 34.1e12,
        torch.int8: 272.8e12,
        "int4": 546.6e12,
    },
    "rtx 3080": {
        torch.float32: 29.8e12,
        "tfloat32": 29.8e12,
        torch.bfloat16: 29.8e12,
        torch.float16: 29.8e12,
        torch.int8: 238e12,
        "int4": 476e12,
    },
    "rtx 3070": {
        torch.float32: 20.3e12,
        "tfloat32": 20.3e12,
        torch.bfloat16: 20.3e12,
        torch.float16: 20.3e12,
        torch.int8: 162.6e12,
        "int4": 325.2e12,
    },
    # Turing
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {
        torch.float32: 8.1e12,
        torch.float16: 65e12,
        torch.int8: 130e12,
        "int4": 260e12,
    },
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
    "quadro rtx 5000": {
        torch.float32: 11.2e12,
        torch.float16: 89.2e12,
    },
    "rtx 2080 super": {
        torch.float32: 11.2e12,
        torch.float16: 22.3e12,
        torch.int8: 178.4e12,
        "int4": 356.8e12,
    },
    "rtx 2080 ti": {
        torch.float32: 14.2e12,
        torch.float16: 28.5e12,
        torch.int8: 227.7e12,
        "int4": 455.4e12,
    },
    "rtx 2080": {
        torch.float32: 10.6e12,
        torch.float16: 21.2e12,
        torch.int8: 169.6e12,
        "int4": 339.1e12,
    },
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    "rtx 2070 super": {
        torch.float32: 9.1e12,
        torch.float16: 18.1e12,
        torch.int8: 145e12,
        "int4": 290e12,
    },
    "titan rtx": {
        torch.float32: 16.3e12,
        torch.float16: 32.6e12,
        torch.int8: 261e12,
        "int4": 522e12,
    },
    # Volta
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100 sxm": {
        torch.float64: 7.8e12,
        torch.float32: 15.7e12,
        torch.float16: 125e12,
    },
    "v100 pcie": {
        torch.float64: 7e12,
        torch.float32: 14e12,
        torch.float16: 112e12,
    },
    "v100s pcie": {
        torch.float64: 8.2e12,
        torch.float32: 16.4e12,
        torch.float16: 130e12,
    },
}


# Adapted from https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/utilities/throughput.py
def get_chip_name(device: int = 0) -> str:
    device_name = torch.cuda.get_device_name(device)
    chip = device_name.lower()

    if "h100" in chip:
        if "hbm3" in chip:
            chip = "h100 sxm"
        elif "nvl" in chip:
            chip = "h100 nvl"
        elif "pcie" in chip or "hbm2e" in chip:
            chip = "h100 pcie"
    elif "l4" in chip:
        chip = "l40" if "tesla" in chip else "l4"
    elif "geforce rtx" in chip:
        number = chip.split(" ")[3]
        extra = ""
        if "super" in chip:
            extra = " super"
        elif "ti" in chip:
            extra = " ti"
        chip = f"rtx {number}{extra}"
    elif "a6000" in chip:
        chip = "a6000"
    elif "a100" in chip:
        chip = "a100"
    elif "a40" in chip:
        chip = "a40"
    elif "a10g" in chip:
        chip = "a10g"
    elif "t4" in chip:
        chip = "t4"
    elif "quadro rtx 5000" in chip:
        chip = "quadro rtx 5000"
    elif "titan rtx" in chip:
        chip = "titan rtx"
    elif "v100-sxm" in chip:
        chip = "v100 sxm"
    elif "v100-pcie" in chip:
        chip = "v100 pcie"
    elif "v100s-pcie" in chip:
        chip = "v100s pcie"
    else:
        chip = None
    return chip


def get_vram(device: int = 0) -> int:
    device_props = torch.cuda.get_device_properties(device)
    return device_props.total_memory


def get_bandwidth(device: int = 0) -> int:
    try:
        from triton.testing import get_dram_gbps

        bandwidth = get_dram_gbps(device) * 1e9
    except ImportError:
        print("Could not import triton to get DRAM Gbps. Please install triton")
        bandwidth = None
    return bandwidth


def get_flops_by_dtype(chip_name: str) -> dict[torch.dtype, float]:
    return _AVAILABLE_GPU_SPECS.get(chip_name, None)


@dataclass
class DeviceSpec:
    """
    Abstract device specs for theoretical peak performance calculations.

    Fields will be auto-populated in __post_init__ if not already specified
    and if data is available
    - bandwidth (bytes /s)
    - flops_per_s (FLOP / s)
    - vram (bytes)
    - dtype (torch.dtype) dtype used for theoretical peak performance
    - flops_by_dtype (dict[Union[torch.dtype, str], float]): mapping from dtype to FLOP / s
    """

    device_type: str
    name: Optional[str] = None
    bandwidth: Optional[int] = None
    flops_per_s: Optional[int] = None
    vram: Optional[int] = None
    dtype: Optional[torch.dtype] = None
    flops_by_dtype: dict = field(default_factory=dict)

    def _post_init_check(self):
        assert (
            self.bandwidth is not None
        ), "GPU bandwidth is None - please specify the bandwidth in GB/s in order to enable speed of light calculations"
        assert (
            self.dtype is not None
        ), "GPU dtype is None - please specify the dtype in order to enable speed of light calculations"
        assert (
            self.flops_per_s is not None
        ), "GPU flops_per_s is None - please specify the flops_per_s in FLOP/s in order to enable speed of light calculations"
        self.flops_by_dtype.update({self.dtype: self.flops_per_s})

        # Not needed for downstream calculations atm, no need to assert
        if self.vram is None:
            print("GPU vram is None - please specify the vram in bytes")

    def __setattr__(self, name, value):
        # Check if the attribute is already defined
        if name in {f.name for f in fields(self)}:
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"Cannot add new attribute '{name}' to {self.__class__.__name__}"
            )

    def __str__(self):
        if self.bandwidth is not None:
            formatted_bw = f"{self.bandwidth / 1e9:,.1f}GB/s"
        if self.flops_per_s is not None:
            formatted_flops = f"{self.flops_per_s / 1e12:,.1f}TFLOPs"
        if self.vram is not None:
            formatted_vram = f"{self.vram / 1e9:,.1f}GB"
        return f"DeviceSpec(device_type={self.device_type}, name={self.name}, dtype={self.dtype}, bandwidth={formatted_bw}, flops={formatted_flops}, vram={formatted_vram})"

    @property
    def roofline_balancepoint(self):
        """
        Arithmetic intensity (FLOP / byte) transition point from
        memory-bound to compute-bound regime.

        This is the ridgepoint of the roofline curve.
        """
        assert (
            self.bandwidth is not None
        ), "Please set bandwidth in order to calculate roofline balancepoint"
        assert (
            self.flops_per_s is not None
        ), "Please set flops_per_s in order to calculate roofline balancepoint"

        return self.flops_per_s / self.bandwidth


@dataclass
class CUDADeviceSpec(DeviceSpec):
    """
    CUDA specs for theoretical peak performance, conformant with DeviceSpec interface.

    Fields will be auto-populated in __post_init__ if not specified
    and if data is available.

    See _AVAILABLE_GPU_SPECS for a list of available chip data.

    Fields and expected units:
        - device (int): CUDA device index
        - name (str): name of the device
        - bandwidth (bytes /s): memory bandwidth in bytes / s
        - flops_per_s (FLOP / s): FLOPs per second
        - vram (bytes): VRAM in bytes
        - dtype (torch.dtype): dtype used for theoretical peak performance
        - flops_by_dtype (dict[Union[torch.dtype, str], float]): mapping from dtype to FLOP / s
        - use_tensorcores (bool): whether to use tensorcores if dtype == torch.float32
    """

    device_type: str = "cuda"
    # Device index
    device: Optional[int] = 0
    # Whether to use tfloat32 FLOPs for dtype == torch.float32
    # We assume that tensorcores will always be used for fp16, int8, and other sub-single precision dtypes
    use_tensorcores: bool = True

    def __post_init__(self):
        # Populate fields if not already populated
        self.name = torch.cuda.get_device_name(self.device)

        # Memory bandwidth in bytes / s
        if self.bandwidth is None:
            self.bandwidth = get_bandwidth()

        # FLOPs / s
        if self.flops_per_s is None:
            chip_name = get_chip_name(self.device)
            if chip_name is None:
                print(f"No FLOPs data available for device name {self.name}")
            else:
                flops_by_dtype = get_flops_by_dtype(chip_name)
                if flops_by_dtype is not None:
                    self.flops_by_dtype.update(flops_by_dtype)

                # Populate flops if not already populated
                if flops_by_dtype is not None and self.dtype in flops_by_dtype:
                    self.flops_per_s = flops_by_dtype[self.dtype]

                    if self.dtype == torch.float32:
                        use_tf32 = "tfloat32" in flops_by_dtype and self.use_tensorcores

                        if use_tf32:
                            self.flops_per_s = flops_by_dtype["tfloat32"]
                else:
                    print(
                        f"Could not find FLOPs for dtype {self.dtype} for device {self.name}"
                    )
        # Vram
        if self.vram is None:
            self.vram = get_vram()

        # Issue post check warnings
        self._post_init_check()
