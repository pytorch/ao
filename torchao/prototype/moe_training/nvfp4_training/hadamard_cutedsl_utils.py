# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime availability checks for the CuteDSL NVFP4 RHT kernels."""

import importlib.util

import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

# Import name -> pip package name for the required runtime packages.
_CUTEDSL_RUNTIME_PACKAGES = {
    "cuda.bindings.driver": "cuda-python",
    "cutlass": "nvidia-cutlass-dsl",
    "cutlass.cute": "nvidia-cutlass-dsl",
    "tvm_ffi": "apache-tvm-ffi",
}


def _missing_cutedsl_runtime_packages() -> list:
    """Return the list of missing CuteDSL runtime packages (empty if all present)."""
    missing = []
    for module_name, package_name in _CUTEDSL_RUNTIME_PACKAGES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            # ModuleNotFoundError: parent module absent (e.g. 'cuda' on CPU)
            # ValueError: malformed module name
            spec = None
        if spec is None and package_name not in missing:
            missing.append(package_name)
    return missing


def _cutedsl_runtime_available() -> bool:
    """True iff the cuda-python + nvidia-cutlass-dsl + apache-tvm-ffi packages are importable."""
    return len(_missing_cutedsl_runtime_packages()) == 0


def cutedsl_nvfp4_kernels_available() -> bool:
    """True iff the CuteDSL NVFP4 RHT kernels can run here.

    Requires a Blackwell (SM 10.x) GPU, CUDA 12.8+, and the cuda-python +
    nvidia-cutlass-dsl + apache-tvm-ffi runtime packages.
    """
    return (
        torch.cuda.is_available()
        and is_sm_at_least_100()
        and is_cuda_version_at_least(12, 8)
        and _cutedsl_runtime_available()
    )


# Full requirement list, for user-facing error messages.
CUTEDSL_NVFP4_REQUIREMENTS = (
    "a Blackwell (SM100) GPU, CUDA 12.8+, and the cuda-python, nvidia-cutlass-dsl, "
    "and apache-tvm-ffi packages"
)


def cutedsl_nvfp4_unavailable_reason() -> str:
    """Reason the CuteDSL NVFP4 kernels are unavailable: the unmet hardware, CUDA, or
    package requirement(s). Returns ``""`` when everything is available."""
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("no CUDA device")
    elif not is_sm_at_least_100():
        reasons.append("requires a Blackwell (SM 10.x) GPU")
    if not is_cuda_version_at_least(12, 8):
        reasons.append("requires CUDA 12.8+")
    missing = _missing_cutedsl_runtime_packages()
    if missing:
        reasons.append("missing packages: " + ", ".join(sorted(missing)))
    return "; ".join(reasons)
