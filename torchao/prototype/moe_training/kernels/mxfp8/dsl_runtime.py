# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Check whether the CuTeDSL and FlyDSL runtimes are installed, without importing them.

Importing cutlass can break CUDA in processes forked afterwards (seen with
nvidia-cutlass-dsl 4.6.0), so `quant.py` takes these checks from here rather than
from `cute_utils.py` and `flydsl_utils.py`, which import the runtimes.
"""

import importlib.machinery
import importlib.util


def _is_installed(module_name: str) -> bool:
    # find_spec("a.b") imports package "a" to read its path, so walk the package
    # path with PathFinder instead and never run a package's __init__.
    top, _, rest = module_name.partition(".")
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return False
    name = top
    for part in rest.split(".") if rest else ():
        if spec is None or spec.submodule_search_locations is None:
            return False
        name = f"{name}.{part}"
        spec = importlib.machinery.PathFinder.find_spec(
            name, spec.submodule_search_locations
        )
    return spec is not None


_CUTEDSL_RUNTIME_PACKAGES = {
    "cuda.bindings.driver": "cuda-python",
    "cutlass": "nvidia-cutlass-dsl",
    "cutlass.cute": "nvidia-cutlass-dsl",
    "tvm_ffi": "apache-tvm-ffi",
}


def _missing_cutedsl_runtime_packages() -> list[str]:
    """Check which CuTeDSL runtime packages are missing.

    Returns:
        List of missing package names
    """
    missing = []
    for module_name, package_name in _CUTEDSL_RUNTIME_PACKAGES.items():
        if not _is_installed(module_name) and package_name not in missing:
            missing.append(package_name)
    return missing


def _cutedsl_runtime_available() -> bool:
    """Check if all CuTeDSL runtime packages are available.

    Returns:
        True if all required packages are installed
    """
    return len(_missing_cutedsl_runtime_packages()) == 0


_FLYDSL_RUNTIME_PACKAGES = {
    "flydsl": "flydsl",
    "flydsl.compiler": "flydsl",
    "flydsl.expr": "flydsl",
}


def _missing_flydsl_runtime_packages() -> list[str]:
    missing = []
    for module_name, package_name in _FLYDSL_RUNTIME_PACKAGES.items():
        if not _is_installed(module_name) and package_name not in missing:
            missing.append(package_name)
    return missing


def _flydsl_runtime_available() -> bool:
    return len(_missing_flydsl_runtime_packages()) == 0
