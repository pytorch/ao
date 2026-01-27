# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os

import torch


def _get_torchao_mps_lib_path():
    """Get the path to the MPS ops library.

    Searches in the following locations:
    1. The torchao package directory (for pip-installed packages)
    2. The build directory (for development installs from source)
    3. The cmake-out directory relative to this file (for standalone CMake builds)
    """
    import torchao

    libname = "libtorchao_ops_mps_aten.dylib"

    # Try the torchao package directory first (pip install location)
    torchao_dir = os.path.dirname(torchao.__file__)
    pip_libpath = os.path.join(torchao_dir, libname)
    if os.path.exists(pip_libpath):
        return pip_libpath

    # Try the build directory (for editable/development installs)
    # The build directory is typically at the repo root level
    repo_root = os.path.dirname(torchao_dir)
    build_pattern = os.path.join(repo_root, "build", "lib.*", "torchao", libname)
    build_matches = glob.glob(build_pattern)
    if build_matches:
        return build_matches[0]

    # Fall back to cmake-out directory (standalone CMake build)
    cmake_libpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "cmake-out/lib/", libname)
    )
    if os.path.exists(cmake_libpath):
        return cmake_libpath

    return None


def _load_torchao_mps_lib():
    """Load the MPS ops library."""
    try:
        for nbit in range(1, 8):
            getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
            getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
    except AttributeError:
        libpath = _get_torchao_mps_lib_path()
        if libpath is None:
            raise RuntimeError(
                "Could not find libtorchao_ops_mps_aten.dylib. "
                "Please build with TORCHAO_BUILD_EXPERIMENTAL_MPS=1"
            )
        try:
            torch.ops.load_library(libpath)
        except Exception as e:
            raise RuntimeError(f"Failed to load library {libpath}: {e}")

        for nbit in range(1, 8):
            getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
            getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
