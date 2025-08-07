# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import glob
import subprocess
import tempfile

import torch


def cmake_build_torchao_ops(cmake_lists_path, temp_build_dir):
    from distutils.sysconfig import get_python_lib

    print("Building torchao ops for ATen target")
    cmake_prefix_path = get_python_lib()
    subprocess.run(
        [
            "cmake",
            "-DCMAKE_PREFIX_PATH=" + cmake_prefix_path,
            "-DCMAKE_INSTALL_PREFIX=" + temp_build_dir.name,
            "-S " + cmake_lists_path,
            "-B " + temp_build_dir.name,
        ]
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            temp_build_dir.name,
            "-j 16",
            "--target install",
            "--config Release",
        ]
    )


def temp_build_and_load_torchao_ops(cmake_lists_path):
    temp_build_dir = tempfile.TemporaryDirectory()
    cmake_build_torchao_ops(cmake_lists_path, temp_build_dir)
    libs = glob.glob(f"{temp_build_dir.name}/lib/libtorchao_ops_aten.*")
    libs = list(filter(lambda l: (l.endswith("so") or l.endswith("dylib")), libs))
    assert len(libs) == 1
    torch.ops.load_library(libs[0])
    print(f"TorchAO ops are loaded from {libs[0]}")
