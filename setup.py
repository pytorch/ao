# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import subprocess
import sys
import time
from datetime import datetime

from setuptools import find_packages, setup

current_date = datetime.now().strftime("%Y%m%d")


def get_git_commit_id():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return ""


def read_requirements(file_path):
    with open(file_path, "r") as file:
        return file.read().splitlines()


def read_version(file_path="version.txt"):
    with open(file_path, "r") as file:
        return file.readline().strip()


# Use Git commit ID if VERSION_SUFFIX is not set
version_suffix = os.getenv("VERSION_SUFFIX")
if version_suffix is None:
    version_suffix = f"+git{get_git_commit_id()}"

use_cpp = os.getenv("USE_CPP")

version_prefix = read_version()
# Version is version.dev year month date if using nightlies and version if not
version = (
    f"{version_prefix}.dev{current_date}"
    if os.environ.get("TORCHAO_NIGHTLY")
    else version_prefix
)

import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_WINDOWS,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# Constant known variables used throughout this file
cwd = os.path.abspath(os.path.curdir)
third_party_path = os.path.join(cwd, "third_party")


def get_submodule_folders():
    git_modules_path = os.path.join(cwd, ".gitmodules")
    default_modules_path = [
        os.path.join(third_party_path, name)
        for name in [
            "cutlass",
        ]
    ]
    if not os.path.exists(git_modules_path):
        return default_modules_path
    with open(git_modules_path) as f:
        return [
            os.path.join(cwd, line.split("=", 1)[1].strip())
            for line in f
            if line.strip().startswith("path")
        ]


def check_submodules():
    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            print("Could not find any of {} in {}".format(", ".join(files), folder))
            print("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (
            os.path.isdir(folder) and len(os.listdir(folder)) == 0
        )

    if bool(os.getenv("USE_SYSTEM_LIBS", False)):
        return
    folders = get_submodule_folders()
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(" --- Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd
            )
            end = time.time()
            print(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            print(" --- Submodule initalization failed")
            print("Please run:\n\tgit submodule update --init --recursive")
            sys.exit(1)
    for folder in folders:
        check_for_files(
            folder,
            [
                "CMakeLists.txt",
                "Makefile",
                "setup.py",
                "LICENSE",
                "LICENSE.md",
                "LICENSE.txt",
            ],
        )


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    if not torch.cuda.is_available():
        print(
            "PyTorch GPU support is not available. Skipping compilation of CUDA extensions"
        )
    if CUDA_HOME is None and torch.cuda.is_available():
        print("CUDA toolkit is not available. Skipping compilation of CUDA extensions")
        print(
            "If you'd like to compile CUDA extensions locally please install the cudatoolkit from https://anaconda.org/nvidia/cuda-toolkit"
        )

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "-t=0",
        ]
    }

    if not IS_WINDOWS:
        extra_compile_args["cxx"] = [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ]

        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.extend(["-O0", "-g"])
    else:
        extra_compile_args["cxx"] = ["/O2" if not debug_mode else "/Od", "/permissive-"]

        if debug_mode:
            extra_compile_args["cxx"].append("/ZI")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.append("/DEBUG")

    use_cutlass = False
    if use_cuda and not IS_WINDOWS:
        use_cutlass = True
        cutlass_dir = os.path.join(third_party_path, "cutlass")
        cutlass_include_dir = os.path.join(cutlass_dir, "include")
    if use_cutlass:
        extra_compile_args["nvcc"].extend(
            [
                "-DTORCHAO_USE_CUTLASS",
                "-I" + cutlass_include_dir,
            ]
        )

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "torchao", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(
        glob.glob(os.path.join(extensions_cuda_dir, "**/*.cu"), recursive=True)
    )

    if use_cuda:
        sources += cuda_sources

    if len(sources) == 0:
        return None

    ext_modules = [
        extension(
            "torchao._C",
            sources,
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


check_submodules()

setup(
    name="torchao",
    version=version + version_suffix,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "torchao.kernel.configs": ["*.pkl"],
    },
    ext_modules=get_extensions() if use_cpp != "0" else None,
    extras_require={"dev": read_requirements("dev-requirements.txt")},
    description="Package for applying ao techniques to GPU models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/ao",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
