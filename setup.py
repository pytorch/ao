# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
from datetime import datetime

from setuptools import find_packages, setup

current_date = datetime.now().strftime("%Y.%m.%d")

def read_requirements(file_path):
    with open(file_path, "r") as file:
        return file.read().splitlines()

def read_version(file_path="version.txt"):
    with open(file_path, "r") as file:
        return file.readline().strip()

# Determine the package name based on the presence of an environment variable
package_name = "torchao-nightly" if os.environ.get("TORCHAO_NIGHTLY") else "torchao"
version_suffix = os.getenv("VERSION_SUFFIX", "")
use_cpp = os.getenv('USE_CPP')


# Version is year.month.date if using nightlies
version = current_date if package_name == "torchao-nightly" else read_version()

import torch

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
    IS_WINDOWS
)


def get_extensions():
    debug_mode = os.getenv('DEBUG', '0') == '1'
    if debug_mode:
        print("Compiling in debug mode")

    if not torch.cuda.is_available():
        print("PyTorch GPU support is not available. Skipping compilation of CUDA extensions")
    if CUDA_HOME is None and torch.cuda.is_available():
        print("CUDA toolkit is not available. Skipping compilation of CUDA extensions")
        print("If you'd like to compile CUDA extensions locally please install the cudatoolkit from https://anaconda.org/nvidia/cuda-toolkit")

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    if not IS_WINDOWS:
        extra_link_args = []
        extra_compile_args = {
            "cxx": [
                "-O3" if not debug_mode else "-O0",
                "-fdiagnostics-color=always",
            ],
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
                "-t=0",
            ]
        }

        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.extend(["-O0", "-g"])

    else:
        extra_link_args = []
        extra_compile_args = {
            "cxx": [
                "/O2" if not debug_mode else "/Od",
                "/permissive-"
            ],
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
                "-t=0",
            ]
        }

        if debug_mode:
            extra_compile_args["cxx"].append("/ZI")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.append("/DEBUG")

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "torchao", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "**/*.cu"), recursive=True))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            "torchao._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules

setup(
    name=package_name,
    version=version+version_suffix,
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
    url="https://github.com/pytorch-labs/ao",
    cmdclass={"build_ext": BuildExtension},
)
