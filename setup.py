# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import glob
import re
from distutils import log
from distutils.sysconfig import get_python_lib

from datetime import datetime
from pathlib import Path
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from setuptools import find_packages, setup, Extension

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
install_executorch_kernels = os.getenv('USE_EXECUTORCH', '0')

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
    use_mps = torch.backends.mps.is_available()
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
    sources = [file_path for file_path in sources if "executorch" not in file_path.split(os.sep)]
    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "**/*.cu"), recursive=True))

    if use_cuda:
        sources += cuda_sources

    if use_mps:
        extensions_mps_dir = os.path.join(extensions_dir, "metal")
        mps_sources = list(glob.glob(os.path.join(extensions_mps_dir, "**/*.mm"), recursive=True))
        sources += mps_sources
    ext_modules = [
        extension(
            "torchao._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    if install_executorch_kernels != '0':
        ext_modules.append(
            CMakeExtension(
                "torchao.executorch_kernels", "torchao/csrc/executorch"
            )
        )

    return ext_modules

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(BuildExtension):
    def build_extension(self, ext: Extension):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        default_parallel = str(os.cpu_count() - 1)
        self.parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", default_parallel)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # get_python_lib() typically returns the path to site-packages, where
        # all pip packages in the environment are installed.
        cmake_prefix_path = os.environ.get("CMAKE_PREFIX_PATH", get_python_lib())

        # The root of the repo should be the current working directory. Get
        # the absolute path.
        repo_root = os.fspath(Path.cwd() / ext.sourcedir)
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",            
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        build_args = [f"-j{self.parallel}"]

        build_args += ["--target", "executorch_kernels", "--clean-first"]

        # Put the cmake cache under the temp directory, like
        # "pip-out/temp.<plat>/cmake-out".
        cmake_cache_dir = os.path.join(repo_root, self.build_temp, "cmake-out")
        self.mkpath(cmake_cache_dir)

        if not self.dry_run:
            # Dry run should log the command but not actually run it.
            (Path(cmake_cache_dir) / "CMakeCache.txt").unlink(missing_ok=True)

        self.spawn(["cmake", "-S", repo_root, "-B", cmake_cache_dir, *cmake_args])

        # Build the system.
        self.spawn(["cmake", "--build", cmake_cache_dir, *build_args])
    

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
    cmdclass={"build_ext": CMakeBuild},
)
