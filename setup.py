# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import subprocess
from datetime import datetime

from setuptools import Extension, find_packages, setup

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

import platform

build_torchao_experimental = (
    use_cpp == "1"
    and platform.machine().startswith("arm64")
    and platform.system() == "Darwin"
)

version_prefix = read_version()
# Version is version.dev year month date if using nightlies and version if not
version = (
    f"{version_prefix}.dev{current_date}"
    if os.environ.get("TORCHAO_NIGHTLY")
    else version_prefix
)


def use_debug_mode():
    return os.getenv("DEBUG", "0") == "1"


import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_WINDOWS,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


# BuildExtension is a subclass of from setuptools.command.build_ext.build_ext
class TorchAOBuildExt(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        cmake_extensions = [
            ext for ext in self.extensions if isinstance(ext, CMakeExtension)
        ]
        other_extensions = [
            ext for ext in self.extensions if not isinstance(ext, CMakeExtension)
        ]
        for ext in cmake_extensions:
            self.build_cmake(ext)

        # Use BuildExtension to build other extensions
        self.extensions = other_extensions
        super().build_extensions()

        self.extensions = other_extensions + cmake_extensions

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        build_type = "Debug" if use_debug_mode() else "Release"

        from distutils.sysconfig import get_python_lib

        torch_dir = get_python_lib() + "/torch/share/cmake/Torch"

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            [
                "cmake",
                ext.sourcedir,
                "-DCMAKE_BUILD_TYPE=" + build_type,
                "-DTORCHAO_BUILD_EXECUTORCH_OPS=OFF",
                "-DTorch_DIR=" + torch_dir,
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            ],
            cwd=self.build_temp,
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def get_extensions():
    debug_mode = use_debug_mode()
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
        this_dir = os.path.abspath(os.path.curdir)
        cutlass_dir = os.path.join(this_dir, "third_party", "cutlass")
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

    ext_modules = []
    if len(sources) > 0:
        ext_modules.append(
            extension(
                "torchao._C",
                sources,
                py_limited_api=True,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        )

    if build_torchao_experimental:
        ext_modules.append(
            CMakeExtension(
                "torchao.experimental",
                sourcedir="torchao/experimental",
            )
        )

    return ext_modules


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
    cmdclass={"build_ext": TorchAOBuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
