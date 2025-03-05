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
from typing import List, Optional

from setuptools import Extension, find_packages, setup

current_date = datetime.now().strftime("%Y%m%d")

PY3_9_HEXCODE = "0x03090000"


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
    ROCM_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

IS_ROCM = (torch.version.hip is not None) and (ROCM_HOME is not None)


class BuildOptions:
    def __init__(self):
        # TORCHAO_BUILD_CPU_AARCH64 is enabled by default on Arm-based Apple machines
        # The kernels require sdot/udot, which are not required on Arm until Armv8.4 or later,
        # but are available on Arm-based Apple machines.  On non-Apple machines, the kernels
        # can be built by explicitly setting TORCHAO_BUILD_CPU_AARCH64=1
        self.build_cpu_aarch64 = self._os_bool_var(
            "TORCHAO_BUILD_CPU_AARCH64",
            default=(self._is_arm64() and self._is_macos()),
        )
        if self.build_cpu_aarch64:
            assert (
                self._is_arm64()
            ), "TORCHAO_BUILD_CPU_AARCH64 requires an arm64 machine"

        # TORCHAO_BUILD_KLEIDIAI is disabled by default for now because
        # 1) It increases the build time
        # 2) It has some accuracy issues in CI tests due to BF16
        self.build_kleidi_ai = self._os_bool_var(
            "TORCHAO_BUILD_KLEIDIAI", default=False
        )
        if self.build_kleidi_ai:
            assert (
                self.build_cpu_aarch64
            ), "TORCHAO_BUILD_KLEIDIAI requires TORCHAO_BUILD_CPU_AARCH64 be set"

        # TORCHAO_BUILD_EXPERIMENTAL_MPS is disabled by default.
        self.build_experimental_mps = self._os_bool_var(
            "TORCHAO_BUILD_EXPERIMENTAL_MPS", default=False
        )
        if self.build_experimental_mps:
            assert self._is_macos(), "TORCHAO_BUILD_EXPERIMENTAL_MPS requires MacOS"
            assert self._is_arm64(), "TORCHAO_BUILD_EXPERIMENTAL_MPS requires arm64"
            assert (
                torch.mps.is_available()
            ), "TORCHAO_BUILD_EXPERIMENTAL_MPS requires MPS be available"

    def _is_arm64(self) -> bool:
        return platform.machine().startswith("arm64")

    def _is_macos(self) -> bool:
        return platform.system() == "Darwin"

    def _os_bool_var(self, var, default) -> bool:
        default_val = "1" if default else "0"
        return os.getenv(var, default_val) == "1"


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

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            [
                "cmake",
                ext.cmake_lists_dir,
            ]
            + ext.cmake_args
            + ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir],
            cwd=self.build_temp,
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)


class CMakeExtension(Extension):
    def __init__(
        self, name, cmake_lists_dir: str = "", cmake_args: Optional[List[str]] = None
    ):
        Extension.__init__(self, name, sources=[])
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        if cmake_args is None:
            cmake_args = []
        self.cmake_args = cmake_args


def get_extensions():
    debug_mode = use_debug_mode()
    if debug_mode:
        print("Compiling in debug mode")

    if not torch.cuda.is_available():
        print(
            "PyTorch GPU support is not available. Skipping compilation of CUDA extensions"
        )
    if (CUDA_HOME is None and ROCM_HOME is None) and torch.cuda.is_available():
        print(
            "CUDA toolkit or ROCm is not available. Skipping compilation of CUDA extensions"
        )
        print(
            "If you'd like to compile CUDA extensions locally please install the cudatoolkit from https://anaconda.org/nvidia/cuda-toolkit"
        )

    use_cuda = torch.cuda.is_available() and (
        CUDA_HOME is not None or ROCM_HOME is not None
    )
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [f"-DPy_LIMITED_API={PY3_9_HEXCODE}"],
        "nvcc": ["-O3" if not debug_mode else "-O0", "-t=0", "-std=c++17"],
    }

    if not IS_WINDOWS:
        extra_compile_args["cxx"].extend(
            ["-O3" if not debug_mode else "-O0", "-fdiagnostics-color=always"]
        )

        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            if "nvcc" in extra_compile_args:
                extra_compile_args["nvcc"].append("-g")
            extra_link_args.extend(["-O0", "-g"])
    else:
        extra_compile_args["cxx"].extend(
            ["/O2" if not debug_mode else "/Od", "/permissive-"]
        )

        if debug_mode:
            extra_compile_args["cxx"].append("/ZI")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.append("/DEBUG")

    curdir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(curdir, "torchao", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(
        glob.glob(os.path.join(extensions_cuda_dir, "**/*.cu"), recursive=True)
    )

    if use_cuda:
        sources += cuda_sources

    use_cutlass = False
    if use_cuda and not IS_WINDOWS:
        use_cutlass = True
        cutlass_dir = os.path.join(third_party_path, "cutlass")
        cutlass_include_dir = os.path.join(cutlass_dir, "include")
        cutlass_tools_include_dir = os.path.join(
            cutlass_dir, "tools", "util", "include"
        )
        cutlass_extensions_include_dir = os.path.join(cwd, extensions_cuda_dir)
    if use_cutlass:
        extra_compile_args["nvcc"].extend(
            [
                "-DTORCHAO_USE_CUTLASS",
                "-I" + cutlass_include_dir,
                "-I" + cutlass_tools_include_dir,
                "-I" + cutlass_extensions_include_dir,
            ]
        )

    # Get base directory and source paths
    curdir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(curdir, "torchao", "csrc")

    # Collect C++ source files
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(
        glob.glob(os.path.join(extensions_cuda_dir, "**/*.cu"), recursive=True)
    )

    extensions_hip_dir = os.path.join(
        extensions_dir, "cuda", "tensor_core_tiled_layout"
    )
    hip_sources = list(
        glob.glob(os.path.join(extensions_hip_dir, "*.cu"), recursive=True)
    )
    extensions_hip_dir = os.path.join(extensions_dir, "cuda", "sparse_marlin")
    hip_sources += list(
        glob.glob(os.path.join(extensions_hip_dir, "*.cu"), recursive=True)
    )

    # Collect CUDA source files if needed
    if not IS_ROCM and use_cuda:
        sources += cuda_sources
    else:
        # Remove CUTLASS-based kernels from the cuda_sources list.  An
        # assumption is that these files will have "cutlass" in its
        # name.
        cutlass_sources = list(
            glob.glob(
                os.path.join(extensions_cuda_dir, "**/*cutlass*.cu"), recursive=True
            )
        )
        sources = [s for s in sources if s not in cutlass_sources]

    # TOOD: Remove this and use what CUDA has once we fix all the builds.
    if IS_ROCM and use_cuda:
        # Add ROCm GPU architecture check
        gpu_arch = torch.cuda.get_device_properties(0).name
        if gpu_arch != "gfx942":
            print(f"Warning: Unsupported ROCm GPU architecture: {gpu_arch}")
            print(
                "Currently only gfx942 is supported. Skipping compilation of ROCm extensions"
            )
            return None
        sources += hip_sources

    if len(sources) == 0:
        return None

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
        build_options = BuildOptions()

        def bool_to_on_off(value):
            return "ON" if value else "OFF"

        from distutils.sysconfig import get_python_lib

        torch_dir = get_python_lib() + "/torch/share/cmake/Torch"

        ext_modules.append(
            CMakeExtension(
                "torchao.experimental",
                cmake_lists_dir="torchao/experimental",
                cmake_args=(
                    [
                        f"-DCMAKE_BUILD_TYPE={'Debug' if use_debug_mode() else 'Release'}",
                        f"-DTORCHAO_BUILD_CPU_AARCH64={bool_to_on_off(build_options.build_cpu_aarch64)}",
                        f"-DTORCHAO_BUILD_KLEIDIAI={bool_to_on_off(build_options.build_kleidi_ai)}",
                        f"-DTORCHAO_BUILD_MPS_OPS={bool_to_on_off(build_options.build_experimental_mps)}",
                        "-DTorch_DIR=" + torch_dir,
                    ]
                    + (
                        ["-DCMAKE_INSTALL_PREFIX=cmake-out"]
                        if build_options.build_experimental_mps
                        else []
                    )
                ),
            )
        )

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
    ext_modules=get_extensions(),
    extras_require={"dev": read_requirements("dev-requirements.txt")},
    description="Package for applying ao techniques to GPU models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/ao",
    cmdclass={"build_ext": TorchAOBuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
