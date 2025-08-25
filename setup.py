# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
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

use_cpp_avx512 = os.getenv("USE_AVX512", "1") == "1" and platform.system() == "Linux"

from torchao.utils import TORCH_VERSION_AT_LEAST_2_7

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
    _get_cuda_arch_flags,
)


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
            assert self._is_arm64(), (
                "TORCHAO_BUILD_CPU_AARCH64 requires an arm64 machine"
            )

        # TORCHAO_BUILD_KLEIDIAI is disabled by default for now because
        # 1) It increases the build time
        # 2) It has some accuracy issues in CI tests due to BF16
        self.build_kleidi_ai = self._os_bool_var(
            "TORCHAO_BUILD_KLEIDIAI", default=False
        )
        if self.build_kleidi_ai:
            assert self.build_cpu_aarch64, (
                "TORCHAO_BUILD_KLEIDIAI requires TORCHAO_BUILD_CPU_AARCH64 be set"
            )

        # TORCHAO_BUILD_EXPERIMENTAL_MPS is disabled by default.
        self.build_experimental_mps = self._os_bool_var(
            "TORCHAO_BUILD_EXPERIMENTAL_MPS", default=False
        )
        if self.build_experimental_mps:
            assert self._is_macos(), "TORCHAO_BUILD_EXPERIMENTAL_MPS requires MacOS"
            assert self._is_arm64(), "TORCHAO_BUILD_EXPERIMENTAL_MPS requires arm64"
            assert torch.mps.is_available(), (
                "TORCHAO_BUILD_EXPERIMENTAL_MPS requires MPS be available"
            )

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
    if CUDA_HOME is None and torch.cuda.is_available() and torch.version.cuda:
        print("CUDA toolkit is not available. Skipping compilation of CUDA extensions")
        print(
            "If you'd like to compile CUDA extensions locally please install the cudatoolkit from https://anaconda.org/nvidia/cuda-toolkit"
        )
    if ROCM_HOME is None and torch.cuda.is_available() and torch.version.hip:
        print("ROCm is not available. Skipping compilation of ROCm extensions")
        print("If you'd like to compile ROCm extensions locally please install ROCm")

    use_cuda = (
        torch.cuda.is_available() and torch.version.cuda and CUDA_HOME is not None
    )
    use_hip = torch.cuda.is_available() and torch.version.hip and ROCM_HOME is not None
    extension = CUDAExtension if (use_cuda or use_hip) else CppExtension

    nvcc_args = [
        "-DNDEBUG" if not debug_mode else "-DDEBUG",
        "-O3" if not debug_mode else "-O0",
        "-t=0",
        "-std=c++17",
    ]
    hip_args = [
        "-DNDEBUG" if not debug_mode else "-DDEBUG",
        "-O3" if not debug_mode else "-O0",
        "-std=c++17",
    ]

    extra_link_args = []
    extra_compile_args = {
        "cxx": [f"-DPy_LIMITED_API={PY3_9_HEXCODE}"],
        "nvcc": nvcc_args if use_cuda else hip_args,
    }

    if not IS_WINDOWS:
        extra_compile_args["cxx"].extend(
            ["-O3" if not debug_mode else "-O0", "-fdiagnostics-color=always"]
        )

        if use_cpp_avx512 and TORCH_VERSION_AT_LEAST_2_7:
            if torch._C._cpu._is_avx512_supported():
                extra_compile_args["cxx"].extend(
                    [
                        "-DCPU_CAPABILITY_AVX512",
                        "-march=native",
                        "-mfma",
                        "-fopenmp",
                    ]
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

    hip_sparse_marlin_supported = True
    if use_hip:
        # naive search for hipblalst.h, if any found contain HIPBLASLT_ORDER_COL16 and VEC_EXT
        found_col16 = False
        found_vec_ext = False
        print("ROCM_HOME", ROCM_HOME)
        hipblaslt_headers = list(
            glob.glob(os.path.join(ROCM_HOME, "include", "hipblaslt", "hipblaslt.h"))
        )
        print("hipblaslt_headers", hipblaslt_headers)
        for header in hipblaslt_headers:
            with open(header) as f:
                text = f.read()
                if "HIPBLASLT_ORDER_COL16" in text:
                    found_col16 = True
                if "HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT" in text:
                    found_vec_ext = True
        if found_col16:
            extra_compile_args["cxx"].append("-DHIPBLASLT_HAS_ORDER_COL16")
            print("hipblaslt found extended col order enums")
        else:
            print("hipblaslt does not have extended col order enums")
        if found_vec_ext:
            extra_compile_args["cxx"].append("-DHIPBLASLT_VEC_EXT")
            print("hipblaslt found vec ext")
        else:
            print("hipblaslt does not have vec ext")

        # sparse_marlin depends on features in ROCm 6.4, __builtin_amdgcn_global_load_lds
        ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split(".")[:2])
        hip_sparse_marlin_supported = ROCM_VERSION >= (6, 4)

    # Get base directory and source paths
    curdir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(curdir, "torchao", "csrc")

    # Collect C++ source files
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))
    if IS_WINDOWS:
        # Remove csrc/cpu/*.cpp on Windows due to the link issue: unresolved external symbol PyInit__C
        excluded_sources = list(
            glob.glob(os.path.join(extensions_dir, "cpu/*.cpp"), recursive=True)
        )
        sources = [s for s in sources if s not in excluded_sources]

    # Collect CUDA source files
    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(
        glob.glob(os.path.join(extensions_cuda_dir, "**/*.cu"), recursive=True)
    )

    # Collect HIP source files
    extensions_hip_dir = os.path.join(
        extensions_dir, "cuda", "tensor_core_tiled_layout"
    )
    hip_sources = list(
        glob.glob(os.path.join(extensions_hip_dir, "*.cu"), recursive=True)
    )
    if hip_sparse_marlin_supported:
        extensions_hip_dir = os.path.join(extensions_dir, "cuda", "sparse_marlin")
        hip_sources += list(
            glob.glob(os.path.join(extensions_hip_dir, "*.cu"), recursive=True)
        )
    extensions_hip_dir = os.path.join(extensions_dir, "rocm")
    hip_sources += list(
        glob.glob(os.path.join(extensions_hip_dir, "**/*.hip"), recursive=True)
    )
    hip_sources += list(
        glob.glob(os.path.join(extensions_hip_dir, "**/*.cpp"), recursive=True)
    )

    # Add CUDA source files if needed
    if use_cuda:
        sources += cuda_sources

    # TODO: Remove this and use what CUDA has once we fix all the builds.
    # Add HIP source files if needed
    if use_hip:
        # Add ROCm GPU architecture check
        gpu_arch = torch.cuda.get_device_properties(0).name
        if gpu_arch != "gfx942":
            print(f"Warning: Unsupported ROCm GPU architecture: {gpu_arch}")
            print("Currently only gfx942 is supported. Compiling only for gfx942.")
        extra_compile_args["nvcc"].append("--offload-arch=gfx942")
        sources += hip_sources

    use_cutlass = False
    cutlass_90a_sources = None
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
                "-DCUTE_USE_PACKED_TUPLE=1",
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                "--ftemplate-backtrace-limit=0",
                # "--keep",
                # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",
                # "--resource-usage",
                # "-lineinfo",
                # "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # https://github.com/NVIDIA/cutlass/blob/main/media/docs/dependent_kernel_launch.md
            ]
        )

        cuda_arch_flags = _get_cuda_arch_flags()
        build_for_sm90 = "-gencode=arch=compute_90,code=sm_90" in cuda_arch_flags
        build_for_sm90a = "-gencode=arch=compute_90a,code=sm_90a" in cuda_arch_flags
        if build_for_sm90 and not build_for_sm90a:
            cutlass_90a_sources = [
                os.path.join(
                    extensions_cuda_dir,
                    "rowwise_scaled_linear_sparse_cutlass",
                    "rowwise_scaled_linear_sparse_cutlass_f8f8.cu",
                ),
                os.path.join(
                    extensions_cuda_dir,
                    "to_sparse_semi_structured_cutlass_sm9x",
                    "to_sparse_semi_structured_cutlass_sm9x_f8.cu",
                ),
            ]
            for dtypes in ["e4m3e4m3", "e4m3e5m2", "e5m2e4m3", "e5m2e5m2"]:
                cutlass_90a_sources.append(
                    os.path.join(
                        extensions_cuda_dir,
                        "rowwise_scaled_linear_sparse_cutlass",
                        "rowwise_scaled_linear_sparse_cutlass_" + dtypes + ".cu",
                    )
                )
            sources = [s for s in sources if s not in cutlass_90a_sources]
    else:
        # Remove CUTLASS-based kernels from the sources list.  An
        # assumption is that these files will have "cutlass" in its
        # name.
        cutlass_sources = list(
            glob.glob(
                os.path.join(extensions_cuda_dir, "**/*cutlass*.cu"), recursive=True
            )
        )
        sources = [s for s in sources if s not in cutlass_sources]

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

    if cutlass_90a_sources is not None and len(cutlass_90a_sources) > 0:
        cutlass_90a_extra_compile_args = copy.deepcopy(extra_compile_args)
        cutlass_90a_extra_compile_args["nvcc"].extend(
            cuda_arch_flags + ["-gencode=arch=compute_90a,code=sm_90a"]
        )
        ext_modules.append(
            extension(
                "torchao._C",
                cutlass_90a_sources,
                py_limited_api=True,
                extra_compile_args=cutlass_90a_extra_compile_args,
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
    packages=find_packages(exclude=["benchmarks", "benchmarks.*"]),
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
