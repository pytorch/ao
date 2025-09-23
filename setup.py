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
from setuptools.command.build_ext import build_ext as _setuptools_build_ext

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

import platform

################################################################################
# Build Configuration - Environment Variables and Build Options
################################################################################

# Core build toggles
use_cpp = os.getenv("USE_CPP", "1")
use_cpu_kernels = os.getenv("USE_CPU_KERNELS", "0") == "1"

# Platform detection
is_arm64 = platform.machine().startswith("arm64") or platform.machine() == "aarch64"
is_macos = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"

# Auto-enable experimental builds on ARM64 macOS when USE_CPP=1
build_macos_arm_auto = use_cpp == "1" and is_arm64 and is_macos

# Build configuration hierarchy and relationships:
#
# Level 1: USE_CPP (Primary gate)
#   ├── "0" → Skip all C++ extensions (Python-only mode)
#   └── "1"/None → Build C++ extensions
#
# Level 2: Platform-specific optimizations
#   ├── USE_CPU_KERNELS="1" + Linux → Include optimized CPU kernels (AVX512, etc.)
#   └── ARM64 + macOS → Auto-enable experimental builds (build_macos_arm_auto)
#
# Level 3: Shared CPU kernel builds (cmake-based)
#   ├── BUILD_TORCHAO_EXPERIMENTAL="1" → Force experimental builds
#   ├── build_macos_arm_auto → Auto-enable on ARM64 macOS
#   └── When enabled, provides access to:
#       ├── TORCHAO_BUILD_CPU_AARCH64 → ARM64 CPU kernels
#       ├── TORCHAO_BUILD_KLEIDIAI → Kleidi AI library integration
#       ├── TORCHAO_BUILD_EXPERIMENTAL_MPS → MPS acceleration (macOS only)
#       ├── TORCHAO_ENABLE_ARM_NEON_DOT → ARM NEON dot product instructions
#       ├── TORCHAO_ENABLE_ARM_I8MM → ARM 8-bit integer matrix multiply
#       └── TORCHAO_PARALLEL_BACKEND → Backend selection (aten_openmp, executorch, etc.)


version_prefix = read_version()
# Version is version.dev year month date if using nightlies and version if not
version = (
    f"{version_prefix}.dev{current_date}"
    if os.environ.get("TORCHAO_NIGHTLY")
    else version_prefix
)


def use_debug_mode():
    return os.getenv("DEBUG", "0") == "1"


# Heavy imports (torch, torch.utils.cpp_extension) are deferred to build time


class BuildOptions:
    def __init__(self):
        # TORCHAO_BUILD_CPU_AARCH64 is enabled by default on Arm-based Apple machines
        # The kernels require sdot/udot, which are not required on Arm until Armv8.4 or later,
        # but are available on Arm-based Apple machines.  On non-Apple machines, the kernels
        # can be built by explicitly setting TORCHAO_BUILD_CPU_AARCH64=1
        self.build_cpu_aarch64 = self._os_bool_var(
            "TORCHAO_BUILD_CPU_AARCH64",
            default=(is_arm64 and is_macos),
        )
        if self.build_cpu_aarch64:
            assert is_arm64, "TORCHAO_BUILD_CPU_AARCH64 requires an arm64 machine"

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
            import torch  # Lazy import

            assert is_macos, "TORCHAO_BUILD_EXPERIMENTAL_MPS requires macOS"
            assert is_arm64, "TORCHAO_BUILD_EXPERIMENTAL_MPS requires arm64"
            assert torch.mps.is_available(), (
                "TORCHAO_BUILD_EXPERIMENTAL_MPS requires MPS be available"
            )

        # TORCHAO_PARALLEL_BACKEND specifies which parallel backend to use
        # Possible values: aten_openmp, executorch, openmp, pthreadpool, single_threaded
        self.parallel_backend = os.getenv("TORCHAO_PARALLEL_BACKEND", "aten_openmp")

        # TORCHAO_ENABLE_ARM_NEON_DOT enable ARM NEON Dot Product extension
        # Enabled by default on macOS silicon
        self.enable_arm_neon_dot = self._os_bool_var(
            "TORCHAO_ENABLE_ARM_NEON_DOT",
            default=(is_arm64 and is_macos),
        )
        if self.enable_arm_neon_dot:
            assert self.build_cpu_aarch64, (
                "TORCHAO_ENABLE_ARM_NEON_DOT requires TORCHAO_BUILD_CPU_AARCH64 be set"
            )

        # TORCHAO_ENABLE_ARM_I8MM enable ARM 8-bit Integer Matrix Multiply instructions
        # Not enabled by default on macOS as not all silicon mac supports it
        self.enable_arm_i8mm = self._os_bool_var(
            "TORCHAO_ENABLE_ARM_I8MM", default=False
        )
        if self.enable_arm_i8mm:
            assert self.build_cpu_aarch64, (
                "TORCHAO_ENABLE_ARM_I8MM requires TORCHAO_BUILD_CPU_AARCH64 be set"
            )

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


def get_cuda_version_from_nvcc():
    """Get CUDA version from nvcc if available."""
    try:
        result = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.STDOUT
        )
        output = result.decode("utf-8")
        # Look for version line like "release 12.6"
        for line in output.split("\n"):
            if "release" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() == "release" and i + 1 < len(parts):
                        return parts[i + 1].rstrip(",")

    except:
        return None


def get_cutlass_build_flags():
    """Determine which CUTLASS kernels to build based on CUDA version.
    SM90a: CUDA 12.6+, SM100a: CUDA 12.8+
    """
    # Lazy import torch and helper; only needed when building CUDA extensions
    import torch
    from torch.utils.cpp_extension import _get_cuda_arch_flags

    # Try nvcc then torch version
    cuda_version = get_cuda_version_from_nvcc() or torch.version.cuda

    try:
        if not cuda_version:
            raise ValueError("No CUDA version found")

        major, minor = map(int, cuda_version.split(".")[:2])
        build_sm90a = major > 12 or (major == 12 and minor >= 6)
        build_sm100a = major > 12 or (major == 12 and minor >= 8)

        if build_sm90a:
            print(f"CUDA {cuda_version}: Enabling SM90a CUTLASS kernels")
        if build_sm100a:
            print(f"CUDA {cuda_version}: Enabling SM100a CUTLASS kernels")

        return build_sm90a, build_sm100a
    except:
        # Fallback to architecture flags
        cuda_arch_flags = _get_cuda_arch_flags()
        return (
            "-gencode=arch=compute_90a,code=sm_90a" in cuda_arch_flags,
            "-gencode=arch=compute_100a,code=sm_100a" in cuda_arch_flags,
        )


class LazyTorchAOBuildExt(_setuptools_build_ext):
    def run(self):
        # Import heavy torch build only when actually running build_ext
        from torch.utils.cpp_extension import BuildExtension as _BuildExtension

        class _TorchAOBuildExt(_BuildExtension):
            def run(self_inner):
                if os.getenv("USE_CPP", "1") != "0":
                    check_submodules()
                if not self_inner.distribution.ext_modules:
                    self_inner.distribution.ext_modules = get_extensions()
                super(_TorchAOBuildExt, self_inner).run()

            def build_extensions(self_inner):
                cmake_extensions = [
                    ext
                    for ext in self_inner.extensions
                    if isinstance(ext, CMakeExtension)
                ]
                other_extensions = [
                    ext
                    for ext in self_inner.extensions
                    if not isinstance(ext, CMakeExtension)
                ]
                for ext in cmake_extensions:
                    self_inner.build_cmake(ext)

                self_inner.extensions = other_extensions
                super(_TorchAOBuildExt, self_inner).build_extensions()
                self_inner.extensions = other_extensions + cmake_extensions

            def build_cmake(self_inner, ext):
                extdir = os.path.abspath(
                    os.path.dirname(self_inner.get_ext_fullpath(ext.name))
                )
                if not os.path.exists(self_inner.build_temp):
                    os.makedirs(self_inner.build_temp)
                ext_filename = os.path.basename(self_inner.get_ext_filename(ext.name))
                ext_basename = os.path.splitext(ext_filename)[0]
                if os.getenv("VERBOSE_BUILD", "0") == "1" or use_debug_mode():
                    print(
                        "CMAKE COMMAND",
                        [
                            "cmake",
                            ext.cmake_lists_dir,
                        ]
                        + ext.cmake_args
                        + [
                            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                            "-DTORCHAO_CMAKE_EXT_SO_NAME=" + ext_basename,
                        ],
                    )
                subprocess.check_call(
                    [
                        "cmake",
                        ext.cmake_lists_dir,
                    ]
                    + ext.cmake_args
                    + [
                        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                        "-DTORCHAO_CMAKE_EXT_SO_NAME=" + ext_basename,
                    ],
                    cwd=self_inner.build_temp,
                )
                subprocess.check_call(
                    ["cmake", "--build", "."], cwd=self_inner.build_temp
                )

        # Morph this instance into the real BuildExtension subclass and run
        self.__class__ = _TorchAOBuildExt
        return _TorchAOBuildExt.run(self)


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
    # Skip building C++ extensions if USE_CPP is set to "0"
    if use_cpp == "0":
        print("USE_CPP=0: Skipping compilation of C++ extensions")
        return []

    debug_mode = use_debug_mode()
    if debug_mode:
        print("Compiling in debug mode")

    # Heavy imports moved here to minimize setup.py import overhead
    import torch
    from torch.utils.cpp_extension import (
        CUDA_HOME,
        IS_WINDOWS,
        ROCM_HOME,
        CppExtension,
        CUDAExtension,
    )

    # Only skip CUDA extensions if neither CUDA_HOME nor nvcc is available.
    # In many CI environments CUDA_HOME may be unset even though nvcc is on PATH.
    if torch.version.cuda and CUDA_HOME is None and get_cuda_version_from_nvcc() is None:
        print("CUDA toolkit is not available (CUDA_HOME unset and nvcc not found). Skipping compilation of CUDA extensions")
        print(
            "If you'd like to compile CUDA extensions locally please install the cudatoolkit from https://anaconda.org/nvidia/cuda-toolkit"
        )
    if ROCM_HOME is None and torch.version.hip:
        print("ROCm is not available. Skipping compilation of ROCm extensions")
        print("If you'd like to compile ROCm extensions locally please install ROCm")

    # Build CUDA extensions if CUDA is available and either CUDA_HOME is set or nvcc is present
    use_cuda = bool(torch.version.cuda) and (CUDA_HOME is not None or get_cuda_version_from_nvcc() is not None)
    use_rocm = torch.version.hip and ROCM_HOME is not None
    extension = CUDAExtension if (use_cuda or use_rocm) else CppExtension

    nvcc_args = [
        "-DNDEBUG" if not debug_mode else "-DDEBUG",
        "-O3" if not debug_mode else "-O0",
        "-t=0",
        "-std=c++17",
    ]
    rocm_args = [
        "-DNDEBUG" if not debug_mode else "-DDEBUG",
        "-O3" if not debug_mode else "-O0",
        "-std=c++17",
    ]

    extra_link_args = []
    extra_compile_args = {
        "cxx": [f"-DPy_LIMITED_API={PY3_9_HEXCODE}"],
        "nvcc": nvcc_args if use_cuda else rocm_args,
    }

    if not IS_WINDOWS:
        extra_compile_args["cxx"].extend(
            ["-O3" if not debug_mode else "-O0", "-fdiagnostics-color=always"]
        )

        if use_cpu_kernels and is_linux:
            if (
                hasattr(torch._C._cpu, "_is_avx512_supported")
                and torch._C._cpu._is_avx512_supported()
            ):
                extra_compile_args["cxx"].extend(
                    [
                        "-DCPU_CAPABILITY_AVX512",
                        "-march=native",
                        "-mfma",
                        "-fopenmp",
                    ]
                )
            if (
                hasattr(torch._C._cpu, "_is_avx512_vnni_supported")
                and torch._C._cpu._is_avx512_vnni_supported()
            ):
                extra_compile_args["cxx"].extend(
                    [
                        "-DCPU_CAPABILITY_AVX512_VNNI",
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

    rocm_sparse_marlin_supported = False
    rocm_tiled_layout_supported = False
    if use_rocm:
        # naive search for hipblalst.h, if any found contain HIPBLASLT_ORDER_COL16 and VEC_EXT
        found_col16 = False
        found_vec_ext = False
        found_outer_vec = False
        if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
            print("ROCM_HOME", ROCM_HOME)
        hipblaslt_headers = list(
            glob.glob(os.path.join(ROCM_HOME, "include", "hipblaslt", "hipblaslt.h"))
        )
        if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
            print("hipblaslt_headers", hipblaslt_headers)
        for header in hipblaslt_headers:
            with open(header) as f:
                text = f.read()
                if "HIPBLASLT_ORDER_COL16" in text:
                    found_col16 = True
                if "HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT" in text:
                    found_vec_ext = True
                if "HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F" in text:
                    found_outer_vec = True
        if found_col16:
            extra_compile_args["cxx"].append("-DHIPBLASLT_HAS_ORDER_COL16")
            if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
                print("hipblaslt found extended col order enums")
        else:
            if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
                print("hipblaslt does not have extended col order enums")
        if found_outer_vec:
            extra_compile_args["cxx"].append("-DHIPBLASLT_OUTER_VEC")
            if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
                print("hipblaslt found outer vec")
        elif found_vec_ext:
            extra_compile_args["cxx"].append("-DHIPBLASLT_VEC_EXT")
            if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
                print("hipblaslt found vec ext")
        else:
            if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
                print("hipblaslt does not have vec ext")

    # Get base directory and source paths
    curdir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(curdir, "torchao", "csrc")

    # Collect C++ source files
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))

    # Exclude C++ CPU sources that are built by CMake
    cpu_cmake_sources = glob.glob(
        os.path.join(extensions_dir, "cpu", "torch_free_kernels", "**", "*.cpp"),
        recursive=True,
    )
    cpu_cmake_sources += glob.glob(
        os.path.join(extensions_dir, "cpu", "shared_kernels", "**", "*.cpp"),
        recursive=True,
    )
    sources = [s for s in sources if s not in cpu_cmake_sources]

    if not use_cpu_kernels or not is_linux:
        # Remove csrc/cpu/*.cpp
        excluded_sources = list(
            glob.glob(
                os.path.join(extensions_dir, "cpu/aten_kernels/*.cpp"), recursive=False
            )
        )
        sources = [s for s in sources if s not in excluded_sources]

    # Collect CUDA source files
    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(
        glob.glob(os.path.join(extensions_cuda_dir, "**/*.cu"), recursive=True)
    )

    # Define ROCm source directories
    rocm_source_dirs = [
        os.path.join(extensions_dir, "rocm", "swizzle"),
    ]
    if rocm_tiled_layout_supported:
        rocm_source_dirs.append(
            os.path.join(extensions_dir, "cuda", "tensor_core_tiled_layout")
        )
    if rocm_sparse_marlin_supported:
        rocm_source_dirs.extend([os.path.join(extensions_dir, "cuda", "sparse_marlin")])

    # Collect all ROCm sources from the defined directories
    rocm_sources = []
    for rocm_dir in rocm_source_dirs:
        rocm_sources.extend(glob.glob(os.path.join(rocm_dir, "*.cu"), recursive=True))
        rocm_sources.extend(glob.glob(os.path.join(rocm_dir, "*.hip"), recursive=True))
        rocm_sources.extend(glob.glob(os.path.join(rocm_dir, "*.cpp"), recursive=True))

    # Add CUDA source files if needed
    if use_cuda:
        sources += cuda_sources

    # Add MXFP8 cuda extension dir
    mxfp8_extension_dir = os.path.join(extensions_dir, "cuda", "mx_kernels")
    mxfp8_sources_to_exclude = list(
        glob.glob(os.path.join(mxfp8_extension_dir, "**/*"), recursive=True)
    )
    sources = [s for s in sources if s not in mxfp8_sources_to_exclude]

    # TOOD: Remove this and use what CUDA has once we fix all the builds.
    # TODO: Add support for other ROCm GPUs
    if use_rocm:
        extra_compile_args["nvcc"].append("--offload-arch=gfx942")
        sources += rocm_sources
    else:
        # Remove ROCm-based sources from the sources list.
        extensions_rocm_dir = os.path.join(extensions_dir, "rocm")
        rocm_sources = list(
            glob.glob(os.path.join(extensions_rocm_dir, "**/*.cpp"), recursive=True)
        )
        sources = [s for s in sources if s not in rocm_sources]

    use_cutlass = False
    cutlass_90a_sources = None
    cutlass_100a_sources = None
    build_for_sm90a = False
    build_for_sm100a = False
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

        build_for_sm90a, build_for_sm100a = get_cutlass_build_flags()
        # Define sm90a sources
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
            os.path.join(extensions_cuda_dir, "activation24", "sparsify24.cu"),
            os.path.join(extensions_cuda_dir, "activation24", "sparse_gemm.cu"),
        ]
        for dtypes in ["e4m3e4m3", "e4m3e5m2", "e5m2e4m3", "e5m2e5m2"]:
            cutlass_90a_sources.append(
                os.path.join(
                    extensions_cuda_dir,
                    "rowwise_scaled_linear_sparse_cutlass",
                    "rowwise_scaled_linear_sparse_cutlass_" + dtypes + ".cu",
                )
            )
        # Always remove sm90a sources from main sources
        sources = [s for s in sources if s not in cutlass_90a_sources]

        # Always compile mx_fp_cutlass_kernels.cu ONLY with sm100a architecture
        cutlass_100a_sources = [
            os.path.join(
                extensions_cuda_dir,
                "mx_kernels",
                "mx_fp_cutlass_kernels.cu",
            ),
        ]
        # Remove from main sources to prevent compilation with other architectures
        sources = [
            s for s in sources if os.path.basename(s) != "mx_fp_cutlass_kernels.cu"
        ]

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
        if os.getenv("VERBOSE_BUILD", "0") == "1" or debug_mode:
            print("SOURCES", sources)
        # Double-check to ensure mx_fp_cutlass_kernels.cu is not in sources
        sources = [
            s for s in sources if os.path.basename(s) != "mx_fp_cutlass_kernels.cu"
        ]

        ext_modules.append(
            extension(
                "torchao._C",
                sources,
                py_limited_api=True,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        )

    # Add the mxfp8 casting CUDA extension
    if use_cuda:
        mxfp8_sources = [
            os.path.join(mxfp8_extension_dir, "mxfp8_extension.cpp"),
            os.path.join(mxfp8_extension_dir, "mxfp8_cuda.cu"),
        ]

        # Only add the extension if the source files exist AND we are building for sm100
        mxfp8_src_files_exist = all(os.path.exists(f) for f in mxfp8_sources)
        if mxfp8_src_files_exist and build_for_sm100a:
            print("Building mxfp8_cuda extension")
            ext_modules.append(
                CUDAExtension(
                    name="torchao.prototype.mxfp8_cuda",
                    sources=mxfp8_sources,
                    include_dirs=[
                        mxfp8_extension_dir,  # For mxfp8_quantize.cuh, mxfp8_extension.cpp, and mxfp8_cuda.cu
                    ],
                    extra_compile_args={
                        "cxx": ["-std=c++17", "-O3"],
                        "nvcc": nvcc_args
                        + [
                            "-gencode=arch=compute_100,code=sm_100",
                            "-gencode=arch=compute_120,code=compute_120",
                        ],
                    },
                ),
            )

    # Only build the cutlass_90a extension if sm90a is in the architecture flags
    if (
        cutlass_90a_sources is not None
        and len(cutlass_90a_sources) > 0
        and build_for_sm90a
    ):
        cutlass_90a_extra_compile_args = copy.deepcopy(extra_compile_args)
        # Only use sm90a architecture for these sources, ignoring other flags
        cutlass_90a_extra_compile_args["nvcc"].append(
            "-gencode=arch=compute_90a,code=sm_90a"
        )
        ext_modules.append(
            extension(
                "torchao._C_cutlass_90a",
                cutlass_90a_sources,
                py_limited_api=True,
                extra_compile_args=cutlass_90a_extra_compile_args,
                extra_link_args=extra_link_args,
            )
        )

    # Only build the cutlass_100a extension if sm100a is in the architecture flags
    if (
        cutlass_100a_sources is not None
        and len(cutlass_100a_sources) > 0
        and build_for_sm100a
    ):
        cutlass_100a_extra_compile_args = copy.deepcopy(extra_compile_args)
        # Only use sm100a architecture for these sources, ignoring cuda_arch_flags
        cutlass_100a_extra_compile_args["nvcc"].append(
            "-gencode=arch=compute_100a,code=sm_100a"
        )
        ext_modules.append(
            extension(
                "torchao._C_cutlass_100a",
                cutlass_100a_sources,
                py_limited_api=True,
                extra_compile_args=cutlass_100a_extra_compile_args,
                extra_link_args=extra_link_args,
            )
        )

    # Build CMakeLists from /torchao/csrc/cpu - additional options become available : TORCHAO_BUILD_CPU_AARCH64, TORCHAO_BUILD_KLEIDIAI, TORCHAO_BUILD_MPS_OPS, TORCHAO_PARALLEL_BACKEND
    if build_macos_arm_auto or os.getenv("BUILD_TORCHAO_EXPERIMENTAL") == "1":
        build_options = BuildOptions()

        def bool_to_on_off(value):
            return "ON" if value else "OFF"

        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            raise RuntimeError("Unable to locate 'torch' package for CMake config")
        torch_pkg_dir = os.path.dirname(spec.origin)
        torch_dir = os.path.join(torch_pkg_dir, "share", "cmake", "Torch")

        ext_modules.append(
            CMakeExtension(
                "torchao._C_cpu_shared_kernels_aten",
                cmake_lists_dir="torchao/csrc/cpu",
                cmake_args=(
                    [
                        f"-DCMAKE_BUILD_TYPE={'Debug' if use_debug_mode() else 'Release'}",
                        f"-DTORCHAO_BUILD_CPU_AARCH64={bool_to_on_off(build_options.build_cpu_aarch64)}",
                        f"-DTORCHAO_BUILD_KLEIDIAI={bool_to_on_off(build_options.build_kleidi_ai)}",
                        f"-DTORCHAO_ENABLE_ARM_NEON_DOT={bool_to_on_off(build_options.enable_arm_neon_dot)}",
                        f"-DTORCHAO_ENABLE_ARM_I8MM={bool_to_on_off(build_options.enable_arm_i8mm)}",
                        f"-DTORCHAO_PARALLEL_BACKEND={build_options.parallel_backend}",
                        "-DTORCHAO_BUILD_TESTS=OFF",
                        "-DTORCHAO_BUILD_BENCHMARKS=OFF",
                        "-DTorch_DIR=" + torch_dir,
                    ]
                ),
            )
        )

    return ext_modules


# Defer submodule checks to build time via build_ext

setup(
    name="torchao",
    version=version + version_suffix,
    packages=find_packages(include=["torchao*"]),
    include_package_data=True,
    package_data={
        "torchao.kernel.configs": ["*.pkl"],
    },
    # Defer extension discovery to build time for performance
    ext_modules=[],
    extras_require={"dev": read_requirements("dev-requirements.txt")},
    description="Package for applying ao techniques to GPU models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/ao",
    cmdclass={"build_ext": LazyTorchAOBuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
