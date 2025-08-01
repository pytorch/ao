"""
Modern build backend for torchao using setuptools hooks.
Handles complex C++ extension building cleanly.
"""
import os
import sys
import glob
import copy
import subprocess
import platform
from typing import List, Optional, Dict
from pathlib import Path


def get_version():
    """Get version from version.txt file."""
    version_file = Path(__file__).parent / "version.txt"
    with open(version_file, "r") as f:
        version = f.readline().strip()
    
    # Add dev suffix for nightlies
    if os.environ.get("TORCHAO_NIGHTLY"):
        from datetime import datetime
        current_date = datetime.now().strftime("%Y%m%d")
        version = f"{version}.dev{current_date}"
    
    # Add version suffix
    version_suffix = os.getenv("VERSION_SUFFIX")
    if version_suffix:
        version += version_suffix
    elif version_suffix is None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL
            ).decode("ascii").strip()
            version += f"+git{git_commit}"
        except:
            pass
    
    return version


def check_submodules():
    """Initialize git submodules if needed."""
    if os.getenv("USE_SYSTEM_LIBS"):
        return
    
    cwd = Path(__file__).parent
    cutlass_path = cwd / "third_party" / "cutlass"
    
    if not cutlass_path.exists() or not any(cutlass_path.iterdir()):
        print("Initializing git submodules...")
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=cwd
        )


class CMakeExtension:
    """CMake-based extension for experimental builds."""
    def __init__(self, name: str, cmake_lists_dir: str, cmake_args: Optional[List[str]] = None):
        self.name = name
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        self.cmake_args = cmake_args or []


def get_extensions():
    """Build all extensions for torchao."""
    if os.getenv("USE_CPP", "1") == "0":
        print("USE_CPP=0: Skipping C++ extensions")
        return []
    
    try:
        import torch
        from torch.utils.cpp_extension import (
            BuildExtension, CppExtension, CUDAExtension,
            CUDA_HOME, ROCM_HOME, IS_WINDOWS, _get_cuda_arch_flags
        )
    except ImportError:
        print("PyTorch not available: Skipping C++ extensions")
        return []
    
    # Platform detection
    is_arm64 = platform.machine() in ["arm64", "aarch64"]
    is_macos = platform.system() == "Darwin"
    is_linux = platform.system() == "Linux"
    
    # Configuration
    debug_mode = os.getenv("DEBUG", "0") == "1"
    # Enable CPU kernels by default on macOS, Linux requires explicit opt-in
    use_cpu_kernels = os.getenv("USE_CPU_KERNELS", "1" if is_macos else "0") == "1"
    # Skip GPU kernels on M1 Mac since no discrete GPU
    use_cuda = False if is_macos else (torch.version.cuda and CUDA_HOME is not None)
    use_rocm = False if is_macos else (torch.version.hip and ROCM_HOME is not None)
    
    # Paths
    cwd = Path(__file__).parent
    extensions_dir = cwd / "torchao" / "csrc"
    
    # Compile arguments
    nvcc_args = [
        "-DNDEBUG" if not debug_mode else "-DDEBUG",
        "-O3" if not debug_mode else "-O0",
        "-t=0", "-std=c++17",
    ]
    
    cxx_args = ["-DPy_LIMITED_API=0x03090000"]  # Python 3.9+
    link_args = []
    
    if not IS_WINDOWS:
        cxx_args.extend([
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ])
        
        # AVX512 support on Linux
        if use_cpu_kernels and is_linux:
            if hasattr(torch._C._cpu, "_is_avx512_supported") and torch._C._cpu._is_avx512_supported():
                cxx_args.extend(["-DCPU_CAPABILITY_AVX512", "-march=native", "-mfma", "-fopenmp"])
            if hasattr(torch._C._cpu, "_is_avx512_vnni_supported") and torch._C._cpu._is_avx512_vnni_supported():
                cxx_args.append("-DCPU_CAPABILITY_AVX512_VNNI")
        
        if debug_mode:
            cxx_args.append("-g")
            nvcc_args.append("-g")
            link_args.extend(["-O0", "-g"])
    else:
        cxx_args.extend(["/O2" if not debug_mode else "/Od", "/permissive-"])
        if debug_mode:
            cxx_args.extend(["/ZI"])
            nvcc_args.append("-g")
            link_args.append("/DEBUG")
    
    # Collect sources (make them relative paths)
    all_sources = glob.glob(str(extensions_dir / "**/*.cpp"), recursive=True)
    sources = [os.path.relpath(s, cwd) for s in all_sources]
    
    # Filter CPU kernels if not enabled
    if not use_cpu_kernels:
        cpu_sources = [os.path.relpath(s, cwd) for s in glob.glob(str(extensions_dir / "cpu/*.cpp"))]
        sources = [s for s in sources if s not in cpu_sources]
    
    # Remove CUDA-specific sources if not using CUDA
    if not use_cuda:
        cuda_cpp_sources = [os.path.relpath(s, cwd) for s in glob.glob(str(extensions_dir / "cuda/**/*.cpp"), recursive=True)]
        sources = [s for s in sources if s not in cuda_cpp_sources]
    
    # Add CUDA sources
    if use_cuda:
        cuda_sources = glob.glob(str(extensions_dir / "cuda/**/*.cu"), recursive=True)
        sources.extend(cuda_sources)
        
        # CUTLASS support
        if not IS_WINDOWS:
            cutlass_dir = cwd / "third_party" / "cutlass"
            if cutlass_dir.exists():
                nvcc_args.extend([
                    "-DTORCHAO_USE_CUTLASS",
                    f"-I{cutlass_dir / 'include'}",
                    f"-I{cutlass_dir / 'tools/util/include'}",
                    f"-I{extensions_dir / 'cuda'}",
                    "-DCUTE_USE_PACKED_TUPLE=1",
                    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
                    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                ])
    
    # Add ROCm sources
    if use_rocm:
        rocm_dirs = [
            extensions_dir / "rocm" / "swizzle",
            extensions_dir / "cuda" / "tensor_core_tiled_layout",
        ]
        for rocm_dir in rocm_dirs:
            sources.extend(glob.glob(str(rocm_dir / "*.cu")))
            sources.extend(glob.glob(str(rocm_dir / "*.hip")))
            sources.extend(glob.glob(str(rocm_dir / "*.cpp")))
        nvcc_args.append("--offload-arch=gfx942")
    else:
        # Remove ROCm sources
        rocm_sources = [os.path.relpath(s, cwd) for s in glob.glob(str(extensions_dir / "rocm/**/*.cpp"), recursive=True)]
        sources = [s for s in sources if s not in rocm_sources]
    
    # Build extensions
    extensions = []
    extension_type = CUDAExtension if (use_cuda or use_rocm) else CppExtension
    
    if sources:
        extensions.append(
            extension_type(
                "torchao._C",
                sources,
                extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
                extra_link_args=link_args,
                py_limited_api=True,
            )
        )
    
    # Experimental CMake extensions (explicit request OR specific experimental flags)
    experimental_flags = [
        "BUILD_TORCHAO_EXPERIMENTAL",
        "TORCHAO_BUILD_CPU_AARCH64", 
        "TORCHAO_BUILD_KLEIDIAI",
        "TORCHAO_BUILD_EXPERIMENTAL_MPS",
    ]
    build_experimental = any(os.getenv(flag, "0") == "1" for flag in experimental_flags)
    
    if build_experimental:
        # Custom CMake build extension
        class CMakeBuildExtension(BuildExtension):
            def build_extensions(self):
                cmake_exts = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
                regular_exts = [ext for ext in self.extensions if not isinstance(ext, CMakeExtension)]
                
                for ext in cmake_exts:
                    self.build_cmake(ext)
                
                self.extensions = regular_exts
                if regular_exts:
                    super().build_extensions()
                self.extensions = regular_exts + cmake_exts
            
            def build_cmake(self, ext):
                extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
                os.makedirs(self.build_temp, exist_ok=True)
                
                subprocess.check_call([
                    "cmake", ext.cmake_lists_dir,
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                ] + ext.cmake_args, cwd=self.build_temp)
                
                subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)
        
        # Add CMake extension with a dummy Extension wrapper
        from setuptools import Extension
        cmake_ext = Extension("torchao.experimental", [])
        cmake_ext._cmake_info = CMakeExtension(
            "torchao.experimental",
            "torchao/experimental",
            [
                f"-DCMAKE_BUILD_TYPE={'Debug' if debug_mode else 'Release'}",
                f"-DTORCHAO_BUILD_CPU_AARCH64={'ON' if os.getenv('TORCHAO_BUILD_CPU_AARCH64', '1' if is_arm64 and is_macos else '0') == '1' else 'OFF'}",
                f"-DTORCHAO_BUILD_KLEIDIAI={'ON' if os.getenv('TORCHAO_BUILD_KLEIDIAI', '0') == '1' else 'OFF'}",
                f"-DTORCHAO_BUILD_MPS_OPS={'ON' if os.getenv('TORCHAO_BUILD_EXPERIMENTAL_MPS', '0') == '1' else 'OFF'}",
                f"-DTORCHAO_ENABLE_ARM_NEON_DOT={'ON' if os.getenv('TORCHAO_ENABLE_ARM_NEON_DOT', '1' if is_arm64 and is_macos else '0') == '1' else 'OFF'}",
                f"-DTORCHAO_ENABLE_ARM_I8MM={'ON' if os.getenv('TORCHAO_ENABLE_ARM_I8MM', '0') == '1' else 'OFF'}",
                f"-DTORCHAO_PARALLEL_BACKEND={os.getenv('TORCHAO_PARALLEL_BACKEND', 'aten_openmp')}",
            ]
        )
        
        # Add torch directory
        try:
            from distutils.sysconfig import get_python_lib
            torch_dir = os.path.join(get_python_lib(), "torch", "share", "cmake", "Torch")
            cmake_ext._cmake_info.cmake_args.append(f"-DTorch_DIR={torch_dir}")
        except:
            pass
        extensions.append(cmake_ext)
    
    return extensions


def get_cmdclass():
    """Get command classes for setuptools."""
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        return {}
    
    class TorchAOBuildExt(BuildExtension):
        def build_extensions(self):
            # Handle CMake extensions
            cmake_exts = [ext for ext in self.extensions if hasattr(ext, '_cmake_info')]
            regular_exts = [ext for ext in self.extensions if not hasattr(ext, '_cmake_info')]
            
            for ext in cmake_exts:
                self.build_cmake(ext._cmake_info)
            
            self.extensions = regular_exts
            if regular_exts:
                super().build_extensions()
            self.extensions = regular_exts + cmake_exts
        
        def build_cmake(self, cmake_ext):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(cmake_ext.name)))
            os.makedirs(self.build_temp, exist_ok=True)
            
            subprocess.check_call([
                "cmake", cmake_ext.cmake_lists_dir,
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            ] + cmake_ext.cmake_args, cwd=self.build_temp)
            
            subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)
    
    return {"build_ext": TorchAOBuildExt}