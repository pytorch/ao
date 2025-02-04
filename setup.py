import platform
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

import torch
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, IS_WINDOWS
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

version_prefix = read_version()
version = (
    f"{version_prefix}.dev{current_date}"
    if os.environ.get("TORCHAO_NIGHTLY")
    else version_prefix
)

def get_submodule_folders():
    git_modules_path = os.path.join(os.path.abspath(os.path.curdir), ".gitmodules")
    default_modules_path = [
        os.path.join("third_party", name)
        for name in ["cutlass"]
    ]
    if not os.path.exists(git_modules_path):
        return default_modules_path
    with open(git_modules_path) as f:
        return [
            os.path.join(line.split("=", 1)[1].strip())
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
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(" --- Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"]
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

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class TorchAOBuildExt(BuildExtension):
    def build_extensions(self):
        cmake_extensions = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
        other_extensions = [ext for ext in self.extensions if not isinstance(ext, CMakeExtension)]

        for ext in cmake_extensions:
            self.build_cmake(ext)

        # Use BuildExtension to build other extensions
        self.extensions = other_extensions
        super().build_extensions()

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        debug_mode = os.getenv("DEBUG", "0") == "1"
        build_type = "Debug" if debug_mode else "Release"

        # Get PyTorch's cmake directory
        torch_cmake_dir = os.path.join(torch.utils.cmake_prefix_path, "Torch")

        # Try to find Ninja
        try:
            subprocess.check_output(['ninja', '--version'])
            use_ninja = True
        except (subprocess.SubprocessError, FileNotFoundError):
            use_ninja = False

        # Build the cmake arguments
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DTorch_DIR={torch_cmake_dir}",
            f"-DUSE_CUDA={'ON' if torch.cuda.is_available() and CUDA_HOME else 'OFF'}",
        ]

        # Add CUDA architecture flags if CUDA is enabled
        if torch.cuda.is_available() and CUDA_HOME:
            # Get CUDA compute capability of the current GPU
            capability = torch.cuda.get_device_capability()
            arch_list = f"{capability[0]}.{capability[1]}"
            cmake_args.append(f"-DTORCH_CUDA_ARCH_LIST={arch_list}")

        # Add Ninja generator if available
        if use_ninja:
            cmake_args += ["-GNinja"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )

        # Use ninja if available, otherwise default to standard build
        if use_ninja:
            subprocess.check_call(["ninja"], cwd=build_temp)
        else:
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", build_type], cwd=build_temp
            )

def get_extensions():
    if CUDA_HOME is None and torch.cuda.is_available():
        print("CUDA toolkit is not available. Skipping compilation of CUDA extensions")
        print(
            "If you'd like to compile CUDA extensions locally please install the cudatoolkit from https://anaconda.org/nvidia/cuda-toolkit"
        )

    # Check for experimental build conditions
    build_torchao_experimental = (
        os.getenv("USE_CPP") == "1"
        and platform.machine().startswith("arm64")
        and platform.system() == "Darwin"
    )

    use_cpp = os.getenv("USE_CPP")
    ext_modules = []

    if use_cpp != "0":
        ext_modules.append(
            CMakeExtension(
                "torchao._C",
                sourcedir=".",
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
