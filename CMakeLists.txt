cmake_minimum_required(VERSION 3.19)
project(torchao LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_CPP "Build C++ extensions" ON)
option(USE_CPU_KERNELS "Build CPU kernels" OFF)
option(USE_CUDA "Build CUDA extensions" ON)
option(USE_ROCM "Build ROCm extensions" OFF)
option(BUILD_TORCHAO_EXPERIMENTAL "Build experimental extensions" OFF)
option(DEBUG "Build in debug mode" OFF)

# Platform detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

if(APPLE)
    set(IS_MACOS TRUE)
else()
    set(IS_MACOS FALSE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(IS_LINUX TRUE)
else()
    set(IS_LINUX FALSE)
endif()

# Auto-enable experimental builds on ARM64 macOS when USE_CPP=ON
if(USE_CPP AND IS_ARM64 AND IS_MACOS)
    set(BUILD_MACOS_ARM_AUTO TRUE)
else()
    set(BUILD_MACOS_ARM_AUTO FALSE)
endif()

if(BUILD_MACOS_ARM_AUTO OR BUILD_TORCHAO_EXPERIMENTAL)
    set(BUILD_EXPERIMENTAL TRUE)
else()
    set(BUILD_EXPERIMENTAL FALSE)
endif()

# Handle AUTO values from pyproject.toml
if(USE_CUDA STREQUAL "AUTO")
    find_package(CUDAToolkit QUIET)
    # Also check if PyTorch was built with CUDA by looking for CUDA libraries in TORCH_LIBRARIES
    if(CUDAToolkit_FOUND AND TORCH_LIBRARIES MATCHES "cuda")
        set(USE_CUDA ON)
        message(STATUS "CUDA enabled: found CUDA toolkit and PyTorch CUDA libraries")
    else()
        set(USE_CUDA OFF)
        message(STATUS "CUDA disabled: CUDAToolkit=${CUDAToolkit_FOUND}, PyTorch CUDA libs=${TORCH_LIBRARIES}")
    endif()
endif()

if(USE_ROCM STREQUAL "AUTO")
    find_package(hip QUIET)
    if(hip_FOUND)
        set(USE_ROCM ON)
    else()
        set(USE_ROCM OFF)
    endif()
endif()

if(BUILD_TORCHAO_EXPERIMENTAL STREQUAL "AUTO")
    set(BUILD_TORCHAO_EXPERIMENTAL ${BUILD_EXPERIMENTAL})
endif()

if(TORCHAO_BUILD_CPU_AARCH64 STREQUAL "AUTO")
    if(IS_ARM64 AND IS_MACOS)
        set(TORCHAO_BUILD_CPU_AARCH64 ON)
    else()
        set(TORCHAO_BUILD_CPU_AARCH64 OFF)
    endif()
endif()

if(TORCHAO_ENABLE_ARM_NEON_DOT STREQUAL "AUTO")
    if(IS_ARM64 AND IS_MACOS)
        set(TORCHAO_ENABLE_ARM_NEON_DOT ON)
    else()
        set(TORCHAO_ENABLE_ARM_NEON_DOT OFF)
    endif()
endif()

# Skip building if USE_CPP is OFF
if(NOT USE_CPP)
    message(STATUS "USE_CPP=OFF: Skipping compilation of C++ extensions")
    return()
endif()

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find PyTorch
execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" 
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" TORCH_CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED CONFIG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Check submodules
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "
import os
import subprocess
import sys

def check_submodules():
    if os.getenv('USE_SYSTEM_LIBS', '0') == '1':
        return
    
    git_modules_path = '.gitmodules'
    if not os.path.exists(git_modules_path):
        default_folders = ['third_party/cutlass']
    else:
        with open(git_modules_path) as f:
            folders = [
                line.split('=', 1)[1].strip()
                for line in f
                if line.strip().startswith('path')
            ]
    
    # Check if submodules exist
    missing = []
    for folder in folders:
        if not os.path.exists(folder) or (os.path.isdir(folder) and len(os.listdir(folder)) == 0):
            missing.append(folder)
    
    if missing:
        print(f'Initializing submodules: {missing}')
        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])

check_submodules()
"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Base compile options
set(BASE_COMPILE_OPTIONS)
set(BASE_COMPILE_DEFINITIONS)

if(DEBUG)
    list(APPEND BASE_COMPILE_OPTIONS -O0 -g)
    list(APPEND BASE_COMPILE_DEFINITIONS DEBUG)
else()
    list(APPEND BASE_COMPILE_OPTIONS -O3)
    list(APPEND BASE_COMPILE_DEFINITIONS NDEBUG)
endif()

if(NOT MSVC)
    list(APPEND BASE_COMPILE_OPTIONS -fdiagnostics-color=always)
endif()

# Collect source files
file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/**/*.cpp")

# Remove CPU sources if not enabled
if(NOT USE_CPU_KERNELS OR NOT IS_LINUX)
    file(GLOB CPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cpu/*.cpp")
    list(REMOVE_ITEM CPP_SOURCES ${CPU_SOURCES})
endif()

# CUDA/ROCm configuration
set(CUDA_SOURCES)
set(CUDA_COMPILE_OPTIONS)
set(CUDA_COMPILE_DEFINITIONS)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/**/*.cu")
    
    # Remove mx_fp_cutlass_kernels.cu - handled separately
    list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*mx_fp_cutlass_kernels\\.cu$")
    
    list(APPEND CUDA_COMPILE_OPTIONS 
        -t=0
        --std=c++17
        ${BASE_COMPILE_OPTIONS}
    )
    
    list(APPEND CUDA_COMPILE_DEFINITIONS ${BASE_COMPILE_DEFINITIONS})
    
    # Check for CUTLASS
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include")
        set(USE_CUTLASS TRUE)
        set(CUTLASS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include")
        set(CUTLASS_TOOLS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include")
        
        list(APPEND CUDA_COMPILE_DEFINITIONS TORCHAO_USE_CUTLASS)
        list(APPEND CUDA_COMPILE_OPTIONS
            -I${CUTLASS_INCLUDE_DIR}
            -I${CUTLASS_TOOLS_INCLUDE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda
            -DCUTE_USE_PACKED_TUPLE=1
            -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
            -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
            -DCUTLASS_DEBUG_TRACE_LEVEL=0
            --ftemplate-backtrace-limit=0
        )
        
        # Check CUDA version for SM90a/SM100a support
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "
import torch
cuda_version = torch.version.cuda
if cuda_version:
    major, minor = map(int, cuda_version.split('.')[:2])
    build_sm90a = major > 12 or (major == 12 and minor >= 6)
    build_sm100a = major > 12 or (major == 12 and minor >= 8)
    print(f'{int(build_sm90a)},{int(build_sm100a)}')
else:
    print('0,0')
"
            OUTPUT_VARIABLE CUDA_ARCH_SUPPORT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        string(REPLACE "," ";" CUDA_ARCH_SUPPORT_LIST "${CUDA_ARCH_SUPPORT}")
        list(GET CUDA_ARCH_SUPPORT_LIST 0 BUILD_SM90A)
        list(GET CUDA_ARCH_SUPPORT_LIST 1 BUILD_SM100A)
        
        # Collect SM90a sources
        file(GLOB SM90A_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/rowwise_scaled_linear_sparse_cutlass/*.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/to_sparse_semi_structured_cutlass_sm9x/*.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/activation24/*.cu"
        )
        
        # Remove SM90a sources from main list
        foreach(src ${SM90A_SOURCES})
            list(REMOVE_ITEM CUDA_SOURCES ${src})
        endforeach()
        
        # Collect SM100a sources
        set(SM100A_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/mx_kernels/mx_fp_cutlass_kernels.cu"
        )
    endif()
elseif(USE_ROCM)
    # ROCm configuration would go here
    message(STATUS "ROCm support not yet implemented in CMake build")
endif()

# Main extension library
if(CPP_SOURCES OR CUDA_SOURCES)
    add_library(torchao_C MODULE ${CPP_SOURCES} ${CUDA_SOURCES})
    
    target_compile_features(torchao_C PRIVATE cxx_std_17)
    target_compile_options(torchao_C PRIVATE ${BASE_COMPILE_OPTIONS})
    target_compile_definitions(torchao_C PRIVATE ${BASE_COMPILE_DEFINITIONS} Py_LIMITED_API=0x03090000)
    
    target_link_libraries(torchao_C PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(torchao_C PRIVATE ${TORCH_INCLUDE_DIRS})
    
    if(USE_CUDA)
        target_compile_options(torchao_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)
        target_compile_definitions(torchao_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_DEFINITIONS}>)
        # Link system libraries needed by CUDA
        if(UNIX AND NOT APPLE)
            target_link_libraries(torchao_C PRIVATE dl rt)
        endif()
    endif()
    
    # Set output properties
    set_target_properties(torchao_C PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_NAME "_C"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    
    install(TARGETS torchao_C DESTINATION torchao)
endif()

# SM90a extension
if(USE_CUTLASS AND BUILD_SM90A AND SM90A_SOURCES)
    add_library(torchao_C_cutlass_90a MODULE ${SM90A_SOURCES})
    
    target_compile_features(torchao_C_cutlass_90a PRIVATE cxx_std_17)
    target_compile_options(torchao_C_cutlass_90a PRIVATE 
        ${BASE_COMPILE_OPTIONS}
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS} -gencode=arch=compute_90a,code=sm_90a>
    )
    target_compile_definitions(torchao_C_cutlass_90a PRIVATE 
        ${BASE_COMPILE_DEFINITIONS} 
        Py_LIMITED_API=0x03090000
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_DEFINITIONS}>
    )
    
    target_link_libraries(torchao_C_cutlass_90a PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(torchao_C_cutlass_90a PRIVATE ${TORCH_INCLUDE_DIRS})
    
    set_target_properties(torchao_C_cutlass_90a PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_NAME "_C_cutlass_90a"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    
    install(TARGETS torchao_C_cutlass_90a DESTINATION torchao)
endif()

# SM100a extension
if(USE_CUTLASS AND BUILD_SM100A AND SM100A_SOURCES)
    add_library(torchao_C_cutlass_100a MODULE ${SM100A_SOURCES})
    
    target_compile_features(torchao_C_cutlass_100a PRIVATE cxx_std_17)
    target_compile_options(torchao_C_cutlass_100a PRIVATE 
        ${BASE_COMPILE_OPTIONS}
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS} -gencode=arch=compute_100a,code=sm_100a>
    )
    target_compile_definitions(torchao_C_cutlass_100a PRIVATE 
        ${BASE_COMPILE_DEFINITIONS} 
        Py_LIMITED_API=0x03090000
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_DEFINITIONS}>
    )
    
    target_link_libraries(torchao_C_cutlass_100a PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(torchao_C_cutlass_100a PRIVATE ${TORCH_INCLUDE_DIRS})
    
    set_target_properties(torchao_C_cutlass_100a PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_NAME "_C_cutlass_100a"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    
    install(TARGETS torchao_C_cutlass_100a DESTINATION torchao)
endif()

# MXFP8 extension
if(USE_CUDA AND BUILD_SM100A)
    set(MXFP8_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/mx_kernels/mxfp8_extension.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/mx_kernels/mxfp8_cuda.cu"
    )
    
    if(EXISTS "${MXFP8_SOURCES}")
        add_library(mxfp8_cuda MODULE ${MXFP8_SOURCES})
        
        target_compile_features(mxfp8_cuda PRIVATE cxx_std_17)
        target_compile_options(mxfp8_cuda PRIVATE 
            ${BASE_COMPILE_OPTIONS}
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
        )
        
        target_link_libraries(mxfp8_cuda PRIVATE ${TORCH_LIBRARIES} cuda cudart)
        target_include_directories(mxfp8_cuda PRIVATE 
            ${TORCH_INCLUDE_DIRS}
            "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/mx_kernels"
        )
        
        set_target_properties(mxfp8_cuda PROPERTIES
            PREFIX ""
            SUFFIX ".so"
            LIBRARY_OUTPUT_NAME "mxfp8_cuda"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
        
        install(TARGETS mxfp8_cuda DESTINATION torchao/prototype)
    endif()
endif()

# Experimental builds
if(BUILD_TORCHAO_EXPERIMENTAL)
    add_subdirectory(torchao/experimental)
endif()