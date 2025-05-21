# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

project(torchao_ops_mps_linear_fp_act_xbit_weight)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if (NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(FATAL_ERROR "Unified Memory requires Apple Silicon architecture")
    endif()
else()
    message(FATAL_ERROR "Torchao experimental mps ops can only be built on macOS/iOS")
endif()

find_package(Torch REQUIRED)

# Generate metal_shader_lib.h by running gen_metal_shader_lib.py
set(METAL_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../kernels/mps/metal)
file(GLOB METAL_FILES ${METAL_SHADERS_DIR}/*.metal)
set(METAL_SHADERS_YAML ${CMAKE_CURRENT_SOURCE_DIR}/../../kernels/mps/metal.yaml)
set(GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../../kernels/mps/codegen/gen_metal_shader_lib.py)
set(GENERATED_METAL_SHADER_LIB ${CMAKE_INSTALL_PREFIX}/include/torchao/experimental/kernels/mps/src/metal_shader_lib.h)
add_custom_command(
    OUTPUT ${GENERATED_METAL_SHADER_LIB}
    COMMAND python ${GEN_SCRIPT} ${GENERATED_METAL_SHADER_LIB}
    DEPENDS ${METAL_FILES} ${METAL_SHADERS_YAML} ${GEN_SCRIPT}
    COMMENT "Generating metal_shader_lib.h using gen_metal_shader_lib.py"
)
add_custom_target(generated_metal_shader_lib ALL DEPENDS ${GENERATED_METAL_SHADER_LIB})

if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()
message(STATUS "TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")

include_directories(${TORCHAO_INCLUDE_DIRS})
include_directories(${CMAKE_INSTALL_PREFIX}/include)
add_library(torchao_ops_mps_linear_fp_act_xbit_weight_aten OBJECT linear_fp_act_xbit_weight_aten.mm)
add_dependencies(torchao_ops_mps_linear_fp_act_xbit_weight_aten generated_metal_shader_lib)

target_include_directories(torchao_ops_mps_linear_fp_act_xbit_weight_aten PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(torchao_ops_mps_linear_fp_act_xbit_weight_aten PRIVATE "${TORCH_LIBRARIES}")
target_compile_definitions(torchao_ops_mps_linear_fp_act_xbit_weight_aten PRIVATE USE_ATEN=1)

# Enable Metal support
find_library(METAL_LIB Metal)
find_library(FOUNDATION_LIB Foundation)
target_link_libraries(torchao_ops_mps_linear_fp_act_xbit_weight_aten PRIVATE ${METAL_LIB} ${FOUNDATION_LIB})

add_library(torchao_ops_mps_aten SHARED)
target_link_libraries(torchao_ops_mps_aten PRIVATE
    torchao_ops_mps_linear_fp_act_xbit_weight_aten
)
install(TARGETS torchao_ops_mps_aten DESTINATION lib)

if(TORCHAO_BUILD_EXECUTORCH_OPS)
    include_directories(${CMAKE_INSTALL_PREFIX}/../..)
    include_directories(${CMAKE_INSTALL_PREFIX}/schema/include)
    include_directories(${CMAKE_INSTALL_PREFIX}/../third-party/flatbuffers/include)
    add_library(torchao_ops_mps_linear_fp_act_xbit_weight_executorch OBJECT linear_fp_act_xbit_weight_executorch.mm)
    add_dependencies(torchao_ops_mps_linear_fp_act_xbit_weight_executorch generated_metal_shader_lib)
    target_compile_definitions(torchao_ops_mps_linear_fp_act_xbit_weight_executorch PRIVATE USE_EXECUTORCH=1)
    target_link_libraries(torchao_ops_mps_linear_fp_act_xbit_weight_executorch PRIVATE executorch executorch_core mpsdelegate)
    target_link_libraries(torchao_ops_mps_linear_fp_act_xbit_weight_executorch PRIVATE ${METAL_LIB} ${FOUNDATION_LIB})

    add_library(torchao_ops_mps_executorch STATIC)
    target_link_libraries(torchao_ops_mps_executorch PRIVATE
        torchao_ops_mps_linear_fp_act_xbit_weight_executorch
    )
    install(TARGETS torchao_ops_mps_executorch DESTINATION lib)
endif()
