# Used by ExecuTorch build
cmake_minimum_required(VERSION 3.19)

project(executorch_kernels)

set(CMAKE_CXX_STANDARD 17)

set(executorch_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/executorch)
# include(ExternalProject)
include(FetchContent)
FetchContent_Declare(
    executorch_external
    GIT_REPOSITORY https://github.com/pytorch/executorch.git
    GIT_TAG v0.2.1
    SOURCE_DIR ${executorch_SOURCE_DIR}
    GIT_SUBMODULES ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
message("Downloading executorch...")
FetchContent_GetProperties(executorch_external)
if(NOT executorch_external_POPULATED)
  FetchContent_Populate(executorch_external)
endif()
message("executorch downloaded")

find_library(executorch PATH ${executorch_SOURCE_DIR})
add_library(executorch_kernels SHARED int4mv_executorch.mm)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../metal/mps mps)
target_link_options(executorch_kernels PUBLIC -undefined dynamic_lookup)
target_link_libraries(executorch_kernels PUBLIC mps)
target_include_directories(executorch_kernels PUBLIC ${executorch_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/..)

install(
    TARGETS executorch_kernels
    DESTINATION lib
    INCLUDES
    DESTINATION ${executorch_SOURCE_DIR}
)