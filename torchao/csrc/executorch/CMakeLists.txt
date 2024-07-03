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
    GIT_TAG a418f5c280a9211b4f1bc78d510c9851eb66c2c3
    SOURCE_DIR ${executorch_SOURCE_DIR}
    # GIT_SUBMODULES ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
FetchContent_GetProperties(executorch_external)
if(NOT executorch_external_POPULATED)
  message("Downloading executorch...")
  FetchContent_Populate(executorch_external)
endif()
message("executorch downloaded")

add_library(portable_lib STATIC IMPORTED)
find_library(LIB_portable_lib 
    "_portable_lib.cpython-311-darwin.so" 
    HINTS ${CMAKE_PREFIX_PATH}/executorch/extension/pybindings 
    CMAKE_FIND_ROOT_PATH_BOTH 
    REQUIRED
)
set_target_properties(portable_lib PROPERTIES IMPORTED_LOCATION "${LIB_portable_lib}")
# build mps backend, executorch
set(EXECUTORCH_BUILD_MPS ON)
set(EXECUTORCH_BUILD_SDK ON)
# set(EXECUTORCH_BUILD_PYBIND ON)
# set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON) # TODO: Fix this in executorch
# set(EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT ON) # TODO: Fix this in executorch
add_subdirectory(${executorch_SOURCE_DIR} executorch_build)
add_library(executorch_kernels SHARED int4mv_executorch.mm)

set_target_properties(
    executorch_kernels
      PROPERTIES # Assume that this library will be installed in
                 # `site-packages/executorch/extension/pybindings`, and that
                 # the torch libs are in `site-packages/torch/lib`.
                 BUILD_RPATH "@loader_path/../../executorch/extension/pybindings"
                 INSTALL_RPATH "@loader_path/../../executorch/extension/pybindings"
    )
target_link_options(executorch_kernels PUBLIC -undefined dynamic_lookup)
target_link_libraries(executorch_kernels PUBLIC portable_lib mpsdelegate)
target_include_directories(executorch_kernels PUBLIC 
    ${executorch_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/schema/include ${executorch_SOURCE_DIR}/third-party/flatbuffers/include)

install(
    TARGETS executorch_kernels
    DESTINATION lib
    INCLUDES
    DESTINATION ${executorch_SOURCE_DIR}
)