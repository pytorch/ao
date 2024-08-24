# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

cmake_minimum_required(VERSION 3.19)
project(tests)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_compile_options("-Wall" "-Werror")

include(CMakePrintHelpers)
message("TORCHAO_LIBRARIES: ${TORCHAO_LIBRARIES}")
include_directories(${TORCHAO_LIBRARIES})

add_library(
  dep
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/reduction/find_min_and_max.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/reduction/compute_sum.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/quantization/quantize.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/valpacking/interleave.cpp
)

enable_testing()

add_definitions(-DTORCHAO_PARALLEL_TEST_DUMMY=1)
add_executable(test_linear_operator test_linear_operator.cpp)
target_link_libraries(
    test_linear_operator
    PRIVATE
    GTest::gtest_main
    dep
)

include(GoogleTest)
gtest_discover_tests(test_linear_operator)
