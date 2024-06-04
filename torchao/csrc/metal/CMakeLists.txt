cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(_common_compile_options -Wno-deprecated-declarations)

find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)

add_compile_options("-Wall" "-Werror")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(_common_compile_options -Wno-deprecated-declarations -Os)

# ios can only build library but not binary
if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  # portable_ops_lib

  #set(mps_executor_runner_libs
  #    "-framework Foundation" "-weak_framework MetalPerformanceShaders"
  #    "-weak_framework MetalPerformanceShadersGraph" "-weak_framework Metal"
  #)

  add_executable(test_int4_q_kernels test_int4_q_kernels.mm)

  target_link_libraries(
    test_int4_q_kernels
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
  )

  target_compile_options(test_int4_q_kernels PUBLIC ${_common_compile_options})
endif()
