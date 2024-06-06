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

  add_executable(int4_q_kernels int4_q_kernels.mm)

  target_link_libraries(
    int4_q_kernels
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
  )

  target_compile_options(int4_q_kernels PUBLIC ${_common_compile_options})

  add_executable(llama_bench llama_bench.mm)

  target_link_libraries(
    llama_bench
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
  )

  target_compile_options(llama_bench PUBLIC ${_common_compile_options})
endif()
