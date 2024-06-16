# Used by ExecuTorch build
cmake_minimum_required(VERSION 3.19)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_library(mps MPSDevice.mm MPSStream.mm OperationUtils.mm)

target_include_directories(mps PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)
find_library(MPS_FRAMEWORK MetalPerformanceShaders)
find_library(MPS_GRAPG_FRAMEWORK MetalPerformanceShadersGraph)

target_link_libraries(mps PUBLIC 
    "-framework Foundation"
    "-weak_framework MetalPerformanceShaders"
    "-weak_framework MetalPerformanceShadersGraph"
    "-weak_framework Metal"
)
