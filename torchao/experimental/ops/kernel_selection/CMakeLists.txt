# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)



project(torchao)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

include(FetchContent)
FetchContent_Declare(cpuinfo
    GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
    GIT_TAG main) # need main for benchmark::benchmark
FetchContent_MakeAvailable(
    cpuinfo)

add_executable(main main.cpp)
target_link_libraries(main cpuinfo)
