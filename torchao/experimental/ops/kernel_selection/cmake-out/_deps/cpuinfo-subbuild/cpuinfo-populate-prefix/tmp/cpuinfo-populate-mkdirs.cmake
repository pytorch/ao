# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-src")
  file(MAKE_DIRECTORY "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-src")
endif()
file(MAKE_DIRECTORY
  "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-build"
  "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix"
  "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/tmp"
  "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp"
  "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src"
  "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
