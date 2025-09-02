# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

function(target_link_torchao_parallel_backend target_name torchao_parallel_backend)
    string(TOUPPER ${torchao_parallel_backend} TORCHAO_PARALLEL_BACKEND_TOUPPER)
    if(TORCHAO_PARALLEL_BACKEND_TOUPPER STREQUAL "ATEN_OPENMP")
        message(STATUS "Building with TORCHAO_PARALLEL_BACKEND=ATEN_OPENMP")

        set(_OMP_CXX_COMPILE_FLAGS "-fopenmp")
        if (APPLE)
            set(_OMP_CXX_COMPILE_FLAGS "-Xclang -fopenmp")
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_OMP_CXX_COMPILE_FLAGS}" PARENT_SCOPE)

        find_package(Torch REQUIRED)
        include_directories("${TORCH_INCLUDE_DIRS}")
        target_link_libraries(${target_name} PRIVATE "${TORCH_LIBRARIES}")

        target_compile_definitions(${target_name} PRIVATE TORCHAO_PARALLEL_ATEN=1 AT_PARALLEL_OPENMP=1 INTRA_OP_PARALLEL=1)
        
        # Try to find OpenMP library in PyTorch installation
        set(_OMP_LIB_PATH "${TORCH_INSTALL_PREFIX}/lib/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}")
        if(EXISTS "${_OMP_LIB_PATH}")
            target_link_libraries(${target_name} PRIVATE "${_OMP_LIB_PATH}")
        else()
            # Fallback: let CMake find system OpenMP
            find_package(OpenMP QUIET)
            if(OpenMP_CXX_FOUND)
                target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
            else()
                message(WARNING "OpenMP not found in PyTorch installation or system. Parallel operations may not work optimally.")
            endif()
        endif()

    elseif(TORCHAO_PARALLEL_BACKEND_TOUPPER STREQUAL "EXECUTORCH")
        message(STATUS "Building with TORCHAO_PARALLEL_BACKEND=TORCHAO_PARALLEL_EXECUTORCH")
        message(STATUS "EXECUTORCH_INCLUDE_DIRS: ${EXECUTORCH_INCLUDE_DIRS}")
        message(STATUS "EXECUTORCH_LIBRARIES: ${EXECUTORCH_LIBRARIES}")
        target_include_directories(${target_name} PRIVATE "${EXECUTORCH_INCLUDE_DIRS}")
        target_link_libraries(${target_name} PRIVATE executorch_core)
        target_compile_definitions(${target_name} PRIVATE TORCHAO_PARALLEL_EXECUTORCH=1)

    elseif(TORCHAO_PARALLEL_BACKEND_TOUPPER STREQUAL "OPENMP")
        message(STATUS "Building with TORCHAO_PARALLEL_BACKEND=OPENMP.  You must set the CMake variable OpenMP_ROOT to the OMP library location before compiling.  Do not use this option if Torch was built with OPENMP; use ATEN_OPENMP instead.")
        find_package(OpenMP REQUIRED)
        target_compile_definitions(${target_name} PRIVATE TORCHAO_PARALLEL_OPENMP=1)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)

    elseif(TORCHAO_PARALLEL_BACKEND_TOUPPER STREQUAL "PTHREADPOOL")
        message(STATUS "Building with TORCHAO_PARALLEL_BACKEND=PTHREADPOOL")
        include(FetchContent)
        FetchContent_Declare(pthreadpool
            GIT_REPOSITORY https://github.com/Maratyszcza/pthreadpool.git
            GIT_TAG master)

        FetchContent_MakeAvailable(
            pthreadpool)

        target_compile_definitions(${target_name} PRIVATE TORCHAO_PARALLEL_PTHREADPOOL=1)
        target_link_libraries(${target_name} PRIVATE pthreadpool)

    elseif(TORCHAO_PARALLEL_BACKEND_TOUPPER STREQUAL "SINGLE_THREADED")
        message(STATUS "Building with TORCHAO_PARALLEL_BACKEND=SINGLE_THREADED")
        target_compile_definitions(${target_name} PRIVATE TORCHAO_PARALLEL_SINGLE_THREADED=1)

    elseif(TORCHAO_PARALLEL_BACKEND_TOUPPER STREQUAL "TEST_DUMMY")
        message(STATUS "Building with TORCHAO_PARALLEL_BACKEND=TEST_DUMMY")
        target_compile_definitions(${target_name} PRIVATE TORCHAO_PARALLEL_TEST_DUMMY=1)

    else()
        message(FATAL_ERROR "Unknown TORCHAO_PARALLEL_BACKEND: ${TORCHAO_PARALLEL_BACKEND}. Please choose one of: aten_openmp, executorch, openmp, pthreadpool, single_threaded.")
    endif()
endfunction()
