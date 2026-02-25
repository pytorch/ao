# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Arm Limited and/or its affiliates.
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

        target_compile_definitions(${target_name} PRIVATE TORCHAO_PARALLEL_ATEN=1 AT_PARALLEL_OPENMP=1 INTRA_OP_PARALLEL=1)

        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            set(_TORCH_LIBDIR "${TORCH_INSTALL_PREFIX}/lib")
            find_library(_TORCH_OMP_RUNTIME
                NAMES
                gomp libgomp.so.1 libgomp.so
                omp  libomp.so.1  libomp.so
                HINTS "${_TORCH_LIBDIR}"
                NO_DEFAULT_PATH
            )
            if(_TORCH_OMP_RUNTIME)
                target_link_libraries(${target_name} PRIVATE "${_TORCH_OMP_RUNTIME}")
            else()
                target_link_libraries(${target_name} PRIVATE ${TORCH_INSTALL_PREFIX}/lib/libomp${CMAKE_SHARED_LIBRARY_SUFFIX})
            endif()
        else()
            target_link_libraries(${target_name} PRIVATE ${TORCH_INSTALL_PREFIX}/lib/libomp${CMAKE_SHARED_LIBRARY_SUFFIX})
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
