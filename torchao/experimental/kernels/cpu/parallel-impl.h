// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef TORCHAO_PARALLEL_ATEN
#pragma message("TORCHAO_PARALLEL_ATEN is set.  Using ATen parallel backend.")

// TODO(T200106949): reconcile at::parallel_for's grain_size with what is needed
// in torchao::parallel_for
#error "TORCHAO_PARALLEL_ATEN is not implemented yet"

#else
#ifdef TORCHAO_PARALLEL_EXECUTORCH
#pragma message( \
    "TORCHAO_PARALLEL_EXECUTORCH is set.  Using ExecuTorch parallel backend.")

#error "TORCHAO_PARALLEL_EXECUTORCH is not implemented yet"

#else
#ifdef TORCHAO_PARALLEL_PTHREADPOOL
#pragma message( \
    "TORCHAO_PARALLEL_PTHREADPOOL is set.  Using pthreadpool parallel backend.")
#include <torchao/experimental/kernels/cpu/parallel-pthreadpool-impl.h>

#else
#ifdef TORCHAO_PARALLEL_OMP
#pragma message("TORCHAO_PARALLEL_OMP is set.  Using OMP parallel backend.")
#include <torchao/experimental/kernels/cpu/parallel-omp-impl.h>

#else
#if defined TORCHAO_PARALLEL_SINGLE_THREADED
#pragma message( \
    "TORCHAO_PARALLEL_SINGLE_THREADED is set.  Using single-threaded parallel backend.")
#include <torchao/experimental/kernels/cpu/parallel-single_threaded-impl.h>

#else
#if defined TORCHAO_PARALLEL_TEST_DUMMY
#pragma message( \
    "TORCHAO_PARALLEL_TEST_DUMMY is set.  Using test dummy parallel backend.")
#include <torchao/experimental/kernels/cpu/parallel-test_dummy-impl.h>

#else
#error \
    "Set parallel backend by defining one of the following: \
 TORCHAO_PARALLEL_ATEN, \
 TORCHAO_PARALLEL_EXECUTORCH, \
 TORCHAO_PARALLEL_PTHREADPOOL, \
 TORCHAO_PARALLEL_OMP, \
 TORCHAO_PARALLEL_SINGLE_THREADED, \
 TORCHAO_PARALLEL_TEST_DUMMY"

#endif // TORCHAO_PARALLEL_TEST_DUMMY
#endif // TORCHAO_PARALLEL_SINGLE_THREADED
#endif // TORCHAO_PARALLEL_OMP
#endif // TORCHAO_PARALLEL_PTHREADPOOL
#endif // TORCHAO_PARALLEL_EXECUTORCH
#endif // TORCHAO_PARALLEL_ATEN
