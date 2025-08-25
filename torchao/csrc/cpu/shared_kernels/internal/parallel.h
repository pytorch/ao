// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace torchao {

// F has signature [&](int64_t idx)
template <typename F>
void parallel_1d(const int64_t begin, const int64_t end, const F& f);

int get_num_threads();

} // namespace torchao

#ifdef TORCHAO_PARALLEL_ATEN
#pragma message("TORCHAO_PARALLEL_ATEN is set.  Using ATen parallel backend.")
#ifndef INTRA_OP_PARALLEL
#pragma message( \
    "INTRA_OP_PARALLEL is not set; TORCHAO_PARALLEL_ATEN may be single-threaded.")
#endif
#ifndef AT_PARALLEL_OPENMP
#pragma message( \
    "AT_PARALLEL_OPENMP is not set; TORCHAO_PARALLEL_ATEN may be single-threaded.")
#endif
#include <torchao/csrc/cpu/shared_kernels/internal/parallel-aten-impl.h>

#else
#ifdef TORCHAO_PARALLEL_EXECUTORCH
#pragma message( \
    "TORCHAO_PARALLEL_EXECUTORCH is set.  Using ExecuTorch parallel backend.")
#include <torchao/csrc/cpu/shared_kernels/internal/parallel-executorch-impl.h>

#else
#ifdef TORCHAO_PARALLEL_PTHREADPOOL
#pragma message( \
    "TORCHAO_PARALLEL_PTHREADPOOL is set.  Using pthreadpool parallel backend.")
#include <torchao/csrc/cpu/shared_kernels/internal/parallel-pthreadpool-impl.h>

#else
#ifdef TORCHAO_PARALLEL_OPENMP
#pragma message( \
    "TORCHAO_PARALLEL_OPENMP is set.  Using OPENMP parallel backend.")
#include <torchao/csrc/cpu/shared_kernels/internal/parallel-openmp-impl.h>

#else
#if defined TORCHAO_PARALLEL_SINGLE_THREADED
#pragma message( \
    "TORCHAO_PARALLEL_SINGLE_THREADED is set.  Using single-threaded parallel backend.")
#include <torchao/csrc/cpu/shared_kernels/internal/parallel-single_threaded-impl.h>

#else
#if defined TORCHAO_PARALLEL_TEST_DUMMY
#pragma message( \
    "TORCHAO_PARALLEL_TEST_DUMMY is set.  Using test dummy parallel backend.")
#include <torchao/csrc/cpu/shared_kernels/internal/parallel-test_dummy-impl.h>

#else
#error \
    "Set parallel backend by defining one of the following: \
 TORCHAO_PARALLEL_ATEN, \
 TORCHAO_PARALLEL_EXECUTORCH, \
 TORCHAO_PARALLEL_PTHREADPOOL, \
 TORCHAO_PARALLEL_OPENMP, \
 TORCHAO_PARALLEL_SINGLE_THREADED, \
 TORCHAO_PARALLEL_TEST_DUMMY"

#endif // TORCHAO_PARALLEL_TEST_DUMMY
#endif // TORCHAO_PARALLEL_SINGLE_THREADED
#endif // TORCHAO_PARALLEL_OPENMP
#endif // TORCHAO_PARALLEL_PTHREADPOOL
#endif // TORCHAO_PARALLEL_EXECUTORCH
#endif // TORCHAO_PARALLEL_ATEN
