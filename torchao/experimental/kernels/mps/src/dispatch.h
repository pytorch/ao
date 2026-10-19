// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>

namespace torchao::kernels::mps::lowbit::dispatch {

inline void dispatch_mm(
    id<MTLComputeCommandEncoder> encoder,
    int32_t maxThreadsPerGroup,
    int32_t M,
    int32_t N,
    [[maybe_unused]] int32_t K) {
  [encoder dispatchThreads:MTLSizeMake(N, M, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min(maxThreadsPerGroup, M), 1, 1)];
}

inline void dispatch_mm_Mr1xNr4_per_TG(
    id<MTLComputeCommandEncoder> encoder,
    int32_t maxThreadsPerGroup,
    int32_t M,
    int32_t N,
    int32_t K) {
  (void)K;
  if (maxThreadsPerGroup < 32) {
    throw std::runtime_error("Can't dispatch!");
  }
  [encoder dispatchThreads:MTLSizeMake(N / 4 * 32, 1, M)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

inline void dispatch_qmv(
    id<MTLComputeCommandEncoder> encoder,
    int32_t maxThreadsPerGroup,
    int32_t M,
    int32_t N,
    int32_t K) {
  (void)K;
  if (maxThreadsPerGroup < 64) {
    throw std::runtime_error("Can't dispatch!");
  }
  [encoder dispatchThreadgroups:MTLSizeMake(M, (N + 7) / 8, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

// Dispatch for the generalized intN GEMM kernel (intNgemm_mm).
// Uses 32x64 output tiles with 128 threads per threadgroup (4 simdgroups).
// Threadgroups: ceil(M/32) x ceil(N/64)
// Threadgroup memory: 12288 bytes (8192 for A + 4096 for B). The result
// matrix reuses the A region after the K-loop (see intNgemm.metal for the
// threadgroup_barrier that makes this safe).
inline void dispatch_gemm(
    id<MTLComputeCommandEncoder> encoder,
    int32_t maxThreadsPerGroup,
    int32_t M,
    int32_t N,
    int32_t K) {
  (void)K;
  if (maxThreadsPerGroup < 128) {
    throw std::runtime_error("Can't dispatch GEMM: need at least 128 threads per threadgroup");
  }
  // Set threadgroup memory length for the dynamic shared_memory allocation.
  // A: 8192 bytes (over-allocated; 32*32*4=4096 used but 8192 required —
  // same quirk as PyTorch's kernel_mul_mm, see intNgemm.metal comment),
  // B: 4096 bytes (64*32*2).
  // Results reuse the A region after the K-loop completes (requires
  // threadgroup_barrier, see kernel comment). Total: 12288 bytes.
  // Within Apple Silicon's 32KB threadgroup memory limit (Apple WWDC22
  // "Scale compute workloads across Apple GPUs": "threads can share up to
  // 32K of threadgroup memory" on all Apple GPUs).
  [encoder setThreadgroupMemoryLength:12288 atIndex:0];
  [encoder dispatchThreadgroups:MTLSizeMake((M + 31) / 32, (N + 63) / 64, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

} // namespace torchao::kernels::mps::lowbit::dispatch
