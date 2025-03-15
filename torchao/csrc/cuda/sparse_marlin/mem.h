// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
/*
 * Copyright (C) 2024 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All
 * Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "base.h"

namespace torchao {

#ifdef USE_ROCM
#include <hip/hip_runtime.h>

// Convert generic pointer to shared memory address for ROCm
template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(const T* ptr) {
    // First get the address as a size_t to handle all pointer sizes
    size_t addr = reinterpret_cast<size_t>(ptr);

    // Extract the lower 32 bits which represent the shared memory offset
    // This is safe because shared memory addresses are always within 32-bit range
    return static_cast<uint32_t>(addr & 0xFFFFFFFF);
}
#else
// For CUDA, use the native intrinsic
template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(const T* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
#endif

// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
__device__ inline void cp_async4_pred_zfill(void* smem_ptr,
                                            const void* glob_ptr,
                                            bool pred = true,
                                            const bool zfill = false) {
  const int BYTES = 16;
  int src_in_bytes = (zfill ? 0 : BYTES);
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  __builtin_amdgcn_global_load_lds(static_cast<const uint32_t*>(glob_ptr), &smem, BYTES, 0, 0);
  #else
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES), "r"(src_in_bytes));
  #endif
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  __builtin_amdgcn_global_load_lds(static_cast<const uint32_t*>(glob_ptr), &smem, BYTES, 0, 0);
  #else
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
  #endif
}

// Asynchronous global->shared copy
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  __builtin_amdgcn_global_load_lds(static_cast<const uint32_t*>(glob_ptr), &smem, BYTES, 0, 0);
  #else
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
  #endif
}

// Async copy fence.
__device__ inline void cp_async_fence() {
#ifdef USE_ROCM
  __builtin_amdgcn_s_waitcnt(0);
#else
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
#ifdef USE_ROCM
  // For AMD GPUs, we use s_waitcnt
  // This waits for all outstanding memory operations to complete
  __builtin_amdgcn_s_waitcnt(0);
#else
  // For NVIDIA GPUs, use the original instruction
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  asm volatile(
      "ds_read_b128 %0, %1 offset:0\n"
      "ds_read_b128 %2, %1 offset:16\n"
      : "=v"(a[0]), "=v"(a[1]), "=v"(a[2]), "=v"(a[3])
      : "v"(smem));
  #else
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
  #endif
}

__device__ inline void ldsm4_m(FragM& frag_m, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_m);
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  asm volatile(
      "ds_read_b64 %0, %2 offset:0\n"
      : "=v"(a[0]), "=v"(a[1])
      : "v"(smem));
  #else
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a[0]), "=r"(a[1])
               : "r"(smem));
  #endif
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
__device__ inline void ldsm4_t(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  asm volatile(
      "ds_read_b128 %0, %1 offset:0\n"
      "ds_read_b128 %2, %1 offset:16\n"
      : "=v"(a[0]), "=v"(a[1]), "=v"(a[2]), "=v"(a[3])
      : "v"(smem));
  #else
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(smem));
  #endif
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do {
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      #ifdef USE_ROCM
      asm volatile("flat_load_dword %0, %1 glc\n\t"
                   "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                   : "=v"(state)
                   : "v"(lock));
      #else
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
      #endif
    } while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    #ifdef USE_ROCM
    asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                 "s_memrealtime\n\t"
                 "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                 "flat_atomic_add_i32 %0, %1\n\t"
                 : "+v"(*lock)
                 : "v"(val));
    #else
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
    #endif
  }
}
}  // namespace torchao
