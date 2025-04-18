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

#ifndef USE_ROCM
#include <cudaTypedefs.h>
#endif

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <device_functions.h>  // For some ROCm versions
// Some intrinsics might require the compiler to be in the right mode
// with the correct target architecture flags (-march=gfx942)
#endif

namespace torchao {

// On CUDA earlier than 12.5, the ordered_metadata version of this instruction
// is not supported. On later versions of CUDA the version without ordered
// metadata results in the following warning:
//  | Advisory: Modifier 'sp::ordered_metadata' should be used on instruction
//  | 'mma' instead of modifier 'sp' as it is expected to have substantially
//  | reduced performance on some future architectures

#if defined(USE_ROCM)
  // Correct MFMA instruction for AMD GPUs
  #define MMA_SP_INST "v_mfma_f32_16x16x16_f16 "
#elif defined(CUDA_VERSION) && CUDA_VERSION >= 12050
  #define MMA_SP_INST \
    "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
#else
  #define MMA_SP_INST "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
#endif

// m16n8k32 sparse tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
__device__ inline void mma_sp(const FragB& a_frag0, const FragB& a_frag1,
                              const FragA& frag_b, FragC& frag_c, FragM& frag_m,
                              const int psel) {
  const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a_frag0);
  const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a_frag1);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* e = reinterpret_cast<const uint32_t*>(&frag_m);

  float* c = reinterpret_cast<float*>(&frag_c);
  if (psel == 0) {
    #ifdef USE_ROCM
    // AMD GPUs use a different syntax for MFMA instructions
    // The operands need to be listed individually, not in curly braces
    asm volatile(MMA_SP_INST
                 "%0, %4, %8, %12\n"
                 : "=v"(c[0]), "=v"(c[1]), "=v"(c[2]), "=v"(c[3])
                 : "v"(a0[0]), "v"(a1[0]), "v"(a0[1]), "v"(a1[1]), 
                   "v"(b[0]), "v"(b[2]), "v"(b[4]), "v"(b[6]), 
                   "v"(c[0]), "v"(c[1]), "v"(c[2]), "v"(c[3]));
    
    asm volatile(MMA_SP_INST
                 "%0, %4, %8, %12\n"
                 : "=v"(c[4]), "=v"(c[5]), "=v"(c[6]), "=v"(c[7])
                 : "v"(a0[0]), "v"(a1[0]), "v"(a0[1]), "v"(a1[1]), 
                   "v"(b[1]), "v"(b[3]), "v"(b[5]), "v"(b[7]), 
                   "v"(c[4]), "v"(c[5]), "v"(c[6]), "v"(c[7]));
    #else
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[0]),
                   "r"(b[2]), "r"(b[4]), "r"(b[6]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]), "r"(e[0]));
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[1]),
                   "r"(b[3]), "r"(b[5]), "r"(b[7]), "f"(c[4]), "f"(c[5]),
                   "f"(c[6]), "f"(c[7]), "r"(e[0]));
    #endif
  } else {
    #ifdef USE_ROCM
   asm volatile(MMA_SP_INST
                 "%0, %4, %8, %12\n"
                 : "=v"(c[0]), "=v"(c[1]), "=v"(c[2]), "=v"(c[3])
                 : "v"(a0[0]), "v"(a1[0]), "v"(a0[1]), "v"(a1[1]), 
                   "v"(b[0]), "v"(b[2]), "v"(b[4]), "v"(b[6]), 
                   "v"(c[0]), "v"(c[1]), "v"(c[2]), "v"(c[3]));
    asm volatile(MMA_SP_INST
                 "%0, %4, %8, %12\n"
                 : "=v"(c[4]), "=v"(c[5]), "=v"(c[6]), "=v"(c[7])
                 : "v"(a0[0]), "v"(a1[0]), "v"(a0[1]), "v"(a1[1]), 
                   "v"(b[1]), "v"(b[3]), "v"(b[5]), "v"(b[7]), 
                   "v"(c[4]), "v"(c[5]), "v"(c[6]), "v"(c[7])); 
    #else
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x1;\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[0]),
                   "r"(b[2]), "r"(b[4]), "r"(b[6]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]), "r"(e[0]));
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x1;\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[1]),
                   "r"(b[3]), "r"(b[5]), "r"(b[7]), "f"(c[4]), "f"(c[5]),
                   "f"(c[6]), "f"(c[7]), "r"(e[0]));
    #endif
  }
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  #ifdef USE_ROCM
  // AMD GPUs don't have a direct equivalent to lop3, so we implement it using bitwise operations
  res = (a & b & c) | (a & b & ~c) | (a & ~b & c) | (~a & b & c);
  // Apply the LUT
  res = (res & lut) | (~res & ~lut);
  #else
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  #endif
  return res;
}

__device__ __forceinline__ uint2 to_half4(float c0, float c1, float c2,
                                          float c3) {
  uint2 r;
  #ifdef USE_ROCM
  // AMD implementation
  r.x = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(c0, c1));
  r.y = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(c2, c3));
  #else
  // NVIDIA implementation
  asm("{\n\t"
      ".reg .f16 a, b, c, d; \n\t"
      "cvt.rn.f16.f32 a, %2; \n\t"
      "cvt.rn.f16.f32 b, %3; \n\t"
      "cvt.rn.f16.f32 c, %4; \n\t"
      "cvt.rn.f16.f32 d, %5; \n\t"
      "mov.b32 %0, {a, b};   \n\t"
      "mov.b32 %1, {c, d};   \n\t"
      "}"
      : "=r"(r.x), "=r"(r.y)
      : "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  #endif
  return r;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  #ifdef USE_ROCM
  // AMD implementation
  res = ((a & 0xFF) << 24) | ((a & 0xFF00) << 8) | ((a & 0xFF0000) >> 8) | ((a & 0xFF000000) >> 24);
  res = (res >> (start_byte * 8)) & mask;
  #else
  // NVIDIA implementation
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  #endif
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant_4bit(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;

  FragB frag_b;
  #ifdef USE_ROCM
  // AMD implementation
  __half2* lo_ptr = reinterpret_cast<__half2*>(&lo);
  __half2* hi_ptr = reinterpret_cast<__half2*>(&hi);
  const __half2* SUB_ptr = reinterpret_cast<const __half2*>(&SUB);
  const __half2* MUL_ptr = reinterpret_cast<const __half2*>(&MUL);
  const __half2* ADD_ptr = reinterpret_cast<const __half2*>(&ADD);

  frag_b[0] = __hsub2(*lo_ptr, *SUB_ptr);
  frag_b[1] = __hfma2(*hi_ptr, *MUL_ptr, *ADD_ptr);
  #else
  // NVIDIA implementation
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  #endif
  return frag_b;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant_8bit(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  FragB frag_b;
  #ifdef USE_ROCM
  // AMD implementation
  __half2* lo_ptr = reinterpret_cast<__half2*>(&lo);
  __half2* hi_ptr = reinterpret_cast<__half2*>(&hi);
  const __half2* magic_num_ptr = reinterpret_cast<const __half2*>(&I8s_TO_F16s_MAGIC_NUM);

  frag_b[0] = __hsub2(*lo_ptr, *magic_num_ptr);
  frag_b[1] = __hsub2(*hi_ptr, *magic_num_ptr);
  #else
  // NVIDIA implementation
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  #endif
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  #ifdef USE_ROCM
  // AMD implementation
  __half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
  #else
  // NVIDIA implementation
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
  #endif
}

__device__ inline void scale_floats(float* c0, float* c1, float* c2, float* c3,
                                    FragS& s0, float* c4, float* c5, float* c6,
                                    float* c7, FragS& s1) {
  #ifdef USE_ROCM
// AMD MI300X implementation
  *c0 = *c0 * __half2float(s0[0].x);
  *c1 = *c1 * __half2float(s0[0].y);
  *c2 = *c2 * __half2float(s0[1].x);
  *c3 = *c3 * __half2float(s0[1].y);

  *c4 = *c4 * __half2float(s1[0].x);
  *c5 = *c5 * __half2float(s1[0].y);
  *c6 = *c6 * __half2float(s1[1].x);
  *c7 = *c7 * __half2float(s1[1].y); 
  #else
  // NVIDIA implementation
  *c0 = __fmul_rn(*c0, __half2float(s0[0].x));
  *c1 = __fmul_rn(*c1, __half2float(s0[0].y));
  *c2 = __fmul_rn(*c2, __half2float(s0[1].x));
  *c3 = __fmul_rn(*c3, __half2float(s0[1].y));

  *c4 = __fmul_rn(*c4, __half2float(s1[0].x));
  *c5 = __fmul_rn(*c5, __half2float(s1[0].y));
  *c6 = __fmul_rn(*c6, __half2float(s1[1].x));
  *c7 = __fmul_rn(*c7, __half2float(s1[1].y));
  #endif
}

}  // namespace torchao
