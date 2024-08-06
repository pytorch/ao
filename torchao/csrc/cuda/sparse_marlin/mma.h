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

namespace marlin {
// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
__device__ inline void mma(const FragA &a_frag, const FragB &frag_b,
                           FragC &frag_c) {
  const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
  float *c = reinterpret_cast<float *>(&frag_c);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ inline void mma_t(const FragB &a_frag0, const FragB &a_frag1,
                             const FragA &frag_b, FragC &frag_c) {
  const uint32_t *a0 = reinterpret_cast<const uint32_t *>(&a_frag0);
  const uint32_t *a1 = reinterpret_cast<const uint32_t *>(&a_frag1);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
  float *c = reinterpret_cast<float *>(&frag_c);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[0]),
                 "r"(b[2]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
               : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[1]),
                 "r"(b[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}

// m16n8k32 sparse tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
__device__ inline void mma_sp(const FragB &a_frag0, const FragB &a_frag1,
                              const FragA &frag_b, FragC &frag_c, FragM &frag_m,
                              const int psel) {
  const uint32_t *a0 = reinterpret_cast<const uint32_t *>(&a_frag0);
  const uint32_t *a1 = reinterpret_cast<const uint32_t *>(&a_frag1);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
  const uint32_t *e = reinterpret_cast<const uint32_t *>(&frag_m);
  float *c = reinterpret_cast<float *>(&frag_c);
  if (psel == 0) {
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[0]),
                   "r"(b[2]), "r"(b[4]), "r"(b[6]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]), "r"(e[0]));
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[1]),
                   "r"(b[3]), "r"(b[5]), "r"(b[7]), "f"(c[4]), "f"(c[5]),
                   "f"(c[6]), "f"(c[7]), "r"(e[0]));
  } else {
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x1;\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[0]),
                   "r"(b[2]), "r"(b[4]), "r"(b[6]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]), "r"(e[0]));
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x1;\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[1]),
                   "r"(b[3]), "r"(b[5]), "r"(b[7]), "f"(c[4]), "f"(c[5]),
                   "f"(c[6]), "f"(c[7]), "r"(e[0]));
  }
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut> __device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

__device__ __forceinline__ uint2 to_half4(float c0, float c1, float c2,
                                          float c3) {
  uint2 r;
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
  return r;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q) {
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
  frag_b[0] = __hsub2(*reinterpret_cast<half2 *>(&lo),
                      *reinterpret_cast<const half2 *>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2 *>(&hi),
                      *reinterpret_cast<const half2 *>(&MUL),
                      *reinterpret_cast<const half2 *>(&ADD));
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
__device__ inline void scale(FragB &frag_b, FragS &frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half *>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}
} // namespace marlin