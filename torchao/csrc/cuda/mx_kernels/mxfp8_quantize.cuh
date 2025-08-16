// Adapted from https://github.com/NVIDIA/TransformerEngine
// License - Apache-2.0
// https://github.com/NVIDIA/TransformerEngine/blob/main/LICENSE
// * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Use official CUDA PTX library
#include "ptx.cuh"
#include <cuda/barrier>
#include <cuda/ptx>

#define MIN_CUDA_SM 1000 // SM90 = 900, SM100 = 1000

// Check if we're compiling for supported architecture
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < MIN_CUDA_SM)
#warning                                                                       \
    "MXFP8 quantization requires SM90+ (Hopper) or SM100+ (Blackwell) architecture. Kernel will be disabled for this architecture."
#endif

// Architecture detection for native FP8 support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define HAS_NATIVE_FP8_CONVERSION 1
#else
#define HAS_NATIVE_FP8_CONVERSION 0
#endif

enum class DType {
  kByte,
  kFloat32,
  kFloat16,
  kBFloat16,
  kFloat8E4M3,
  kFloat8E5M2
};

enum class ScaleCalculationMode {
  FLOOR, // uses software scaling
  RCEIL, // uses hardware scaling
};

// Data types
using e8m0_t = uint8_t;
using bfloat16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;

constexpr size_t get_dtype_bits(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return 32;
  case DType::kBFloat16:
    return 16;
  case DType::kFloat8E4M3:
    return 8;
  default:
    // TODO: something smarter than this
    return 0;
  }
}

// FP32 constants
constexpr int32_t FP32_MANTISSA_BITS = 23;
constexpr int32_t FP32_EXPONENT_BIAS = 127;

// BF16 constants
constexpr int32_t BF16_MANTISSA_BITS = 7;
constexpr int32_t BF16_EXPONENT_BIAS = 127;

// FP8E4M3 constants
constexpr int32_t F8E4M3_MAX_POW2 = 8;
constexpr float F8E4M3_MAX = 448.0;

// FP8E8M0 constants
constexpr int32_t E8M0_EXPONENT_BIAS = 127;

// 1. Base template (for unsupported types)
template <typename T> struct DataTypeTraits {
  static constexpr bool is_supported = false;
};

// 2. Specialization for float32
template <> struct DataTypeTraits<float> {
  static constexpr bool is_supported = true;
  static constexpr int mantissa_bits = 23;
  static constexpr int exponent_bias = 127;

  __device__ static __forceinline__ float to_float(const float val) {
    return val;
  }
};

// 3. Specialization for bfloat16
template <> struct DataTypeTraits<nv_bfloat16> {
  static constexpr bool is_supported = true;
  static constexpr int mantissa_bits = 7;
  static constexpr int exponent_bias = 127;

  __device__ static __forceinline__ float to_float(const nv_bfloat16 val) {
    return __bfloat162float(val);
  }
};

__device__ static __forceinline__ e8m0_t
calculate_e8m0_biased_scale(const float amax) {
  // torchao ref:
  // https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L239
  const int32_t int_amax = *reinterpret_cast<const int32_t *>(&amax);
  const int32_t extracted_pow2 =
      ((int_amax >> FP32_MANTISSA_BITS) & 0b11111111) - FP32_EXPONENT_BIAS;

  // torchao ref:
  // https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L244
  int32_t scale_unbiased = extracted_pow2 - F8E4M3_MAX_POW2;

  // torchao ref:
  // https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L256
  scale_unbiased = max(scale_unbiased, -E8M0_EXPONENT_BIAS);
  scale_unbiased = min(scale_unbiased, E8M0_EXPONENT_BIAS + 1);
  int32_t scale_with_e8m0_bias = scale_unbiased + E8M0_EXPONENT_BIAS;

  // torchao ref:
  // https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L261C9-L261C26
  const e8m0_t e8m0_biased_scale =
      *reinterpret_cast<e8m0_t *>(&scale_with_e8m0_bias);
  return e8m0_biased_scale;
}

// Constants for MXFP8 kernel
constexpr size_t MXFP8_CHUNK_DIM_Y = 64;
constexpr size_t MXFP8_CHUNK_DIM_X = 64;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_X = 1;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK =
    MXFP8_CHUNKS_PER_BLOCK_Y * MXFP8_CHUNKS_PER_BLOCK_X; // 1 * 1 = 1
constexpr size_t MXFP8_THREADS_PER_CHUNK = 64;
constexpr size_t MXFP8_BUFFERS_NUM = 2;
constexpr size_t MXFP8_PREFETCH_BUFFERS_NUM = 1;

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t MXFP8_BUFFER_DIM_Y = 32;
constexpr size_t MXFP8_BUFFER_DIM_X = MXFP8_CHUNK_DIM_X; // 64
constexpr size_t MXFP8_SHMEM_DIM_Y = MXFP8_BUFFER_DIM_Y; // 32
constexpr size_t MXFP8_SHMEM_DIM_X = MXFP8_BUFFER_DIM_X; // 64

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE =
    MXFP8_CHUNK_DIM_X / ELEMS_PER_THREAD; // 64/16 = 4
constexpr size_t THREADS_PER_CHUNK_Y_ROWWISE =
    MXFP8_THREADS_PER_CHUNK / THREADS_PER_CHUNK_X_ROWWISE;        // 64 / 4 = 16
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = MXFP8_CHUNK_DIM_X; //  64
constexpr size_t MXFP8_BUFF_STAGES_NUM =
    MXFP8_BUFFER_DIM_Y / THREADS_PER_CHUNK_Y_ROWWISE; //   2 = 32 / 16
constexpr size_t MXFP8_ITERATIONS =
    MXFP8_CHUNK_DIM_Y / MXFP8_BUFFER_DIM_Y; //   2 = 64 / 32
static_assert(MXFP8_ITERATIONS >= MXFP8_PREFETCH_BUFFERS_NUM);

constexpr size_t THREADS_PER_WARP = 32; // lol

// Utility macros
#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// Vector type for loading/storing multiple elements
template <typename T, int N> struct Vec {
  union {
    T elt[N];
  } data;

  __device__ inline void clear() {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      data.elt[i] = T(0);
    }
  }

  __device__ inline void load_from(const T *ptr) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      data.elt[i] = ptr[i];
    }
  }

  __device__ inline void store_to(T *ptr) const {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      ptr[i] = data.elt[i];
    }
  }
};

// Source:
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L971
__device__ __forceinline__ float exp2f_rcp(e8m0_t biased_exp) {
  return (biased_exp == 0)
             ? 1
             : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exp));
}

// Source:
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L937
__device__ __forceinline__ e8m0_t float_to_e8m0(float val) {
  // TODO: nan/inf needs to be set for any value
  // of nan/inf in input not just amax.
  if (isnan(val)) {
    return 0xFF;
  }
  if (isinf(val)) {
    return 0xFE;
  }
#if ((__CUDA_ARCH_HAS_FEATURE__(SM100_ALL)) ||                                 \
     (__CUDA_ARCH_HAS_FEATURE__(SM101_ALL)) ||                                 \
     (__CUDA_ARCH_HAS_FEATURE__(SM120_ALL)))
  uint16_t out;
  asm volatile("{\n"
               "cvt.rp.satfinite.ue8m0x2.f32  %0, 0.0, %1;\n"
               "}"
               : "=h"(out)
               : "f"(val));
  return *reinterpret_cast<e8m0_t *>(&out);
#else
  if (val == 0.0f) {
    return 0x00;
  }
  uint32_t val_u32 = *reinterpret_cast<uint32_t *>(&val);
  e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
  uint32_t mantissa = val_u32 & 0x7FFFFF;
  // Round up exponent and deal with satfinite.
  if ((mantissa > 0 && exponent != 0xFE) &&
      !(exponent == 0 && mantissa <= 0x400000)) {
    ++exponent;
  }
  return exponent;
#endif
}

// Quantization limits
// Source:
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L929
template <typename T> struct Quantized_Limits {
  static constexpr float max_norm = 448.0f; // For E4M3
  static constexpr float max_norm_rcp = 1.0f / max_norm;
};

// Warp reduction utilities
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L867
/**
 * Max reduction in subwarps
 * E.g., if nvec=4, each warp processes 128 elements (32 x 4), that covers four
 * MXFP8 scaling factors. To compute an actual scaling factor for 32
 * consequentive elements, only 8 threads need to participate, thus splitting
 * the warp into 4x smaller subwarps 8-thread width. 'Butterfly' reduction is
 * used inside subwarps.
 */
template <int subwarp_width>
__forceinline__ __device__ float subwarp_reduce_max_broadcast(const float val) {
  float val_tmp = val;
#pragma unroll
  for (int offset = subwarp_width / 2; offset > 0; offset /= 2) {
    const float val_other =
        __shfl_down_sync(0xFFFFFFFF, val_tmp, offset, subwarp_width);
    __builtin_assume(val_tmp >= 0);
    __builtin_assume(val_other >= 0);
    val_tmp = fmaxf(val_tmp, val_other);
  }
  // Broadcast the amax to other threads of the subwarp from the zero subwarp
  // lane_id
  constexpr int subwarp_lane_zero = 0;
  val_tmp = __shfl_sync(0xFFFFFFFF, val_tmp, subwarp_lane_zero, subwarp_width);
  return val_tmp;
}

// Source:
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L813C1-L824C2
template <int num_elems>
__device__ __forceinline__ float warp_reduce_max(const float m) {
  float tmp = m;
#pragma unroll
  for (int delta = num_elems / 2; delta > 0; delta /= 2) {
    const float other_m = __shfl_down_sync(0xFFFFFFFF, tmp, delta);
    __builtin_assume(tmp >= 0);
    __builtin_assume(other_m >= 0);
    tmp = fmaxf(tmp, other_m);
  }
  return tmp;
}

// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/utils.cuh#L841C1-L857C2
template <int num_warps, typename compute_t>
__device__ __forceinline__ compute_t reduce_max(const compute_t m,
                                                const int warpid) {
  __shared__ float staging[num_warps];
  constexpr int warp_size = 32;
  const float my_max = m;
  const float my_warp_max = warp_reduce_max<warp_size>(my_max);
  if (threadIdx.x % 32 == 0) {
    staging[warpid] = my_warp_max;
  }
  __syncthreads();
  compute_t result = 0.f;
  if (warpid == 0) {
    const float my_max = threadIdx.x < num_warps ? staging[threadIdx.x] : 0;
    result = warp_reduce_max<num_warps>(my_max);
  }
  return result;
}

// https://stackoverflow.com/a/51549250
// TODO: handle -0 case
__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

// TMA descriptor creation
inline CUtensorMapDataType get_dtype_for_tma(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  case DType::kFloat16:
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  case DType::kBFloat16:
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  case DType::kFloat8E4M3:
  case DType::kFloat8E5M2:
  case DType::kByte:
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  default:
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
}

// Reference:
// https://github.com/NVIDIA/TransformerEngine/blob/1ae1d228d725a488621deba685bd26d6ee1cdb21/transformer_engine/common/common.cu#L137
// This was modified to make it compatible with our implementation and avoid
// using internal TE types.
inline void create_2D_tensor_map(CUtensorMap &tensorMap, void *data_ptr,
                                 DType dtype, const size_t rows,
                                 const size_t cols, uint32_t shmem_y,
                                 uint32_t shmem_x, const size_t stride_elems,
                                 const size_t type_num_bits) {
  // Get function pointer to cuTensorMapEncodeTiled
  static void *driver_ptr = nullptr;
  if (!driver_ptr) {
    cudaDriverEntryPointQueryResult result;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr,
                            cudaEnableDefault, &result);
  }
  auto cuTensorMapEncodeTiled =
      reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);

  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {cols, rows};
  uint64_t stride[rank - 1] = {(stride_elems * type_num_bits) /
                               8}; // (cols * bits per element) / 8
  uint32_t boxSize[rank] = {shmem_x, shmem_y};
  uint32_t elemStride[rank] = {1, 1};

#if defined(DEBUG)
  printf("TMA Descriptor: global_shape=(%llu, %llu), tile_shape=(%u, %u), "
         "stride_bytes=%llu\n",
         (unsigned long long)size[1], (unsigned long long)size[0], boxSize[1],
         boxSize[0], (unsigned long long)stride[0]);
#endif

  cuTensorMapEncodeTiled(
      &tensorMap, get_dtype_for_tma(dtype), rank, data_ptr, size, stride,
      boxSize, elemStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// Helper functions for TMA operations
__device__ inline void copy_2d_to_shared(void *smem,
                                         const CUtensorMap *tensor_map,
                                         uint32_t x, uint32_t y,
                                         size_t smem_size, uint64_t *mbar,
                                         bool is_master) {
  if (is_master) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(smem),
        reinterpret_cast<const uint64_t *>(tensor_map), x, y, mbar);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(mbar, smem_size);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(mbar);
  }
}

////////////////////////////////////////////////////////////////////////////////
// TorchAO shared quantization utils
////////////////////////////////////////////////////////////////////////////////

/**
 * Convert e8m0 biased scale to float32 scale following torchao implementation
 * torchao ref:
 * https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L275C1-L277C30
 */
__device__ __forceinline__ float e8m0_to_scale_fp32(e8m0_t e8m0_biased_scale) {
  int32_t exponent_as_int32 = static_cast<int32_t>(e8m0_biased_scale);
  int32_t float_bits = exponent_as_int32 << FP32_MANTISSA_BITS;
  float scale_fp32 = *reinterpret_cast<float *>(&float_bits);

  // torchao ref:
  // https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L286
  const float F32_MIN_NORMAL = exp2f(-FP32_EXPONENT_BIAS + 1);
  scale_fp32 = max(scale_fp32, F32_MIN_NORMAL);

  return scale_fp32;
}

/**
 * Quantize a single value using torchao-style clamping and conversion
 * torchao ref:
 * https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L289
 */
template <typename OType>
__device__ __forceinline__ OType torchao_quantize_value(float input_value,
                                                        float inv_scale_fp32) {
  // Scale the input value
  float data_lp = input_value * inv_scale_fp32;

  // Apply torchao-style clamping
  // torchao ref:
  // https://github.com/pytorch/ao/blob/00417b8b33abb75c54cdb347bd320fb6ac0a4d94/torchao/prototype/mx_formats/mx_tensor.py#L301C23-L301C74
  data_lp = min(data_lp, F8E4M3_MAX);
  data_lp = max(data_lp, -F8E4M3_MAX);

  return static_cast<OType>(data_lp);
}

/**
 * Complete torchao-style quantization: calculate scale and convert values
 * Template parameters ensure compile-time array size checking for safety
 */
template <typename OType, int NUM_VALUES, ScaleCalculationMode ScalingMode>
__device__ __forceinline__ void
quantize_block(float amax, e8m0_t &out_scale,
                       const float (&input_values)[NUM_VALUES],
                       OType (&output_values)[NUM_VALUES]) {

  float inv_scale_fp32;
  if constexpr (ScalingMode == ScaleCalculationMode::FLOOR) {
    // FLOOR scaling.
    out_scale = calculate_e8m0_biased_scale(amax);

    // Convert scale to float32
    float scale_fp32 = e8m0_to_scale_fp32(out_scale);

    // Calculate inverse scale for fast multiplication
    inv_scale_fp32 = __fdiv_rn(1.0f, scale_fp32);

    // Quantize all values
#pragma unroll
      for (int i = 0; i < NUM_VALUES; ++i) {
        output_values[i] =
            torchao_quantize_value<OType>(input_values[i], inv_scale_fp32);
      }

  } else {
    // RCEIL scaling.
    out_scale = float_to_e8m0(amax * Quantized_Limits<OType>::max_norm_rcp);
    inv_scale_fp32 = exp2f_rcp(out_scale);

#pragma unroll
      for (int i = 0; i < NUM_VALUES; ++i) {
        output_values[i] =
            static_cast<OType>(input_values[i] * inv_scale_fp32);
      }
  }

}

/**
 * Bounds checking helper for IMA avoidance
 */
struct BoundsChecker {
  const size_t rows, cols;
  const size_t chunk_offset_X, chunk_offset_Y;

  __device__ __forceinline__ BoundsChecker(size_t r, size_t c, size_t cox,
                                           size_t coy)
      : rows(r), cols(c), chunk_offset_X(cox), chunk_offset_Y(coy) {}

  __device__ __forceinline__ bool is_out_of_bounds(size_t row,
                                                   size_t col) const {
    return (row >= rows) || (col >= cols);
  }

  __device__ __forceinline__ bool
  is_rowwise_out_of_bounds(size_t shmem_y, size_t shmem_x, int j,
                           size_t row_base) const {
    const size_t row = row_base + shmem_y;
    const size_t col = chunk_offset_X + shmem_x + j;
    return is_out_of_bounds(row, col);
  }

  __device__ __forceinline__ bool
  is_colwise_out_of_bounds(size_t row_offset, size_t col,
                           size_t row_base) const {
    const size_t row = row_base + row_offset;
    return is_out_of_bounds(row, col);
  }
};

////////////////////////////////////////////////////////////////////////////////
// MXFP8 quantization kernel
////////////////////////////////////////////////////////////////////////////////

// Main MXFP8 quantization kernel (with TMA)
template <typename IType, typename OType, size_t SCALE_DIM_Y,
          size_t SCALE_DIM_X, ScaleCalculationMode ScalingMode>
__global__ void __launch_bounds__(MXFP8_THREADS_PER_CHUNK)
    mxfp8_quantize_kernel(
        const __grid_constant__ CUtensorMap tensor_map_input,
        const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
        const __grid_constant__ CUtensorMap tensor_map_output_colwise,
        e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
        const size_t rows, const size_t cols,
        const size_t scales_rowwise_stride_dim0,
        const size_t scales_rowwise_stride_dim1,
        const size_t scales_colwise_stride_dim0,
        const size_t scales_colwise_stride_dim1) {

#if defined(DEBUG)
  printf("mxfp8_quantize_kernel: rows=%llu, cols=%llu, "
         "scales_rowwise_stride_dim0=%llu, scales_rowwise_stride_dim1=%llu, "
         "scales_colwise_stride_dim0=%llu, scales_colwise_stride_dim1=%llu\n",
         (unsigned long long)rows, (unsigned long long)cols,
         (unsigned long long)scales_rowwise_stride_dim0,
         (unsigned long long)scales_rowwise_stride_dim1,
         (unsigned long long)scales_colwise_stride_dim0,
         (unsigned long long)scales_colwise_stride_dim1);

  if (ScalingMode == ScaleCalculationMode::FLOOR) {
    printf("mxfp8_quantize_kernel: scaling_mode: floor\n");
  } else if (ScalingMode == ScaleCalculationMode::RCEIL) {
    printf("mxfp8_quantize_kernel: scaling_mode: rceil\n");
  } else {
    printf("mxfp8_quanitze_kenrel: unknown scaling mode\n");
  }
#endif


  static_assert(DataTypeTraits<IType>::is_supported,
                "Input data type is not supported by this kernel.");

  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y =
      MXFP8_CHUNK_DIM_Y; //   2 = 64 / 32
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X =
      MXFP8_CHUNK_DIM_X / SCALE_DIM_X; //  64 = 64 / 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_Y =
      SCALES_ROWWISE_PER_CHUNK_Y * MXFP8_CHUNKS_PER_BLOCK_Y; //   2 = 2 * 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_X =
      SCALES_ROWWISE_PER_CHUNK_X * MXFP8_CHUNKS_PER_BLOCK_X; //  64 = 64 * 1

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y =
      MXFP8_CHUNK_DIM_Y / SCALE_DIM_Y; //   2 = 64 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X =
      MXFP8_CHUNK_DIM_X; //  64 = 64 / 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_Y =
      SCALES_COLWISE_PER_CHUNK_Y * MXFP8_CHUNKS_PER_BLOCK_Y; //   2 = 2 * 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_X =
      SCALES_COLWISE_PER_CHUNK_X * MXFP8_CHUNKS_PER_BLOCK_X; //  64 = 64 * 1

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);                     //   2 = 32 / 16
  constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X_ROWWISE; //   2

  const int block_offset_Y =
      blockIdx.y * MXFP8_CHUNKS_PER_BLOCK_Y * MXFP8_CHUNK_DIM_Y;
  const int block_offset_X =
      blockIdx.x * MXFP8_CHUNKS_PER_BLOCK_X * MXFP8_CHUNK_DIM_X;
  const int scales_rowwise_block_offset_Y =
      blockIdx.y * SCALES_ROWWISE_PER_BLOCK_Y;
  const int scales_rowwise_block_offset_X =
      blockIdx.x * SCALES_ROWWISE_PER_BLOCK_X;
  const int scales_colwise_block_offset_Y =
      blockIdx.y * SCALES_COLWISE_PER_BLOCK_Y;
  const int scales_colwise_block_offset_X =
      blockIdx.x * SCALES_COLWISE_PER_BLOCK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;

  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 e8m0_t aligned
  __shared__ alignas(128)
      IType in_sh[MXFP8_BUFFERS_NUM][MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128) OType
      out_rowwise_sh[MXFP8_BUFFERS_NUM][MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128) OType
      out_colwise_sh[MXFP8_BUFFERS_NUM][MXFP8_SHMEM_DIM_X][MXFP8_SHMEM_DIM_Y];

  constexpr int shmem_buff_size = sizeof(in_sh) / MXFP8_BUFFERS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  float block_amax = 0;

// Initialize shared memory barrier with the number of threads participating in
// the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[MXFP8_ITERATIONS];

  initialize_barriers<MXFP8_ITERATIONS, MXFP8_THREADS_PER_CHUNK>(
      mbar, is_master_thread);

  int parity = 0;

// Process chunks
#pragma unroll
  // Calculate chunk offsets
  for (int chunk = 0; chunk < MXFP8_CHUNKS_PER_BLOCK; ++chunk) {
    const int chunk_Y = chunk / MXFP8_CHUNKS_PER_BLOCK_X;
    const int chunk_X = chunk % MXFP8_CHUNKS_PER_BLOCK_X;

    const int chunk_offset_Y = block_offset_Y + chunk_Y * MXFP8_CHUNK_DIM_Y;
    const int chunk_offset_X = block_offset_X + chunk_X * MXFP8_CHUNK_DIM_X;

    const int scales_rowwise_chunk_offset_Y =
        scales_rowwise_block_offset_Y + chunk_Y * SCALES_ROWWISE_PER_CHUNK_Y;
    const int scales_rowwise_chunk_offset_X =
        scales_rowwise_block_offset_X + chunk_X * SCALES_ROWWISE_PER_CHUNK_X;
    const int scales_colwise_chunk_offset_Y =
        scales_colwise_block_offset_Y + chunk_Y * SCALES_COLWISE_PER_CHUNK_Y;
    const int scales_colwise_chunk_offset_X =
        scales_colwise_block_offset_X + chunk_X * SCALES_COLWISE_PER_CHUNK_X;

// Prefetch initial data
#pragma unroll
    // Kick off TMA async copy from global to shared memory
    for (int prefetch_buff = 0; prefetch_buff < MXFP8_PREFETCH_BUFFERS_NUM;
         ++prefetch_buff) {
      const int chunk_stage_offset_Y =
          chunk_offset_Y + prefetch_buff * MXFP8_BUFFER_DIM_Y;
      const int chunk_stage_offset_X = chunk_offset_X;
      copy_2d_to_shared(&in_sh[prefetch_buff], &tensor_map_input,
                        chunk_stage_offset_X, chunk_stage_offset_Y,
                        shmem_buff_size, &mbar[prefetch_buff],
                        is_master_thread);
    }

// Process iterations
#pragma unroll
    // Iterate through the chunk along the Y dim
    for (int iter = 0; iter < MXFP8_ITERATIONS; ++iter) {
      const int buff = iter % MXFP8_BUFFERS_NUM;
      const int next_iter = iter + MXFP8_PREFETCH_BUFFERS_NUM;
      const size_t row_base = chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;

      // Prefetch next iteration data
      if (next_iter < MXFP8_ITERATIONS) {
        const int next_buff = next_iter % MXFP8_BUFFERS_NUM;
        const int chunk_it_offset_y =
            chunk_offset_Y + next_iter * MXFP8_BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        copy_2d_to_shared(&in_sh[next_buff], &tensor_map_input,
                          chunk_it_offset_x, chunk_it_offset_y, shmem_buff_size,
                          &mbar[next_iter], is_master_thread);
      }

      ptx::fence_proxy_async_shared_cta();

      // Wait for the data to have arrived
      ptx::mbarrier_wait_parity(&mbar[iter], parity);

#if defined(DEBUG_SMEM)
      // Debugging smem data
      if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Shared memory values:\n");
        for (int b = 0; b < MXFP8_BUFFERS_NUM; b++) {
          for (int y = 0; y < MXFP8_SHMEM_DIM_Y; y++) {
            for (int x = 0; x < MXFP8_SHMEM_DIM_X; x++) {
              printf("in_sh[%d][%d][%d] = %f\n", b, y, x,
                     (float)in_sh[b][y][x]);
            }
          }
        }
      }
#endif

      // ======== RowWise SCALING ========

      // Updated Row-wise scaling section:
      if constexpr (USE_ROWWISE_SCALING) {
        Vec<IType, ELEMS_PER_THREAD> in;
        Vec<OType, ELEMS_PER_THREAD> out_c;

        // Create bounds checker for this chunk
        BoundsChecker bounds(rows, cols, chunk_offset_X, chunk_offset_Y);

        const int iteration_scale_rowwise_offset_Y =
            scales_rowwise_chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;

#pragma unroll
        for (int stage = 0; stage < MXFP8_BUFF_STAGES_NUM; ++stage) {
          const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y_ROWWISE;
          const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
          const int shmem_offset_x = thread_offset_X_rowwise;

          // Load from shared memory into thread local registers
          in.load_from(&in_sh[buff][shmem_offset_y][shmem_offset_x]);

          float thread_amax = 0;
          float in_compute[ELEMS_PER_THREAD];

          // Calculate thread-local amax and prepare input values
#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
            const bool out_of_bounds = bounds.is_rowwise_out_of_bounds(
                shmem_offset_y, shmem_offset_x, j, row_base);

            // Load and convert to float
            float elt = DataTypeTraits<IType>::to_float(in.data.elt[j]);
            in_compute[j] = elt;

            // Update thread local amax
            if (!out_of_bounds) {
              thread_amax = fmaxf(thread_amax, fabsf(elt));
            }
          }

          // Update block local amax
          block_amax = fmaxf(block_amax, thread_amax);

          // Reduce amax across subwarp
          const float subwarp_amax =
              subwarp_reduce_max_broadcast<SUBWARP_WIDTH>(thread_amax);


          // Apply quantization to the local block.
          e8m0_t e8m0_biased_scale;
          OType quantized_values[ELEMS_PER_THREAD];

          quantize_block<OType, ELEMS_PER_THREAD, ScalingMode>(
              subwarp_amax, e8m0_biased_scale, in_compute, quantized_values);

          // Write scaling factor (only a single thread writes it to global
          // memory)
          if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0) {
            const int global_scales_offset_Y =
                iteration_scale_rowwise_offset_Y + stage_offset_Y +
                tid_rowwise_Y;
            const int global_scales_offset_X =
                scales_rowwise_chunk_offset_X +
                tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE;
            const int scale_idx =
                global_scales_offset_Y * scales_rowwise_stride_dim0 +
                global_scales_offset_X;
            scales_rowwise[scale_idx] = e8m0_biased_scale;
          }

          // Store quantized values
#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
            out_c.data.elt[j] = quantized_values[j];
          }
          out_c.store_to(&out_rowwise_sh[buff][shmem_offset_y][shmem_offset_x]);

#if defined(DEBUG)
          if (tid_rowwise_X == 0 && tid_rowwise_Y == 0) {
            printf("Rowwise: subwarp_amax=%f, e8m0_scale=%u\n", subwarp_amax, e8m0_biased_scale);
          }
#endif

        }
      }
      // ======== End RowWise SCALING ========

      // ======== ColWise SCALING ========
      // Column-wise scaling

      if constexpr (USE_COLWISE_SCALING) {
        // Create bounds checker for this chunk
        BoundsChecker bounds(rows, cols, chunk_offset_X, chunk_offset_Y);

        const size_t col = chunk_offset_X + tid_colwise_X;
        const bool col_out_of_bounds = (col >= cols);

        float in_compute[SCALE_DIM_Y];
        float amax = 0;

        // Calculate amax and prepare input values
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; ++i) {
          const bool out_of_bounds =
              bounds.is_colwise_out_of_bounds(i, col, row_base);

          // Load and convert to float
          float elt =
              DataTypeTraits<IType>::to_float(in_sh[buff][i][tid_colwise_X]);
          in_compute[i] = elt;

          // Update thread local amax
          if (!out_of_bounds) {
            amax = fmaxf(amax, fabsf(elt));
          }
        }

        // Apply quantization to the local block.
        e8m0_t e8m0_biased_scale;
        OType quantized_values[SCALE_DIM_Y];
        quantize_block<OType, SCALE_DIM_Y, ScalingMode>(
            amax, e8m0_biased_scale, in_compute, quantized_values);

        // Write scaling factor to global memory
        const int global_scales_offset_Y = scales_colwise_chunk_offset_Y + iter;
        const int global_scales_offset_X =
            scales_colwise_chunk_offset_X + tid_colwise_X;

        // Write scale in column major memory layout, shape (cols, num_row_blocks, 1).
        // Stride along `cols` dim must be 1, for coalesced writes to global memory.
        const int scale_idx =
            global_scales_offset_Y * scales_colwise_stride_dim1 +
            global_scales_offset_X * scales_colwise_stride_dim0;

        // Bounds check for scale writing
        const bool row_out_of_bounds = (row_base >= rows);
        if (!row_out_of_bounds && !col_out_of_bounds) {
          scales_colwise[scale_idx] = e8m0_biased_scale;
        }

        // Store quantized values to shared memory
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; ++i) {
          out_colwise_sh[buff][tid_colwise_X][i] = quantized_values[i];
        }

#if defined(DEBUG)
        if (tid_colwise_X == 0) {
          printf("Colwise: amax=%f, e8m0_scale=%u\n", amax, e8m0_biased_scale);
        }
#endif
      }

      // Wait for shared memory writes to be visible to TMA engine.
      ptx::fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.

      // Initiate TMA transfer to copy shared memory to global memory
      if (is_master_thread) {
        if constexpr (USE_ROWWISE_SCALING) {
          const int chunk_it_offset_y =
              chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;
          const int chunk_it_offset_x = chunk_offset_X;
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise),
              chunk_it_offset_x, chunk_it_offset_y,
              reinterpret_cast<uint64_t *>(&out_rowwise_sh[buff]));
        }
        if constexpr (USE_COLWISE_SCALING) {
          // Swap logical destination offsets for TMA to write into column major layout.
          const int chunk_it_offset_y = chunk_offset_X;
          const int chunk_it_offset_x = chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise),
              chunk_it_offset_x, chunk_it_offset_y,
              reinterpret_cast<uint64_t *>(&out_colwise_sh[buff]));
        }
        // Create a "bulk async-group" out of the previous bulk copy operation.
        ptx::cp_async_bulk_commit_group();

        // Wait for TMA transfer to have finished reading shared memory.
        ptx::cp_async_bulk_wait_group_read<MXFP8_PREFETCH_BUFFERS_NUM>();
      }
    }
    ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    parity ^= 1;
  }

  destroy_barriers<MXFP8_ITERATIONS>(mbar, is_master_thread);
  // #endif
}

// Simple wrapper class for MXFP8 quantization
class MXFP8Quantizer {
public:
  // Quantize a tensor using MXFP8
  // input: pointer to input data
  // output_rowwise: pointer to row-wise quantized output (can be nullptr)
  // output_colwise: pointer to column-wise quantized output (can be nullptr)
  // scales_rowwise: pointer to row-wise scaling factors (required if
  // output_rowwise is not null) scales_colwise: pointer to column-wise scaling
  // factors (required if output_colwise is not null) rows, cols: tensor
  // dimensions input_dtype: data type of input output_dtype: FP8 output type
  // (fp8e4m3 or fp8e5m2) scale_dim_x: block size for row-wise scaling
  // (typically 32) scale_dim_y: block size for column-wise scaling (typically
  // 32)
  static void
  quantize(const void *input, void *output_rowwise, void *output_colwise,
           e8m0_t *scales_rowwise, e8m0_t *scales_colwise,
           size_t scales_rowwise_stride_dim0, size_t scales_rowwise_stride_dim1,
           size_t scales_colwise_stride_dim0, size_t scales_colwise_stride_dim1,
           size_t rows, size_t cols, DType input_dtype, DType output_dtype,
           size_t scale_dim_x = 32, size_t scale_dim_y = 32,
           ScaleCalculationMode scaling_mode = ScaleCalculationMode::FLOOR,
           cudaStream_t stream = 0) {

    // Check parameters
    assert((scale_dim_x == 1 || scale_dim_x == 32) &&
           (scale_dim_y == 1 || scale_dim_y == 32));
    assert(output_rowwise != nullptr || output_colwise != nullptr);

    if (output_rowwise)
      assert(scales_rowwise != nullptr);
    if (output_colwise)
      assert(scales_colwise != nullptr);

    // Calculate grid dimensions
    const size_t chunks_Y = DIVUP(rows, MXFP8_CHUNK_DIM_Y);
    const size_t chunks_X = DIVUP(cols, MXFP8_CHUNK_DIM_X);
    const size_t blocks_Y = DIVUP(chunks_Y, MXFP8_CHUNKS_PER_BLOCK_Y);
    const size_t blocks_X = DIVUP(chunks_X, MXFP8_CHUNKS_PER_BLOCK_X);

    const dim3 block(MXFP8_THREADS_PER_CHUNK);
    const dim3 grid(blocks_X, blocks_Y);

    // Create TMA descriptors
    alignas(64) CUtensorMap tensor_map_input{};
    alignas(64) CUtensorMap tensor_map_output_rowwise{};
    alignas(64) CUtensorMap tensor_map_output_colwise{};
    int32_t input_bits_per_elem = get_dtype_bits(input_dtype);
    int32_t output_bits_per_elem = get_dtype_bits(output_dtype);

    create_2D_tensor_map(tensor_map_input, const_cast<void *>(input),
                         input_dtype,
                         rows, cols,
                         MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X,
                         cols,                 // stride of "slowest moving" dim
                         input_bits_per_elem); // bits per elem in input

    if (output_rowwise) {
      create_2D_tensor_map(
          tensor_map_output_rowwise, output_rowwise, output_dtype,
          rows, cols,
          MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X,
          cols,                  // stride of "slowest moving" dim
          output_bits_per_elem); // bits per elem in output fp8e4m3
    }

    if (output_colwise) {
      create_2D_tensor_map(
          tensor_map_output_colwise, output_colwise, output_dtype,
          cols, rows,            // Swap for column major layout
          MXFP8_SHMEM_DIM_X, MXFP8_SHMEM_DIM_Y,
          rows,                  // stride of "slowest moving" dim
          output_bits_per_elem); // bits per elem in output fp8e4m3
    }

// Launch kernel based on input/output types and scaling dimensions
// Only compile kernel launches for SM90+
#if defined(__CUDACC__) &&                                                     \
    (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= MIN_CUDA_SM)

    // Use TMA and mbarrier instructions
#define LAUNCH_KERNEL(IType, OType, SCALE_Y, SCALE_X, ScalingMode)                          \
  mxfp8_quantize_kernel<IType, OType, SCALE_Y, SCALE_X, ScalingMode>                        \
      <<<grid, block, 0, stream>>>(                                            \
          tensor_map_input, tensor_map_output_rowwise,                         \
          tensor_map_output_colwise, scales_rowwise, scales_colwise, rows,     \
          cols, scales_rowwise_stride_dim0, scales_rowwise_stride_dim1,        \
          scales_colwise_stride_dim0, scales_colwise_stride_dim1);

    // Validate output dtype.
    if (output_dtype != DType::kFloat8E4M3) {
      printf("unsupported output dtype, must be fp8e4m3\n");
      exit(1);
    }

    if (scaling_mode == ScaleCalculationMode::FLOOR) {
      if (input_dtype == DType::kFloat32) {
        if (scale_dim_x == 32 && scale_dim_y == 32) {
          LAUNCH_KERNEL(float, fp8e4m3, 32, 32, ScaleCalculationMode::FLOOR);
        } else if (scale_dim_x == 32 && scale_dim_y == 1) {
          LAUNCH_KERNEL(float, fp8e4m3, 1, 32, ScaleCalculationMode::FLOOR);
        } else if (scale_dim_x == 1 && scale_dim_y == 32) {
          LAUNCH_KERNEL(float, fp8e4m3, 32, 1, ScaleCalculationMode::FLOOR);
        }
      } else if (input_dtype == DType::kBFloat16) {
        if (scale_dim_x == 32 && scale_dim_y == 32) {
          LAUNCH_KERNEL(bfloat16, fp8e4m3, 32, 32, ScaleCalculationMode::FLOOR);
        } else if (scale_dim_x == 32 && scale_dim_y == 1) {
          LAUNCH_KERNEL(bfloat16, fp8e4m3, 1, 32, ScaleCalculationMode::FLOOR);
        } else if (scale_dim_x == 1 && scale_dim_y == 32) {
          LAUNCH_KERNEL(bfloat16, fp8e4m3, 32, 1, ScaleCalculationMode::FLOOR);
        }
      } else {
        printf("unsupported input dtype, must be float32 or bfloat16\n");
        exit(1);
      }
    } else if (scaling_mode == ScaleCalculationMode::RCEIL) {
        if (input_dtype == DType::kFloat32) {
          if (scale_dim_x == 32 && scale_dim_y == 32) {
            LAUNCH_KERNEL(float, fp8e4m3, 32, 32, ScaleCalculationMode::RCEIL);
          } else if (scale_dim_x == 32 && scale_dim_y == 1) {
            LAUNCH_KERNEL(float, fp8e4m3, 1, 32, ScaleCalculationMode::RCEIL);
          } else if (scale_dim_x == 1 && scale_dim_y == 32) {
            LAUNCH_KERNEL(float, fp8e4m3, 32, 1, ScaleCalculationMode::RCEIL);
          }
        } else if (input_dtype == DType::kBFloat16) {
          if (scale_dim_x == 32 && scale_dim_y == 32) {
            LAUNCH_KERNEL(bfloat16, fp8e4m3, 32, 32, ScaleCalculationMode::RCEIL);
          } else if (scale_dim_x == 32 && scale_dim_y == 1) {
            LAUNCH_KERNEL(bfloat16, fp8e4m3, 1, 32, ScaleCalculationMode::RCEIL);
          } else if (scale_dim_x == 1 && scale_dim_y == 32) {
            LAUNCH_KERNEL(bfloat16, fp8e4m3, 32, 1, ScaleCalculationMode::RCEIL);
          }
        } else {
          printf("unsupported input dtype, must be float32 or bfloat16\n");
          exit(1);
        }
    } else {
      printf("unsupported scaling mode\n");
      exit(1);
    }

#undef LAUNCH_KERNEL

#endif
  }
};
