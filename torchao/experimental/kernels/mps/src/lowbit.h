// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdlib>
#include <string>

#include <torchao/experimental/kernels/mps/src/common.h>
#include <torchao/experimental/kernels/mps/src/dispatch.h>
#include <torchao/experimental/kernels/mps/src/metal_shader_lib.h> // metal_lowbit_quantized_lib
#include <torchao/experimental/kernels/mps/src/packing.h>

namespace torchao::kernels::mps::lowbit {
namespace {

template <int nbit>
struct LowBitConfig {};

template <>
struct LowBitConfig<1> {
  static constexpr std::string_view func_prefix = "int1pack_mm_";
  static constexpr auto packing_fn = packing::pack<1>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm;
  // GEMM kernel for M > 1 (prefill)
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_1bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

template <>
struct LowBitConfig<2> {
  static constexpr std::string_view func_prefix = "int2pack_mm_";
  static constexpr auto packing_fn = packing::pack<2>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm_Mr1xNr4_per_TG;
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_2bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

template <>
struct LowBitConfig<3> {
  static constexpr std::string_view func_prefix = "int3pack_mm_";
  static constexpr auto packing_fn = packing::pack<3>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm_Mr1xNr4_per_TG;
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_3bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

template <>
struct LowBitConfig<4> {
  static constexpr std::string_view func_prefix = "int4pack_mm_";
  static constexpr auto packing_fn = packing::pack<4>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm_Mr1xNr4_per_TG;
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_4bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

template <>
struct LowBitConfig<5> {
  static constexpr std::string_view func_prefix = "int5pack_mm_";
  static constexpr auto packing_fn = packing::pack<5>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm;
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_5bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

template <>
struct LowBitConfig<6> {
  static constexpr std::string_view func_prefix = "int6pack_mm_";
  static constexpr auto packing_fn = packing::pack<6>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm;
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_6bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

template <>
struct LowBitConfig<7> {
  static constexpr std::string_view func_prefix = "int7pack_mm_";
  static constexpr auto packing_fn = packing::pack<7>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm;
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_7bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

template <>
struct LowBitConfig<8> {
  static constexpr std::string_view func_prefix = "int8pack_mm_";
  static constexpr auto packing_fn = packing::pack<8>;
  static constexpr auto dispatch_fn = dispatch::dispatch_mm;
  static constexpr std::string_view gemm_func_prefix = "intNgemm_mm_8bit_";
  static constexpr auto gemm_dispatch_fn = dispatch::dispatch_gemm;
};

using DispatchFn =
    void (*)(id<MTLComputeCommandEncoder>, int32_t, int32_t, int32_t, int32_t);

inline void linear_lowbit_quant_weights_mps_impl(
    std::pair<id<MTLBuffer>, size_t> a_buf_offset,
    std::pair<id<MTLBuffer>, size_t> b_buf_offset,
    std::pair<id<MTLBuffer>, size_t> s_buf_offset,
    std::pair<id<MTLBuffer>, size_t> z_buf_offset,
    std::pair<id<MTLBuffer>, size_t> out_buf_offset,
    int32_t M,
    int32_t K,
    int32_t N,
    const std::string shader_func,
    DispatchFn dispatch_fn) {
  std::array<uint32_t, 4> sizes = {
      static_cast<uint32_t>(M),
      static_cast<uint32_t>(K),
      static_cast<uint32_t>(N),
      0};

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_block(mpsStream, ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> cpl =
          metal_lowbit_quantized_lib.getPipelineStateForFunc(shader_func);
      const auto maxThreadsPerGroup = [cpl maxTotalThreadsPerThreadgroup];
      [computeEncoder setComputePipelineState:cpl];
      [computeEncoder setBuffer:a_buf_offset.first offset:a_buf_offset.second atIndex:0];
      [computeEncoder setBuffer:b_buf_offset.first offset:b_buf_offset.second atIndex:1];
      [computeEncoder setBuffer:s_buf_offset.first offset:s_buf_offset.second atIndex:2];
      [computeEncoder setBuffer:z_buf_offset.first offset:z_buf_offset.second atIndex:3];
      [computeEncoder setBuffer:out_buf_offset.first offset:out_buf_offset.second atIndex:4];
      [computeEncoder setBytes:sizes.data()
                        length:sizeof(uint32_t) * sizes.size()
                       atIndex:5];
      dispatch_fn(computeEncoder, maxThreadsPerGroup, M, N, K);
      optionally_wait_for_command_completion(mpsStream);
    }
  });
}

template <int nbit>
std::tuple<const std::string, DispatchFn> get_shader_func_and_dispatch(
    int64_t qGroupSize,
    const std::string_view type_str,
    int32_t M,
    int32_t N,
    int32_t K) {
  // qmv (decode, M=1) is available for all bit widths 1-8.
  if (M == 1 && N % 8 == 0 && K % 512 == 0) {
    return std::make_tuple(
        std::string("qmv_fast_") + std::to_string(nbit) + "bit_" +
            std::to_string(qGroupSize) + "_" + std::string(type_str),
        dispatch::dispatch_qmv);
  }
  if (M == 1) {
    // qmv_impl handles generic N (any N, not just aligned) via guard logic
    return std::make_tuple(
        std::string("qmv_impl_") + std::to_string(nbit) + "bit_" +
            std::to_string(qGroupSize) + "_" + std::string(type_str),
        dispatch::dispatch_qmv);
  }
  // For M > 1, use the simdgroup-tiled GEMM kernel (prefill).
  // The GEMM kernel requires K to be a multiple of BLOCK_SIZE_K (32) and
  // group_size to be a multiple of BLOCK_SIZE_K (32).  Fall back to the
  // existing per-bitwidth pack_mm path for unaligned K or group_size.
  // Set TORCHAO_MPS_DISABLE_SIMDGROUP_GEMM=1 to force the fallback path
  // (for benchmarking old vs new prefill kernels).
  if (K % 32 == 0 && qGroupSize % 32 == 0) {
    const char* env = std::getenv("TORCHAO_MPS_DISABLE_SIMDGROUP_GEMM");
    if (env == nullptr || std::string(env) != "1") {
      return std::make_tuple(
          std::string(LowBitConfig<nbit>::gemm_func_prefix) +
              std::to_string(qGroupSize) + "_" + std::string(type_str),
          LowBitConfig<nbit>::gemm_dispatch_fn);
    }
  }
  // Fallback: per-bitwidth pack_mm path (no simdgroup MMA).
  // Only has pre-compiled instantiations for {32, 64, 128, 256}.
  return std::make_tuple(
      std::string(LowBitConfig<nbit>::func_prefix) + std::to_string(qGroupSize) +
          "_" + std::string(type_str),
      LowBitConfig<nbit>::dispatch_fn);
}

// LowBit Quantized Weights Linear on Metal
template <int nbit>
void linear_lowbit_quant_weights_mps(
    std::pair<id<MTLBuffer>, size_t> a_buf_offset,
    std::pair<id<MTLBuffer>, size_t> b_buf_offset,
    int64_t qGroupSize,
    std::pair<id<MTLBuffer>, size_t> s_buf_offset,
    std::pair<id<MTLBuffer>, size_t> z_buf_offset,
    std::pair<id<MTLBuffer>, size_t> out_buf_offset,
    int32_t M,
    int32_t K,
    int32_t N,
    const std::string_view type_str) {
  assert(K % 8 == 0);
  assert(N % 4 == 0 || M == 1);
  assert(qGroupSize > 0);
  assert(qGroupSize % 32 == 0);
  // The fallback pack_mm path only has pre-compiled instantiations for
  // {32, 64, 128, 256}.  When simdgroup GEMM is disabled (or K is not a
  // multiple of 32), assert the stricter restriction.
  {
    const char* env = std::getenv("TORCHAO_MPS_DISABLE_SIMDGROUP_GEMM");
    const bool simdgroup_disabled = (env != nullptr && std::string(env) == "1");
    const bool using_fallback = simdgroup_disabled || (K % 32 != 0);
    const bool using_qmv = (M == 1);
    if (using_fallback && !using_qmv) {
      assert(
          qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
          qGroupSize == 256);
    }
  }
  std::tuple<const std::string, DispatchFn> shader_func_and_dispatch =
      get_shader_func_and_dispatch<nbit>(qGroupSize, type_str, M, N, K);
  const std::string shader_func = std::get<0>(shader_func_and_dispatch);
  const DispatchFn dispatch_fn = std::get<1>(shader_func_and_dispatch);

  return linear_lowbit_quant_weights_mps_impl(
      a_buf_offset,
      b_buf_offset,
      s_buf_offset,
      z_buf_offset,
      out_buf_offset,
      M,
      K,
      N,
      shader_func,
      dispatch_fn);
}

} // namespace

// LowBit Quantized Weights Linear & Packing on Metal
template <int nbit>
struct LowBitQuantWeights {
  static constexpr auto linear = linear_lowbit_quant_weights_mps<nbit>;
  static constexpr auto pack = LowBitConfig<nbit>::packing_fn;
};

} // namespace torchao::kernels::mps::lowbit
