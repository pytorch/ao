// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#define BUILD_MX_KERNELS_CUTLASS
#endif

#if defined(BUILD_MX_KERNELS_CUTLASS)

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"


#endif

namespace torchao {

#if defined(BUILD_MX_KERNELS_CUTLASS)
namespace {

using namespace cute;

template<typename Element>
constexpr int GetAlignment() {
    if constexpr (std::is_same_v<Element, cutlass::mx_float4_t<cutlass::float_e2m1_t>>)
        return 32;
    return 16;
}

void run_gemm(at::Tensor& a, at::Tensor& b, at::Tensor& a_scale,
             at::Tensor& b_scale, at::Tensor& out, int M, int K, int N) {

  using MmaTileShape        = Shape<_128,_128,_128>;
  using ClusterShape        = Shape<_1,_1,_1>;
  using PerSmTileShape_MNK  = Shape<_128,_128,_128>;

  // A matrix configuration
  using         ElementA    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 32;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 32;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
  using         ElementD    = cutlass::bfloat16_t;
  using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
  using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
  // Kernel functional config
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      PerSmTileShape_MNK, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,                                                   // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Reference device GEMM implementation type
  using StrideA   = typename Gemm::GemmKernel::StrideA;
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using StrideD   = typename Gemm::GemmKernel::StrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // Initialize strides using packed stride configuration
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

  // Initialize scale factor layouts using block scaled configuration
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  using DtypeA = typename ElementA::DataType;
  using DtypeB = typename ElementB::DataType;
  using DtypeScaleA = typename ElementA::ScaleFactorType;
  using DtypeScaleB = typename ElementB::ScaleFactorType;
  using DtypeOut = ElementD;

  Gemm gemm;

  auto A_ptr = reinterpret_cast<DtypeA*>(a.data_ptr());
  auto B_ptr = reinterpret_cast<DtypeB*>(b.data_ptr());
  auto SFA_ptr = reinterpret_cast<DtypeScaleA*>(a_scale.data_ptr());
  auto SFB_ptr = reinterpret_cast<DtypeScaleB*>(b_scale.data_ptr());
  auto out_ptr = reinterpret_cast<DtypeOut*>(out.data_ptr());

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    { // Mainloop arguments
      A_ptr, stride_A,
      B_ptr, stride_B,
      SFA_ptr, layout_SFA,
      SFB_ptr, layout_SFB
    },
    { // Epilogue arguments
      {1.0, 0.0},
      nullptr, StrideC{},  // No bias for now
      out_ptr, stride_D
    }
  };

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Cutlass cannot implement");
  // Allocate workspace memory
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = a.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Cutlass cannot initialize");

  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Cutlass cannot run", cutlass::cutlassGetStatusString(status));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

}
}
#endif

at::Tensor mx_fp4_bf16(at::Tensor a, at::Tensor b, at::Tensor a_scale,
                       at::Tensor b_scale) {
#if defined(BUILD_MX_KERNELS_CUTLASS)
  TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
  TORCH_CHECK(a_scale.is_cuda(), "a_scale must be CUDA tensor");
  TORCH_CHECK(b_scale.is_cuda(), "b_scale must be CUDA tensor");

  auto M = a.size(0);
  auto K = a.size(1) * 2;
  auto N = b.size(1);

  auto out =
      at::empty({M, N}, a.options().dtype(at::kBFloat16));

  run_gemm(a, b, a_scale, b_scale, out, M, K, N);
  return out;
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, __func__);
  return at::Tensor{};
#endif
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::mx_fp4_bf16", &mx_fp4_bf16);
}

} // namespace torchao
