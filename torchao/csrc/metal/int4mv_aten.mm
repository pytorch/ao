#include "int4mv_kernel.h"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSProfiler.h>

namespace torchao {
using Tensor = at::Tensor;

Tensor& int4mv_out_impl(const Tensor& A, const Tensor& B, int64_t groupSize, const Tensor& scalesAndZeros, Tensor& C) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(A.dtype() == at::kBFloat16 || A.dtype() == at::kHalf || A.dtype() == at::kFloat,
              __func__,
              " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == at::kChar, __func__, " : expect B to be int8 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == int(K / 2), __func__, " : expect B.size(1) == ", int(K / 2));

  TORCH_CHECK(scalesAndZeros.dim() == 3 && scalesAndZeros.size(1) == int(N / groupSize) && scalesAndZeros.size(2) == 2,
              __func__,
              ": expect scalesAndZeros to be 3d tensor with sizes [:, ",
              int(N / groupSize),
              ", 2]");

  TORCH_CHECK(N % 32 == 0 && K % 32 == 0);
  std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), 0};
  std::string A_scalar_type = at::native::mps::scalarToMetalTypeString(A.scalar_type());
  at::mps::MPSStream* mpsStream = at::mps::getCurrentMPSStream();

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
#if _CAPTURE_KERNEL
      if (at::mps::getMPSProfiler().isCaptureEnabled()) {
        at::mps::getMPSProfiler().startCapture("int4pack_vm");
      }
#endif
      id<MTLDevice> device = mpsStream->device();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:QUANTIZED_KERNEL]
                                                                  options:nil
                                                                    error:nil];
      id<MTLFunction> customQuantizedLinearFunction = [customKernelLibrary
        newFunctionWithName:[NSString stringWithFormat:@"int4pack_vm_%" PRId64 "_%s",
                                                       groupSize,
                                                       A_scalar_type.c_str()]];
      id<MTLComputePipelineState> quantizedPSO = [device newComputePipelineStateWithFunction:customQuantizedLinearFunction error:nil];
      [computeEncoder setComputePipelineState:quantizedPSO];
      at::native::mps::mtl_setBuffer(computeEncoder, A, 0);
      at::native::mps::mtl_setBuffer(computeEncoder, B, 1);
      at::native::mps::mtl_setBuffer(computeEncoder, scalesAndZeros, 2);
      at::native::mps::mtl_setBuffer(computeEncoder, C, 3);
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      [computeEncoder dispatchThreads:MTLSizeMake(N / 4 * 32, 1, M)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
#if _CAPTURE_KERNEL
      if (at::mps::getMPSProfiler().isCapturing()) {
        at::mps::getMPSProfiler().stopCapture(mpsStream);
      }
#endif
    }
  });
  return C;
}

Tensor int4mv_impl(const Tensor& A, const Tensor& B, int64_t groupSize, const Tensor& scalesAndZeros) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto C = at::empty({M, N}, A.options());
  return int4mv_out_impl(A, B, groupSize, scalesAndZeros, C);
}


TORCH_LIBRARY_IMPL(torchao, MPS, m) {
  m.impl("torchao::int4mv", &int4mv_impl);
  m.impl("torchao::int4mv.out", &int4mv_out_impl);
}

} // namespace torchao