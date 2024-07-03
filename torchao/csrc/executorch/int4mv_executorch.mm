#include "metal/int4mv_kernel.h"
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/backends/apple/mps/runtime/operations/OperationUtils.h>
#include <executorch/backends/apple/mps/runtime/MPSStream.h>
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>


namespace torch {
namespace executor {
namespace native {

using RuntimeContext = torch::executor::RuntimeContext;
using ScalarType = torch::executor::ScalarType;
Tensor& _int4mv_out(
  RuntimeContext& ctx,
  const Tensor& A, 
  const Tensor& B,
  int64_t groupSize,
  const Tensor& scalesAndZeros, 
  Tensor& C) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  ET_CHECK_MSG(A.scalar_type() == ScalarType::BFloat16 || A.scalar_type() == ScalarType::Half || A.scalar_type() == ScalarType::Float,
                "%s : expect A to be either 32-bit or 16-bit float tensor.",
                __func__
  );
  ET_CHECK_MSG(A.dim() == 2, "%s : expect A to be 2D tensor.", __func__);

  ET_CHECK_MSG(B.scalar_type() == ScalarType::Char, "%s : expect B to be int8 tensor.", __func__);
  ET_CHECK_MSG(B.size(1) == int(K / 2), "%s : expect B.size(1) == %zd", __func__, int(K / 2));

  std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), 0};
  std::string A_scalar_type = mps::delegate::scalarToMetalTypeString(A.scalar_type());
  mps::delegate::MPSStream* mpsStream = mps::delegate::getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLDevice> device = mpsStream->device(); // should be mpsStream->device()
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:torchao::QUANTIZED_KERNEL]
                                                                  options:nil
                                                                    error:nil];
      id<MTLFunction> customQuantizedLinearFunction = [customKernelLibrary
        newFunctionWithName:[NSString stringWithFormat:@"int4pack_vm_%" PRId64 "_%s",
                                                       groupSize,
                                                       A_scalar_type.c_str()]];
      id<MTLComputePipelineState> quantizedPSO = [device newComputePipelineStateWithFunction:customQuantizedLinearFunction error:nil];
      [computeEncoder setComputePipelineState:quantizedPSO];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(A) offset:0 atIndex:0];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(B) offset:0 atIndex:1];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(scalesAndZeros) offset:0 atIndex:2];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(C) offset:0 atIndex:3];
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      [computeEncoder dispatchThreads:MTLSizeMake(N / 4 * 32, 1, M)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    ET_CHECK(mpsStream->synchronize(mps::delegate::SyncType::COMMIT_AND_WAIT) == Error::Ok);
  });
  return C;
}

EXECUTORCH_LIBRARY(
    torchao,
    "int4mv.out",
    _int4mv_out);
} // namespace native
} // namespace executor
} // namespace torch

