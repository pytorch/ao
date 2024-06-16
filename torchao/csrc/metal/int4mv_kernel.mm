#include "int4mv_kernel.h"
#include "mps/MPSStream.h"
#include "mps/OperationUtils.h"
namespace torchao {

void int4mv(const uint8_t * A, const uint8_t * B,  uint32_t groupSize, const uint8_t * scalesAndZeros, uint8_t * C, std::array<uint32_t, 4> sizes, std::array<uint64_t, 4> nbytes, std::string A_scalar_type) {
  mps::MPSStream* mpsStream = mps::getCurrentMPSStream();
  uint32_t M = sizes[0];
  uint32_t K = sizes[1];
  uint32_t N = sizes[2];

  uint32_t A_nbytes = nbytes[0];
  uint32_t B_nbytes = nbytes[1];
  uint32_t scalesAndZeros_nbytes = nbytes[2];
  uint32_t C_nbytes = nbytes[3];
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
// #if _CAPTURE_KERNEL
//       if (getMPSProfiler().isCaptureEnabled()) {
//         getMPSProfiler().startCapture(fmt::format("int8pack_mm_{}x{}x{}", M, N, K), mpsStream);
//       }
// #endif
      id<MTLDevice> device = mpsStream->device();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:QUANTIZED_KERNEL]
                                                                  options:nil
                                                                    error:nil];
      id<MTLFunction> customQuantizedLinearFunction = [customKernelLibrary
        newFunctionWithName:[NSString stringWithFormat:@"int4pack_vm_%u_%s",
                                                       groupSize,
                                                       A_scalar_type.c_str()]];
      id<MTLComputePipelineState> quantizedPSO = [device newComputePipelineStateWithFunction:customQuantizedLinearFunction error:nil];
      [computeEncoder setComputePipelineState:quantizedPSO];
      [computeEncoder setBuffer:mps::getMTLBufferStorage(A, A_nbytes) offset:0 atIndex:0];
      [computeEncoder setBuffer:mps::getMTLBufferStorage(B, B_nbytes) offset:0 atIndex:1];
      [computeEncoder setBuffer:mps::getMTLBufferStorage(scalesAndZeros, scalesAndZeros_nbytes) offset:0 atIndex:2];
      [computeEncoder setBuffer:mps::getMTLBufferStorage(C, C_nbytes) offset:0 atIndex:3];
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      [computeEncoder dispatchThreads:MTLSizeMake(N / 4 * 32, 1, M)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
// #if _CAPTURE_KERNEL
//       if (getMPSProfiler().isCapturing()) {
//         getMPSProfiler().stopCapture(mpsStream);
//       }
// #endif
    }
  });
}

} // namespace torchao