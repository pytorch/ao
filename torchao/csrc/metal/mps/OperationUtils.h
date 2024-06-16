#include <numeric>

#include <Foundation/Foundation.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "MPSDevice.h"
namespace torchao::mps {
id<MTLBuffer> getMTLBufferStorage(const uint8_t *data, size_t nbytes);
static inline void checkSupportsBFloat16();
} // namespace torchao::mps