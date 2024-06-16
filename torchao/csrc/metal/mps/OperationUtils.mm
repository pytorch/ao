#include "OperationUtils.h"

namespace torchao::mps {
  id<MTLBuffer> getMTLBufferStorage(const uint8_t * data, size_t nbytes) {
    return [MPSDevice::getInstance()->device() newBufferWithBytesNoCopy:(void*)data
                                                                length:nbytes
                                                                options:0
                                                            deallocator:nil];
  }
  static inline void checkSupportsBFloat16() {
    assert(isMacOS13OrNewer(MacOSVersion::MACOS_VER_14_0_PLUS) &&
                   "MPS bfloat16 type is supported on MacOS 14.0 or newer.");
  }
} // namespace torchao::mps