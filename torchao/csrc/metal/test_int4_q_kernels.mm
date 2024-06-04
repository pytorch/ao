#include <Metal/Metal.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>

/*
   This code is largely copy paste from Nikita Shulga's llm_experiments repo
*/

void fail(const std::string &str) {
  std::cerr << str << std::endl;
  abort();
}

void fail(const std::string &str1, const std::string &str2) {
  std::cerr << str1 << str2 << std::endl;
  abort();
}

id<MTLDevice> getMetalDevice() {
  NSArray *devices = [MTLCopyAllDevices() autorelease];
  if (devices.count == 0) {
    fail("Metal is not supported");
  }
  return devices[0];
}

id<MTLLibrary> compileLibraryFromSource(id<MTLDevice> device,
                                        const std::string &source) {
  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion3_1];
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                   options:options
                     error:&error];
  if (library == nil) {
    fail("Failed to compile: ", error.description.UTF8String);
  }
  return library;
}

id<MTLLibrary> compileLibraryFromFile(id<MTLDevice> device,
                                      const std::string &fname) {
  std::ifstream ifs(fname);
  std::stringstream ss;
  ss << ifs.rdbuf();
  ifs.close();
  return compileLibraryFromSource(device, ss.str());
}

id<MTLBuffer> allocSharedBuffer(id<MTLDevice> device, unsigned length) {
  id<MTLBuffer> rc = [device newBufferWithLength:length
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    fail("Can't allocate " + std::to_string(length) + " bytes on GPU");
  }
  return rc;
}

inline uint32_t float_as_int(float f) {
  union {
    float f;
    uint32_t i;
  } x;
  x.f = f;
  return x.i;
}

inline float int_as_float(uint32_t i) {
  union {
    float f;
    uint32_t i;
  } x;
  x.i = i;
  return x.f;
}

struct BFloat16 {
  BFloat16(float x) : val(float_as_int(x) >> 16) {}
  operator float() const { return int_as_float(val << 16); }

  uint16_t val;
};
using Float16 = _Float16;

template <unsigned groupSize> struct Int4MMBase {
  Int4MMBase(id<MTLDevice> device, const std::string &lib_name_, unsigned M_,
             unsigned N_, unsigned K_)
      : Int4MMBase(device, M_, N_, K_) {
    lib_name = lib_name_;
    lib = compileLibraryFromFile(device, lib_name + ".metal");
  }
  Int4MMBase(id<MTLDevice> device, unsigned M_, unsigned N_, unsigned K_)
      : M(M_), N(N_), K(K_), lib(nil) {
    allocBuffers(device);
  }

  virtual void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                               unsigned maxThreadsPerGroup) const {}

  void encodeMM(id<MTLCommandBuffer> cmdBuffer,
                id<MTLComputePipelineState> cpl) const {
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    std::vector<unsigned> sizes = {M, K, N, 0};
    const auto maxThreadsPerGroup =
        static_cast<decltype(M)>([cpl maxTotalThreadsPerThreadgroup]);
    [encoder setComputePipelineState:cpl];
    [encoder setBuffer:buf_A offset:0 atIndex:0];
    [encoder setBuffer:buf_B offset:0 atIndex:1];
    [encoder setBuffer:buf_scales offset:0 atIndex:2];
    [encoder setBuffer:buf_zero_point offset:0 atIndex:3];
    [encoder setBuffer:buf_C offset:0 atIndex:4];
    [encoder setBytes:sizes.data()
               length:sizeof(uint32_t) * sizes.size()
              atIndex:5];
    dispatchThreads(encoder, maxThreadsPerGroup);
    [encoder endEncoding];
  }

  template <typename T> void init() {
    T *a_ptr = reinterpret_cast<T *>([buf_A contents]);
    uint8_t *b_ptr = reinterpret_cast<uint8_t *>([buf_B contents]);
    T *c_ptr = reinterpret_cast<T *>([buf_C contents]);
    T *s_ptr = reinterpret_cast<T *>([buf_scales contents]);
    T *z_ptr = reinterpret_cast<T *>([buf_zero_point contents]);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> int_distrib(-8, 7);
    std::uniform_real_distribution<> real_distrib(-1.0, 1.0);

    for (unsigned idx = 0; idx < M * K; ++idx) {
      a_ptr[idx] = real_distrib(generator);
    }
    for (unsigned idx = 0; idx < N * K / 2; ++idx) {
      int32_t b0 = int_distrib(generator);
      int32_t b1 = int_distrib(generator);
      b_ptr[idx] = ((b1 + 8) << 4) | (b0 + 8);
    }
    for (unsigned idx = 0; idx < N * K / groupSize; ++idx) {
      s_ptr[idx] = (idx + 1.0) / N;
      z_ptr[idx] = 0;
    }
    for (unsigned idx = 0; idx < M * N; ++idx) {
      c_ptr[idx] = -1.0;
    }
  }

  template <typename T>
  bool validate(float atol_lim = 5e-4, float rtol_lim = 5e-3) const {
    T *a_ptr = reinterpret_cast<T *>([buf_A contents]);
    uint8_t *b_ptr = reinterpret_cast<uint8_t *>([buf_B contents]);
    T *c_ptr = reinterpret_cast<T *>([buf_C contents]);
    T *s_ptr = reinterpret_cast<T *>([buf_scales contents]);
    T *z_ptr = reinterpret_cast<T *>([buf_zero_point contents]);

    for (unsigned m = 0; m < M; m++) {
      for (unsigned n = 0; n < N; n++) {
        float expected = float(c_ptr[m * N + n]);
        const uint32_t k_block = (K + groupSize - 1) / groupSize;
        const T *A_ptr = a_ptr + m * K;

        float rc = 0.0;
        uint k = 0;
        for (uint32_t kb = 0; kb < k_block; kb++) {
          const T scale = s_ptr[(kb * N + n)];
          const T zero = z_ptr[(kb * N + n)] - scale * T(8);
          for (uint idx = 0; idx < groupSize && k < K; idx++, k++) {
            const auto a_val = float(A_ptr[k]);
            uint8_t b_val = b_ptr[(n * K + k) / 2];
            b_val = (k & 1) == 0 ? b_val & 0x0f : (b_val >> 4);
            rc += a_val * float(scale * T(b_val) + zero);
          }
        }

        auto atol = std::abs(rc - expected);
        auto rtol =
            atol / std::max(std::min(std::abs(expected), std::abs(rc)), 1e-6f);
        if (rtol > rtol_lim && atol > atol_lim) {
          std::cerr << "Error at:(" << m << ", " << n << ")\n";
          std::cerr << "Result " << expected << " vs expected " << rc
                    << " (atol=" << atol << " ,rtol=" << rtol << ") at " << m
                    << ":" << n << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  template <typename T> bool run_and_validate() {
    init<T>();
    id<MTLFunction> func = [lib
        newFunctionWithName:[NSString
                                stringWithFormat:@"int4pack_mv_%u_%s",
                                                 groupSize,
                                                 type_string<T>().c_str()]];
    if (func == nil) {
      fail("Can:t get function");
    }
    NSError *error = nil;
    auto cpl = [lib.device newComputePipelineStateWithFunction:func
                                                         error:&error];
    if (cpl == nil) {
      fail("Failed to construct pipeline state: ",
           error.description.UTF8String);
    }
    id<MTLCommandQueue> queue = [lib.device newCommandQueue];
    auto do_compute = ^() {
      @autoreleasepool {
        auto desc = [MTLCommandBufferDescriptor new];
        desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        id<MTLCommandBuffer> cmdBuffer =
            [queue commandBufferWithDescriptor:desc];
        encodeMM(cmdBuffer, cpl);
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
      }
    };

    do_compute();

    if (!validate<T>()) {
      return false;
    }
    return true;
  }

private:
  template <typename T> std::string type_string() const;
  template <> std::string type_string<BFloat16>() const { return "bfloat"; }
  template <> std::string type_string<float>() const { return "float"; }
  template <> std::string type_string<Float16>() const { return "half"; }
  void allocBuffers(id<MTLDevice> device, const unsigned elem_size = 4) {
    buf_A = allocSharedBuffer(device, M * K * elem_size);
    buf_B = allocSharedBuffer(device, N * K / 2);
    buf_C = allocSharedBuffer(device, M * N * elem_size);
    buf_scales = allocSharedBuffer(device, N * K / groupSize * elem_size);
    buf_zero_point = allocSharedBuffer(device, N * K / groupSize * elem_size);
  }

public:
  unsigned M, N, K;     // Input-output matirx dims
  id<MTLBuffer> buf_A;  // MxK elements
  id<MTLBuffer> buf_B;  // NxK elements
  id<MTLBuffer> buf_C;  // MxN elements
  id<MTLBuffer> buf_scales; // (K/groupSize)xNx2 elements
  id<MTLBuffer> buf_zero_point; // (K/groupSize)xNx2 elements
  id<MTLLibrary> lib;
  std::string lib_name;
};

template <unsigned groupSize> struct Int4MV : public Int4MMBase<groupSize> {
  using Int4MMBase<groupSize>::M;
  using Int4MMBase<groupSize>::N;
  Int4MV(id<MTLDevice> device, const std::string &lib_name_, unsigned M_,
         unsigned N_, unsigned K_)
      : Int4MMBase<groupSize>(device, lib_name_, M_, N_, K_) {
    if (M != 1) {
      fail("Value of M must be 1");
    }
  }
  void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                       unsigned maxThreadsPerGroup) const override {
    constexpr auto blockSize = 8;
    if (maxThreadsPerGroup < blockSize * blockSize) {
      throw std::runtime_error("Can't dispatch!");
    }
    [encoder dispatchThreads:MTLSizeMake(N / 4 * 32, 1, M)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  }
};

int main() {
  unsigned M, N, K;
  std::tie(M, N, K) = std::make_tuple(1, 4096, 4096);
  constexpr unsigned groupSize = 32;
  @autoreleasepool {
    id<MTLDevice> device = getMetalDevice();
    std::cout << "Using device " << device.name.UTF8String << std::endl;
    Int4MV<groupSize> int4mv_tester(device, "int4_quantized_kernels", M, N, K);

    // Benchmarks
    if (!int4mv_tester.run_and_validate<BFloat16>()) {
      fail("Failed to validate");
    };
  }
  return 0;
}
