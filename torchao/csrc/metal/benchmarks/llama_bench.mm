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
#include <optional>

struct ModelArgs {
  int32_t dim = 4096;
  int32_t n_layers = 32;
  int32_t n_heads = 32;
  int32_t n_kv_heads = 32;
  int32_t vocab_size = 32000;
  int32_t multiple_of = 256;
  float ffn_dim_multiplier;
  float norm_eps = 1e-5;
  int32_t max_batch_size = 32;
  int32_t max_seq_len = 256;
  int32_t kv_cache_size = 256;
};

enum class FullyConnectedOpType : uint8_t {
  F32_QC4W = 0,
  F32_QC8W,
  BF16_QC4W,
  BF16_QC8W,
};

enum class ComputeType : uint8_t {
  F16 = 0,
  F32,
  QuantizedInt8,
};

namespace {

class MetalProfiler {
  public:
  MetalProfiler(std::string profile_name, id<MTLCommandQueue> queue) {
    capture_manager = [MTLCaptureManager sharedCaptureManager];
    auto capture_descriptor = [MTLCaptureDescriptor new];
    capture_descriptor.captureObject = queue;
    if (not [capture_manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
      std::cout << "Capturing to a GPU trace file isn't supported.\n";
    }
    capture_descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    capture_descriptor.outputURL =
        [NSURL fileURLWithPath:[NSString stringWithFormat:@"%s.gputrace",
                                                          profile_name.c_str()]];
    [capture_manager startCaptureWithDescriptor:capture_descriptor error:nil];
  }

  ~MetalProfiler() {
    [capture_manager stopCapture];
  }

  MTLCaptureManager* capture_manager;
};

void fail(const std::string &str) {
  std::cerr << str << std::endl;
  abort();
}

class BFloat16{};
class Float32{};

template <typename T>
struct type_traits;

template<>
struct type_traits<BFloat16> {
  static const std::string type_string;
  static const size_t elem_size = 2;
};
const std::string type_traits<BFloat16>::type_string="bfloat";

id<MTLDevice> getMetalDevice() {
  NSArray *devices = [MTLCopyAllDevices() autorelease];
  if (devices.count == 0) {
    fail("Metal is not supported");
  }
  return devices[0];
}

id<MTLBuffer> allocSharedBuffer(id<MTLDevice> device, unsigned length) {
  id<MTLBuffer> rc = [device newBufferWithLength:length
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    fail("Can't allocate " + std::to_string(length) + " bytes on GPU");
  }
  return rc;
}

class MetalShaderLibrary {
public:
  MetalShaderLibrary(id<MTLDevice> device, const std::string& file_name) {
    std::ifstream ifs(file_name);
    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();
    library = compileLibrary(device, ss.str());
  }
  MetalShaderLibrary(const MetalShaderLibrary&) = delete;

  id<MTLComputePipelineState> getPipelineStateForFunc(const std::string& fname) {
    auto cpl = cplMap[fname];
    if (cpl) {
      return cpl;
    }

    NSError* error = nil;
    id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
    if (func == nil) {
     fail("Failed to create function state object for: " + fname);
    }
    cpl = [[library device] newComputePipelineStateWithFunction:func error:&error];
    if (cpl == nil) {
      fail("Failed to created pipeline state object, error: " + std::string(error.description.UTF8String));
    }

    return cplMap[fname] = cpl;
  }

private:
  id<MTLLibrary> compileLibrary(id<MTLDevice> device, const std::string& src) {
    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
    [options setLanguageVersion:MTLLanguageVersion3_1];
    auto str = [NSString stringWithCString:src.c_str() encoding:NSASCIIStringEncoding];
    library = [device newLibraryWithSource:str options:options error:&error];
    if (library == nil) {
      fail("Failed to compile: " + std::string(error.description.UTF8String));
    }
    return library;
  }

  id<MTLLibrary> library = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
};

} // namespace

class Linear {
 public:
  explicit Linear(
      id<MTLDevice> device,
      const FullyConnectedOpType linear_type,
      const int32_t input_channels,
      const int32_t output_channels) : lib(device, "int4_quantized_kernels.metal") {

    linear_type_ = linear_type;
    auto M = benchmarking_batch_size_;
    auto N = output_channels;
    auto K = input_channels;
    auto group_size = 32;
    input_channels_ = input_channels; 
    output_channels_ = output_channels; 
    auto elem_size = type_traits<BFloat16>::elem_size;
    input = allocSharedBuffer(device, M * K * elem_size);
    weight = allocSharedBuffer(device, N * K / 2);
    output = allocSharedBuffer(device, M * N * elem_size);
    scales = allocSharedBuffer(device, N * K / group_size * elem_size);
    zero_points = allocSharedBuffer(device, N * K / group_size * elem_size);
  }

  Linear(const Linear& other) = delete;
  Linear& operator=(const Linear& other) = delete;
  Linear(Linear&& other) = delete; // Move constructor
  Linear& operator=(Linear&& other) noexcept =
      delete; // Move assignment operator

  ~Linear() {
  }

  void run_bench(id<MTLComputeCommandEncoder> encoder) {
    // Update this with picking appropriate kernel based on input params.
    std::string func = "int4pack_mv_32_bfloat";//" + std::to_string(group_size) + "_bfloat";
    auto cpl = lib.getPipelineStateForFunc(func);

    @autoreleasepool {
      // Every instance of linear is submitting and waiting
      unsigned int M = benchmarking_batch_size_;
      unsigned int N = output_channels_;
      unsigned int K = input_channels_;
      std::vector<unsigned> sizes = {M, K, N, 0};
      //const auto maxThreadsPerGroup =
      //    static_cast<decltype(M)>([cpl maxTotalThreadsPerThreadgroup]);
      [encoder setComputePipelineState:cpl];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:weight offset:0 atIndex:1];
      [encoder setBuffer:scales offset:0 atIndex:2];
      [encoder setBuffer:zero_points offset:0 atIndex:3];
      [encoder setBuffer:output offset:0 atIndex:4];
      [encoder setBytes:sizes.data()
                 length:sizeof(uint32_t) * sizes.size()
                atIndex:5];
      [encoder dispatchThreads:MTLSizeMake(N/4 * 32, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
  }

 private:
  FullyConnectedOpType linear_type_;
  int32_t benchmarking_batch_size_{1};
  int32_t input_channels_;
  int32_t output_channels_;
  //int32_t group_size;
  id<MTLBuffer> input; // MxK elements
  id<MTLBuffer> output; // NxK elements
  id<MTLBuffer> weight; // MxN elements
  id<MTLBuffer> scales; // (K/groupSize)xN elements
  id<MTLBuffer> zero_points; // (K/groupSize)xN elements
  MetalShaderLibrary lib;
};

class MultiHeadedAttention {
 public:
  explicit MultiHeadedAttention(
      id<MTLDevice> device,
      const ModelArgs& args,
      const FullyConnectedOpType linear_type) {
    int32_t input_channels = args.dim;
    int32_t output_channels = args.n_heads * (args.dim / args.n_heads);
    // int32_t head_dim = args.dim / args.n_heads;

    q_proj_.emplace(device, linear_type, input_channels, output_channels);
    k_proj_.emplace(device, linear_type, input_channels, output_channels);
    v_proj_.emplace(device, linear_type, input_channels, output_channels);
    o_proj_.emplace(device, linear_type, output_channels, input_channels);
  }

  MultiHeadedAttention(const MultiHeadedAttention& other) =
      delete; // Copy constructor
  MultiHeadedAttention& operator=(const MultiHeadedAttention& other) =
      delete; // Assignment operator
  MultiHeadedAttention(MultiHeadedAttention&& other) noexcept =
      delete; // Move constructor
  MultiHeadedAttention& operator=(MultiHeadedAttention&& other) noexcept =
      delete; // Move assignment operator

  ~MultiHeadedAttention() {
  }

  void run_bench(id<MTLComputeCommandEncoder> encoder) {
    q_proj_.value().run_bench(encoder);
    k_proj_.value().run_bench(encoder);
    v_proj_.value().run_bench(encoder);
    o_proj_.value().run_bench(encoder);
  }

 private:
  std::optional<Linear> q_proj_;
  std::optional<Linear> k_proj_;
  std::optional<Linear> v_proj_;
  std::optional<Linear> o_proj_;
};

class FeedForward {
 public:
  explicit FeedForward(
      id<MTLDevice> device,
      const ModelArgs& args,
      const FullyConnectedOpType linear_type) {
    int32_t hidden_dim = 4 * args.dim;
    int32_t n_hidden = 2 * hidden_dim / 3;
    int32_t mask = ~(args.multiple_of - 1);
    int32_t intermediate_size = (n_hidden + args.multiple_of - 1) & mask;

    w1_.emplace(device, linear_type, args.dim, intermediate_size);
    w3_.emplace(device, linear_type, args.dim, intermediate_size);
    w2_.emplace(device, linear_type, intermediate_size, args.dim);

  }

  FeedForward(const FeedForward& other) = delete;
  FeedForward& operator=(const FeedForward& other) = delete;
  FeedForward(FeedForward&& other) = delete; // Move constructor
  FeedForward& operator=(FeedForward&& other) noexcept =
      delete; // Move assignment operator

  ~FeedForward() {
    // xnn_delete_operator(silu_);
  }

  void run_bench(id<MTLComputeCommandEncoder> encoder) {
    w1_.value().run_bench(encoder);
    w2_.value().run_bench(encoder);
    w3_.value().run_bench(encoder);
  }

 private:
  std::optional<Linear> w1_;
  std::optional<Linear> w2_;
  std::optional<Linear> w3_;
};

class TransformerBlock {
 public:
  explicit TransformerBlock(
      id<MTLDevice> device,
      const ModelArgs& args,
      const FullyConnectedOpType linear_type)
      : multi_headed_attention_(device, args, linear_type),
        feedforward_(device, args, linear_type){}

  void run_bench(id<MTLComputeCommandEncoder> encoder) {
    multi_headed_attention_.run_bench(encoder);
    feedforward_.run_bench(encoder);
  }

 private:
  MultiHeadedAttention multi_headed_attention_;
  FeedForward feedforward_;
};

class Transformer {
 public:
  explicit Transformer(
      id<MTLDevice> device,
      const ModelArgs& args,
      const FullyConnectedOpType linear_type) {
    transformer_blocks_.reserve(args.n_layers);
    for (int i = 0; i < args.n_layers; ++i) {
      transformer_blocks_.emplace_back(
          std::make_unique<TransformerBlock>(device, args, linear_type));
    }

    // Not sure if we should quantize the last linear layer or not. For now
    // assuming we do.
    out_logits_.emplace(device, linear_type, args.dim, args.vocab_size);
  }

  Transformer(const Transformer& other) = delete;
  Transformer& operator=(const Transformer& other) = delete;
  Transformer(Transformer&& other) = delete; // Move constructor
  Transformer& operator=(Transformer&& other) noexcept =
      delete; // Move assignment operator

  void run_bench(id<MTLComputeCommandEncoder> encoder) {
    for (auto& transformer_block : transformer_blocks_) {
      transformer_block->run_bench(encoder);
    }
    out_logits_.value().run_bench(encoder);
  }

 private:
  std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks_;
  std::optional<Linear> out_logits_;
};

// #define BENCHMARK_FP32
static void benchmark_llama2_7b() {
  ModelArgs args;
  const int32_t kWarmupIterations = 10;
  const int32_t kIterations = 50;
// Need to benchmark pre-fill separately.
  id<MTLDevice> device = getMetalDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
#if defined(BENCHMARK_FP32)
  Transformer transformer(
      device, args, FullyConnectedOpType::F32_QC4W);
#else
  Transformer transformer(
      device, args, FullyConnectedOpType::BF16_QC4W);
#endif
  for (int i = 0; i < kWarmupIterations; ++i) {
    auto desc = [MTLCommandBufferDescriptor new];
    desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> cmdBuffer = [queue commandBufferWithDescriptor:desc];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    transformer.run_bench(encoder);
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
  }
  auto start_time = std::chrono::steady_clock::now();
  // MetalProfiler profiler("llama2_7b_4bit", queue);
  for (int i = 0; i < kIterations; ++i) {
    auto desc = [MTLCommandBufferDescriptor new];
    desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> cmdBuffer = [queue commandBufferWithDescriptor:desc];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    transformer.run_bench(encoder);
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
  }
  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();
  std::cout << "Elapsed time: " << elapsed_us << " microseconds" << std::endl;
  std::cout << "Elapsed time per iter(" << kIterations
            << "): " << elapsed_us / kIterations << " microseconds"
            << std::endl;
}

// BENCHMARK(benchmark_llama2_7b)->Unit(benchmark::kMicrosecond)->UseRealTime();
int main(int argc, char** argv) {
  @autoreleasepool {
    benchmark_llama2_7b();
  }
}
