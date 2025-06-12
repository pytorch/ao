#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include "compute_sparse_tile.h"
#include "sparse24_metadata.h"
#include "warp_tensor.h"

using namespace torchao;

namespace {
// ############################################
// # CUSPARSELT - 16bits
// ############################################
template <typename P>
__global__ void sparse24_sm90_cusparselt16bits_sparsify_kernel(P p);
struct MetadataCusparseLt16bits {
  static constexpr auto kBlockSize0 = 64;
  static constexpr auto kBlockSize1 = 64;
  static constexpr auto kNumWarpsPerCTA = 2;

  template <typename ElementOut>
  static std::tuple<at::Tensor, at::Tensor> createTensors(at::Tensor input) {
    return MetadataCusparseLt16bitsSm90::template createTensors<ElementOut>(
        input);
  }

  template <typename P>
  CUTLASS_DEVICE static void run(P p) {
    using Element = typename P::ElementIn; // same as ElementOut

    constexpr int kSmemStride0 = kBlockSize1 / 8 + 1;
    __shared__ uint32_t smem[kBlockSize0 * kSmemStride0];

    int block_id = blockIdx.x;
    int num_blocks_rows = p.n_rows / kBlockSize0;
    int num_blocks_cols = p.n_cols / kBlockSize1;

    int block_row = (block_id / num_blocks_cols) * kBlockSize0;
    int block_col = (block_id % num_blocks_cols) * kBlockSize1;

    int warp_id = threadIdx.x / 32;
    WarpTensor<Element, 4, 64> load_dense_tensor;
    int warp_row = warp_id * load_dense_tensor.kRows;
    CUTLASS_PRAGMA_UNROLL
    for (int it_row = 0; it_row < kBlockSize0;
         it_row += kNumWarpsPerCTA * load_dense_tensor.kRows) {
      CUTLASS_PRAGMA_UNROLL
      for (int it_col = 0; it_col < kBlockSize1;
           it_col += load_dense_tensor.kCols) {
        // gmem -> RF
        load_dense_tensor.load(
            p.input_ptr + (it_row + warp_row + block_row) * p.input_s0 +
                it_col + block_col,
            p.input_s0);

        // RF -> RF (sparsify)
        auto [packed, mdata] = load_dense_tensor.sparsify_pack(p.algo);

        // RF -> RF (apply sparsity)
        packed.data = p.activation(packed.data);

        // RF -> gmem (packed data)
        packed.store(
            p.output_ptr + (it_row + warp_row + block_row) * p.output_s0 +
                (it_col + block_col) / 2,
            p.output_s0);

        // RF -> smem (mdata)
        mdata.template store_32bits<kSmemStride0, 1>(
            smem + (warp_row + it_row) * kSmemStride0 + it_col / 8);
      }
    }
    __syncthreads();

    WarpTensor<uint8_t, 16, 32 / 8> mdata_tensor;
    int warp_col = warp_id * (8 * mdata_tensor.kCols);
    static_assert(kBlockSize0 % mdata_tensor.kRows == 0);
    static_assert(kBlockSize1 % mdata_tensor.kCols == 0);
    CUTLASS_PRAGMA_UNROLL
    for (int it_row = 0; it_row < kBlockSize0; it_row += mdata_tensor.kRows) {
      CUTLASS_PRAGMA_UNROLL
      for (int it_col = 0; it_col < kBlockSize1;
           it_col += kNumWarpsPerCTA * (8 * mdata_tensor.kCols)) {
        mdata_tensor.template load_32bits<kSmemStride0, 1>(
            smem + it_row * kSmemStride0 + (it_col + warp_col) / 8);

        int current_col = warp_col + it_col + block_col;
        int current_row = it_row + block_row;
        int idx = (current_col / 32) * 256;
        idx +=
            ((current_row % 8) * 8 + ((current_row % 64) / 16) * 64 +
             (current_row / 64) * 8 * p.n_cols);
        store_metadata_reordered(mdata_tensor, p.mdata_ptr + idx);
      }
    }
  }

  template <typename P>
  static void launch_kernel(P p) {
    TORCH_CHECK(
        p.scale_ptr == nullptr, "cusparselt kernel does not support scaling");
    int num_blocks = cutlass::ceil_div(p.n_cols, kBlockSize1) *
        cutlass::ceil_div(p.n_rows, kBlockSize0);
    sparse24_sm90_cusparselt16bits_sparsify_kernel<P>
        <<<num_blocks,
           kNumWarpsPerCTA * 32,
           0,
           at::cuda::getCurrentCUDAStream()>>>(p);
  }
};

template <typename P>
__global__ void sparse24_sm90_cusparselt16bits_sparsify_kernel(P p) {
  MetadataCusparseLt16bits::run(p);
}

// ############################################
// # CUTLASS - 8bits
// ############################################
template <typename P>
__global__ void sparse24_sm90_cutlass8bits_sparsify_kernel(P p);

struct MetadataCutlass8bits {
  static constexpr int64_t kBlockSize0 = 32;
  static constexpr int64_t kBlockSize1 = 128;
  static constexpr int64_t kNumWarpsPerCTA = 2;
  static constexpr int64_t kThreadsPerCTA = kNumWarpsPerCTA * 32;

  template <typename ElementOut>
  static std::tuple<at::Tensor, at::Tensor> createTensors(at::Tensor input) {
    TORCH_CHECK(input.size(0) % kBlockSize0 == 0);
    TORCH_CHECK(input.size(1) % kBlockSize1 == 0);
    return MetadataCutlass8bitsSm90::template createTensors<ElementOut>(input);
  }

  template <typename P>
  CUTLASS_DEVICE static void run(P p) {
    using ElementIn = typename P::ElementIn;
    using ElementOut = typename P::ElementOut;
    using ElementScale = typename P::ElementScale;

    __shared__ ElementScale smem_scales[kBlockSize0];

    int block_id = blockIdx.x;
    int num_blocks_rows = p.n_rows / kBlockSize0;
    int num_blocks_cols = p.n_cols / kBlockSize1;

    int block_row = (block_id / num_blocks_cols) * kBlockSize0;
    int block_col = (block_id % num_blocks_cols) * kBlockSize1;

    int warp_id = threadIdx.x / 32;

    if (p.scale_ptr) {
      int thread_row = threadIdx.x * 4;
      cutlass::arch::cp_async<
          sizeof(ElementScale[4]),
          cutlass::arch::CacheOperation::Global>(
          smem_scales + thread_row,
          p.scale_ptr + block_row + thread_row,
          thread_row < kBlockSize0);
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kBlockSize0; i += kThreadsPerCTA) {
        int row = i * kThreadsPerCTA + threadIdx.x;
        if (row < kBlockSize0) {
          smem_scales[row] = ElementScale(1);
        }
      }
    }
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    WarpTensor<ElementIn, 2, 128> load_dense_tensor;
    int warp_row = warp_id * load_dense_tensor.kRows;
    CUTLASS_PRAGMA_UNROLL
    for (int it_row = 0; it_row < kBlockSize0;
         it_row += kNumWarpsPerCTA * load_dense_tensor.kRows) {
      CUTLASS_PRAGMA_UNROLL
      for (int it_col = 0; it_col < kBlockSize1;
           it_col += load_dense_tensor.kCols) {
        // gmem -> RF
        load_dense_tensor.load(
            p.input_ptr + (it_row + warp_row + block_row) * p.input_s0 +
                it_col + block_col,
            p.input_s0);

        // RF -> RF (sparsify)
        auto [packed, mdata] = load_dense_tensor.sparsify_pack(p.algo);

        // RF -> RF (cvt to f32, activation and then scale)
        auto packedf32 = packed.template to<float>();
        packedf32.data = p.activation(packedf32.data);
        auto scale =
            (1 / smem_scales[it_row + warp_row + packedf32.thread_row()]);
        packedf32.data = packedf32.data * scale;

        // RF -> RF (convert to fp8)
        auto packedCvt = packedf32.template to<ElementOut>();

        // RF -> gmem (packed data)
        packedCvt.store(
            p.output_ptr + (it_row + warp_row + block_row) * p.output_s0 +
                (it_col + block_col) / 2,
            p.output_s0);

        // RF -> gmem (mdata)
        constexpr int kStrideRow = 16;
        int col = (it_col + block_col);
        mdata.store(
            p.mdata_ptr + (it_row + warp_row + block_row) * kStrideRow +
                (col / 128 * p.n_rows * 16) + (col % 128) / 8,
            16);
      }
    }
  }

  template <typename P>
  static void launch_kernel(P p) {
    int num_blocks = cutlass::ceil_div(p.n_cols, kBlockSize1) *
        cutlass::ceil_div(p.n_rows, kBlockSize0);
    sparse24_sm90_cutlass8bits_sparsify_kernel<P>
        <<<num_blocks,
           kNumWarpsPerCTA * 32,
           0,
           at::cuda::getCurrentCUDAStream()>>>(p);
  }
};

template <typename P>
__global__ void __launch_bounds__(MetadataCutlass8bits::kThreadsPerCTA, 32)
    sparse24_sm90_cutlass8bits_sparsify_kernel(P p) {
  MetadataCutlass8bits::run(p);
}

template <
    typename _ElementIn,
    typename _ElementOut,
    typename _Algorithm,
    typename _PostSparsityActivation>
struct SparsifyKernelParams {
  using ElementIn = _ElementIn;
  using ElementOut = _ElementOut;
  using ElementScale = float;
  using Algorithm = _Algorithm;
  using PostSparsityActivation = _PostSparsityActivation;

  Algorithm algo;
  PostSparsityActivation activation;

  ElementIn const* input_ptr = nullptr;
  int64_t input_s0 = -1;
  ElementOut* output_ptr = nullptr;
  int64_t output_s0 = -1;
  uint8_t* mdata_ptr = nullptr;
  ElementScale* scale_ptr = nullptr;
  int64_t n_rows = -1;
  int64_t n_cols = -1;
  uint16_t* positive_count_ptr = nullptr;
};

template <
    typename MetadataFormat,
    typename ElementIn,
    typename ElementOut,
    typename PostSparsityActivation>
std::tuple<at::Tensor, at::Tensor> sparse24_sm90_sparsify_specialized(
    at::Tensor input,
    PostSparsityActivation activation,
    std::string sp_selection_algo,
    std::optional<at::Tensor> scale) {
  std::optional<at::cuda::CUDAGuard> device_guard;
  TORCH_CHECK(input.is_cuda(), "All tensors must be on GPU");
  device_guard.emplace(input.device());

  TORCH_CHECK(input.dim() == 2, "Can only sparsify 2d tensors");
  TORCH_CHECK(
      input.stride(1) == 1,
      "Can only sparsify contiguous tensors. Sparsify the transpose otherwise.");
  TORCH_CHECK(input.size(1) % 32 == 0);
  if (scale.has_value()) {
    TORCH_CHECK(scale->dim() == 2);
    TORCH_CHECK(
        scale->size(0) == input.size(0), "only row-wise scale is supported");
    TORCH_CHECK(scale->size(1) == 1);
    TORCH_CHECK(scale->is_contiguous());
    TORCH_CHECK(scale->scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(scale->device() == input.device());
  }

  int n_rows = input.size(0);
  int n_cols = input.size(1);

  // Half the storage + 1 bit per element in original tensor (metadata)
  at::Tensor packed, mdata;
  std::tie(packed, mdata) =
      MetadataFormat::template createTensors<ElementOut>(input);

  bool kernel_launched = false;
  auto launchKernel = [&](auto algo, std::string const& algo_name) {
    if (algo_name == sp_selection_algo) {
      kernel_launched = true;
      using Params = SparsifyKernelParams<
          ElementIn,
          ElementOut,
          decltype(algo),
          decltype(activation)>;
      Params p;
      p.algo = algo;
      p.activation = activation;
      p.input_ptr = ((ElementIn const*)input.data_ptr());
      p.input_s0 = input.stride(0);
      p.output_ptr = (ElementOut*)(packed.data_ptr());
      p.output_s0 = input.size(1) / 2;
      p.mdata_ptr = (uint8_t*)(mdata.data_ptr());
      p.scale_ptr = (float*)(scale.has_value() ? scale->data_ptr() : nullptr);
      p.n_rows = n_rows;
      p.n_cols = n_cols;

      MetadataFormat::launch_kernel(p);
    }
  };
  named_algorithms_oneway(launchKernel);
  TORCH_CHECK(kernel_launched, "Unknown algorithm \"", sp_selection_algo, "\"");
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(packed, mdata);
}

struct SquaredReLU {
  template <typename T, int N>
  CUTLASS_DEVICE cutlass::Array<T, N> operator()(cutlass::Array<T, N> x) const {
    cutlass::multiplies<cutlass::Array<T, N>> mul;
    cutlass::maximum<cutlass::Array<T, N>> max;
    x = max(x, T(0));
    x = mul(x, x);
    return x;
  }
};

std::tuple<at::Tensor, at::Tensor> sparse24_sm90_sparsify(
    at::Tensor input,
    std::string metadata_fmt,
    std::string activation,
    std::string sp_selection_algo,
    std::optional<at::ScalarType> out_dtype_,
    std::optional<at::Tensor> scale) {
  auto out_dtype =
      out_dtype_.has_value() ? out_dtype_.value() : input.scalar_type();

  auto runTypedWithAct =
      [&](auto in_type, auto out_type, auto mdatafmt, auto act) {
        using ElementIn = decltype(in_type);
        using ElementOut = decltype(out_type);
        return sparse24_sm90_sparsify_specialized<
            decltype(mdatafmt),
            ElementIn,
            ElementOut>(input, act, sp_selection_algo, scale);
      };

  auto runTyped = [&](auto in_type, auto out_type, auto mdatafmt) {
    if (activation == "identity") {
      return runTypedWithAct(in_type, out_type, mdatafmt, Identity());
    } else if (activation == "srelu") {
      return runTypedWithAct(in_type, out_type, mdatafmt, SquaredReLU());
    } else {
      TORCH_CHECK(false, "Unknown activation:", activation);
    }
  };

  TORCH_CHECK(metadata_fmt == "cusparselt" || metadata_fmt == "cutlass");
  TORCH_CHECK(
      !scale.has_value() || scale->scalar_type() == at::ScalarType::Float);
  if (metadata_fmt == "cusparselt") {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Half ||
        input.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(out_dtype == input.scalar_type());
    if (input.scalar_type() == at::ScalarType::Half) {
      return runTyped(
          cutlass::half_t(), cutlass::half_t(), MetadataCusparseLt16bits());
    } else {
      return runTyped(
          cutlass::bfloat16_t(),
          cutlass::bfloat16_t(),
          MetadataCusparseLt16bits());
    }
  } else if (metadata_fmt == "cutlass") {
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(out_dtype == at::ScalarType::Float8_e4m3fn);
    return runTyped(
        cutlass::bfloat16_t(), cutlass::float_e4m3_t(), MetadataCutlass8bits());
  }
  TORCH_CHECK(false, "Unknown metadata format: '", metadata_fmt, "'")
}
} // namespace

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchao::sparse24_sm90_sparsify"),
      TORCH_FN(sparse24_sm90_sparsify));
}
