#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "ComputeSparseTile.cuh"
#include "SparseSemiStructuredPack.cuh"
#include <cuda_runtime.h>

namespace torchao {

struct MetadataCutlass {
  // Layout needed to run 2:4 gemms in CUTLASS
  // There is basically a hardware specific value for every
  // 32x32 dense tile (1024 bits). Then these tiles are
  // stored in a Column-Major fashion
  ElementInputE *_meta;
  ElementInputE *_meta_trans;
  int64_t _meta_reordered_sy;
  int64_t _meta_trans_reordered_sx;

  // Define create_compressed_representation
  static std::tuple<at::Tensor, // return value of the function
                    at::Tensor, // packed
                    at::Tensor  // packed_meta
                    >
  create_compressed_representation(int rows, int cols, at::Tensor const &like) {
    TORCH_CHECK(like.scalar_type() == at::ScalarType::Half ||
                like.scalar_type() == at::ScalarType::BFloat16 ||
                like.scalar_type() == at::ScalarType::Float8_e4m3fn);
    auto roundedx = cutlass::round_up(rows, kWarpX);
    auto roundedy = cutlass::round_up(cols, kWarpY);

    // NB: Writing to `packed` tensors in transposed manner
    at::Tensor packed =
        at::empty({roundedx, cutlass::ceil_div(roundedy, 2)}, like.options());
    at::Tensor packed_meta =
        at::empty({roundedx * roundedy / 16},
                  like.options().dtype(at::ScalarType::Byte))
            .view({roundedy / 32, roundedx, 2})
            .permute({1, 2, 0});
    return std::make_tuple(packed, packed, packed_meta);
  }

  // define get_meta_offset
  MetadataCutlass(at::Tensor metaN, at::Tensor metaT, int rows, int cols) {
    _meta = (ElementInputE *)metaN.data_ptr();
    _meta_reordered_sy = metaN.stride(2);
    _meta_trans = (ElementInputE *)metaT.data_ptr();
    _meta_trans_reordered_sx = metaT.stride(2);
  }
  CUTLASS_HOST_DEVICE
  int64_t _get_meta_offset(int warp_row, int thread_row, int warp_col,
                           int thread_col, int64_t stride) const {
    int64_t offset = 0;
    offset += warp_row * 2 + (warp_col / 32) * stride;
    // A single warp is 32x64. The right 32x32 tile is at a different position
    offset += 64 * (thread_row / 32);
    offset += (thread_col / 32) * stride;
    // Top/bottom 16x16 tile
    offset += ((thread_row % 32) / 16) * 4;
    // Left/right 16x16 tile
    offset += ((thread_col % 32) / 16) * 2;
    return offset;
  }

  // Define get_metaN and get_metaT
  CUTLASS_HOST_DEVICE
  ElementInputE *get_metaN(int warp_row, int thread_row, int warp_col,
                           int thread_col) const {
    return _meta + _get_meta_offset(warp_row, thread_row, warp_col, thread_col,
                                    _meta_reordered_sy);
  }
  CUTLASS_HOST_DEVICE
  ElementInputE *get_metaT(int warp_row, int thread_row, int warp_col,
                           int thread_col) const {
    return _meta_trans + _get_meta_offset(warp_col, thread_col, warp_row,
                                          thread_row, _meta_trans_reordered_sx);
  }
};

template <typename KT, typename Metadata, typename Algorithm>
__global__ void __launch_bounds__(32 /* num_threads */, 20)
    sparse_semi_structured_tile_kernel(typename KT::Params p, Metadata metadata,
                                       Algorithm algo) {
  KT::sparse_semi_structured_tile_kernel(p, metadata, algo);
}

template <typename Element, typename MetadataFormat>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
sparse_semi_structured_tile_typed(const at::Tensor input,
                                  std::string algorithm) {

  printf("sparse_semi_structured_tile_typed... \n");
  using KT = KernelTypes<Element>;
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!input.is_meta()) {
    device_guard.emplace(input.device());
  }

  TORCH_CHECK(input.dim() == 2, "Can only sparsify 2d tensors");
  TORCH_CHECK(input.stride(1) == 1, "Can only sparsify contiguous tensors. "
                                    "Sparsify the transpose otherwise.");

  auto rows = input.size(0);
  auto cols = input.size(1);

  auto [compressed, packed, packed_meta_reordered] =
      MetadataFormat::create_compressed_representation(rows, cols, input);
  auto [compressed_trans, packed_trans, packed_trans_meta_reordered] =
      MetadataFormat::create_compressed_representation(cols, rows, input);
  TORCH_CHECK(input.size(1) % 32 == 0,
              "Number of cols should be multiple of 32");

  typename KT::Params p;
  p.input = (Element const *)input.data_ptr();
  p.input_s0 = input.stride(0);
  p.input_dim0 = input.size(0);
  p.input_dim1 = input.size(1);

  p.packed = (Element *)packed.data_ptr();
  p.packed_stride = packed.stride(0);
  p.packed_trans = (Element *)packed_trans.data_ptr();
  p.packed_trans_stride = packed_trans.stride(0);

  MetadataFormat metadata = MetadataFormat(
      packed_meta_reordered, packed_trans_meta_reordered, rows, cols);
  at::Tensor threads_masks = at::empty(
      {p.getBlocksGrid().x * p.getThreadsGrid().x,
       p.getBlocksGrid().y * p.getThreadsGrid().y, sizeof(p.threads_masks[0])},
      input.options().dtype(at::ScalarType::Byte));
  p.threads_masks = (uint64_t *)threads_masks.data_ptr();

  printf("launching kernel ... \n");
  bool kernel_launched = false;
  auto launchKernel = [&](auto algo, std::string const &algo_name) {
    if (algo_name == algorithm) {
      kernel_launched = true;
      if (input.is_meta()) {
        return;
      }
      size_t smem_bytes = 0;
      sparse_semi_structured_tile_kernel<KT>
          <<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes,
             at::cuda::getCurrentCUDAStream()>>>(p, metadata, algo);
    }
  };
  named_algorithms(launchKernel);
  TORCH_CHECK(kernel_launched, "Unknown algorithm \"", algorithm, "\"");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(compressed, packed_meta_reordered, compressed_trans,
                         packed_trans_meta_reordered, threads_masks);
}

// <packed, packed_meta_reordered, packed_trans, packed_trans_meta_reorderd,
// threads_masks>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_sparse_semi_structured_tile(const at::Tensor &input,
                             std::string_view algorithm, bool use_cutlass) {
  std::string algo(algorithm.data(), algorithm.size());

  printf("Start debugging here ...\n");

  auto runTyped = [&](auto type) {
    using ElementT = decltype(type);
    return sparse_semi_structured_tile_typed<ElementT, MetadataCutlass>(input,
                                                                        algo);
  };

  if (input.scalar_type() == at::ScalarType::Half) {
    return runTyped(cutlass::half_t());
  } else if (input.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    return runTyped(cutlass::float_e4m3_t());
  } else {
    // TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
    //                 input.scalar_type() == at::ScalarType::BFloat16,
    //             input.scalar_type());
    return runTyped(cutlass::bfloat16_t());
  }
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::sparse_semi_structured_tile", &_sparse_semi_structured_tile);
}

} // namespace torchao
