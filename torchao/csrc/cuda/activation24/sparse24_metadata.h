#pragma once
#include <torch/library.h>
#include <torch/types.h>
#include "compute_sparse_tile.h"

namespace torchao{
template <typename ElementCutlass>
struct CutlassToAt;

template <>
struct CutlassToAt<cutlass::half_t> {
  static auto constexpr value = at::ScalarType::Half;
};
template <>
struct CutlassToAt<cutlass::bfloat16_t> {
  static auto constexpr value = at::ScalarType::BFloat16;
};
template <>
struct CutlassToAt<cutlass::float_e4m3_t> {
  static auto constexpr value = at::ScalarType::Float8_e4m3fn;
};
template <>
struct CutlassToAt<uint16_t> {
  static auto constexpr value = at::ScalarType::UInt16;
};
template <>
struct CutlassToAt<int32_t> {
  static auto constexpr value = at::ScalarType::Int;
};
template <>
struct CutlassToAt<uint8_t> {
  static auto constexpr value = at::ScalarType::Byte;
};
template <>
struct CutlassToAt<float> {
  static auto constexpr value = at::ScalarType::Float;
};

struct MetadataCutlass8bitsSm90 {
  template <typename ElementOut>
  static std::tuple<at::Tensor, at::Tensor> createTensors(at::Tensor input) {
    auto n_rows = input.size(0);
    auto n_cols = input.size(1);
    TORCH_CHECK(n_cols % 128 == 0); // aligned metadata
    TORCH_CHECK(n_rows % 64 == 0); // aligned metadata

    at::Tensor packed = at::empty(
        {n_rows, n_cols / 2},
        input.options().dtype(CutlassToAt<ElementOut>::value));
    at::Tensor mdata =
        at::empty({n_rows, n_cols / 8}, input.options().dtype(at::ScalarType::Byte));
    return std::make_tuple(packed, mdata);
  }
  static CUTLASS_HOST_DEVICE int64_t
  mdataBlockPtrOffset(int row, int col, int64_t n_rows) {
    constexpr int kStrideRow = 16;
    return row * kStrideRow + (col / 128 * n_rows * 16) + (col % 128) / 8;
  }
};

struct MetadataCusparseLt16bitsSm90 {
  template <typename ElementOut>
  static std::tuple<at::Tensor, at::Tensor> createTensors(at::Tensor input) {
    auto n_rows = input.size(0);
    auto n_cols = input.size(1);
    int packed_elements = n_rows * n_cols / 2;
    int mdata_bytes = n_rows * n_cols / 8;

    // We assume 2 bytes per element
    at::Tensor sparse_packed = at::empty(
        {int64_t(packed_elements + mdata_bytes / sizeof(ElementOut))},
        input.options().dtype(CutlassToAt<ElementOut>::value));
    using namespace torch::indexing;
    return std::make_tuple(
        sparse_packed,
        sparse_packed.index({Slice(packed_elements, None)})
            .view(at::ScalarType::Byte));
  }
};

} // namespace torchao
