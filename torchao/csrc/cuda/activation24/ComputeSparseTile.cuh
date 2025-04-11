#pragma once

#include "SparseSemiStructuredPack.cuh"
#include "StaticSort.h"
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/platform/platform.h>
#include <cutlass/version.h>
// Basic FP8 type definitions
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// // For FP8 E4M3 format (4 exponent bits, 3 mantissa bits)
// #include <cutlass/float8_e4m3.h>

// // For FP8 E5M2 format (5 exponent bits, 2 mantissa bits)
// #include <cutlass/float8_e5m2.h>

// Given 4x4 values, computes the selected indices that will remain after 2:4
// sparsification, as a bitmask.
// NOTE: Algorithms might select LESS than 8 values in total in some cases.

namespace torchao {

template <typename Element, typename Pointwise> struct TileValueOrderedT {
  union {
    struct {
      Element value;
      uint2b_t inner_index;
      uint2b_t outer_index;
    } parts;
    uint32_t raw;
  };
  CUTLASS_DEVICE bool
  operator<(TileValueOrderedT<Element, Pointwise> const &other) const {
    return Pointwise::apply(parts.value) < Pointwise::apply(other.parts.value);
  }
  CUTLASS_DEVICE TileValueOrderedT() {}
};

// Operations that we can apply to rank the values
struct IdentityOp {
  template <typename T> static T CUTLASS_HOST_DEVICE apply(T const &x) {
    return x;
  }
};

// Given 1x4 values (a row), computes the selected indices that will remain
// after 2:4 sparsification, as a bitmask. We have 1 constraint: (1) Exactly 2
// values per row ALGO: We use a simple algorithm that selects the 2 largest
// values in the row. NOTE: RF are not indexable, so we shouldn't rely on
// indexing
//   values at any point, otherwise they will be stored in local memory.
template <typename Op = IdentityOp> struct LargestValuesRowwise {
  template <typename T> static CUTLASS_DEVICE T outOfBoundsFillValue() {
    return -cutlass::platform::numeric_limits<T>::infinity();
  }

  template <typename Tile1x16Accessor>
  CUTLASS_DEVICE Indices1x16 operator()(Tile1x16Accessor values) {
    using TileValueOrdered =
        TileValueOrderedT<typename Tile1x16Accessor::Element, Op>;
    using TileValuesFragment = cutlass::Array<TileValueOrdered, 4 * 4>;

    Indices1x16 indices;
    TileValuesFragment values_ordered;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        TileValueOrdered &v = values_ordered[i * 4 + j];
        v.parts.value = values.at(0, i * 4 + j).get();
        v.parts.inner_index = uint2b_t(j);
        v.parts.outer_index = uint2b_t(i);
      }
    }
    // Use a sorting network (aka without branches) to avoid
    // warp divergence
    StaticSort<TileValuesFragment::kElements> sorter;
    sorter(values_ordered);

    // bitmask to store how many we have selected on a given row
    // 0 selected: (numPerRow >> 2*row) = 00 (0)
    // 1 selected: (numPerRow >> 2*row) = 01 (1)
    // 2 selected: (numPerRow >> 2*row) = 11 (3)
    uint32_t numPer1x4Strip = 0;
    indices = 0;

    // Take as many as we can, starting with the largest values
    CUTLASS_PRAGMA_UNROLL
    for (int i = values_ordered.size() - 1; i >= 0; i--) {
      auto &e = values_ordered[i];

      uint32_t rcount = uint2b_t(numPer1x4Strip >> 2 * e.parts.outer_index);
      // NOTE: This is more efficient (yet equivalent) to:
      // `rcount != 3 && ccount != 3`
      bool selected = rcount <= 2;
      indices |= selected << (e.parts.inner_index + 4 * e.parts.outer_index);

      numPer1x4Strip |= (rcount + selected) << 2 * e.parts.outer_index;
    }
    return indices;
  }
};

template <typename T> void named_algorithms(T callback) {
  // default one
  callback(LargestValuesRowwise<IdentityOp>(), "");
}

} // namespace torchao
