#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

// #include "sparse24_pack.h"
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/fast_math.h>
#include <cutlass/half.h>
#include <cutlass/integer_subbyte.h>
#include "static_sort.h"

// Given 4x4 values, computes the selected indices that will remain after 2:4
// sparsification, as a bitmask.
// NOTE: Algorithms might select LESS than 8 values in total in some cases.

namespace torchao {

using cutlass::uint1b_t;
using cutlass::uint2b_t;
using cutlass::uint4b_t;
using uint8b_t = cutlass::integer_subbyte<8, false>;
using ElementInputE = uint16_t;

// Operations that we can apply to rank the values
struct IdentityOp {
  template <typename T>
  static T CUTLASS_HOST_DEVICE to_ordered(T const& x) {
    return x;
  }
};
// Can be applied to rank based on absolute value
struct AbsOp {
  template <typename T>
  static uint16_t CUTLASS_HOST_DEVICE to_ordered(T const& x) {
    return cutlass::abs(x).storage;
  }
};

template <typename Element, typename Pointwise>
struct TileValueOrderedT {
  using ElementCmp = decltype(Pointwise::to_ordered(Element(0)));
  union {
    struct {
      ElementCmp cmp;
      Element value;
      uint2b_t col;
      uint2b_t row;
    } parts;
    uint32_t raw;
  };
  CUTLASS_DEVICE bool operator<(
      TileValueOrderedT<Element, Pointwise> const& other) const {
    return parts.cmp < other.parts.cmp;
  }
  CUTLASS_DEVICE TileValueOrderedT() {}
  CUTLASS_DEVICE TileValueOrderedT(Element value, int col, int row = 0) {
    parts.value = value;
    parts.row = uint2b_t{row};
    parts.col = uint2b_t{col};
    parts.cmp = Pointwise::to_ordered(value);
  }
};

template <typename Op>
struct Top2 {
  template <typename ElementT>
  CUTLASS_DEVICE int operator()(
      cutlass::Array<ElementT, 4> values,
      cutlass::Array<ElementT, 2>& packed) const {
    using TileValueOrdered = TileValueOrderedT<ElementT, Op>;
    cutlass::Array<TileValueOrdered, 4> values_ordered;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      values_ordered[i] = TileValueOrdered(values[i].get(), i);
    }
    StaticSort<4> sorter;
    sorter(values_ordered);
    TileValueOrdered first, second;
    first = values_ordered[3];
    second = values_ordered[2];
    if (first.parts.col > second.parts.col) {
      TileValueOrdered tmp;
      tmp = first;
      first = second;
      second = tmp;
    }
    packed[0] = first.parts.value;
    packed[1] = second.parts.value;
    // returns bitmask of select elements
    return first.parts.col | (second.parts.col << 2);
  }
};

template <typename T>
void named_algorithms_oneway(T callback) {
  callback(Top2<IdentityOp>(), "largest");
  callback(Top2<AbsOp>(), "largest_abs");
}

} // namespace torchao
