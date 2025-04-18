#pragma once

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/fast_math.h>
#include <cutlass/half.h>
#include <cutlass/integer_subbyte.h>
#include "StaticSort.h"

namespace xformers {
namespace sp24 {
using cutlass::uint1b_t;
using cutlass::uint2b_t;
using cutlass::uint4b_t;
using uint8b_t = cutlass::integer_subbyte<8, false>;
using ReorderedLayoutInputE = cutlass::layout::ColumnMajorInterleaved<2>;
using ElementInputE = uint16_t;
constexpr int kWarpX = 32;
constexpr int kWarpY = 64;
constexpr int kThreadX = 8;
constexpr int kThreadY = 8;

} // namespace sp24
} // namespace xformers
