#pragma once

#include "StaticSort.h"
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/fast_math.h>
#include <cutlass/half.h>
#include <cutlass/integer_subbyte.h>

namespace torchao {

using cutlass::uint1b_t;
using cutlass::uint2b_t;
using cutlass::uint4b_t;
using uint8b_t = cutlass::integer_subbyte<8, false>;
using ReorderedLayoutInputE = cutlass::layout::ColumnMajorInterleaved<2>;
using ElementInputE = uint16_t;
constexpr int kWarpX = 4;
constexpr int kWarpY = 128;
constexpr int kThreadX = 1;
constexpr int kThreadY = 16;

// bitmask of selected values, in col-major storage
// eg: indices & (1 << col))
using Indices1x16 = uint16_t;

struct Tile1x16Masks {
  Indices1x16 a;
  CUTLASS_DEVICE Tile1x16Masks() { a = 0; }
};

template <typename Element_> struct KernelTypes {
  using Element = Element_;
  // always read from gmem in chunks of 128bits
  using Fragment = cutlass::Array<Element, 16>;
  using Fragment8 = cutlass::Array<Element, 8>;

  struct Params {
    /// inputs
    Element const *input;
    int64_t input_s0;
    int64_t input_dim0;
    int64_t input_dim1;

    /// outputs
    Element *packed;
    int64_t packed_stride;

    __host__ dim3 getBlocksGrid() const {
      return dim3(cutlass::ceil_div(input_dim0, kWarpX),
                  cutlass::ceil_div(input_dim1, kWarpY), 1);
    }

    static CUTLASS_HOST_DEVICE dim3 getThreadsGrid() {
      return dim3(kWarpX / kThreadX, kWarpY / kThreadY, 1);
    }
  };

  // Holds the packed values for a 1x4 segment (2 values)
  struct Strip1x4Packed {
    Element values[2];
    CUTLASS_DEVICE Strip1x4Packed() {
      values[0] = Element(0);
      values[1] = Element(0);
    }
  };

  // Holds the packed values for a 1x16 Tile (4 1x4 segments)
  struct Strip1x16Packed {
    Strip1x4Packed strips[4];
    CUTLASS_DEVICE Strip1x16Packed() {
      strips[0] = Strip1x4Packed();
      strips[1] = Strip1x4Packed();
      strips[2] = Strip1x4Packed();
      strips[3] = Strip1x4Packed();
    }
  };

  struct Tile1x16Meta {
    uint16_t meta;

    CUTLASS_DEVICE Tile1x16Meta() { meta = 0; }
  };

  CUTLASS_DEVICE static void writePacked(Element *ptr, Strip1x16Packed packed) {
    Fragment8 write;
    write[0] = packed.strips[0].values[0];
    write[1] = packed.strips[0].values[1];
    write[2] = packed.strips[1].values[0];
    write[3] = packed.strips[1].values[1];
    write[4] = packed.strips[2].values[0];
    write[5] = packed.strips[2].values[1];
    write[6] = packed.strips[3].values[0];
    write[7] = packed.strips[3].values[1];
    cutlass::arch::global_store<Fragment8, sizeof(Fragment8)>(write, ptr, true);
  }

  struct Tile1x16Accessor {
    using Element = Element_;

    Fragment (&_lines)[1];
    int _start_row;
    int _start_col;

    CUTLASS_DEVICE Tile1x16Accessor(Fragment (&lines)[1], int start_row,
                                    int start_col)
        : _lines(lines), _start_row(start_row), _start_col(start_col) {}

    CUTLASS_DEVICE typename Fragment::reference at(int r, int c) {
      return _lines[r + _start_row][c + _start_col];
    }
  };

  CUTLASS_DEVICE static Strip1x16Packed
  pack_1x16(Indices1x16 indices, Tile1x16Accessor tile, uint16_t &meta) {
    Strip1x16Packed packed;
    CUTLASS_PRAGMA_UNROLL
    for (int strip = 0; strip < 4; ++strip) {
      uint2b_t col0_from, col1_from;
      auto packValue = [&](uint2b_t col_to, uint2b_t col_from) {
        auto value = tile.at(0, (4 * strip + col_from)).get();
        packed.strips[strip].values[col_to] = value;
        if (col_to == uint2b_t(0)) {
          col0_from = col_from;
        } else {
          col1_from = col_from;
        }
      };

      auto isSelected = [&](int col) {
        return indices & (1 << (4 * strip) + col);
      };

      if (isSelected(1)) {
        packValue(0, 1);
      }
      if (isSelected(0)) {
        packValue(0, 0);
      }
      if (isSelected(0) && isSelected(1)) {
        packValue(1, 1);
      }
      // Process cols 2/3
      // same sort of heuristic
      if (isSelected(2)) {
        packValue(1, 2);
      }
      if (isSelected(3)) {
        packValue(1, 3);
      }
      if (isSelected(2) && isSelected(3)) {
        packValue(0, 2);
      }
      int add_mask = (col0_from | (col1_from << 2)) << (4 * strip);
      meta |= add_mask;
    }
    return packed;
  }

  // Every thread runs this kernel
  template <typename Algorithm, typename MetadataStore>
  CUTLASS_DEVICE static void
  sparse_semi_structured_tile_kernel(Params p, MetadataStore metadata_gmem,
                                     Algorithm compute_tile_indices) {
    // Each thread is responsible for an 1x16 tile, which contains 4 1x4 tiles:
    // A, B, C and D, as displayed in the following schema:
    // +---+---+---+---+
    // | A | B | C | D |
    // +---+---+---+---+
    // Each warp (32 threads) will then be responsible for a 4x128 tile of the
    // input.

    // It will be in the format
    // T1 T2 T3 ... T7
    // T8 ...

    // Top-left of the 4x128 tile we own
    int warp_x = blockIdx.x * kWarpX;
    int warp_y = blockIdx.y * kWarpY;
    int x = warp_x + threadIdx.x * kThreadX;
    int y = warp_y + threadIdx.y * kThreadY;

    Element const *input = p.input + x * p.input_s0 + y;
    Element *packed = p.packed + x * p.packed_stride + (y / 2);
    Fragment lines[1]; // Contains all values from the 1x16 tile

    Tile1x16Meta metadata;
    Tile1x16Masks indices;

    // Load/process tiles `A` and `B`
    Element fillValue = Algorithm::template outOfBoundsFillValue<Element>();
    lines[0].fill(fillValue);
    cutlass::arch::global_load<Fragment, sizeof(Fragment)>(lines[0], input,
                                                           x < p.input_dim0);

    indices.a = compute_tile_indices(Tile1x16Accessor(lines, 0, 0));

    Strip1x16Packed packed_a =
        pack_1x16(indices.a, Tile1x16Accessor(lines, 0, 0), metadata.meta);
    writePacked(packed, packed_a);

    // *p.getCurrentThreadIndices() = indices;

    // Writing non-transposed metadata
    {
      ElementInputE *packed_meta_reordered = metadata_gmem.get_metaN(
          warp_x, threadIdx.x * kThreadX, warp_y, threadIdx.y * kThreadY);
      ((uint16_t *)packed_meta_reordered)[0] = metadata.meta;
    }
  }
};

} // namespace torchao
