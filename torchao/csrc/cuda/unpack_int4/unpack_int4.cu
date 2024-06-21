#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  const uint64_t blocks = a / b + (a % b != 0);
  return blocks;
}
constexpr int32_t kWarpSize = 32;

template <int InnerKTiles>
__global__ void unpack_m16n8k16_Bint4_layout(
    // size [ceil(n / 8)][ceil(k / (InnerKTiles * 16))][32][InnerKTiles / 2]
    const at::PackedTensorAccessor32<int32_t, 4, at::RestrictPtrTraits> in,
    // size [n][k]
    at::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> out) {

  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto kOuterTile = blockIdx.x;
  auto nTile = blockIdx.y;
  auto t = threadIdx.x;

  // n dimension that this lane loads from
  auto n0 = nTile * kNTileSize + (t / 4);
  
  // 8 k-tile values, 4 per m16n8k16 mma.sync operand B
  int32_t ks[8];
  
  // int32_t v[8];
  int32_t v[8];

  // Store address base offset  
  auto pOut = &out[n0][0];

// Unpack 2 k-tiles at a time since min pack size is InnerKTiles = 2    
#pragma unroll
  for (int innerKTile = 0; innerKTile < InnerKTiles; innerKTile += 2) {

    // Offsets of innerTile0
    auto kBase0 = (kOuterTile * InnerKTiles + innerKTile) * kKTileSize;
    ks[0] = kBase0 + (t % 4) * 2;
    ks[1] = ks[0] + 1;
    ks[2] = ks[0] + 8;
    ks[3] = ks[0] + 8 + 1;

    // Offsets of innerTile1
    auto kBase1 = kBase0 + kKTileSize;
    ks[4] = kBase1 + (t % 4) * 2;
    ks[5] = ks[4] + 1;
    ks[6] = ks[4] + 8;
    ks[7] = ks[4] + 8 + 1;

    // inner k-tiles unpack two at a time
    int32_t pack = in[nTile][kOuterTile][t][innerKTile / 2];
    v[0] = pack & 0x0000000f;
    v[2] = (pack >> 4) & 0x0000000f;
    v[4] = (pack >> 8) & 0x0000000f;
    v[6] = (pack >> 12) & 0x0000000f;
    v[1] = (pack >> 16) & 0x0000000f;
    v[3] = (pack >> 20) & 0x0000000f;
    v[5] = (pack >> 24) & 0x0000000f;
    v[7] = (pack >> 28) & 0x0000000f;

    // Write out
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      pOut[ks[i]] = v[i];      
    }    
  }
}

// output is [n][k] (int32 dtype)
// input is [n / 8][k / (InnerKTiles * 16)][32][innerKTiles / 2]
at::Tensor unpack_int4_packed(
    const at::Tensor& packed_w,
    int64_t innerKTiles) 
{

  c10::cuda::CUDAGuard g(packed_w.device());

  TORCH_CHECK(packed_w.dim() == 4);
  TORCH_CHECK(packed_w.dtype() == at::kInt);
  TORCH_CHECK(packed_w.is_contiguous());

  TORCH_CHECK(packed_w.size(2) == 32);
  TORCH_CHECK(packed_w.size(3) == innerKTiles / 2);
  TORCH_CHECK(innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8);

  int N = packed_w.size(0) * 8;
  int K = packed_w.size(1) * innerKTiles * 16;

  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto nTiles = divUp(N, kNTileSize);

  auto kSuperTiles = divUp(K, innerKTiles * kKTileSize);

  auto out = at::empty(
      {N, K},
      at::TensorOptions().dtype(at::kInt).device(packed_w.device()));

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(kSuperTiles, nTiles);
  
  if (innerKTiles == 2) {
    unpack_m16n8k16_Bint4_layout<2><<<grid, kWarpSize, 0, stream>>>(
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>());
  }
   else if (innerKTiles == 4) {
    unpack_m16n8k16_Bint4_layout<4><<<grid, kWarpSize, 0, stream>>>(
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>());
  } else if (innerKTiles == 8) {
    unpack_m16n8k16_Bint4_layout<8><<<grid, kWarpSize, 0, stream>>>(
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>());
  }

  return out;
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::unpack_int4_packed", &unpack_int4_packed);
}
