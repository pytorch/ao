#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <typename T> struct Vec4Type {};

template <> struct Vec4Type<float> { using type = float4; };

template <> struct Vec4Type<half> { using type = half4; };

#if __METAL_VERSION__ >= 310
template <> struct Vec4Type<bfloat> { using type = bfloat4; };
#endif

template <typename T, unsigned groupSize>
kernel void int4pack_mv(constant T *A [[buffer(0)]],
                        constant uchar *B [[buffer(1)]],
                        constant T *scalesAndZeros [[buffer(2)]],
                        device T *outputData [[buffer(3)]],
                        constant uint3 &sizes [[buffer(4)]], // M, K, N
                        uint thread_index [[thread_position_in_grid]],
                        uint tid_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr uint threads_per_channel = 32;
  constexpr uint ks_per_thread = 4;
  constexpr uint k_pack_factor = 2;
  const uint K = sizes.y;
  const uint N = sizes.z;
  uint n = thread_index; // 0..N/4-1
  n = n / threads_per_channel;
  n = n * 4;
  uint k = (tid_in_simdgroup % threads_per_channel) * ks_per_thread;
  constexpr int k_jump = threads_per_channel * ks_per_thread;

  using vecT = typename Vec4Type<T>::type;
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A);
  constant uchar *B_ptr = B + ((n * K) / k_pack_factor);

  thread float4 rc = float4(0.0);
  float4 act_div_scales = {1.f, 1 / 16.f, 1 / 256.f, 1 / 4096.f};

  uint k_block_index = k / groupSize;
  uint scales_n_offset = (k_block_index * N + n) * 2;
  uint zeros_n_offset = scales_n_offset + 1;
  uint scales_jump =
      N * 2 *
      (k_jump /
       groupSize); /* the last term accounts for identifying the group this
                      thread will have to process in each iteration. This mean
                      each iteration it must jump to a different group. Thus
                      k_jump must be > grupSize */
  for (; k < K; k += k_jump) {
    const T scale0 = scalesAndZeros[scales_n_offset];
    // Adding zero point results in 10% perf penalty.
    const T zero0 = scalesAndZeros[zeros_n_offset] - scale0 * T(8);

    const T scale1 = scalesAndZeros[scales_n_offset + 2];
    const T zero1 = scalesAndZeros[zeros_n_offset + 2] - scale1 * T(8);

    const T scale2 = scalesAndZeros[scales_n_offset + 4];
    const T zero2 = scalesAndZeros[zeros_n_offset + 4] - scale2 * T(8);

    const T scale3 = scalesAndZeros[scales_n_offset + 6];
    const T zero3 = scalesAndZeros[zeros_n_offset + 6] - scale3 * T(8);

    scales_n_offset += scales_jump;
    zeros_n_offset += scales_jump;

    const float4 zeros = float4(zero0, zero1, zero2, zero3);

    float4 a_val = float4(A_ptr[k / 4]);   // * k_scales;
    float4 a_vec = a_val * act_div_scales; // * k_scales;
    float a_val_sum = a_val[0] + a_val[1] + a_val[2] + a_val[3];

    float4x4 b_mat;
    ushort b_val0 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 0 * K) / k_pack_factor))[0];
    ushort b_val1 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 1 * K) / k_pack_factor))[0];
    ushort b_val2 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 2 * K) / k_pack_factor))[0];
    ushort b_val3 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 3 * K) / k_pack_factor))[0];
    b_mat[0] = scale0 * float4(float(b_val0 & 0x000f), float(b_val0 & 0x00f0),
                               float(b_val0 & 0x0f00), float(b_val0 & 0xf000));
    b_mat[1] = scale1 * float4(float(b_val1 & 0x000f), float(b_val1 & 0x00f0),
                               float(b_val1 & 0x0f00), float(b_val1 & 0xf000));
    b_mat[2] = scale2 * float4(float(b_val2 & 0x000f), float(b_val2 & 0x00f0),
                               float(b_val2 & 0x0f00), float(b_val2 & 0xf000));
    b_mat[3] = scale3 * float4(float(b_val3 & 0x000f), float(b_val3 & 0x00f0),
                               float(b_val3 & 0x0f00), float(b_val3 & 0xf000));

    rc += a_vec * b_mat;
    rc += a_val_sum * zeros;
  }
  rc += simd_shuffle_down(rc, 1);
  rc += simd_shuffle_down(rc, 2);
  rc += simd_shuffle_down(rc, 4);
  rc += simd_shuffle_down(rc, 8);
  rc += simd_shuffle_down(rc, 16);
  if (tid_in_simdgroup % threads_per_channel == 0) {
    reinterpret_cast<device vecT *>(outputData)[n / 4] = vecT(rc);
  }
}

#define INSTANTIATE_INT4MV(DTYPE, GSIZE)                                       \
  template [[host_name("int4pack_mv_" #GSIZE "_" #DTYPE)]] kernel void         \
  int4pack_mv<DTYPE, GSIZE>(                                                   \
      constant DTYPE * A [[buffer(0)]], constant uchar * B [[buffer(1)]],      \
      constant DTYPE * scalesAndZeros [[buffer(2)]],                           \
      device DTYPE * outputData [[buffer(3)]],                                 \
      constant uint3 & sizes [[buffer(4)]],                                    \
      uint thread_index [[thread_position_in_grid]],                           \
      uint tid_in_simdgroup [[thread_index_in_simdgroup]])

INSTANTIATE_INT4MV(float, 32);
INSTANTIATE_INT4MV(half, 32);
INSTANTIATE_INT4MV(float, 64);
INSTANTIATE_INT4MV(half, 64);
INSTANTIATE_INT4MV(float, 128);
INSTANTIATE_INT4MV(half, 128);
INSTANTIATE_INT4MV(float, 256);
INSTANTIATE_INT4MV(half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT4MV(bfloat, 32);
INSTANTIATE_INT4MV(bfloat, 64);
INSTANTIATE_INT4MV(bfloat, 128);
INSTANTIATE_INT4MV(bfloat, 256);
#endif
