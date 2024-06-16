#include <cstdint>
#include <array>
#include <string>
namespace torchao {

static char* QUANTIZED_KERNEL = R"METAL_QUANTIZED(
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <typename T> struct Vec4Type {};

template <> struct Vec4Type<float> { using type = float4; };

template <> struct Vec4Type<half> { using type = half4; };

#if __METAL_VERSION__ >= 310
template <> struct Vec4Type<bfloat> { using type = bfloat4; };
#endif

/*
   This code takes heavy inspiration from MLX qvm kernel here:
   https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.metal#L381
   Specifically:
     - Multiplying activation by inverse scaling factor to reduce compute
   boundedness
     - Handling zero point by accumulating act in separate sum term. Needed with
   optimization done above. MLX MIT License:
   https://github.com/ml-explore/mlx/blob/main/LICENSE
*/

/*
   A matrix is [M x K] (right now this kernel does not support M > 1 but this is
   a very easy fix that will follow right after) B matrix is [N x K]. For 4 bit
   2 of the k values are packed in one byte so you can think of B as [N x K/2]
   matrix from layout perspective.

   Since this kernel is optimizing for gemv case, we split work, along reduction
   dim k, among the threads of same simdgroup. Ex: if K = 4096 and simdgroup
   size is 32 (current algorithm should work as long as simdgroup size is > 32).
   Then each thread will accumulate 4096/32 = 128 k values. However these 128
   values, handled by each thread are not laid out contiguously. Each thread
   handles 4 contiguous k values and then jumps 128 elements, k_jump =
   thread_per_channel (32) * ks_per_thread (4). Take a simpler example where
   simdgroup is of size 4. In this case threads_per_channel = 4. Assume K = 32
      k                thread
   [0, 1, 2, 3,          0
    4, 5, 6, 7,          1
    8, 9, 10, 11,        2
    12, 13, 14, 15,      3
    16, 17, 18, 19,      0
    20, 21, 22, 23,      1
    24, 25, 26, 27,      2
    28, 29, 30, 31]      3
   thread id in simd group that handle corresponding
   ks
   Thread 0 here is handling (0, 1, 2, 3) and then (16, 17, 18, 19). They are
   apart by k_jump = 4 * 4 = 16 This is done to improve memory access locality
   amonng threads that are working co-operatively. Once each thread has their
   partial sums accumulated, we use tree reduction (Metal offers simd_sum but
   not used so that we support simdgroup size = 64). In the
   example above we will have 4 partial sums.

   Each thread also handles 4 different output rows. Thus each simdgroup will be
   responsible for (1x4) tile of the output. We haven't evaluated whether a
   different tile size is better or not. We probably will do some auto-tuning
   once initial work is done.

*/

/*
   @brief This shader implements 4-bit matrix-vector multiplication where A
   matrix is fp16, bfloat or float and B matrix is a 4-bit groupwise-quantized weight
   matrix.
   @param [in] A is activation matrix of size M x K.
   @param [in] B is weight matrix of size M x K. Each byte contains 2 4-bit
   values, along K dim, packed together.
   @param [in] scales_and_zeros is scales and zero points corresponding each
   output channel x groups. These are packed as [num_groups = ceil(K / group_size), N, 2]. N = output
   @param [out] output_data is output matrix of size M x N.
   @param [in] sizes array contains values of M, N and K.
   @param [in] thread_index is global thread id.
   @param [in] tid_in_simdgruop is thread id in simdgroup. e.g. in simdgroup of size 32 it can be in [0-31].
*/
/*
TODOs:
   Right now code handles only M = 1 case. Fix that.
*/
template <typename T, unsigned group_size>
kernel void int4pack_vm(constant T *A [[buffer(0)]],
                        constant uchar *B [[buffer(1)]],
                        constant T *scales_and_zeros [[buffer(2)]],
                        device T *output_data [[buffer(3)]],
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
  // This is starting k for each thread. In the example above, for thread 1 this
  // value will be 4.
  uint k = (tid_in_simdgroup % threads_per_channel) * ks_per_thread;
  constexpr int k_jump = threads_per_channel * ks_per_thread;

  using vecT = typename Vec4Type<T>::type;
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A);
  constant uchar *B_ptr = B + ((n * K) / k_pack_factor);

  thread float4 result = float4(0.0);
  // We multipy group of 4 channels with these scales.
  // Because corresponding values from weight matrix are effectively left
  // shifted. This is to avoid doing right shift on those values which ends up
  // affecting performance. This is the trick applied in MLX kernels.
  float4 act_div_scales = {1.f, 1 / 16.f, 1 / 256.f, 1 / 4096.f};

  // Find specific group to which channels handled by this thread
  // belong.
  uint k_block_index = k / group_size;
  // Since scales_and_zeros are packed as [num_groups, N, 2].
  // Finding a specific's group's scales and zero points requires jump by factor
  // of N*2
  uint scales_group_offset = (k_block_index * N + n) * 2;
  uint zeros_gruop_offset = scales_group_offset + 1;
  const uint scales_jump =
      N * 2 *
      (k_jump /
       group_size); /* the last term accounts for identifying the group this
                      thread will have to process in each iteration. This mean
                      each iteration it must jump to a different group. Thus
                      k_jump must be > group_size */
  for (; k < K; k += k_jump) {
    const T scale0 = scales_and_zeros[scales_group_offset];
    // Adding zero point results in 10% perf penalty.
    const T zero0 = scales_and_zeros[zeros_gruop_offset] - scale0 * T(8);

    const T scale1 = scales_and_zeros[scales_group_offset + 2];
    const T zero1 = scales_and_zeros[zeros_gruop_offset + 2] - scale1 * T(8);

    const T scale2 = scales_and_zeros[scales_group_offset + 4];
    const T zero2 = scales_and_zeros[zeros_gruop_offset + 4] - scale2 * T(8);

    const T scale3 = scales_and_zeros[scales_group_offset + 6];
    const T zero3 = scales_and_zeros[zeros_gruop_offset + 6] - scale3 * T(8);

    scales_group_offset += scales_jump;
    zeros_gruop_offset += scales_jump;

    const float4 zeros = float4(zero0, zero1, zero2, zero3);

    float4 a_val = float4(A_ptr[k / 4]);
    // We are gonna skip right-shifts of the weights and hence divide by corresponding factor.
    float4 a_vec = a_val * act_div_scales;
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

    result += a_vec * b_mat;
    result += a_val_sum * zeros;
  }
  result += simd_shuffle_down(result, 1);
  result += simd_shuffle_down(result, 2);
  result += simd_shuffle_down(result, 4);
  result += simd_shuffle_down(result, 8);
  result += simd_shuffle_down(result, 16);
  if (tid_in_simdgroup % threads_per_channel == 0) {
    reinterpret_cast<device vecT *>(output_data)[n / 4] = vecT(result);
  }
}

#define INSTANTIATE_INT4MV(DTYPE, GSIZE)                                       \
  template [[host_name("int4pack_vm_" #GSIZE "_" #DTYPE)]] kernel void         \
  int4pack_vm<DTYPE, GSIZE>(                                                   \
      constant DTYPE * A [[buffer(0)]], constant uchar * B [[buffer(1)]],      \
      constant DTYPE * scales_and_zeros [[buffer(2)]],                         \
      device DTYPE * output_data [[buffer(3)]],                                 \
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
)METAL_QUANTIZED";

void int4mv(const uint8_t * A, const uint8_t * B, uint32_t groupSize, const uint8_t * scalesAndZeros, uint8_t * C, std::array<uint32_t, 4> sizes, std::array<uint64_t, 4> nbytes, std::string A_scalar_type);

} // namespace torchao