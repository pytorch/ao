#include <metal_stdlib>
using namespace metal;

/**
 * LowBit Quantized Linear for bitwidths that are divisors of 8. Hence the name.
 *
 * @param[A] M x K unquantized input tensor of floating point dtype (Float, Half, BFloat16)
 * @param[B] Packed & quantized weight tensor of uint8 dtype. Expected shape is N x (nbit * K / 8)
 * @param[scalesAndZeros] 3D tensor containg the scales and zero point for each group. Expected shape is #groups x N x 2
 * @param[outputData] M x N output tensor of floating point dtype (same as input)
 * @param[sizes] The sizes involved in the order: M, K, N
 *
 * Dispatched threads: N x M x 1
 */
template<typename T, unsigned nbit, unsigned groupSize>
kernel void divbit_mm(
    constant T                 * A              [[buffer(0)]],
    constant uchar             * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device   T                 * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]], // M, K, N
    uint2                        thread_index   [[thread_position_in_grid]]) {
    const uint K = sizes.y;
    const uint N = sizes.z;
    const uint m = thread_index.y; // 0..M-1
    const uint n = thread_index.x; // 0..N-1
    const uint32_t k_block = (K + groupSize - 1) / groupSize;
    constant T *A_ptr = A + m * K;
    constant uchar *B_ptr = B;

    constexpr uint8_t zero_shift = 1 << (nbit - 1);
    constexpr uint8_t values_per_byte = 8 / nbit;
    constexpr uint8_t minimask = (1 << nbit) - 1;

    float rc = 0.0;
    uint k = 0;
    for (uint32_t kb = 0; kb < k_block ; kb ++) {
      const T scale = scalesAndZeros[(kb * N + n) * 2 + 0];
      const T zero = scalesAndZeros[(kb * N + n) * 2 + 1] - scale * T(zero_shift);
      for(uint idx = 0; idx < groupSize && k < K; idx++, k++) {
        const auto a_val = float(A_ptr[k]);
        uint8_t b_val = B_ptr[(n * K + k) / values_per_byte];
        uint8_t shift = nbit * (k % values_per_byte);
        uint8_t mask = minimask << shift;
        b_val = (b_val & mask) >> shift;
        rc += a_val * float(scale * T(b_val) + zero);
      }
    }
    outputData[m * N + n] = T(rc);
}

#define INSTANTIATE_DIVBIT_MM(NBIT, DTYPE, GSIZE)                           \
template                                                                 \
[[host_name("int" #NBIT "pack_mm_" #GSIZE "_" #DTYPE)]]                  \
kernel void divbit_mm<DTYPE, NBIT, GSIZE>(                               \
    constant DTYPE             * A              [[buffer(0)]],           \
    constant uchar             * B              [[buffer(1)]],           \
    constant DTYPE             * scalesAndZeros [[buffer(2)]],           \
    device   DTYPE             * outputData     [[buffer(3)]],           \
    constant uint3             & sizes          [[buffer(4)]],           \
    uint2                        thread_index [[thread_position_in_grid]])

INSTANTIATE_DIVBIT_MM(1, float, 32);
INSTANTIATE_DIVBIT_MM(1, half, 32);
INSTANTIATE_DIVBIT_MM(1, float, 64);
INSTANTIATE_DIVBIT_MM(1, half, 64);
INSTANTIATE_DIVBIT_MM(1, float, 128);
INSTANTIATE_DIVBIT_MM(1, half, 128);
INSTANTIATE_DIVBIT_MM(1, float, 256);
INSTANTIATE_DIVBIT_MM(1, half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_DIVBIT_MM(1, bfloat, 32);
INSTANTIATE_DIVBIT_MM(1, bfloat, 64);
INSTANTIATE_DIVBIT_MM(1, bfloat, 128);
INSTANTIATE_DIVBIT_MM(1, bfloat, 256);
#endif

INSTANTIATE_DIVBIT_MM(2, float, 32);
INSTANTIATE_DIVBIT_MM(2, half, 32);
INSTANTIATE_DIVBIT_MM(2, float, 64);
INSTANTIATE_DIVBIT_MM(2, half, 64);
INSTANTIATE_DIVBIT_MM(2, float, 128);
INSTANTIATE_DIVBIT_MM(2, half, 128);
INSTANTIATE_DIVBIT_MM(2, float, 256);
INSTANTIATE_DIVBIT_MM(2, half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_DIVBIT_MM(2, bfloat, 32);
INSTANTIATE_DIVBIT_MM(2, bfloat, 64);
INSTANTIATE_DIVBIT_MM(2, bfloat, 128);
INSTANTIATE_DIVBIT_MM(2, bfloat, 256);
#endif

INSTANTIATE_DIVBIT_MM(4, float, 32);
INSTANTIATE_DIVBIT_MM(4, half, 32);
INSTANTIATE_DIVBIT_MM(4, float, 64);
INSTANTIATE_DIVBIT_MM(4, half, 64);
INSTANTIATE_DIVBIT_MM(4, float, 128);
INSTANTIATE_DIVBIT_MM(4, half, 128);
INSTANTIATE_DIVBIT_MM(4, float, 256);
INSTANTIATE_DIVBIT_MM(4, half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_DIVBIT_MM(4, bfloat, 32);
INSTANTIATE_DIVBIT_MM(4, bfloat, 64);
INSTANTIATE_DIVBIT_MM(4, bfloat, 128);
INSTANTIATE_DIVBIT_MM(4, bfloat, 256);
#endif
