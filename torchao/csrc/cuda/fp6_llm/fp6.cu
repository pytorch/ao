#include <stdint.h>
#include <stdexcept>
#include <cstring>

// reference implementation. this doesn't have a lot of bit manipulation, so it's less error-prone
// this is not exposed to PyTorch
__device__ __host__ static uint8_t fp32_to_fp6_ref(float a) {
#ifndef __CUDA_ARCH__
    if (std::isnan(a) | std::isinf(a))
        throw std::invalid_argument("Encounter +/-inf or NaN, which is not representable in FP6.");
    if (std::abs(a) >= 30.0f)
        throw std::invalid_argument("FP6 overflow. FP6 cannot represent +/-inf.");
#endif

    a *= 0x1p-124;  // 2^(127-3)
    uint32_t bits;
    std::memcpy(&bits, &a, sizeof(a));

    uint8_t sign = bits >> 31u << 5u;
    uint8_t exp_and_man = (bits >> 21u) & 0x1Fu;
    uint8_t result = sign | exp_and_man;

    // round to nearest even
    uint32_t remainder = bits << 11u;
    if ((remainder > 0x8000'0000u) || ((remainder == 0x8000'0000u) && (result & 1u))) {
        result += 1;
    }

    return result;
}

// we need to do this because C++17 does not allow using struct as template non-type parameter
// use the upper 16 bits for num exponent, lower 16 bits for num mantissa
static constexpr uint32_t encode_fp_spec(uint32_t n_exp_bits, uint32_t n_man_bits) {
    return (n_exp_bits << 16u) | n_man_bits;
}

static constexpr uint32_t FP32_SPEC = encode_fp_spec(8u, 23u);
static constexpr uint32_t FP16_SPEC = encode_fp_spec(5u, 10u);
static constexpr uint32_t BF16_SPEC = encode_fp_spec(8u, 7u);

// NOTE: only works for len < 32
__device__ __host__ static constexpr uint32_t ones_mask(uint32_t len) { return (1u << len) - 1u; }

// inspired by __internal_float2half() and float2half() from "cuda_fp16.hpp"
template <typename T, uint32_t FP_SPEC>
__device__ __host__ static uint8_t bits_to_fp6(T bits) {
    constexpr uint32_t N_EXP = FP_SPEC >> 16u;
    constexpr uint32_t N_MAN = FP_SPEC & ones_mask(16u);
    constexpr uint32_t N_EXP_MAN = N_EXP + N_MAN;

    // sanity checks. will be removed in template specialization.
#ifndef __CUDA_ARCH__
    if (N_EXP < 3)
        throw std::invalid_argument("Number of exponent bits must be >= 3.");
    if (N_MAN < 3)
        throw std::invalid_argument("Number of mantissa bits must be >= 3.");
#endif

    T remainder = 0u;
    T sign = bits >> N_EXP_MAN << 5u;
    bits &= ones_mask(N_EXP_MAN);  // clear sign bit
    T result;

    constexpr uint32_t EXP_BIAS_DIFF = ones_mask(N_EXP - 1u) - 3u;

    // only checks for invalid values on CPU, since we can't throw exception in CUDA
#ifndef __CUDA_ARCH__
    // all exponent bits are 1s
    if (bits >= (ones_mask(N_EXP) << N_MAN))
        throw std::invalid_argument("Encounter +/-inf or NaN, which is not representable in FP6.");
    // max FP6 (28) + half of least significand (2) = 30 (assume N_MAN >= 3)
    if (bits >= (((EXP_BIAS_DIFF + 7u) << N_MAN) | (0x7u << (N_MAN - 3u))))
        throw std::invalid_argument("FP6 overflow. FP6 cannot represent +/-inf.");
#endif

    // FP6 normal number (E>=001)
    if (bits >= ((EXP_BIAS_DIFF + 1u) << N_MAN)) {
        remainder = bits << (1u + N_EXP + 2u);
        bits -= (EXP_BIAS_DIFF << N_MAN);  // update exponent
        result = sign | (bits >> (N_MAN - 2u));
    }
    // FP6 subnormal number (more than half of min FP6 subnormal = 0.0625 * 0.5)
    else if (bits > ((EXP_BIAS_DIFF - 2u) << N_MAN)) {
        T exp = bits >> N_MAN;
        T man = bits & ones_mask(N_MAN);

        // to make subnormal FP6 from normal FP16
        // step 1: add implicit 1 to mantissa
        man |= (1u << N_MAN);

        // step 2: shift mantissa right so that exponent value is equal to
        // exponent value of FP6 subnormal, which is -2 (equivalent to E=001)
        T shift = EXP_BIAS_DIFF + 1u - exp;
        remainder = man << (1u + N_EXP + 2u + shift);
        result = sign | (man >> (shift + (N_MAN - 2u)));  // implicit E=000
    }
    // FP6 underflow. E=000, M=00
    else {
        result = sign;
    }

    // round to nearest even
    constexpr T HALF_REMAINDER = 1u << N_EXP_MAN;
    if ((remainder > HALF_REMAINDER) || ((remainder == HALF_REMAINDER) && (result & 0x1u))) {
        result += 1;
    }
    return result;
}

template <typename T, uint32_t FP_SPEC>
__device__ __host__ static void bits_4_to_fp6_4_packed(const T *bits_ptr, uint8_t *fp6_ptr) {
    uint8_t val0 = bits_to_fp6<T, FP_SPEC>(bits_ptr[0]);
    uint8_t val1 = bits_to_fp6<T, FP_SPEC>(bits_ptr[1]);
    uint8_t val2 = bits_to_fp6<T, FP_SPEC>(bits_ptr[2]);
    uint8_t val3 = bits_to_fp6<T, FP_SPEC>(bits_ptr[3]);

    fp6_ptr[0] = (val0 << 2) | (val1 >> 4);  // 0000 0011
    fp6_ptr[1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
    fp6_ptr[2] = (val2 << 6) | (val3);       // 2233 3333
}

// assume the lower 6 bits contain the data
__device__ __host__ static float fp6_to_fp32(const uint8_t a) {
    // we shift the bits so that sign, exponent, and mantissa bits are in their correct positions in FP32.
    // this also handles subnormal numbers correctly.
    // FP6:                                  SE EEMM
    // FP32: S000 00EE EMM0 0000 0000 0000 0000 0000
    uint32_t bits = a;  // bit extension
    uint32_t sign = bits >> 5u << 31u;
    uint32_t exp_and_man = (bits & 0x1Fu) << 21u;
    uint32_t result_bits = sign | exp_and_man;

    // the result will be off by the difference in exponent bias (3 in FP6 and 127 in FP32)
    // we can correct this by direct FP32 multiplication, which also handles subnormal numbers.
    float result;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result * 0x1p124;  // 2^(127-3)
}

__device__ __host__ static void fp6_4_packed_to_fp32_4(const uint8_t *fp6_ptr, float *fp32_ptr) {
    uint8_t bits0 = fp6_ptr[0];  // 0000 0011
    uint8_t bits1 = fp6_ptr[1];  // 1111 2222
    uint8_t bits2 = fp6_ptr[2];  // 2233 3333

    fp32_ptr[0] = fp6_to_fp32(bits0 >> 2);
    fp32_ptr[1] = fp6_to_fp32(((bits0 & 0x3u) << 4) | (bits1 >> 4));
    fp32_ptr[2] = fp6_to_fp32(((bits1 & 0xFu) << 2) | (bits2 >> 6));
    fp32_ptr[3] = fp6_to_fp32(bits2 & 0x3Fu);
}

__global__ void fp6_packed_to_fp32_kernel(const uint8_t *fp6_ptr, float *fp32_ptr, int n) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    if (idx < n)
        fp6_4_packed_to_fp32_4(fp6_ptr + idx, fp32_ptr + idx / 3 * 4);
}

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchao {

// this is useful for debugging
at::Tensor to_fp6_unpacked_cpu(at::Tensor fp_tensor) {
    TORCH_CHECK(fp_tensor.is_contiguous());
    TORCH_CHECK(fp_tensor.is_cpu());

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp_tensor.device());
    at::Tensor fp6_tensor = at::empty(fp_tensor.sizes(), options);
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();

    int n = fp_tensor.numel();
    auto dtype = fp_tensor.dtype();

    if (dtype == torch::kFloat32) {
        const uint32_t *fp32_ptr = reinterpret_cast<uint32_t *>(fp_tensor.data_ptr<float>());

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            fp6_ptr[i] = bits_to_fp6<uint32_t, FP32_SPEC>(fp32_ptr[i]);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            fp6_ptr[i] = bits_to_fp6<uint16_t, FP16_SPEC>(fp16_ptr[i]);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            fp6_ptr[i] = bits_to_fp6<uint16_t, BF16_SPEC>(bf16_ptr[i]);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}

template <typename T, uint32_t FP_SPEC>
__global__ void bits_to_fp6_unpacked_kernel(const T *bits_ptr, uint8_t *fp6_ptr, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        fp6_ptr[idx] = bits_to_fp6<T, FP_SPEC>(bits_ptr[idx]);
}

// this is useful for debugging
at::Tensor to_fp6_unpacked_cuda(at::Tensor fp_tensor) {
    TORCH_CHECK(fp_tensor.is_contiguous());
    TORCH_CHECK(fp_tensor.is_cuda());

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp_tensor.device());
    at::Tensor fp6_tensor = at::empty(fp_tensor.sizes(), options);
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();

    int n = fp_tensor.numel();
    auto dtype = fp_tensor.dtype();

    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    if (dtype == torch::kFloat32) {
        const uint32_t *fp32_ptr = reinterpret_cast<uint32_t *>(fp_tensor.data_ptr<float>());
        bits_to_fp6_unpacked_kernel<uint32_t, FP32_SPEC><<<grid_size, block_size>>>(fp32_ptr, fp6_ptr, n);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        bits_to_fp6_unpacked_kernel<uint16_t, FP16_SPEC><<<grid_size, block_size>>>(fp16_ptr, fp6_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        bits_to_fp6_unpacked_kernel<uint16_t, BF16_SPEC><<<grid_size, block_size>>>(bf16_ptr, fp6_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}

at::Tensor to_fp6_packed_cpu(at::Tensor fp_tensor) {
    TORCH_CHECK(fp_tensor.is_contiguous());
    TORCH_CHECK(fp_tensor.is_cpu());
    TORCH_CHECK(fp_tensor.ndimension() == 2);

    int M = fp_tensor.size(0);
    int N = fp_tensor.size(1);
    TORCH_CHECK(N % 4 == 0, "Last dimension must be a multiple of 4, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp_tensor.device());
    at::Tensor fp6_tensor = at::empty({M, N * 3 / 4}, options);
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();

    int n = fp_tensor.numel();
    auto dtype = fp_tensor.dtype();

    if (dtype == torch::kFloat32) {
        const uint32_t *fp32_ptr = reinterpret_cast<uint32_t *>(fp_tensor.data_ptr<float>());

        #pragma omp parallel for
        for (int i = 0; i < n; i += 4)
            bits_4_to_fp6_4_packed<uint32_t, FP32_SPEC>(fp32_ptr + i, fp6_ptr + i / 4 * 3);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());

        #pragma omp parallel for
        for (int i = 0; i < n; i += 4)
            bits_4_to_fp6_4_packed<uint16_t, FP16_SPEC>(fp16_ptr + i, fp6_ptr + i / 4 * 3);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());

        #pragma omp parallel for
        for (int i = 0; i < n; i += 4)
            bits_4_to_fp6_4_packed<uint16_t, BF16_SPEC>(bf16_ptr + i, fp6_ptr + i / 4 * 3);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}

template <typename T, uint32_t FP_SPEC, int BLOCK_SIZE>
__global__ void bits_to_fp6_packed_kernel(const T *bits_ptr, uint8_t *fp6_ptr, int n) {
    // naive version
    // times 4 since each thread will handle 4 values
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < n)
        bits_4_to_fp6_4_packed<T, FP_SPEC>(bits_ptr + idx, fp6_ptr + idx / 4 * 3);
    return;

    // more optimized version. coalesced memory write (speedup is minimal)
    // const int tid = threadIdx.x;
    // const int input_offset = (blockIdx.x * blockDim.x) * 4;
    // const int output_offset = (blockIdx.x * blockDim.x) * 3;

    // bits_ptr += input_offset;
    // fp6_ptr += output_offset;

    // __shared__ uint8_t shmem[BLOCK_SIZE * 3];

    // if (input_offset + tid * 4 < n) {
    //     uint8_t val0, val1, val2, val3;
    //     if (std::is_same_v<T, uint32_t>) {
    //         uint4 values = reinterpret_cast<const uint4 *>(bits_ptr)[tid * 4];
    //         val0 = bits_to_fp6<T, FP_SPEC>(values.x);
    //         val1 = bits_to_fp6<T, FP_SPEC>(values.y);
    //         val2 = bits_to_fp6<T, FP_SPEC>(values.z);
    //         val3 = bits_to_fp6<T, FP_SPEC>(values.w);
    //     } else if (std::is_same_v<T, uint16_t>) {
    //         ushort4 values = reinterpret_cast<const ushort4 *>(bits_ptr)[tid * 4];
    //         val0 = bits_to_fp6<T, FP_SPEC>(values.x);
    //         val1 = bits_to_fp6<T, FP_SPEC>(values.y);
    //         val2 = bits_to_fp6<T, FP_SPEC>(values.z);
    //         val3 = bits_to_fp6<T, FP_SPEC>(values.w);
    //     } else {
    //         val0 = bits_to_fp6<T, FP_SPEC>(bits_ptr[tid * 4]);
    //         val1 = bits_to_fp6<T, FP_SPEC>(bits_ptr[tid * 4 + 1]);
    //         val2 = bits_to_fp6<T, FP_SPEC>(bits_ptr[tid * 4 + 2]);
    //         val3 = bits_to_fp6<T, FP_SPEC>(bits_ptr[tid * 4 + 3]);
    //     }
    //     shmem[tid * 3]     = (val0 << 2) | (val1 >> 4);  // 0000 0011
    //     shmem[tid * 3 + 1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
    //     shmem[tid * 3 + 2] = (val2 << 6) | (val3);       // 2233 3333
    // }
    // __syncthreads();

    // // TODO: write in larger word size
    // for (int i = 0; i < 3; i++) {
    //     if (output_offset + BLOCK_SIZE * i + tid < n / 4 * 3) {
    //         fp6_ptr[BLOCK_SIZE * i + tid] = shmem[BLOCK_SIZE * i + tid];
    //     }
    // }
}

at::Tensor to_fp6_packed_cuda(at::Tensor fp_tensor) {
    TORCH_CHECK(fp_tensor.is_contiguous());
    TORCH_CHECK(fp_tensor.is_cuda());
    TORCH_CHECK(fp_tensor.ndimension() == 2);

    int M = fp_tensor.size(0);
    int N = fp_tensor.size(1);
    TORCH_CHECK(N % 4 == 0, "Last dimension must be a multiple of 4, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp_tensor.device());
    at::Tensor fp6_tensor = at::empty({M, N * 3 / 4}, options);
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();

    int n = fp_tensor.numel();
    auto dtype = fp_tensor.dtype();

    // times 4 since each thread will handle 4 values
    constexpr int block_size = 256;
    int grid_size = (n + (block_size * 4) - 1) / (block_size * 4);

    if (dtype == torch::kFloat32) {
        const uint32_t *fp32_ptr = reinterpret_cast<uint32_t *>(fp_tensor.data_ptr<float>());
        bits_to_fp6_packed_kernel<uint32_t, FP32_SPEC, block_size><<<grid_size, block_size>>>(fp32_ptr, fp6_ptr, n);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        bits_to_fp6_packed_kernel<uint16_t, FP16_SPEC, block_size><<<grid_size, block_size>>>(fp16_ptr, fp6_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        bits_to_fp6_packed_kernel<uint16_t, BF16_SPEC, block_size><<<grid_size, block_size>>>(bf16_ptr, fp6_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}

__global__ void fp6_unpacked_to_fp32_kernel(const uint8_t *fp6_ptr, float *fp32_ptr, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        fp32_ptr[idx] = fp6_to_fp32(fp6_ptr[idx]);
}

at::Tensor fp6_unpacked_to_fp32(at::Tensor fp6_tensor) {
    TORCH_CHECK(fp6_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(fp6_tensor.is_contiguous());
    TORCH_CHECK(fp6_tensor.is_cpu() || fp6_tensor.is_cuda());

    at::TensorOptions options = at::TensorOptions().dtype(torch::kFloat32).device(fp6_tensor.device());
    at::Tensor fp32_tensor = at::empty(fp6_tensor.sizes(), options);

    const uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    float *fp32_ptr = fp32_tensor.data_ptr<float>();
    int n = fp6_tensor.numel();

    if (fp6_tensor.is_cpu()) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            fp32_ptr[i] = fp6_to_fp32(fp6_ptr[i]);
    } else {
        constexpr int block_size = 256;
        int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
        fp6_unpacked_to_fp32_kernel<<<grid_size, block_size>>>(fp6_ptr, fp32_ptr, n);
    }

    return fp32_tensor;
}

at::Tensor fp6_packed_to_fp32(at::Tensor fp6_tensor) {
    TORCH_CHECK(fp6_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(fp6_tensor.is_contiguous());
    TORCH_CHECK(fp6_tensor.is_cpu() || fp6_tensor.is_cuda());
    TORCH_CHECK(fp6_tensor.ndimension() == 2);

    int M = fp6_tensor.size(0);
    int N = fp6_tensor.size(1);
    TORCH_CHECK(N % 3 == 0, "Last dimension must be a multiple of 3, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kFloat32).device(fp6_tensor.device());
    at::Tensor fp32_tensor = at::empty({M, N / 3 * 4}, options);

    const uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    float *fp32_ptr = fp32_tensor.data_ptr<float>();
    int n = fp6_tensor.numel();

    if (fp6_tensor.is_cpu()) {
        #pragma omp parallel for
        for (int i = 0; i < n; i += 3)
            fp6_4_packed_to_fp32_4(fp6_ptr + i, fp32_ptr + i / 3 * 4);
    } else {
        constexpr int block_size = 256;
        int grid_size = (n + block_size * 3 - 1) / (block_size * 3);
        fp6_packed_to_fp32_kernel<<<grid_size, block_size>>>(fp6_ptr, fp32_ptr, n);
    }

    return fp32_tensor;
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::to_fp6_unpacked", &to_fp6_unpacked_cpu);
  m.impl("torchao::to_fp6_packed", &to_fp6_packed_cpu);
  m.impl("torchao::fp6_unpacked_to_fp32", &fp6_unpacked_to_fp32);
  m.impl("torchao::fp6_packed_to_fp32", &fp6_packed_to_fp32);
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::to_fp6_unpacked", &to_fp6_unpacked_cuda);
  m.impl("torchao::to_fp6_packed", &to_fp6_packed_cuda);
  m.impl("torchao::fp6_unpacked_to_fp32", &fp6_unpacked_to_fp32);
  m.impl("torchao::fp6_packed_to_fp32", &fp6_packed_to_fp32);
}

}
