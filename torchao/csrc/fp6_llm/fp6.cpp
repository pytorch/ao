#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <stdint.h>
#include <stdexcept>
#include <cstring>


class fp6_nan_inf : public std::invalid_argument {
public:
    fp6_nan_inf() : std::invalid_argument("Encounter +/-inf or NaN, which is not representable in FP6.") { }
};

class fp6_overflow : public std::invalid_argument {
public:
    fp6_overflow() : std::invalid_argument("FP6 overflow. FP6 cannot represent +/-inf. Make sure input < 30.0") { }
};

// we need to do this because C++17 does not allow using struct as template non-type parameter
// use the upper 16 bits for num exponent, lower 16 bits for num mantissa
static constexpr uint32_t encode_fp_spec(uint32_t n_exp, uint32_t n_man) { return (n_exp << 16u) | n_man; }
static constexpr uint32_t FP32_SPEC = encode_fp_spec(8u, 23u);
static constexpr uint32_t FP16_SPEC = encode_fp_spec(5u, 10u);
static constexpr uint32_t BF16_SPEC = encode_fp_spec(8u, 7u);

// NOTE: only works for len < 32
static constexpr uint32_t ones_mask(uint32_t len) { return (1u << len) - 1u; }

// inspired by __internal_float2half() and float2half() from "cuda_fp16.hpp"
template <typename T, uint32_t FP_SPEC>
static uint8_t to_fp6_bits(T bits) {
    constexpr uint32_t N_EXP = FP_SPEC >> 16u;
    constexpr uint32_t N_MAN = FP_SPEC & ones_mask(16u);
    constexpr uint32_t N_EXP_MAN = N_EXP + N_MAN;

    // sanity checks. will be removed in template instantiation.
    // minimum 1 bit above FP6 (3 exponent bits and 2 mantissa bits) to avoid edge cases.
    static_assert(N_EXP >= 4, "Number of exponent bits must be >= 4.");
    static_assert(N_MAN >= 3, "Number of mantissa bits must be >= 3.");

    T remainder = 0u;
    T sign = bits >> N_EXP_MAN << 5u;
    bits &= ones_mask(N_EXP_MAN);  // clear sign bit
    T result;

    constexpr uint32_t EXP_BIAS_DIFF = ones_mask(N_EXP - 1u) - 3u;

    // all exponent bits are 1s
    if (bits >= (ones_mask(N_EXP) << N_MAN)) throw fp6_nan_inf();

    // max FP6 (28) + half of least significand (2) = 30 (assume N_MAN >= 3)
    if (bits >= (((EXP_BIAS_DIFF + 7u) << N_MAN) | (0x7u << (N_MAN - 3u)))) throw fp6_overflow();

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
        remainder = man << (1u + N_EXP + 2u - shift);
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

namespace torchao {

template <typename T, uint32_t FP_SPEC> void to_fp6_unpacked_cpu_impl(const T *bits_ptr, uint8_t *fp6_ptr, int n) {
    // exception within OpenMP parallel region must be caught.
    // set a flag when exception occurs, then re-raise it.
    bool found_nan_inf = false;
    bool found_overflow = false;

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        try { fp6_ptr[i] = to_fp6_bits<T, FP_SPEC>(bits_ptr[i]); }
        catch (fp6_nan_inf const &) { found_nan_inf = true; }
        catch (fp6_overflow const &) { found_overflow = true; }
    }

    if (found_nan_inf) throw fp6_nan_inf();
    if (found_overflow) throw fp6_overflow();
}

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
        to_fp6_unpacked_cpu_impl<uint32_t, FP32_SPEC>(fp32_ptr, fp6_ptr, n);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        to_fp6_unpacked_cpu_impl<uint16_t, FP16_SPEC>(fp16_ptr, fp6_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        to_fp6_unpacked_cpu_impl<uint16_t, BF16_SPEC>(bf16_ptr, fp6_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}

template <typename T, uint32_t FP_SPEC> void to_fp6_packed_cpu_impl(const T *bits_ptr, uint8_t *fp6_ptr, int n) {
    // exception within OpenMP parallel region must be caught.
    // set a flag when exception occurs, then re-raise it.
    bool found_nan_inf = false;
    bool found_overflow = false;

#pragma omp parallel for
    for (int i = 0; i < n / 4; i++) {
        try {
            uint8_t val0 = to_fp6_bits<T, FP_SPEC>(bits_ptr[i * 4]);
            uint8_t val1 = to_fp6_bits<T, FP_SPEC>(bits_ptr[i * 4 + 1]);
            uint8_t val2 = to_fp6_bits<T, FP_SPEC>(bits_ptr[i * 4 + 2]);
            uint8_t val3 = to_fp6_bits<T, FP_SPEC>(bits_ptr[i * 4 + 3]);

            fp6_ptr[i * 3]     = (val0 << 2) | (val1 >> 4);  // 0000 0011
            fp6_ptr[i * 3 + 1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
            fp6_ptr[i * 3 + 2] = (val2 << 6) | (val3);       // 2233 3333
        }
        catch (fp6_nan_inf const &) { found_nan_inf = true; }
        catch (fp6_overflow const &) { found_overflow = true; }
    }

    if (found_nan_inf) throw fp6_nan_inf();
    if (found_overflow) throw fp6_overflow();
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
        to_fp6_packed_cpu_impl<uint32_t, FP32_SPEC>(fp32_ptr, fp6_ptr, n);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        to_fp6_packed_cpu_impl<uint16_t, FP16_SPEC>(fp16_ptr, fp6_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        to_fp6_packed_cpu_impl<uint16_t, BF16_SPEC>(bf16_ptr, fp6_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}


TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::to_fp6_unpacked_cpu", &to_fp6_unpacked_cpu);
  m.impl("torchao::to_fp6_packed_cpu", &to_fp6_packed_cpu);
}

}
