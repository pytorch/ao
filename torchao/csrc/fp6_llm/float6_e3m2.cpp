#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <stdint.h>
#include <stdexcept>
#include <cstring>


class float6_e3m2_nan_inf : public std::invalid_argument {
public:
    float6_e3m2_nan_inf() : std::invalid_argument("Encounter +/-inf or NaN, which is not representable in float6_e3m2.") { }
};

class float6_e3m2_overflow : public std::invalid_argument {
public:
    float6_e3m2_overflow() : std::invalid_argument("float6_e3m2 overflow. float6_e3m2 cannot represent +/-inf. Make sure input < 30.0") { }
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
static uint8_t to_float6_e3m2_bits(T bits_) {
    constexpr uint32_t N_EXP = FP_SPEC >> 16u;
    constexpr uint32_t N_MAN = FP_SPEC & ones_mask(16u);
    constexpr uint32_t N_EXP_MAN = N_EXP + N_MAN;
    constexpr uint32_t EXP_BIAS_DIFF = ones_mask(N_EXP - 1u) - 3u;

    // sanity checks. will be removed in template instantiation.
    // minimum 1 bit above FP6 (3 exponent bits and 2 mantissa bits) to avoid edge cases.
    static_assert(N_EXP >= 4, "Number of exponent bits must be >= 4.");
    static_assert(N_MAN >= 3, "Number of mantissa bits must be >= 3.");

    uint32_t bits = bits_;  // bit extension
    uint32_t sign = bits >> N_EXP_MAN << 5u;
    bits &= ones_mask(N_EXP_MAN);  // clear sign bit
    uint32_t result, remainder;

    // all exponent bits are 1s
    if (bits >= (ones_mask(N_EXP) << N_MAN)) throw float6_e3m2_nan_inf();

    // max FP6 (28) + half of least significand (2) = 30 (assume N_MAN >= 3)
    if (bits >= (((EXP_BIAS_DIFF + 7u) << N_MAN) | (0x7u << (N_MAN - 3u)))) throw float6_e3m2_overflow();

    // FP6 normal number (E>=001)
    if (bits >= ((EXP_BIAS_DIFF + 1u) << N_MAN)) {
        remainder = bits << (32u - (N_MAN - 2u));  // shift the truncated bits to most significant position
        bits -= (EXP_BIAS_DIFF << N_MAN);          // update exponent
        result = sign | (bits >> (N_MAN - 2u));
    }
    // FP6 subnormal number (more than half of min FP6 subnormal = 0.0625 * 0.5)
    else if (bits > ((EXP_BIAS_DIFF - 2u) << N_MAN)) {
        uint32_t exp = bits >> N_MAN;
        uint32_t man = bits & ones_mask(N_MAN);

        // to make subnormal FP6 from normal FP16
        // step 1: add implicit 1 to mantissa
        man |= (1u << N_MAN);

        // step 2: shift mantissa right so that exponent value is equal to
        // exponent value of FP6 subnormal, which is -2 (equivalent to E=001)
        uint32_t shift = EXP_BIAS_DIFF + 1u - exp;
        remainder = man << (32u - (N_MAN - 2u + shift));  // shift the truncated bits to most significant position
        result = sign | (man >> (shift + (N_MAN - 2u)));  // implicit E=000
    }
    // FP6 underflow. E=000, M=00
    else {
        remainder = 0u;
        result = sign;
    }

    // round to nearest even
    if ((remainder > 0x8000'0000u) || ((remainder == 0x8000'0000u) && (result & 0x1u))) {
        result += 1;
    }
    return result;
}

// assume the lower 6 bits contain the data.
template <typename T, uint32_t FP_SPEC>
static T from_float6_e3m2_bits(uint8_t a) {
    constexpr uint32_t N_EXP = FP_SPEC >> 16u;
    constexpr uint32_t N_MAN = FP_SPEC & ones_mask(16u);
    constexpr uint32_t N_EXP_MAN = N_EXP + N_MAN;
    constexpr uint32_t EXP_BIAS_DIFF = ones_mask(N_EXP - 1u) - 3u;

    uint32_t bits = a;  // bit extension
    uint32_t sign = bits >> 5u;
    uint32_t exp = (bits >> 2u) & 0x7u;
    uint32_t man = bits & 0x3u;

    if (exp > 0u) {         // FP6 normal numbers
        exp += EXP_BIAS_DIFF;
    } else if (man > 0u) {  // FP6 denormal numbers
        uint32_t shift = (man >= 0b10u) ? 1u : 2u;
        man = (man << shift) & 0x3u;  // shift and remove explicit 1
        exp = 1u + EXP_BIAS_DIFF - shift;
    }
    // don't need to handle zero, since E=000 and M=00

    uint32_t result = (sign << N_EXP_MAN) | (exp << N_MAN) | (man << (N_MAN - 2u));
    return static_cast<T>(result);
}

namespace torchao {

template <typename T, uint32_t FP_SPEC> void to_float6_e3m2_unpacked_cpu_impl(const T *bits_ptr, uint8_t *fp6_ptr, int n) {
    // exception within OpenMP parallel region must be caught.
    // set a flag when exception occurs, then re-raise it.
    bool found_nan_inf = false;
    bool found_overflow = false;

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        try { fp6_ptr[i] = to_float6_e3m2_bits<T, FP_SPEC>(bits_ptr[i]); }
        catch (float6_e3m2_nan_inf const &) { found_nan_inf = true; }
        catch (float6_e3m2_overflow const &) { found_overflow = true; }
    }

    if (found_nan_inf) throw float6_e3m2_nan_inf();
    if (found_overflow) throw float6_e3m2_overflow();
}

// this is useful for debugging
at::Tensor to_float6_e3m2_unpacked_cpu(at::Tensor fp_tensor) {
    TORCH_CHECK(fp_tensor.is_contiguous());
    TORCH_CHECK(fp_tensor.is_cpu());

    at::TensorOptions options = at::TensorOptions().dtype(torch::kUInt8).device(fp_tensor.device());
    at::Tensor fp6_tensor = at::empty(fp_tensor.sizes(), options);
    uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();

    int n = fp_tensor.numel();
    auto dtype = fp_tensor.dtype();

    if (dtype == torch::kFloat32) {
        const uint32_t *fp32_ptr = reinterpret_cast<uint32_t *>(fp_tensor.data_ptr<float>());
        to_float6_e3m2_unpacked_cpu_impl<uint32_t, FP32_SPEC>(fp32_ptr, fp6_ptr, n);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        to_float6_e3m2_unpacked_cpu_impl<uint16_t, FP16_SPEC>(fp16_ptr, fp6_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        to_float6_e3m2_unpacked_cpu_impl<uint16_t, BF16_SPEC>(bf16_ptr, fp6_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}

template <typename T, uint32_t FP_SPEC> void to_float6_e3m2_packed_cpu_impl(const T *bits_ptr, uint8_t *fp6_ptr, int n) {
    // exception within OpenMP parallel region must be caught.
    // set a flag when exception occurs, then re-raise it.
    bool found_nan_inf = false;
    bool found_overflow = false;

#pragma omp parallel for
    for (int i = 0; i < n / 4; i++) {
        try {
            uint8_t val0 = to_float6_e3m2_bits<T, FP_SPEC>(bits_ptr[i * 4]);
            uint8_t val1 = to_float6_e3m2_bits<T, FP_SPEC>(bits_ptr[i * 4 + 1]);
            uint8_t val2 = to_float6_e3m2_bits<T, FP_SPEC>(bits_ptr[i * 4 + 2]);
            uint8_t val3 = to_float6_e3m2_bits<T, FP_SPEC>(bits_ptr[i * 4 + 3]);

            fp6_ptr[i * 3]     = (val0 << 2) | (val1 >> 4);  // 0000 0011
            fp6_ptr[i * 3 + 1] = (val1 << 4) | (val2 >> 2);  // 1111 2222
            fp6_ptr[i * 3 + 2] = (val2 << 6) | (val3);       // 2233 3333
        }
        catch (float6_e3m2_nan_inf const &) { found_nan_inf = true; }
        catch (float6_e3m2_overflow const &) { found_overflow = true; }
    }

    if (found_nan_inf) throw float6_e3m2_nan_inf();
    if (found_overflow) throw float6_e3m2_overflow();
}

at::Tensor to_float6_e3m2_packed_cpu(at::Tensor fp_tensor) {
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
        to_float6_e3m2_packed_cpu_impl<uint32_t, FP32_SPEC>(fp32_ptr, fp6_ptr, n);

    } else if (dtype == torch::kFloat16) {
        const uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        to_float6_e3m2_packed_cpu_impl<uint16_t, FP16_SPEC>(fp16_ptr, fp6_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        const uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        to_float6_e3m2_packed_cpu_impl<uint16_t, BF16_SPEC>(bf16_ptr, fp6_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp6_tensor;
}

template <typename T, uint32_t FP_SPEC>
void from_float6_e3m2_unpacked_cpu_impl(const uint8_t *fp6_ptr, T *fp_ptr, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        fp_ptr[i] = from_float6_e3m2_bits<T, FP_SPEC>(fp6_ptr[i]);
}

at::Tensor from_float6_e3m2_unpacked_cpu(at::Tensor fp6_tensor, c10::ScalarType dtype) {
    TORCH_CHECK(fp6_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(fp6_tensor.is_contiguous());
    TORCH_CHECK(fp6_tensor.is_cpu());

    at::TensorOptions options = at::TensorOptions().dtype(dtype).device(fp6_tensor.device());
    at::Tensor fp_tensor = at::empty(fp6_tensor.sizes(), options);

    const uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp6_tensor.numel();

    if (dtype == torch::kFloat32) {
        uint32_t *fp32_ptr = reinterpret_cast<uint32_t *>(fp_tensor.data_ptr<float>());
        from_float6_e3m2_unpacked_cpu_impl<uint32_t, FP32_SPEC>(fp6_ptr, fp32_ptr, n);

    } else if (dtype == torch::kFloat16) {
        uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        from_float6_e3m2_unpacked_cpu_impl<uint16_t, FP16_SPEC>(fp6_ptr, fp16_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        from_float6_e3m2_unpacked_cpu_impl<uint16_t, BF16_SPEC>(fp6_ptr, bf16_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp_tensor;
}

template <typename T, uint32_t FP_SPEC>
void from_float6_e3m2_packed_cpu_impl(const uint8_t *fp6_ptr, T *fp_ptr, int n) {
#pragma omp parallel for
    for (int i = 0; i < n / 3; i++) {
        uint8_t bits0 = fp6_ptr[i * 3];      // 0000 0011
        uint8_t bits1 = fp6_ptr[i * 3 + 1];  // 1111 2222
        uint8_t bits2 = fp6_ptr[i * 3 + 2];  // 2233 3333

        fp_ptr[i * 4]     = from_float6_e3m2_bits<T, FP_SPEC>(bits0 >> 2);
        fp_ptr[i * 4 + 1] = from_float6_e3m2_bits<T, FP_SPEC>(((bits0 & 0x3u) << 4) | (bits1 >> 4));
        fp_ptr[i * 4 + 2] = from_float6_e3m2_bits<T, FP_SPEC>(((bits1 & 0xFu) << 2) | (bits2 >> 6));
        fp_ptr[i * 4 + 3] = from_float6_e3m2_bits<T, FP_SPEC>(bits2 & 0x3Fu);
    }
}

at::Tensor from_float6_e3m2_packed_cpu(at::Tensor fp6_tensor, c10::ScalarType dtype) {
    TORCH_CHECK(fp6_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(fp6_tensor.is_contiguous());
    TORCH_CHECK(fp6_tensor.is_cpu());
    TORCH_CHECK(fp6_tensor.ndimension() == 2);

    int M = fp6_tensor.size(0);
    int N = fp6_tensor.size(1);
    TORCH_CHECK(N % 3 == 0, "Last dimension must be a multiple of 3, receives ", N);

    at::TensorOptions options = at::TensorOptions().dtype(dtype).device(fp6_tensor.device());
    at::Tensor fp_tensor = at::empty({M, N / 3 * 4}, options);

    const uint8_t *fp6_ptr = fp6_tensor.data_ptr<uint8_t>();
    int n = fp6_tensor.numel();

    if (dtype == torch::kFloat32) {
        uint32_t *fp32_ptr = reinterpret_cast<uint32_t *>(fp_tensor.data_ptr<float>());
        from_float6_e3m2_packed_cpu_impl<uint32_t, FP32_SPEC>(fp6_ptr, fp32_ptr, n);

    } else if (dtype == torch::kFloat16) {
        uint16_t *fp16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::Half>());
        from_float6_e3m2_packed_cpu_impl<uint16_t, FP16_SPEC>(fp6_ptr, fp16_ptr, n);

    } else if (dtype == torch::kBFloat16) {
        uint16_t *bf16_ptr = reinterpret_cast<uint16_t *>(fp_tensor.data_ptr<at::BFloat16>());
        from_float6_e3m2_packed_cpu_impl<uint16_t, BF16_SPEC>(fp6_ptr, bf16_ptr, n);

    } else {
        throw std::invalid_argument("Only FP32, FP16, and BF16 inputs are accepted.");
    }

    return fp_tensor;
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::to_float6_e3m2_unpacked_cpu", &to_float6_e3m2_unpacked_cpu);
  m.impl("torchao::to_float6_e3m2_packed_cpu", &to_float6_e3m2_packed_cpu);
  m.impl("torchao::from_float6_e3m2_unpacked_cpu", &from_float6_e3m2_unpacked_cpu);
  m.impl("torchao::from_float6_e3m2_packed_cpu", &from_float6_e3m2_packed_cpu);
}

}
