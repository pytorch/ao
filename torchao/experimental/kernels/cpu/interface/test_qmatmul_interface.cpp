// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cfenv>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/interface/quantized_matmul.h>

float kTol = 0.0001;

// This is unfortunately had to be copied over because code in test_utils.h
// depends on quantization kernels which are only buildable for ARM.
// I would like the testing code in this folder to be independent of the arch.
namespace {
void get_qvals_range(int& qmin, int& qmax, int nbit, bool is_symmetric) {
  if (is_symmetric) {
    qmin = -(1 << (nbit - 1)) + 1;
    qmax = -qmin;
  } else {
    qmin = -(1 << (nbit - 1));
    qmax = (1 << (nbit - 1)) - 1;
  }
}

void get_scale_and_zero(
    float& scale,
    int& zero,
    float vmin,
    float vmax,
    int qmin,
    int qmax) {
  assert(qmin < qmax);
  assert(vmin < vmax);
  scale = (vmax - vmin) / (qmax - qmin);
  zero = qmin - std::round(vmin / scale);
}

inline std::vector<float>
get_random_vector(int size, float min = -1.0, float max = 1.0) {
  assert(min < max);
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::bind(std::uniform_real_distribution<float>(min, max), rng);
  std::vector<float> res(size);
  std::generate(res.begin(), res.end(), std::ref(dist));
  return res;
}

void quantize(
    // Output
    int8_t* qvals,
    // Inputs
    const float* vals,
    int size,
    float scale,
    int8_t zero,
    int8_t qmin,
    int8_t qmax) {
  float invScale = 1.0 / (scale + 1e-16);
  int i = 0;
  auto curr_rounding_mode = fegetround();
  fesetround(FE_TONEAREST);
  for (; i < size; ++i) {
    // Quantize remaining elements using scalar code
    float val = vals[i];
    float qval_f32 = zero + val * invScale;
    int32_t qval_s32 = static_cast<int32_t>(std::nearbyint(qval_f32));

    // Clip to qmin and qmax
    qval_s32 = std::max(
        static_cast<int32_t>(qmin),
        std::min(qval_s32, static_cast<int32_t>(qmax)));

    // Store the quantized value
    qvals[i] = static_cast<int8_t>(qval_s32);
  }
  fesetround(int(curr_rounding_mode));
}

auto generate_per_token_quantized_tensor(
    int m,
    int n,
    bool transposed = false) {
  auto activations = get_random_vector(m * n, -1.0, 1.0);
  auto activation_qvals = std::vector<int8_t>(m * n, 0);
  auto activation_scales = std::vector<float>(m, 0);
  auto activation_zeros = std::vector<int8_t>(m, 0);

  // Quantize activations with 8-bit asymmetric
  // TODO: replace with generic function that does not use aarch64
  // quantize method after we combine with torchao
  int qmin, qmax, zero;
  float vmin, vmax, scale;
  get_qvals_range(qmin, qmax, /*nbit=*/8, /*is_symmetric=*/false);
  for (int m_idx = 0; m_idx < m; m_idx++) {
    auto minmax = std::minmax_element(
        activations.data() + m_idx * n, activations.data() + (m_idx + 1) * n);
    vmin = *minmax.first;
    vmax = *minmax.second;
    get_scale_and_zero(scale, zero, vmin, vmax, qmin, qmax);
    activation_scales[m_idx] = scale;
    activation_zeros[m_idx] = zero;
    quantize(
        /*qvals=*/activation_qvals.data() + m_idx * n,
        /*vals=*/activations.data() + m_idx * n,
        /*size=*/n,
        scale,
        zero,
        qmin,
        qmax);
  }

  if (transposed) {
    auto activations_t = std::vector<float>(m * n, 0);
    auto activation_qvals_t = std::vector<int8_t>(m * n, 0);
    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int n_idx = 0; n_idx < n; n_idx++) {
        int activation_idx = m_idx * n + n_idx;
        int tranposed_idx = n_idx * m + m_idx;
        activations_t[tranposed_idx] = activations[activation_idx];
        activation_qvals_t[tranposed_idx] = activation_qvals[activation_idx];
      }
    }
    activations = activations_t;
    activation_qvals = activation_qvals_t;
  }

  return std::make_tuple(
      activations, activation_qvals, activation_scales, activation_zeros);
}

struct channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case {
  int m;
  int k;
  int n;
  int stride;

  bool lhs_has_zeros;
  bool rhs_has_zeros;
  bool lhs_is_transposed;
  bool rhs_is_transposed;

  std::vector<float> expected_output;

  std::vector<float> lhs;
  std::vector<int8_t> lhs_qvals;
  std::vector<float> lhs_scales;
  std::vector<int8_t> lhs_zeros;

  std::vector<float> rhs;
  std::vector<int8_t> rhs_qvals;
  std::vector<float> rhs_scales;
  std::vector<int8_t> rhs_zeros;

  channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case(
      int m_,
      int k_,
      int n_,
      int stride_,
      bool lhs_has_zeros_,
      bool rhs_has_zeros_,
      bool lhs_is_transposed_,
      bool rhs_is_transposed_,
      std::vector<float> expected_output_,
      std::vector<float> lhs_,
      std::vector<int8_t> lhs_qvals_,
      std::vector<float> lhs_scales_,
      std::vector<int8_t> lhs_zeros_,
      std::vector<float> rhs_,
      std::vector<int8_t> rhs_qvals_,
      std::vector<float> rhs_scales_,
      std::vector<int8_t> rhs_zeros_)
      : m(m_),
        k(k_),
        n(n_),
        stride(stride_),
        lhs_has_zeros(lhs_has_zeros_),
        rhs_has_zeros(rhs_has_zeros_),
        lhs_is_transposed(lhs_is_transposed_),
        rhs_is_transposed(rhs_is_transposed_),
        expected_output(expected_output_),
        lhs(lhs_),
        lhs_qvals(lhs_qvals_),
        lhs_scales(lhs_scales_),
        lhs_zeros(lhs_zeros_),
        rhs(rhs_),
        rhs_qvals(rhs_qvals_),
        rhs_scales(rhs_scales_),
        rhs_zeros(rhs_zeros_) {
    assert(expected_output.size() == m * n);
    assert(lhs.size() == m * stride * k);
    assert(lhs_qvals.size() == m * stride * k);
    assert(lhs_scales.size() == m * stride);
    assert(lhs_zeros.size() == m * stride);
    assert(rhs.size() == n * stride * k);
    assert(rhs_qvals.size() == n * stride * k);
    assert(rhs_scales.size() == n * stride);
    assert(rhs_zeros.size() == n * stride);
  }

  static channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case generate(
      int m,
      int k,
      int n,
      bool lhs_has_zeros,
      bool rhs_has_zeros,
      bool lhs_is_transposed,
      // rhs_is_transposed means generated b matrix is mxk instead of kxm
      bool rhs_is_transposed,
      int stride = 1) {
    assert(!lhs_is_transposed);
    assert(lhs_has_zeros);
    assert(rhs_has_zeros);
    assert(rhs_is_transposed || stride == 1);
    // Generate activations
    auto [lhs, lhs_qvals, lhs_scales, lhs_zeros] =
        generate_per_token_quantized_tensor(m * stride, k);

    auto [rhs, rhs_qvals, rhs_scales, rhs_zeros] =
        generate_per_token_quantized_tensor(n * stride, k, !rhs_is_transposed);
    // Above function produces nxk matrix and to produce kxn you need transposed
    // = true. we do !rhs_is_transposed becaues when rhs_is_transposed = true
    // the shape should be nxk instead of kxn.

    // Compute expected output
    std::vector<float> expected_output(m * n);

    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int n_idx = 0; n_idx < n; n_idx++) {
        float res = 0.0;
        for (int k_idx = 0; k_idx < k; k_idx++) {
          int lhs_idx = m_idx * stride * k + k_idx;
          int rhs_idx = k_idx * stride * n + n_idx * stride;
          if (rhs_is_transposed) {
            rhs_idx = n_idx * stride * k + k_idx;
          }
          float lhs_dequant = lhs_scales[m_idx * stride] *
              (lhs_qvals[lhs_idx] - lhs_zeros[m_idx * stride]);

          float rhs_dequant = rhs_scales[n_idx * stride] *
              (rhs_qvals[rhs_idx] - rhs_zeros[n_idx * stride]);

          res += lhs_dequant * rhs_dequant;
        }
        expected_output[m_idx * n + n_idx] = res;
      }
    }

    // Return test case
    return channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case(
        m,
        k,
        n,
        stride,
        lhs_has_zeros,
        rhs_has_zeros,
        lhs_is_transposed,
        rhs_is_transposed,
        expected_output,
        lhs,
        lhs_qvals,
        lhs_scales,
        lhs_zeros,
        rhs,
        rhs_qvals,
        rhs_scales,
        rhs_zeros);
  }
};
} // namespace

template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_transposed>
struct test_channelwise_8bit_channelwise_8bit_b {
  static void Run(int m, int k, int n);
};

template <bool a_has_zeros, bool b_has_zeros>
struct test_channelwise_8bit_channelwise_8bit_b<
    a_has_zeros,
    b_has_zeros,
    false,
    true> {
  static void Run(int m, int k, int n, int stride = 1) {
    auto test_case =
        channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case::generate(
            m, k, n, a_has_zeros, a_has_zeros, false, true, stride);

    int a_stride_m, b_stride_n;
    auto kernel = torchao::kernels::cpu::quantized_matmul::
        get_int8_a_int8_b_channelwise_qmatmul(
            m, n, k, false, true, a_stride_m, b_stride_n);
    a_stride_m = a_stride_m * stride;
    b_stride_n = b_stride_n * stride;

    std::vector<float> output(m * n);
    kernel(
        m,
        n,
        k,
        test_case.lhs_qvals.data(),
        a_stride_m /*lsh_stride_m*/,
        test_case.rhs_qvals.data(),
        b_stride_n /*rsh_stride_n*/,
        output.data(),
        n /*out_stride_n*/,
        test_case.lhs_zeros.data(),
        test_case.rhs_zeros.data(),
        test_case.lhs_scales.data(),
        test_case.rhs_scales.data(),
        stride, /*lhs qparams stride*/
        stride /*rhs qparams stride*/);

    for (int i = 0; i < m * n; i++) {
      EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
    }
  }
};

TEST(test_channelwise_8bit_channelwise_8bit_b, TranposedBWithZeroPoints) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/1, /*k=*/128, /*n=*/16);
}

TEST(test_channelwise_8bit_channelwise_8bit_b, TranposeBWithZeroPointsLargeM) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/128, /*n=*/16);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposedBWithZeroPointsOddSizes) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/24);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposedBWithZeroPointsOddSizes2) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/19);
}

// Test shapes for which we have to use fallback kernel
TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposedBWithZeroPointsOddSizesFallback) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/5);
}

// Test shapes for which we have to use fallback kernel
TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposedBWithZeroPointsOddSizesFallback2) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/2, /*n=*/1);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposeBWithZeroPointsLargeMStrided) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/128, /*n=*/16, 5);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposedBWithZeroPointsOddSizes2Strided) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/19, 16);
}

// Test shapes for which we have to use fallback kernel
TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposedBWithZeroPointsOddSizesFallbackStrided) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/5, 7);
}

// Test shapes for which we have to use fallback kernel
TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TranposedBWithZeroPointsOddSizesFallback2Strided) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/2, /*n=*/1, 32);
}
