#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>


#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut/groupwise_lowbit_weight_with_lut.h>
// A tolerance for floating-point comparisons
constexpr float kTol = 1e-4;

// Use the kernel's namespace
using namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut;

template <typename T>
std::vector<T> transpose_matrix(const std::vector<T>& matrix, int rows, int cols) {
    // --- Core Transpose Logic (Unchanged) ---
    std::vector<T> transposed_matrix(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // The core transpose formula: (row, col) -> (col, row)
            transposed_matrix[j * rows + i] = matrix[i * cols + j];
        }
    }
    return transposed_matrix;
}

// The main test driver function, templated on tile size
template <int MR, int NR>
void test_groupwise_lowbit_lut_kernel(
    int m,
    int k,
    int n,
    int packing_group_size,
    bool has_scales,
    bool has_bias,
    bool has_clamp) {
    using namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut;

    ASSERT_EQ(m % MR, 0);
    ASSERT_EQ(n % NR, 0);
    ASSERT_EQ(k % packing_group_size, 0);

    // 1. Generate test data and golden reference using the provided factory
    auto test_case = torchao::groupwise_lowbit_weight_lut_test_case::generate_per_group(
        m, k, n,
        packing_group_size,
        /*weight_nbit=*/4,
        /*has_scales=*/has_scales,
        has_bias, has_clamp);

    // 2. Pack Activations
    std::vector<char> packed_activations(m * k * sizeof(float));
    pack_activations(
        packed_activations.data(), m, k, test_case.activations.data(), MR);

    // 3. Pack Weights using the new packer
    std::vector<char> packed_weights(packed_weights_size(4, n, k, has_bias, packing_group_size, packing_group_size, NR, true));

    auto transposed_weight_qval_indices = transpose_matrix(test_case.weight_qval_indices, n, k);

    pack_weights(
        packed_weights.data(),
        transposed_weight_qval_indices.data(),
        test_case.weight_scales,
        test_case.weight_luts,
        test_case.bias,
        test_case.weight_nbit,
        has_bias,
        n,
        k,
        test_case.scale_group_size,
        test_case.lut_group_size,
        NR,
        16,  // kr
        2    // sr
    );

    // 4. Run the kernel
    std::vector<float> output(m * n);
    groupwise_lowbit_lut_kernel(
        output.data(),
        n,
        m, n, k,
        packing_group_size,
        packing_group_size,
        packed_weights.data(),
        packed_activations.data(),
        test_case.clamp_min,
        test_case.clamp_max,
        has_bias,
        has_clamp,
        true);  // has_scale parameter (ignored by the kernel)

    // 5. Compare results
    for (int i = 0; i < m * n; i++) {
        EXPECT_NEAR(output[i], test_case.expected_output[i], kTol)
            << "Mismatch at index " << i;
    }
}

TEST(TestGroupwiseLowbitLut, Tile_4x16_FastPath) {
    constexpr int MR = 4;
    constexpr int NR = 16;

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*packing_group_size=*/32,/*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false);

    // With bias
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/4, /*k=*/32, /*n=*/16, /*packing_group_size=*/16,/*has_scales=*/false,
        /*has_bias=*/true, /*has_clamp=*/false);

    // With clamp
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/12, /*k=*/64, /*n=*/16, /*packing_group_size=*/64,/*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/true);

    // With bias and clamp
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/4, /*k=*/128, /*n=*/48, /*packing_group_size=*/64,/*has_scales=*/false,
        /*has_bias=*/true, /*has_clamp=*/true);

    // With scales
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*packing_group_size=*/32,/*has_scales=*/true,
        /*has_bias=*/false, /*has_clamp=*/false);

}
