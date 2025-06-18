#include <gtest/gtest.h>
#include <vector>

#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut/groupwise_lowbit_weight_with_lut.h>

// Use the kernel API
using namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut;

template <int MR, int NR>
void test_groupwise_lowbit_lut_kernel(
    int m,
    int k,
    int n,
    int scale_group_size,
    int lut_group_size,
    bool has_scales,
    bool has_bias,
    bool has_clamp,
    int weight_nbit,
    bool promote_to_4bit_layout) {
    using namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut;
    ASSERT_EQ(m % MR, 0) << "M must be a multiple of MR";
    ASSERT_EQ(n % NR, 0) << "N must be a multiple of NR";
    ASSERT_EQ(k % scale_group_size, 0) << "K must be a multiple of scale_group_size";
    ASSERT_EQ(k % lut_group_size, 0) << "K must be a multiple of lut_group_size";

    // 1. Generate test case
    auto test_case = torchao::groupwise_lowbit_weight_lut_test_case::generate_with_decoupled_grouping(
        m, k, n,
        /*scale_group_size=*/scale_group_size,
        /*lut_group_size=*/lut_group_size,
        /*weight_nbit=*/weight_nbit,
        /*has_scales=*/has_scales,
        has_bias, has_clamp);

    // 2. Pack Activations
    const auto& source_activations = test_case.activations;
    std::vector<float> packed_activations_buffer(packed_activations_size_float(m, k, MR));
    pack_activations(packed_activations_buffer.data(), m, k, source_activations.data(), MR);

    // 3. Pack Weights
    std::vector<char> packed_weights(packed_weights_size(4, n, k, has_bias, scale_group_size, lut_group_size, NR, promote_to_4bit_layout));

    pack_weights(
        packed_weights.data(),
        test_case.weight_qval_indices.data(),
        test_case.weight_scales,
        test_case.weight_luts,
        test_case.weight_nbit,
        n,
        k,
        test_case.scale_group_size,
        test_case.lut_group_size,
        NR, promote_to_4bit_layout);

    // 4. Run the kernel
    std::vector<float> output(m * n);
    groupwise_lowbit_lut_kernel(
        output.data(),
        n,
        m, n, k,
        scale_group_size, lut_group_size,
        packed_weights.data(),
        packed_activations_buffer.data(),
        test_case.bias.data(),
        test_case.clamp_min,
        test_case.clamp_max,
        has_bias,
        has_clamp,
        weight_nbit,
        MR, NR);

    // 5. Compare results
    constexpr float kTol = 1e-4;
    for (int i = 0; i < m * n; i++) {
        EXPECT_NEAR(output[i], test_case.expected_output[i], kTol)
            << "Mismatch at index " << i;
    }
}

TEST(test_groupwise_lowbit_lut_kernel, tile_4x16_aligned_scale_lut_group_size) {
    // MR and NR are fixed for current kernel.
    constexpr int MR = 4; // Micro-kernel Row height
    constexpr int NR = 16; // Micro-kernel Register width

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/32,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/4, true);

    // With bias
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/4, /*k=*/32, /*n=*/16,/*scale_group_size=*/32,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/true, /*has_clamp=*/false, /*weight_nbit=*/4, true);

    // With clamp
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/12, /*k=*/64, /*n=*/16, /*scale_group_size=*/32,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/true, /*weight_nbit=*/4, true);

    // With bias and clamp
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/4, /*k=*/128, /*n=*/48,
        /*scale_group_size=*/32,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/true, /*has_clamp=*/true, /*weight_nbit=*/4, true);

    // With scales
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32,
        /*scale_group_size=*/32,
        /*lut_group_size=*/32,
        /*has_scales=*/true,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/4, true);

    // With scales clamp, and bias
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32,
        /*scale_group_size=*/32,
        /*lut_group_size=*/32,
        /*has_scales=*/true,
        /*has_bias=*/true, /*has_clamp=*/true, /*weight_nbit=*/4, true);

}


TEST(test_groupwise_lowbit_lut_kernel, tile_4x16_misaligned_scale_lut_group_size) {
    constexpr int MR = 4;
    constexpr int NR = 16;

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/32,
        /*lut_group_size=*/16, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/4, true);

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/32,
        /*lut_group_size=*/64, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/4, true);

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/64,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/4, true);

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/16,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/4, true);

}


TEST(test_groupwise_lowbit_lut_kernel, lower_indice_bit) {
    // MR and NR are fixed for current kernel.
    constexpr int MR = 4; // Micro-kernel Row height
    constexpr int NR = 16; // Micro-kernel Register width

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/32,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/1, true);

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/32,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/2, true);

    // Standard case
    test_groupwise_lowbit_lut_kernel<MR, NR>(
        /*m=*/8, /*k=*/64, /*n=*/32, /*scale_group_size=*/32,
        /*lut_group_size=*/32, /*has_scales=*/false,
        /*has_bias=*/false, /*has_clamp=*/false, /*weight_nbit=*/3, true);
}
