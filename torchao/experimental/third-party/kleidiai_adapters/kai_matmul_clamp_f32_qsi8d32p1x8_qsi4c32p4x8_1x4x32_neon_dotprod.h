#include <torchao/experimental/third-party/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <torchao/experimental/third-party/kleidiai/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h>
#include <torchao/experimental/third-party/kleidiai/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h>



namespace torchao::kleidiai_adapters::kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod {

int activation_data_size(int m, int k, int group_size);

void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations);

int weight_data_size(int n, int k, int group_size);

void prepare_weight_data(
    void* weight_data,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros);

void kernel(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* weight_data,
    const void* activation_data,
    // Not applied if nullptr
    const float* bias,
    // Ignored if has_clamp = false
    float clamp_min,
    float clamp_max);

} // namespace torchao::kleidiai_adapters::kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod


int torchao::kleidiai_adapters::kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod::activation_data_size(int m, int k, int group_size) {
  auto mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
  auto kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
  auto sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
  return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, group_size, mr, kr, sr);
}

void torchao::kleidiai_adapters::kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod::prepare_activation_data(
    void* activation_data,
    int m,
    int k,
    int group_size,
    const float* activations) {
      auto mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
      auto kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
      auto sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
      kai_run_lhs_quant_pack_qsi8d32p_f32(m, k, group_size, /*mr=*/mr, /*kr=*/kr, /*sr=*/sr, /*m_idx_start=*/0, activations, /*lhs_stride=*/k * sizeof(float), /*lhs_packed=*/activation_data);
}

int torchao::kleidiai_adapters::kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod::weight_data_size(int n, int k, int group_size) {
  auto nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
  auto kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, group_size);
}

void torchao::kleidiai_adapters::kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod::prepare_weight_data(
    void* weight_data,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros) {
    auto nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    auto kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    auto sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    auto lhs_zero_point = 8; // @nocommit: check
    auto rhs_zero_point = 8; // @nocommit: check

    kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0_params params{lhs_zero_point, rhs_zero_point};
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
      /*num_groups=*/1, n, k, nr, kr, sr, group_size, /*rhs=*/weight_qvals, /*bias=*/nullptr, /*rhs_packed=*/weight_data, /*extra_bytes*/0, params);
}

void torchao::kleidiai_adapters::kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod::kernel(
    float32_t* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* weight_data,
    const void* activation_data,
    // Not applied if nullptr
    const float* bias,
    // Ignored if has_clamp = false
    float clamp_min,
    float clamp_max) {
      assert(group_size == 32);
      assert(bias == nullptr);
      kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(m, n, k, group_size, activation_data, weight_data, output, output_m_stride, 1, clamp_min, clamp_max);
}
