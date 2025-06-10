#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>

// ============================================================================
// 1. Core Enums and Configuration Struct
// ============================================================================
namespace torchao::test_utils::lut {

enum class QuantizationGranularity { PER_TENSOR, PER_CHANNEL, PER_GROUP };
enum class GroundTruthStrategy { IDEAL_DEQUANT, LUT_DEQUANT, RANDOM_LUT_DIRECT };

struct TestCaseConfig {
  int rows;
  int cols;
  QuantizationGranularity granularity;
  GroundTruthStrategy strategy;
  int group_size = -1;
  int nbit = 4;
  int random_seed = 123;
};



std::pair<std::vector<int8_t>, std::vector<int8_t>>
generate_random_int8_lut_and_indices(
    int weight_nbit,
    const std::vector<int8_t>& weight_qvals) {
  // 1. Define the offset used to map between signed and unsigned representations.
  const int offset = (1 << (weight_nbit - 1));

  // 2. Create the simple LUT that maps an unsigned index back to a signed qval.
  //    e.g., for 4-bit, maps [0, 15] -> [-8, 7]
  const int lut_size = (1 << weight_nbit);
  std::vector<int8_t> lut(lut_size);
  for (int i = 0; i < lut_size; i++) {
    lut[i] = i - offset;
  }

  // 3. Create the vector of unsigned indices for the packing function.
  //    This converts the signed qvals (e.g., [-8, 7]) into unsigned indices
  //    (e.g., [0, 15]) that the packing function will use.
  std::vector<int8_t> weight_qval_idxs(weight_qvals.size());
  for (size_t i = 0; i < weight_qvals.size(); i++) {
    weight_qval_idxs[i] = weight_qvals[i] + offset;
  }

  return {lut, weight_qval_idxs};
}


  /**
  * @brief Generates a Look-Up Table (LUT) from pre-computed quantization parameters.
  *
  * This function creates a dequantization LUT that maps every possible value of the
  * input data type (T_in) to its corresponding floating-point representation for
  * each quantization group.
  *
  * @tparam T_in The data type of the quantized values (e.g., int8_t). This
  *              determines the size of each individual LUT (e.g., 256 entries for int8_t).
  * @tparam T_zp The data type of the zero-points (e.g., int8_t).
  * @param scales A vector of scale values, one for each quantization group.
  * @param zeros A vector of zero-point values, one for each quantization group.
  * @param has_zeros A flag indicating if the zero-points should be used.
  * @return A flattened std::vector<float> containing all the group LUTs concatenated.
  */
  std::vector<int8_t> generate_requant_lut_from_params(
    int nbit,
    const std::vector<int8_t>& zeros,
    bool has_zeros) {
  // Determine the range of the low-bit quantized values.
  const int q_min = -(1 << (nbit - 1));
  const int q_max = (1 << (nbit - 1)) - 1;
  const size_t lut_size_per_group = static_cast<size_t>(q_max) - q_min + 1;
  const int lut_index_offset = q_min;

  const int num_groups = zeros.size();
  std::vector<int8_t> luts(num_groups * lut_size_per_group);

  for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
    const int8_t zero_point = has_zeros ? zeros[group_idx] : 0;

    for (int q_val = q_min; q_val <= q_max; ++q_val) {
      size_t lut_idx = group_idx * lut_size_per_group + (q_val - lut_index_offset);
      // The LUT stores the result of (quantized_value - zero_point)
      luts[lut_idx] = static_cast<int8_t>(q_val - zero_point);
    }
  }
  return luts;
  }



/**
 * @brief A generic, extensible test case for LUT-based quantization.
 *
 * This struct holds all the necessary data to test a LUT-based dequantization
 * kernel. It is templated to support various input, zero-point, and LUT types.
 *
 * @tparam T_in The data type of the quantized input values (e.g., int8_t).
 * @tparam T_zp The data type of the zero-points (e.g., int8_t, float).
 * @tparam T_lut The data type used to store the Look-Up Table (e.g., float, bfloat16).
 */
template <typename T_in = int8_t, typename T_zp = int8_t, typename T_lut = float>
struct lut_quantization_test_case {
  // Config
  int rows;
  int cols;
  QuantizationGranularity granularity;
  int group_size;

  // Data
  std::vector<T_in> input_qvals;
  std::vector<float> scales;
  std::vector<T_zp> zeros;
  std::vector<T_lut> lut;
  std::vector<float> expected_output;
  int nbit;

  lut_quantization_test_case(
      int rows_, int cols_, int nbit_, QuantizationGranularity granularity_, int group_size_,
      std::vector<T_in> input_qvals_, std::vector<float> scales_,
      std::vector<T_zp> zeros_, std::vector<T_lut> lut_,
      std::vector<float> expected_output_)
      : rows(rows_),
        cols(cols_),
        granularity(granularity_),
        group_size(group_size_),
        nbit(nbit_),
        input_qvals(std::move(input_qvals_)),
        scales(std::move(scales_)),
        zeros(std::move(zeros_)),
        lut(std::move(lut_)),
        expected_output(std::move(expected_output_)) {
    // Assertions to ensure data integrity remain crucial
    assert(input_qvals.size() == rows * cols);
    assert(expected_output.size() == rows * cols);

    const size_t lut_size_per_param_set = (1 << nbit);
    size_t expected_param_sets = 0;
    if (granularity == QuantizationGranularity::PER_TENSOR) {
        expected_param_sets = 1;
    } else if (granularity == QuantizationGranularity::PER_CHANNEL) {
        expected_param_sets = rows;
    } else { // PER_GROUP
        assert(group_size > 0 && cols % group_size == 0);
        expected_param_sets = rows * (cols / group_size);
    }
    assert(scales.size() == expected_param_sets);
    assert(zeros.size() == expected_param_sets);
    assert(lut.size() == expected_param_sets * lut_size_per_param_set);
  }
};

/**
 * @brief Generates test cases for LUT quantization on the CPU.
 *
 * This class encapsulates the logic for creating test data, quantizing it,
 * building the LUT, and calculating the expected result.
 */
template <typename T_in = int8_t, typename T_zp = int8_t, typename T_lut = float>
class CpuTestCaseGenerator {
public:
  static lut_quantization_test_case<T_in, T_zp, T_lut> generate(const TestCaseConfig& config) {
    std::mt19937 gen(config.random_seed);

    // Check if the user requested the simple, direct LUT generation strategy (Unit test for LUT).
    if (config.strategy == GroundTruthStrategy::RANDOM_LUT_DIRECT) {
      return generate_random_lut_direct(config, gen);
    }

    // 1. Generate random floating-point data
    auto input_float = generate_random_data(config.rows * config.cols, gen);

    // 2. Determine quantization parameters (scales and zero-points)
    int num_param_sets, groups_per_row;
    std::tie(num_param_sets, groups_per_row) = get_param_set_counts(config);
    auto [scales, zeros] = determine_quant_params(num_param_sets, gen);

    // 3. Quantize the input data
    auto input_qvals = quantize_input(input_float, scales, zeros, config, groups_per_row);

    // 4. Build the Look-Up Table
    auto lut = build_lut(scales, zeros, num_param_sets, config);

    // 5. Compute the expected output based on the chosen strategy
    auto expected_output = compute_expected_output(input_qvals, scales, zeros, lut, config, groups_per_row);

    // 6. Return the fully constructed test case object
    return lut_quantization_test_case<T_in, T_zp, T_lut>(
      config.rows, config.cols, config.granularity, config.group_size,
      config.nbit, // <-- Pass nbit here
      std::move(input_qvals), std::move(scales), std::move(zeros),
      std::move(lut), std::move(expected_output));
  }

private:


  /**
   * @brief Helper for the simple RANDOM_LUT_DIRECT strategy.
   *
   * Bypasses quantization simulation. It directly creates a random LUT and
   * random indices, with the ground truth being the result of the lookup.
   * This path always generates a PER_TENSOR style test case.
   */
  static lut_quantization_test_case<T_in, T_zp, T_lut>
  generate_random_lut_direct(const TestCaseConfig& config, std::mt19937& gen) {
    const int num_elements = config.rows * config.cols;
    const T_in q_min = -(1 << (config.nbit - 1));
    const T_in q_max = (1 << (config.nbit - 1)) - 1;
    const size_t lut_size = (1 << config.nbit);
    const int lut_idx_offset = q_min;

    // 1. Generate a completely random LUT.
    std::vector<T_lut> lut(lut_size);
    std::uniform_real_distribution<float> lut_val_dist(-5.0f, 5.0f);
    for (size_t i = 0; i < lut_size; ++i) {
      lut[i] = static_cast<T_lut>(lut_val_dist(gen));
    }

    // 2. Generate random quantized values to serve as our input data.
    std::vector<T_in> input_qvals(num_elements);
    std::uniform_int_distribution<int> q_val_dist(q_min, q_max);
    for (int i = 0; i < num_elements; ++i) {
      input_qvals[i] = static_cast<T_in>(q_val_dist(gen));
    }

    // 3. The ground truth is simply the result of looking up the q_vals in the LUT.
    std::vector<float> expected_output(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      T_in q_val = input_qvals[i];
      size_t lut_idx = q_val - lut_idx_offset; // Map q_val (e.g., -8..7) to index (e.g., 0..15)
      assert(lut_idx < lut.size());
      expected_output[i] = static_cast<float>(lut[lut_idx]);
    }

    // 4. Create dummy scale/zero to satisfy the test case struct assertions.
    //    This strategy is inherently per-tensor.
    std::vector<float> scales = {1.0f};
    std::vector<T_zp> zeros = {static_cast<T_zp>(0)};

    return lut_quantization_test_case<T_in, T_zp, T_lut>(
        config.rows, config.cols, QuantizationGranularity::PER_TENSOR, /*group_size=*/-1, config.nbit,
        std::move(input_qvals), std::move(scales), std::move(zeros),
        std::move(lut), std::move(expected_output));
  }

  // Helper to generate random float data
  static std::vector<float> generate_random_data(int size, std::mt19937& gen) {
      std::vector<float> data(size);
      std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
      for (int i = 0; i < size; ++i) data[i] = dis(gen);
      return data;
  }

  // Helper to get number of parameter sets
  static std::pair<int, int> get_param_set_counts(const TestCaseConfig& config) {
      if (config.granularity == QuantizationGranularity::PER_TENSOR) {
          return {1, 0};
      }
      if (config.granularity == QuantizationGranularity::PER_CHANNEL) {
          return {config.rows, 0};
      }
      // PER_GROUP
      assert(config.group_size > 0 && config.cols % config.group_size == 0);
      int groups_per_row = config.cols / config.group_size;
      return {config.rows * groups_per_row, groups_per_row};
  }

  // Helper to generate scales and zero points
  static std::pair<std::vector<float>, std::vector<T_zp>>
  determine_quant_params(int num_param_sets, std::mt19937& gen) {
      std::vector<float> scales(num_param_sets);
      std::vector<T_zp> zeros(num_param_sets);
      std::uniform_real_distribution<float> scale_dis(0.001f, 0.1f);
      std::uniform_int_distribution<int> zero_dis(
          std::numeric_limits<T_zp>::min(), std::numeric_limits<T_zp>::max());

      for (int i = 0; i < num_param_sets; ++i) {
          scales[i] = scale_dis(gen);
          zeros[i] = static_cast<T_zp>(zero_dis(gen));
      }
      return {scales, zeros};
  }

  // Helper to perform quantization
  static std::vector<T_in> quantize_input(
      const std::vector<float>& input_float, const std::vector<float>& scales,
      const std::vector<T_zp>& zeros, const TestCaseConfig& config, int groups_per_row) {
        assert(config.nbit >= 1 && config.nbit <= 8);
        const T_in q_min = -(1 << (config.nbit - 1));
        const T_in q_max = (1 << (config.nbit - 1)) - 1;
        std::vector<T_in> input_qvals(config.rows * config.cols);
        for (int i = 0; i < config.rows; ++i) {
          for (int j = 0; j < config.cols; ++j) {
              int param_set_idx = get_param_idx(i, j, config, groups_per_row);
              float val = input_float[i * config.cols + j];
              float q_float = std::nearbyint(val / scales[param_set_idx] + zeros[param_set_idx]);
              q_float = std::max(static_cast<float>(q_min),
                               std::min(static_cast<float>(q_max), q_float));
              input_qvals[i * config.cols + j] = static_cast<T_in>(q_float);
          }
        }
      return input_qvals;
  }

  // Helper to build the LUT
  static std::vector<T_lut> build_lut(
      const std::vector<float>& scales, const std::vector<T_zp>& zeros, int num_param_sets, const TestCaseConfig& config) {
        assert(config.nbit >= 1 && config.nbit <= 8);
        const T_in q_min = -(1 << (config.nbit - 1));
        const T_in q_max = (1 << (config.nbit - 1)) - 1;

        // The size of the LUT is 2^nbit
        const size_t lut_size_per_set = (1 << config.nbit);
        const int lut_idx_offset = q_min;

        std::vector<T_lut> lut(num_param_sets * lut_size_per_set);
        for (int i = 0; i < num_param_sets; ++i) {
          for (int q_val_int = q_min; q_val_int <= q_max; ++q_val_int) {
              size_t lut_idx = i * lut_size_per_set + (q_val_int - lut_idx_offset);
              float dequant_val = scales[i] * (static_cast<float>(q_val_int) - static_cast<float>(zeros[i]));
              lut[lut_idx] = static_cast<T_lut>(dequant_val); // Cast to final LUT type
          }
        }
      return lut;
  }

  // Helper to compute the reference output
  static std::vector<float> compute_expected_output(
      const std::vector<T_in>& input_qvals, const std::vector<float>& scales,
      const std::vector<T_zp>& zeros, const std::vector<T_lut>& lut,
      const TestCaseConfig& config, int groups_per_row) {

      const T_in q_min = -(1 << (config.nbit - 1));
      const size_t lut_size_per_set = (1 << config.nbit);
      const int lut_idx_offset = q_min;


      std::vector<float> expected_output(config.rows * config.cols);
      for (int i = 0; i < config.rows; ++i) {
          for (int j = 0; j < config.cols; ++j) {
              size_t linear_idx = i * config.cols + j;
              int param_set_idx = get_param_idx(i, j, config, groups_per_row);
              T_in q_val = input_qvals[linear_idx];
              float dequantized_val = 0.0f;

              if (config.strategy == GroundTruthStrategy::IDEAL_DEQUANT) {
                  dequantized_val = scales[param_set_idx] * (static_cast<float>(q_val) - static_cast<float>(zeros[param_set_idx]));
              } else { // LUT_DEQUANT
                  size_t lut_idx = param_set_idx * lut_size_per_set + (q_val - lut_idx_offset);
                  dequantized_val = static_cast<float>(lut[lut_idx]);
              }
              expected_output[linear_idx] = dequantized_val;
          }
      }
      return expected_output;
  }

  // Helper to get the index for the quantization parameters (scale/zero) for a given element
  static int get_param_idx(int row, int col, const TestCaseConfig& config, int groups_per_row) {
    switch(config.granularity) {
      case QuantizationGranularity::PER_TENSOR: return 0;
      case QuantizationGranularity::PER_CHANNEL: return row;
      case QuantizationGranularity::PER_GROUP: return row * groups_per_row + (col / config.group_size);
    }
    return 0; // Should not be reached
  }


};
} // namespace torchao::test_utils::lut
