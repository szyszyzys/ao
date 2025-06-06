// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <vector>

#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/channelwise_8bit_activation_groupwise_lowbit_weight.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
\

float kTol = 0.0001;

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 1;
  constexpr int kr = 32;
  constexpr int sr = 1;

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  std::vector<char> packed_weights(packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x1x32_f32_neondot<weight_nbit>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 16;
  constexpr int sr = 2;

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  std::vector<char> packed_weights(packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x4x16_f32_neondot<weight_nbit>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 8;
  constexpr int kr = 16;
  constexpr int sr = 2;

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  std::vector<char> packed_weights(packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, /*has_lut*/ false>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, tile_1x1x32) {
  constexpr int weight_nbit = 4;

  // Standard
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With weight zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/true>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // With clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/false,
      /*has_clamp=*/true);
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, tile_1x4x16) {
  constexpr int weight_nbit = 4;

  // Standard
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With weight zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/true>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // With clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/true);

  // n less than 4
  for (int n = 1; n < 4; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
        weight_nbit,
        /*has_weight_zeros=*/false>(
        /*m=*/7,
        /*k=*/64,
        /*n=*/n,
        /*group_size=*/16,
        /*has_bias=*/false,
        /*has_clamp=*/false);
  }
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, tile_1x8x16) {
  constexpr int weight_nbit = 4;

  // Standard
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With weight zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/true>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // With clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/true);

  // n less than 8
  for (int n = 1; n < 8; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
        weight_nbit,
        /*has_weight_zeros=*/false>(
        /*m=*/7,
        /*k=*/64,
        /*n=*/n,
        /*group_size=*/16,
        /*has_bias=*/false,
        /*has_clamp=*/false);
  }
}

void determine_zero_point_and_scale(const std::vector<int8_t>& weight_qvals, int weight_nbit, int& zero_point, float& scale) {
    int min_val = *std::min_element(weight_qvals.begin(), weight_qvals.end());
    int max_val = *std::max_element(weight_qvals.begin(), weight_qvals.end());
    zero_point = (min_val + max_val) / 2;
    int num_quant_levels = 1 << weight_nbit;
    scale = (max_val - min_val) / (num_quant_levels - 1);
}

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_lut(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 8;
  constexpr int kr = 16;
  constexpr int sr = 2;

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);
  torchao::general_lut_test_case lut_test_case(weight_nbit, test_case.weight_qvals);
  // Access the LUT and weight_qval_idxs
  const std::vector<int8_t>& lut = lut_test_case.getLUT();
  const std::vector<int8_t>& weight_qval_idxs = lut_test_case.getWeightQvalIdxs();

  // assigne space
  std::vector<char> packed_weights(packed_weights_with_lut_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  // fill in the packed weights
  // fill in the packed weights
  pack_weights_with_lut<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      weight_qval_idxs.data(),
      /*n_luts*/ 1,
      lut.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, /*has_lut*/ true>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, LUT) {
  constexpr int weight_nbit = 4;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros*/ false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // has_weight_zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros*/ true>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // has_bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // has_clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros*/ false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/true);

  // n less than 8 (nr)
  for (int n = 1; n < 8; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
        weight_nbit,
        /*has_weight_zeros=*/false>(
        /*m=*/7,
        /*k=*/64,
        /*n=*/n,
        /*group_size=*/16,
        /*has_bias=*/false,
        /*has_clamp=*/false);
  }

  // Other bitwidths
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      /*weight_nbit*/ 1,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      /*weight_nbit*/ 2,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      /*weight_nbit*/ 3,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);
}

// Function to test LUT quantization with configurable parameters
template <int weight_bit_width>
void test_general_lut_quantization_and_dequantization(
    int num_values,
    unsigned seed = 42,
    std::function<float(int)> custom_mapping = nullptr) {

  // Generate random weight quantization values with the specified seed
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-(1 << (weight_bit_width - 1)),
                                         (1 << (weight_bit_width - 1)) - 1);

  std::vector<int8_t> weight_qvals(num_values);
  for (int i = 0; i < num_values; i++) {
    weight_qvals[i] = static_cast<int8_t>(dist(rng));
  }

  // Create a general_lut_test_case instance
  torchao::general_lut_test_case lut_test_case(weight_bit_width, weight_qvals);

  // Get the LUT and weight_qval_idxs
  const std::vector<int8_t>& lut = lut_test_case.getLUT();
  const std::vector<int8_t>& weight_qval_idxs = lut_test_case.getWeightQvalIdxs();

  // Verify LUT size
  ASSERT_EQ(lut.size(), 1 << weight_bit_width);

  // Verify weight_qval_idxs size
  ASSERT_EQ(weight_qval_idxs.size(), weight_qvals.size());

  // Calculate expected values directly
  int min_val = *std::min_element(weight_qvals.begin(), weight_qvals.end());
  int max_val = *std::max_element(weight_qvals.begin(), weight_qvals.end());
  int zero_point = (min_val + max_val) / 2;
  int num_quant_levels = 1 << weight_bit_width;
  float scale = (max_val - min_val) / (num_quant_levels - 1);

  // Verify dequantization for each value
  for (size_t i = 0; i < weight_qvals.size(); ++i) {
    uint8_t idx = weight_qval_idxs[i];
    int8_t dequantized_from_lut = lut[idx];
    float dequant_float = scale * (weight_qvals[i] - zero_point);
    int8_t expected_dequantized = static_cast<int8_t>(std::round(dequant_float));

    if (dequantized_from_lut != expected_dequantized) {
      std::cout << "Mismatch at index " << i << ": LUT=" << static_cast<int>(dequantized_from_lut)
               << ", Expected=" << static_cast<int>(expected_dequantized) << ", Original=" << static_cast<int>(weight_qvals[i])
               << ", Scale=" << scale << ", ZeroPoint=" << zero_point << ", Float=" << dequant_float << std::endl;
    }
    ASSERT_EQ(dequantized_from_lut, expected_dequantized);
  }

  // If a custom mapping function is provided, test it
  if (custom_mapping) {
    // Create a copy of the original LUT test case
    torchao::general_lut_test_case custom_lut_test_case = lut_test_case;

    // Update the LUT with the custom mapping function
    custom_lut_test_case.updateMapping(custom_mapping);

    // Get the updated LUT
    const std::vector<int8_t>& custom_lut = custom_lut_test_case.getLUT();

    // Verify the custom mapping was applied correctly
    for (int i = 0; i < custom_lut.size(); ++i) {
      ASSERT_EQ(custom_lut[i], static_cast<int8_t>(custom_mapping(i)));
    }
  }
}

// Test case that calls the function with different parameters
TEST(test_general_lut, quantization_and_dequantization) {
  // Test with different bit widths
  test_general_lut_quantization_and_dequantization<1>(1000);
  test_general_lut_quantization_and_dequantization<2>(1000);
  test_general_lut_quantization_and_dequantization<3>(1000);
  test_general_lut_quantization_and_dequantization<4>(1000);

  // Test with different numbers of values
  test_general_lut_quantization_and_dequantization<3>(10);
  test_general_lut_quantization_and_dequantization<3>(100);
  test_general_lut_quantization_and_dequantization<3>(5000);

  // Test with different seeds
  test_general_lut_quantization_and_dequantization<2>(1000, 123);
  test_general_lut_quantization_and_dequantization<2>(1000, 456);
  test_general_lut_quantization_and_dequantization<2>(1000, 789);

  // Test with custom mapping functions
  auto linear_mapping = [](int idx) -> float {
    return static_cast<float>((idx % 32) - 16);
  };

  auto sine_mapping = [](int idx) -> float {
    return std::sin(idx * 0.5f) * 64;
  };

  test_general_lut_quantization_and_dequantization<4>(1000, 42, linear_mapping);
  test_general_lut_quantization_and_dequantization<4>(1000, 42, sine_mapping);
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
