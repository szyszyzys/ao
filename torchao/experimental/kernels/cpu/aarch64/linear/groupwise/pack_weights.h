#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <array>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::
    groupwise_lowbit_weight_with_lut::weight_packing {

namespace internal {

// Packs a buffer of (kr * nr) low-bit indices (stored as uint8_t) down to bits.
// This is the core bit-packing routine for weight indices.
template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void pack_indices_buffer(
    uint8_t* __restrict__ packed_indices,
    const uint8_t* __restrict__ buffer) {
  // This would delegate to an optimized bitpacking implementation.
  // The number of values to pack is kr * nr.
  constexpr int num_vals = kr * nr;
  torchao::bitpacking::pack_lowbit_indices<num_vals, weight_nbit>(
      packed_indices, buffer);
}

// Reorders nr * kr values for GEMM with packing params (nr, kr).
// It takes kr values from each of nr columns and interleaves them.
// This is essential for arranging the indices in the order the kernel expects.
template <typename T>
void reorder_for_packing(
    // Output
    T* __restrict__ reordered_values,
    // Inputs
    const T* __restrict__ values,
    int nr,
    int kr) {
  int dst_idx = 0;
  for (int n_idx = 0; n_idx < nr; n_idx++) {
    // Take kr values from column n_idx
    std::memcpy(
        reordered_values + dst_idx,
        values + n_idx * kr,
        sizeof(T) * kr);
    dst_idx += kr;
  }
}

} // namespace internal

// Calculates the total size in bytes required for the packed weights buffer.
// The layout is defined by what the FMA kernel needs.
size_t packed_weights_with_lut_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_bias,
    int nr) {
  
  // Pad n to the next multiple of nr for consistent tile processing.
  int n_padded = ((n + nr - 1) / nr) * nr;
  size_t total_size = 0;

  // Size of bias (one float per column)
  if (has_bias) {
    total_size += n_padded * sizeof(float);
  }

  // Size of all float LUTs
  // There is one LUT per group.
  assert(k % group_size == 0);
  int groups_per_col = k / group_size;
  int lut_entries = 1 << weight_nbit;
  total_size += n_padded * groups_per_col * lut_entries * sizeof(float);

  // Size of all bit-packed indices
  size_t total_indices = (size_t)n_padded * k;
  size_t indices_bytes = (total_indices * weight_nbit + 7) / 8;
  total_size += indices_bytes;

  return total_size;
}

// Main function to pack weights, LUTs, and bias for the FMA LUT kernel.
// The template parameters <weight_nbit, nr> define the format.
template <int weight_nbit, int nr>
void pack_weights_with_lut(
    // Output
    void* __restrict__ packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* __restrict__ weight_qval_idxs, // Note: int8_t from PyTorch, but we treat as uint
    const float* __restrict__ luts,
    const float* __restrict__ bias) {

  // We assume a KR (k-unrolling factor) that is a multiple of the group size
  // or that the k-loop in the kernel handles group boundaries correctly.
  // For packing, a kr of 16 is a reasonable choice.
  constexpr int kr = 16;
  static_assert(16 % kr == 0, "Group size must be a multiple of kr for simple packing.");

  bool has_bias = (bias != nullptr);

  // Buffer to hold a temporary nr * kr block of indices.
  std::array<uint8_t, nr * kr> buffer;
  // Buffer to hold the reordered indices before bit-packing.
  std::array<uint8_t, nr * kr> reordered_buffer;

  // The size of a bit-packed block of (nr * kr) indices.
  constexpr int packed_block_bytes = (nr * kr * weight_nbit + 7) / 8;
  
  const uint8_t* indices_in = reinterpret_cast<const uint8_t*>(weight_qval_idxs);

  // Data pointer for writing to the packed weights buffer.
  auto packed_weights_byte_ptr = reinterpret_cast<char*>(packed_weights);

  // Loop over columns of the weight matrix in tiles of `nr`.
  for (int n_idx = 0; n_idx < n; n_idx += nr) {
    // --- Step 1: Pack Bias (optional) ---
    if (has_bias) {
      std::memset(packed_weights_byte_ptr, 0, nr * sizeof(float));
      if (n_idx < n) {
        std::memcpy(packed_weights_byte_ptr, bias + n_idx, std::min(nr, n - n_idx) * sizeof(float));
      }
      packed_weights_byte_ptr += nr * sizeof(float);
    }
    
    // --- Step 2: Pack LUTs ---
    // The LUTs are stored for each column in the tile.
    int groups_per_col = k / group_size;
    int lut_entries = 1 << weight_nbit;
    size_t luts_per_col_bytes = groups_per_col * lut_entries * sizeof(float);
    std::memset(packed_weights_byte_ptr, 0, nr * luts_per_col_bytes);
    if (n_idx < n) {
        std::memcpy(packed_weights_byte_ptr, luts + n_idx * groups_per_col * lut_entries, std::min(nr, n-n_idx) * luts_per_col_bytes);
    }
    packed_weights_byte_ptr += nr * luts_per_col_bytes;
    
    // --- Step 3: Pack Indices ---
    // Loop over the k-dimension in chunks of `kr`.
    for (int k_idx = 0; k_idx < k; k_idx += kr) {
      // Gather the nr x kr block of indices.
      buffer.fill(0);
      for (int j = 0; j < nr; j++) {
        if (n_idx + j < n) {
          std::memcpy(
              buffer.data() + kr * j,
              indices_in + (n_idx + j) * k + k_idx,
              kr);
        }
      }

      // Reorder and bit-pack the indices.
      internal::reorder_for_packing(reordered_buffer.data(), buffer.data(), nr, kr);
      internal::pack_indices_buffer<weight_nbit, kr, nr>(
          reinterpret_cast<uint8_t*>(packed_weights_byte_ptr),
          reordered_buffer.data());
      packed_weights_byte_ptr += packed_block_bytes;
    } // k_idx loop
  } // n_idx loop
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
