// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <cassert>
#include <cstring>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>

namespace torchao::kernels::cpu::aarch64::linear::
    groupwise_lowbit_weight_with_lut::kernel {
namespace internal {

// Helper to clamp a vector between min and max values.
inline float32x4_t
vec_clamp(float32x4_t x, float32x4_t vec_min, float32x4_t vec_max) {
  return vminq_f32(vmaxq_f32(x, vec_min), vec_max);
}

// Helper to unpack NR low-bit indices.
// In a real library, this would be a highly optimized function in bitpack.h
// This is a simplified example for demonstration.
template <int NR, int weight_nbit>
inline void unpack_nr_indices(
    uint8_t* __restrict__ indices,
    const void* __restrict__ packed_data) {
  // A real implementation would have optimized paths for each weight_nbit.
  // We delegate to a hypothetical optimized function.
  torchao::bitpacking::unpack_lowbit_indices<NR, weight_nbit>(
      indices, reinterpret_cast<const uint8_t*>(packed_data));
}

// The generic, templated micro-kernel implementation.
// This is the core logic that will be instantiated for different tile sizes.
template <
    int MR,
    int NR,
    int KR,
    int weight_nbit,
    bool has_bias,
    bool has_clamp>
void kernel_fma_impl(
    // Outputs
    float* __restrict__ output,
    // Inputs
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* __restrict__ packed_weights,
    const void* __restrict__ packed_activations,
    float clamp_min,
    float clamp_max) {
  
  // This implementation requires NR to be a multiple of 4 for float32x4_t vectors.
  static_assert(NR % 4 == 0, "NR must be a multiple of 4 for this NEON kernel.");

  // Pre-calculate compile-time constants.
  constexpr int VEC_NR = NR / 4;
  constexpr int lut_size = 1 << weight_nbit;

  const char* act_ptr_base = reinterpret_cast<const char*>(packed_activations);
  const char* wgt_ptr_base = reinterpret_cast<const char*>(packed_weights);

  // Loop over the output matrix C in tiles of MR x NR.
  for (int m_idx = 0; m_idx < m; m_idx += MR) {
    // Get base pointers for the MR rows of activations we will process.
    const float* act_ptrs[MR];
    for (int i = 0; i < MR; ++i) {
      act_ptrs[i] = reinterpret_cast<const float*>(act_ptr_base + packed_activations_offset(m_idx + i, k));
    }

    for (int n_idx = 0; n_idx < n; n_idx += NR) {
      // Create a 2D array of vector registers to hold the MR x NR accumulator tile.
      float32x4_t res[MR][VEC_NR];
      for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < VEC_NR; ++j) {
          res[i][j] = vdupq_n_f32(0.0f);
        }
      }

      // Calculate pointers for the current weight tile.
      const char* wgt_ptr_tile = wgt_ptr_base + packed_weights_with_lut_offset(n_idx, k, group_size, weight_nbit, has_bias);
      const char* wgt_luts_ptr = wgt_ptr_tile;
      if constexpr (has_bias) { wgt_luts_ptr += NR * sizeof(float); }
      const char* wgt_indices_ptr = wgt_luts_ptr + (k / group_size) * lut_size * sizeof(float);
      const size_t indices_k_stride_bytes = (NR * weight_nbit + 7) / 8; // Bit-packing stride

      // Loop over the reduction dimension K, unrolling KR times.
      for (int k_idx = 0; k_idx < k; k_idx += KR) {
        // This is the unrolled inner loop. The compiler will fully unroll this.
        for (int k_unroll = 0; k_unroll < KR; ++k_unroll) {
          const int k_step = k_idx + k_unroll;

          // 1. Unpack and gather weight data for this k_step
          const int group_idx = k_step / group_size;
          const float* lut_ptr_group = reinterpret_cast<const float*>(wgt_luts_ptr) + group_idx * lut_size;
          
          uint8_t indices[NR];
          internal::unpack_nr_indices<NR, weight_nbit>(indices, wgt_indices_ptr + k_step * indices_k_stride_bytes);
          
          float w_vals[NR];
          for(int i = 0; i < NR; ++i) w_vals[i] = lut_ptr_group[indices[i]];

          float32x4_t w_vecs[VEC_NR];
          for(int j = 0; j < VEC_NR; ++j) {
            w_vecs[j] = vld1q_f32(w_vals + j * 4);
          }

          // 2. Perform the rank-1 update (FMA) for the entire MR x NR tile
          for (int i = 0; i < MR; ++i) {
            float32x4_t act_vec = vld1q_dup_f32(act_ptrs[i] + k_step);
            for (int j = 0; j < VEC_NR; ++j) {
              res[i][j] = vmlaq_f32(res[i][j], w_vecs[j], act_vec);
            }
          }
        }
      } // k_idx loop

      // --- Post-processing ---

      // Add bias if enabled
      if constexpr (has_bias) {
        const float* bias_ptr = reinterpret_cast<const float*>(wgt_ptr_tile);
        for (int j = 0; j < VEC_NR; ++j) {
          float32x4_t bias_vec = vld1q_f32(bias_ptr + j * 4);
          for (int i = 0; i < MR; ++i) {
            res[i][j] = vaddq_f32(res[i][j], bias_vec);
          }
        }
      }

      // Apply clamp (e.g., ReLU) if enabled
      if constexpr (has_clamp) {
        float32x4_t vec_min = vdupq_n_f32(clamp_min);
        float32x4_t vec_max = vdupq_n_f32(clamp_max);
        for (int i = 0; i < MR; ++i) {
          for (int j = 0; j < VEC_NR; ++j) {
            res[i][j] = internal::vec_clamp(res[i][j], vec_min, vec_max);
          }
        }
      }

      // Store the final MR x NR tile to the output matrix.
      // This loop handles tail processing for m automatically if m is not a multiple of MR.
      for (int i = 0; i < MR && (m_idx + i < m); ++i) {
        float* out_ptr = output + (m_idx + i) * output_m_stride + n_idx;
        // This loop handles tail processing for n automatically if n is not a multiple of NR.
        for (int j = 0; j < VEC_NR && (n_idx + j * 4 < n); ++j) {
            int remaining = n - (n_idx + j*4);
            if (remaining >= 4) {
                vst1q_f32(out_ptr + j * 4, res[i][j]);
            } else {
                // Handle remaining elements one by one
                float temp[4];
                vst1q_f32(temp, res[i][j]);
                for(int l=0; l<remaining; ++l) {
                    *(out_ptr + j*4 + l) = temp[l];
                }
            }
        }
      }
    } // n_idx loop
  } // m_idx loop
}

} // namespace internal
} // namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
