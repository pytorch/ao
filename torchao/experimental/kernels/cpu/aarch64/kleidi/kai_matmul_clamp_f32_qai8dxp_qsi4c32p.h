// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/experimental/kernels/cpu/aarch64/kleidi/pack.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>

namespace torchao::kernels::cpu::aarch64::kleidi {
  namespace  kai_matmul_clamp_f32_qai8dxp_qsi4c32p {

    using ukernel = struct kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel;

    namespace neon_dotprod_1x4x32 {
      ukernel get_ukernel() {
        return ukernel {
          .get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod
        };
      }

      int activation_data_size(int m, int k, int group_size) {
        auto ukernel = get_ukernel();
        auto lhs_packing = get_lhs_packing();
        return lhs_packing.get_lhs_packed_size(m, k, group_size, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
      }

      void prepare_activation_data(
        void* activation_data,
        // Inputs
        int m,
        int k,
        // Ignored if has_weight_zeros = false
        int group_size,
        const float* activations) {
          auto ukernel = get_ukernel();
          auto lhs_pack = get_lhs_packing();
          lhs_pack.run_lhs_pack(m, k, group_size, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr(), /*m_index_start=*/0,
            activations, /*lhs_stride=*/ k*sizeof(float), activation_data);
      }

      int weight_data_size(int n, int k, int group_size) {
        auto ukernel = get_ukernel();
        auto rhs_pack = get_rhs_packing();
        return rhs_pack.get_rhs_packed_size(n, k, ukernel.get_nr(), ukernel.get_kr(), group_size);
      }

      void prepare_weight_data(
        void* weight_data,
        // Inputs
        int n,
        int k,
        int group_size,
        const int8_t* weight_qvals,
        const float* weight_scales,
        const int8_t* weight_zeros) {
          if (weight_zeros) {
            // TODO check all zeros
            assert (weight_zeros[0] == 8);
          }
          auto ukernel = get_ukernel();
          auto rhs_pack = get_rhs_packing();
          rhs_packing::qparams_t qparams{1, 8};
          // @nocommit - Unsigned hack, add a naive packing routine
          rhs_pack.run_rhs_pack(/*groups=*/1, n, k, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr(), 
            group_size, reinterpret_cast<const uint8_t*>(weight_qvals), /*bias=*/nullptr, weight_data, /*extra_bytes=*/0, &qparams);
      }

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
        float clamp_max) {
          auto ukernel = get_ukernel();
          ukernel.run_matmul(m, n, k, group_size, activation_data, weight_data, output, output_m_stride, /*dst_stride_col=*/1, clamp_min, clamp_max);
      }

      size_t get_alignement() {
          return 16;
      }
    } // namespace neon_dotprod_1x4x32
  } // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
