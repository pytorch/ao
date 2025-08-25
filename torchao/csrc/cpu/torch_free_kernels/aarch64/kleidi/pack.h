// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <kai/kai_common.h>
#include <kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h>

namespace torchao::kernels::cpu::aarch64::kleidi {
namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p {
// All the kernels in this namespace use following packing interface/routines.
// TODO: move these to Kleidi as interfaces?
typedef struct rhs_packing {
  typedef struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params qparams_t;
  typedef size_t (*get_rhs_offset_t)(size_t, size_t);
  typedef size_t (*get_rhs_packed_stride_t)(
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      enum kai_datatype);
  typedef size_t (*get_rhs_packed_offset_t)(
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      enum kai_datatype);
  typedef size_t (*get_rhs_packed_size_t)(
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      enum kai_datatype);
  typedef void (*run_rhs_pack_t)(
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      const uint8_t*,
      size_t,
      const float*,
      const void*,
      size_t,
      void*,
      size_t,
      const qparams_t*);

  get_rhs_offset_t get_rhs_offset;
  get_rhs_packed_stride_t get_rhs_packed_stride;
  get_rhs_packed_offset_t get_rhs_packed_offset;
  get_rhs_packed_size_t get_rhs_packed_size;
  run_rhs_pack_t run_rhs_pack;
} rhs_packing;

// TODO add transpose variant i.e kxn
rhs_packing get_rhs_packing() {
  return rhs_packing{
      .get_rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
      .get_rhs_packed_stride =
          kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
      .get_rhs_packed_offset =
          kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
      .get_rhs_packed_size =
          kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
      .run_rhs_pack = kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0};
}

typedef struct lhs_packing {
  typedef size_t (*get_lhs_m_step_t)(size_t);
  typedef size_t (*get_lhs_offset_t)(size_t, size_t);
  typedef size_t (
      *get_lhs_packed_offset_t)(size_t, size_t, size_t, size_t, size_t);
  typedef size_t (
      *get_lhs_packed_size_t)(size_t, size_t, size_t, size_t, size_t);
  typedef void (*run_lhs_pack_t)(
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      size_t,
      const float*,
      size_t,
      void*);

  get_lhs_m_step_t get_lhs_m_step;
  get_lhs_offset_t get_lhs_offset;
  get_lhs_packed_offset_t get_lhs_packed_offset;
  get_lhs_packed_size_t get_lhs_packed_size;
  run_lhs_pack_t run_lhs_pack;
} lhs_packing;

lhs_packing get_lhs_packing() {
  return lhs_packing{
      .get_lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
      .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
      .get_lhs_packed_offset =
          kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
      .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
      .run_lhs_pack = kai_run_lhs_quant_pack_qai8dxp_f32};
}

} // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
