// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/mps/linear_fp_act_xbit_weight.h>

using torchao::kernels::mps::lowbit::aten::linear_mps_kernel;
using torchao::kernels::mps::lowbit::aten::linear_mps_kernel_meta;
using torchao::kernels::mps::lowbit::aten::pack_weights_cpu_kernel;

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(torchao, m) {
  m.def("_pack_weight_1bit(Tensor W) -> Tensor");
  m.def("_pack_weight_2bit(Tensor W) -> Tensor");
  m.def("_pack_weight_3bit(Tensor W) -> Tensor");
  m.def("_pack_weight_4bit(Tensor W) -> Tensor");
  m.def("_pack_weight_5bit(Tensor W) -> Tensor");
  m.def("_pack_weight_6bit(Tensor W) -> Tensor");
  m.def("_pack_weight_7bit(Tensor W) -> Tensor");
  m.def(
      "_linear_fp_act_1bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_2bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_3bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_4bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_5bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_6bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_7bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("_pack_weight_1bit", &pack_weights_cpu_kernel<1>);
  m.impl("_pack_weight_2bit", &pack_weights_cpu_kernel<2>);
  m.impl("_pack_weight_3bit", &pack_weights_cpu_kernel<3>);
  m.impl("_pack_weight_4bit", &pack_weights_cpu_kernel<4>);
  m.impl("_pack_weight_5bit", &pack_weights_cpu_kernel<5>);
  m.impl("_pack_weight_6bit", &pack_weights_cpu_kernel<6>);
  m.impl("_pack_weight_7bit", &pack_weights_cpu_kernel<7>);
}

TORCH_LIBRARY_IMPL(torchao, MPS, m) {
  m.impl("_linear_fp_act_1bit_weight", &linear_mps_kernel<1>);
  m.impl("_linear_fp_act_2bit_weight", &linear_mps_kernel<2>);
  m.impl("_linear_fp_act_3bit_weight", &linear_mps_kernel<3>);
  m.impl("_linear_fp_act_4bit_weight", &linear_mps_kernel<4>);
  m.impl("_linear_fp_act_5bit_weight", &linear_mps_kernel<5>);
  m.impl("_linear_fp_act_6bit_weight", &linear_mps_kernel<6>);
  m.impl("_linear_fp_act_7bit_weight", &linear_mps_kernel<7>);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  m.impl("_linear_fp_act_1bit_weight", &linear_mps_kernel_meta<1>);
  m.impl("_linear_fp_act_2bit_weight", &linear_mps_kernel_meta<2>);
  m.impl("_linear_fp_act_3bit_weight", &linear_mps_kernel_meta<3>);
  m.impl("_linear_fp_act_4bit_weight", &linear_mps_kernel_meta<4>);
  m.impl("_linear_fp_act_5bit_weight", &linear_mps_kernel_meta<5>);
  m.impl("_linear_fp_act_6bit_weight", &linear_mps_kernel_meta<6>);
  m.impl("_linear_fp_act_7bit_weight", &linear_mps_kernel_meta<7>);
}
