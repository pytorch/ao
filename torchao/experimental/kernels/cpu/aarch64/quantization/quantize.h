// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include <arm_neon.h>

// These methods are here temporarily
// Eventually they will be moved to a non-arch specific location
// or replaced by existing PyTorch functions
// The quantize method in aarch64 namespace will remain here;
// it is used for dynamic activation quantization
namespace torchao {
namespace quantization {

void get_qvals_range(int& qmin, int& qmax, int nbit, bool is_symmetric);

// val = scale * qval
float get_scale(float vmin, float vmax, int qmin, int qmax);

// val = scale * (qval - zero)
void get_scale_and_zero(
    float& scale,
    int& zero,
    float vmin,
    float vmax,
    int qmin,
    int qmax);

} // namespace quantization
} // namespace torchao

namespace torchao {
namespace kernels {
namespace cpu {
namespace aarch64 {
namespace quantization {
void quantize(
    // Output
    int8_t* qvals,
    // Inputs
    const float32_t* vals,
    int size,
    float32_t scale,
    int8_t zero,
    int8_t qmin,
    int8_t qmax);

} // namespace quantization
} // namespace aarch64
} // namespace cpu
} // namespace kernels
} // namespace torchao
