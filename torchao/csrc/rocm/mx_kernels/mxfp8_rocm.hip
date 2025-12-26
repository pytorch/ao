// MXFP8 quantization kernels for ROCm
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

namespace mxfp8 {

constexpr int FP32_MANTISSA_BITS = 23;
constexpr int FP32_EXPONENT_BIAS = 127;
constexpr float E4M3_MAX_NORM = 448.0f;
constexpr int E4M3_MAX_EXPONENT = 8;

enum class ScalingMode { FLOOR, RCEIL };

// Convert amax to e8m0 scale (FLOOR mode: take exponent directly)
__device__ __forceinline__ uint8_t amax_to_scale_e8m0_floor(float amax) {
    int32_t int_amax = *reinterpret_cast<int32_t*>(&amax);
    int32_t extracted_pow2 = ((int_amax >> FP32_MANTISSA_BITS) & 0xFF) - FP32_EXPONENT_BIAS;
    int32_t scale_unbiased = extracted_pow2 - E4M3_MAX_EXPONENT;
    scale_unbiased = max(scale_unbiased, -FP32_EXPONENT_BIAS);
    scale_unbiased = min(scale_unbiased, FP32_EXPONENT_BIAS + 1);
    return static_cast<uint8_t>(scale_unbiased + FP32_EXPONENT_BIAS);
}

// Convert amax to e8m0 scale (RCEIL mode: round up if mantissa > 0)
// CUDA does: float_to_e8m0(amax / 448.0f)
__device__ __forceinline__ uint8_t amax_to_scale_e8m0_rceil(float amax) {
    constexpr float E4M3_MAX_NORM_RCP = 1.0f / 448.0f;
    float val = amax * E4M3_MAX_NORM_RCP;
    
    if (val == 0.0f) return 0x00;
    
    uint32_t val_u32 = *reinterpret_cast<uint32_t*>(&val);
    uint8_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    
    // Round up exponent if mantissa > 0 (with saturation handling)
    if ((mantissa > 0 && exponent != 0xFE) &&
        !(exponent == 0 && mantissa <= 0x400000)) {
        ++exponent;
    }
    
    return exponent;
}

// Convert e8m0 scale to inverse scale: 2^(127 - scale)
__device__ __forceinline__ float e8m0_to_inv_scale(uint8_t scale) {
    if (scale == 0) return 1.0f;
    return exp2f(FP32_EXPONENT_BIAS - static_cast<float>(scale));
}

// MXFP8 colwise quantization kernel
// Input: row-major [rows, cols]
// Output: column-major [rows, cols] with stride {1, rows}
// Scales: column-major [cols, num_row_blocks] with stride {1, cols}
template<typename IType, typename OType, int BLOCK_SIZE, ScalingMode MODE>
__global__ void mxfp8_quantize_colwise_kernel(
    const IType *input,
    OType *output,
    uint8_t *scales,
    int64_t rows,
    int64_t cols) {
    
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    const int row_block = blockIdx.x;
    const int row_start = row_block * BLOCK_SIZE;
    const int row_end = min(row_start + BLOCK_SIZE, static_cast<int>(rows));
    
    if (col >= cols) return;
    
    const IType *input_ptr = input + row_start * cols + col;
    OType *output_ptr = output + row_start + col * rows;
    
    // Pass 1: compute abs-max
    float amax = 0.0f;
    for (int i = 0; i < row_end - row_start; i++) {
        float val = static_cast<float>(input_ptr[i * cols]);
        amax = fmaxf(amax, fabsf(val));
    }
    
    // Compute scale and inverse scale
    uint8_t scale_e8m0;
    if constexpr (MODE == ScalingMode::FLOOR) {
        scale_e8m0 = amax_to_scale_e8m0_floor(amax);
    } else {
        scale_e8m0 = amax_to_scale_e8m0_rceil(amax);
    }
    float inv_scale = e8m0_to_inv_scale(scale_e8m0);
    
    // Pass 2: quantize
    for (int i = 0; i < row_end - row_start; i++) {
        float val = static_cast<float>(input_ptr[i * cols]);
        float scaled_val = val * inv_scale;
        scaled_val = fminf(fmaxf(scaled_val, -E4M3_MAX_NORM), E4M3_MAX_NORM);
        output_ptr[i] = static_cast<OType>(scaled_val);
    }
    
    scales[col + row_block * cols] = scale_e8m0;
}

void mxfp8_quantize_rocm(const at::Tensor &input,
                         at::Tensor &output,
                         at::Tensor &scales,
                         bool colwise,
                         int64_t block_size,
                         const std::string &scaling_mode) {
    TORCH_CHECK(colwise, "Only colwise quantization is supported");
    TORCH_CHECK(block_size == 32, "Only block_size=32 is supported");
    TORCH_CHECK(scaling_mode == "floor" || scaling_mode == "rceil",
                "scaling_mode must be 'floor' or 'rceil'");

    constexpr int THREADBLOCK_SIZE = 512;

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);

    dim3 grid((rows + block_size - 1) / block_size, 
              (cols + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE);
    dim3 block(THREADBLOCK_SIZE);

    auto stream = at::cuda::getCurrentHIPStream();

    if (scaling_mode == "floor") {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            input.scalar_type(), "mxfp8_quantize_colwise_rocm", [&] {
                mxfp8_quantize_colwise_kernel<scalar_t, __hip_fp8_e4m3, 32, ScalingMode::FLOOR>
                    <<<grid, block, 0, stream>>>(
                        input.data_ptr<scalar_t>(),
                        reinterpret_cast<__hip_fp8_e4m3*>(output.data_ptr()),
                        reinterpret_cast<uint8_t*>(scales.data_ptr()),
                        rows,
                        cols);
            });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            input.scalar_type(), "mxfp8_quantize_colwise_rocm", [&] {
                mxfp8_quantize_colwise_kernel<scalar_t, __hip_fp8_e4m3, 32, ScalingMode::RCEIL>
                    <<<grid, block, 0, stream>>>(
                        input.data_ptr<scalar_t>(),
                        reinterpret_cast<__hip_fp8_e4m3*>(output.data_ptr()),
                        reinterpret_cast<uint8_t*>(scales.data_ptr()),
                        rows,
                        cols);
            });
    }
}

} // namespace mxfp8
