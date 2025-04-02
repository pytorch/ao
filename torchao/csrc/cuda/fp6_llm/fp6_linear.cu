// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
//    Copyright 2024 FP6-LLM authors
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
// This file is adapted from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/fp6_linear.cu
//
// MODIFICATION NOTE (2024-09-25): added SM75 support (https://github.com/pytorch/ao/pull/942):
// - Modified the TilingConfig parameters for SM75 to deal with smaller shared memory
// - Added proper architecture check at both host and device level
//


#include "kernel_matmul.cuh"
#include "kernel_reduction.cuh"

#include <stdio.h>
#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>


// https://github.com/Dao-AILab/flash-attention/blob/478ee666cccbd1b8f63648633003059a8dc6827d/hopper/utils.h#L25
#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


template<typename TilingConfig, typename InputDataType, typename OutputDataType, int EXPONENT, int MANTISSA>
static void Kernel_Ex(cudaStream_t    stream,
                      const uint4     *Weight,
                      const half      *Scales,
                      const half      *B,
                      OutputDataType  *C,
                      const size_t    M_Global,
                      const size_t    N_Global,
                      const size_t    K_Global,
                      int             Split_K)
{
    #ifdef DEBUG_MODE
        printf("\n");
        printf("Launcher.cu->Kernel_Ex():\n");
        printf("M: %d, N: %d, K: %d, SplitK: %d\n", M_Global, N_Global, K_Global, Split_K);
        printf("TILE_M: %d, TILE_K: %d, TILE_N: %d\n", TilingConfig::TILE_M, TilingConfig::TILE_K, TilingConfig::TILE_N);
    #endif
    static size_t SHMEM_SZ = max(TilingConfig::SMEM_SIZE_B_TILE+SMEM_SIZE_PER_TB_A_TILE, TilingConfig::SMEM_SIZE_C_TILE);
    cudaFuncSetAttribute(QUANT_GEMM_Kernel<TilingConfig, InputDataType, OutputDataType, EXPONENT, MANTISSA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    size_t  dimN = (N_Global-1) / TilingConfig::TILE_N + 1;
    size_t  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3    GridDim(dimN, dimM, 1);
    dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    //
    #ifdef DEBUG_MODE
        printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
                GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
        printf("\n");
    #endif
    QUANT_GEMM_Kernel<TilingConfig, InputDataType, OutputDataType, EXPONENT, MANTISSA><<<GridDim, BlockDim, SHMEM_SZ, stream>>>
                    (Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename InputDataType, int EXPONENT, int MANTISSA>
void        fpx_linear_kernel(cudaStream_t    stream,
                              const uint4     *Weight,
                              const half      *Scales,
                              const half      *B,
                              InputDataType   *C,
                              const size_t    M_Global,
                              const size_t    N_Global,
                              const size_t    K_Global,
                              float           *Reduction_Workspace,  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
                              int             Split_K)
{
    static_assert(std::is_same<InputDataType, half>::value || std::is_same<InputDataType, __nv_bfloat16>::value, "Type must be 'half' or '__nv_bfloat16'");
    assert(M_Global % 256 == 0);
    assert(K_Global % 64 == 0);
    assert(N_Global > 0);

    // Check GPU Compute Capability before proceeding
    int device, major, minor;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    // Early exit with error for unsupported architectures
    if ((major < 7) || (major == 7 && minor < 5)) {
        TORCH_CHECK(false, "Quant-LLM Error: This kernel requires GPU with SM75 (Turing) or higher architecture. "
                         "Your current device has SM", major, minor, " which is not supported.");
    }

    const bool is_sm75_gpu = (major == 7) && (minor == 5);
    if (is_sm75_gpu && std::is_same<InputDataType, __nv_bfloat16>::value) {
        TORCH_CHECK(false, "Quant-LLM Error: BFloat16 inputs are not supported on SM75 (Turing) GPUs.");
    }

    // Work around to support more N shapes:
    size_t N_PowerOf2;
    if(N_Global>0 &&  N_Global<=8)      N_PowerOf2 = 8;
    if(N_Global>8 &&  N_Global<=16)     N_PowerOf2 = 16;
    if(N_Global>16 && N_Global<=32)     N_PowerOf2 = 32;
    if(N_Global>32 && N_Global<=64)     N_PowerOf2 = 64;
    if(N_Global>64 && N_Global<=128)    N_PowerOf2 = 128;
    if(N_Global>128)                    N_PowerOf2 = ((N_Global-1)/128+1) * 128;

    if (is_sm75_gpu && (N_PowerOf2 == 64 || N_PowerOf2 == 128 || N_PowerOf2 % 128 == 0)) {
        // For SM75 and N >= 64, we use a different TilingConfig to deal with smaller shared memory.
        if (Split_K == 1) {
            Kernel_Ex<TilingConfig<4, 1, 4>, InputDataType, InputDataType, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
        } else {
            Kernel_Ex<TilingConfig<4, 1, 4>, InputDataType, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
        }
    } else {
        if (Split_K == 1) {
            switch (N_PowerOf2) {
                case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, InputDataType, InputDataType, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
                case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, InputDataType, InputDataType, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
                case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, InputDataType, InputDataType, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
                case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, InputDataType, InputDataType, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
                case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, InputDataType, InputDataType, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
                default:    if (N_PowerOf2 % 128 != 0) {
                                TORCH_CHECK(false, "Quant-LLM Error: Unsupported N dimension ", N_PowerOf2);
                            }
                            Kernel_Ex<TilingConfig<4, 1, 8>, InputDataType, InputDataType, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            }
        }
        else {
            switch (N_PowerOf2) {
                case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, InputDataType, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
                case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, InputDataType, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
                case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, InputDataType, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
                case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, InputDataType, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
                case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, InputDataType, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
                default:    if (N_PowerOf2 % 128 != 0) {
                                TORCH_CHECK(false, "Quant-LLM Error: Unsupported N dimension ", N_PowerOf2);
                            }
                            Kernel_Ex<TilingConfig<4, 1, 8>, InputDataType, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            }
        }
    }

    if (Split_K != 1) {
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<InputDataType><<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
        CHECK_CUDA_KERNEL_LAUNCH();
    }
}


// https://github.com/NVIDIA/apex/blob/master/csrc/type_shim.h
#define DISPATCH_HALF_AND_BF16(TYPE, NAME, ...)                                \
  switch (TYPE) {                                                              \
  case at::ScalarType::Half: {                                                 \
    using torch_t = at::Half;                                                  \
    using nv_t = half;                                                         \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  case at::ScalarType::BFloat16: {                                             \
    using torch_t = at::BFloat16;                                              \
    using nv_t = __nv_bfloat16;                                                \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");            \
  }

namespace torchao {
// MODIFICATION NOTE: dtype of _weights is changed to uint8
/*
Computes FPx-FP16 GEMM (PyTorch interface).

[Mathmatical Formula]
Standard definition of linear layer:    Out = In * trans(W), where In, Out, and W are stored in row-major.
After Equivalent transformation    :    trans(Out) = W * trans(In). Note that we do not perform "transpose" during runtime, we instead interpret the In/Out as column-major matrices when calling our CUDA kernel.

[Inputs]
  _in_feats:  tensor of shape [B, IC];                  // half or bf16
  _weights:   int tensor of shape [OC, IC // 8 * x];    // x UINT8 words contains 8 FPx weights.
  _scales:    tensor of shape [OC];                     // half or bf16
  splitK:     spliting the MatMul problem along K dimension for higher GPU utilization, default 1.
[Outputs]
  _out_feats: tensor of shape [B, OC];                  // half or bf16
*/
torch::Tensor fp_eXmY_linear_forward_cuda(
    int64_t         EXPONENT,
    int64_t         MANTISSA,
    torch::Tensor   _in_feats,
    torch::Tensor   _weights,
    torch::Tensor   _scales,
    int64_t         splitK=1)
{
    // Check GPU Compute Capability before proceeding
    int device, major, minor;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    // Early exit with error for unsupported architectures
    if ((major < 7) || (major == 7 && minor < 5)) {
        TORCH_CHECK(false, "Quant-LLM Error: This kernel requires GPU with SM75 (Turing) or higher architecture. "
                         "Your current device has SM", major, minor, " which is not supported.");
    }

    const bool is_sm75_gpu = (major == 7) && (minor == 5);
    if (is_sm75_gpu && _in_feats.scalar_type() == at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "Quant-LLM Error: BFloat16 inputs are not supported on SM75 (Turing) GPUs.");
    }

    const int64_t NBITS   = 1 + EXPONENT + MANTISSA;
    int num_in_feats      = _in_feats.size(0);
    int num_in_channels   = _in_feats.size(1);
    int num_out_channels  = _weights.size(0);
    TORCH_CHECK(num_in_channels % 64 == 0, "Expected in_features to be a multiple of 64, but received ", num_in_channels);
    TORCH_CHECK((num_in_channels / 8 * NBITS) == _weights.size(1));    // Making sure the K dimension is matched.
    //
    int M = num_out_channels;
    int K = num_in_channels;
    int N = num_in_feats;
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, num_out_channels}, options);

    options = torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device());
    at::Tensor _workspace = torch::empty({splitK, num_in_feats, num_out_channels}, options);
    auto Reduction_Workspace = reinterpret_cast<float*>(_workspace.data_ptr<float>());  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)

    // MODIFICATION NOTE: use at::cuda::getCurrentCUDAStream() instead of default stream (0)
    // this fixes problem with CUDA graphs when used with torch.compile()
    auto stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_HALF_AND_BF16(_in_feats.scalar_type(), "fpx_linear_kernel", [&] {
        auto weight = reinterpret_cast<const uint4*>(_weights.data_ptr<uint8_t>());  // weights is [OC, IC] but in FP6.
        auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<torch_t>());
        auto scales = reinterpret_cast<const half*>(_scales.data_ptr<torch_t>());
        auto out_feats = reinterpret_cast<nv_t*>(_out_feats.data_ptr<torch_t>());

        // officially supported in Quant-LLM
        if (EXPONENT == 3 && MANTISSA == 2)
            fpx_linear_kernel<nv_t, 3, 2>(stream, weight, scales, in_feats, out_feats, M, N, K, Reduction_Workspace, splitK);
        else if (EXPONENT == 2 && MANTISSA == 2)
            fpx_linear_kernel<nv_t, 2, 2>(stream, weight, scales, in_feats, out_feats, M, N, K, Reduction_Workspace, splitK);

        // experimental
        else if (EXPONENT == 2 && MANTISSA == 3)
            fpx_linear_kernel<nv_t, 2, 3>(stream, weight, scales, in_feats, out_feats, M, N, K, Reduction_Workspace, splitK);
        else if (EXPONENT == 3 && MANTISSA == 1)
            fpx_linear_kernel<nv_t, 3, 1>(stream, weight, scales, in_feats, out_feats, M, N, K, Reduction_Workspace, splitK);
        // else if (EXPONENT == 2 && MANTISSA == 1)
        //     fpx_linear_kernel<nv_t, 2, 1>(stream, weight, scales, in_feats, out_feats, M, N, K, Reduction_Workspace, splitK);
        // else if (EXPONENT == 3 && MANTISSA == 0)
        //     fpx_linear_kernel<nv_t, 3, 0>(stream, weight, scales, in_feats, out_feats, M, N, K, Reduction_Workspace, splitK);
        // else if (EXPONENT == 2 && MANTISSA == 0)
        //     fpx_linear_kernel<nv_t, 2, 0>(stream, weight, scales, in_feats, out_feats, M, N, K, Reduction_Workspace, splitK);

        else
            TORCH_CHECK(false, "FP", NBITS, " E", EXPONENT, "M", MANTISSA, " is not supported.");
    });

    return _out_feats;
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::quant_llm_linear", &fp_eXmY_linear_forward_cuda);
}

} // namespace torchao
