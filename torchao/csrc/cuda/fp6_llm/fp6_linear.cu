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
// This file is adapted from https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/fp6_linear.cu

#include "kernel_matmul.cuh"
#include "kernel_reduction.cuh"

#include <stdio.h>
#include <assert.h>

template<typename TilingConfig, typename OutputDataType>
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
    static size_t SHMEM_SZ = max(TilingConfig::SMEM_SIZE_B_TILE+SMEM_SIZE_A1_TILE+SMEM_SIZE_A2_TILE, TilingConfig::SMEM_SIZE_C_TILE);
    cudaFuncSetAttribute(QUANT_GEMM_Kernel<TilingConfig, OutputDataType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
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
    QUANT_GEMM_Kernel<TilingConfig, OutputDataType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>
                    (Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
}

/*
 *
 */
cudaError_t fp6_linear_kernel(cudaStream_t    stream,
                              const uint4     *Weight,
                              const half      *Scales,
                              const half      *B,
                              half            *C,
                              const size_t    M_Global,
                              const size_t    N_Global,
                              const size_t    K_Global, 
                              float           *Reduction_Workspace,  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
                              int             Split_K)
{
    assert(M_Global % 256 == 0);
    assert(K_Global % 64 == 0);
    assert(N_Global>0);

    // Work around to support more N shapes:
    size_t N_PowerOf2;
    if(N_Global>0 &&  N_Global<=8)      N_PowerOf2 = 8;
    if(N_Global>8 &&  N_Global<=16)     N_PowerOf2 = 16;
    if(N_Global>16 && N_Global<=32)     N_PowerOf2 = 32;
    if(N_Global>32 && N_Global<=64)     N_PowerOf2 = 64;
    if(N_Global>64 && N_Global<=128)    N_PowerOf2 = 128;
    if(N_Global>128)                    N_PowerOf2 = ((N_Global-1)/128+1) * 128;

    if (Split_K == 1) {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
        }
    }
    else {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
        }
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    return cudaGetLastError();
}


#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchao {
/*
Computes FP6-FP16 GEMM (PyTorch interface).

[Mathmatical Formula]
Standard definition of linear layer:    Out = In * trans(W), where In, Out, and W are stored in row-major.
After Equivalent transformation    :    trans(Out) = W * trans(In). Note that we do not perform "transpose" during runtime, we instead interpret the In/Out as column-major matrices when calling our CUDA kernel.

[Inputs]
  _in_feats:  tensor of shape [B, IC];                  // half 
  _weights:   int tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6 weights.
  _scales:    tensor of shape [OC];                     // half
  splitK:     spliting the MatMul problem along K dimension for higher GPU utilization, default 1.
[Outputs]
  _out_feats: tensor of shape [B, OC];                  // half
*/
torch::Tensor fp6_linear_forward_cuda(torch::Tensor _in_feats,
                                      torch::Tensor _weights,
                                      torch::Tensor _scales,
                                      int64_t       splitK=1)
{
    int num_in_feats      = _in_feats.size(0);
    int num_in_channels   = _in_feats.size(1);
    int num_out_channels  = _weights.size(0);
    TORCH_CHECK(num_in_channels%64 == 0, "Expected in_features to be a multiple of 64, but received ", num_in_channels);
    TORCH_CHECK((num_in_channels/16*3) == _weights.size(1));    // Making sure the K dimension is matched.
    //
    int M = num_out_channels;
    int K = num_in_channels;
    int N = num_in_feats;
    // Input Tensors
    auto weight = reinterpret_cast<const uint4*>(_weights.data_ptr<int>());  // weights is [OC, IC] but in FP6.
    auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales   = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
    // Output Tensors
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, num_out_channels}, options);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    options = torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device());
    at::Tensor _workspace = torch::empty({splitK, num_in_feats, num_out_channels}, options);
    auto Reduction_Workspace = reinterpret_cast<float*>(_workspace.data_ptr<float>());  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
      
    fp6_linear_kernel(0, // Using default stream here.
                      weight,
                      scales,
                      in_feats,
                      out_feats,
                      M,
                      N,
                      K, 
                      Reduction_Workspace,  
                      splitK);

    return _out_feats;
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::fp6_llm_linear", &fp6_linear_forward_cuda);
}

} // namespace torchao
