#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/*
* Computes FP6-FP16 GEMM (C++ interface).
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
                              int             Split_K);

/*
 * In-place weight prepacking (C++ interface).
 */
void weight_matrix_prepacking(int* packed_weights, int *FP6Weights, size_t M, size_t K);

/*
 * Dequant a FP6 matrix to a equivalent FP16 matrix using CPUs.
 */
void DeQuantMatrix_FP6_To_FP16(half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale);

#ifndef NO_PYTORCH
#include <torch/extension.h>
#include <torch/library.h>

namespace torchao {
/*
* Computes FP6-FP16 GEMM (PyTorch interface).
*/
torch::Tensor fp6_linear_forward_cuda(torch::Tensor _in_feats,
                                      torch::Tensor _weights,
                                      torch::Tensor _scales,
                                      int           splitK=1);

/*
 * Weight prepacking (Pytorch interface).
 */
torch::Tensor weight_matrix_prepacking_cpu(torch::Tensor fp6_tensor);

/*
 * Dequant a FP6 matrix to a equivalent FP16 matrix using CPUs.
 * A useful tool to construct input matrices for the FP16 GEMM baseline.
 * [Input]
 *  fp6_tensor:  int  tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6  weights.
 *  fp16_scale:  half tensor of shape [OC];                 // for row-wise quantization.
 * [Output]
 *  fp16_tensor: half tensor of shape [OC, IC].     
 */
torch::Tensor weight_matrix_dequant_cpu(torch::Tensor fp6_tensor, torch::Tensor fp16_scale);

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::fp16act_fp6weight_linear", &fp6_linear_forward_cuda);
  m.impl("torchao::fp6_weight_prepacking_cpu", &weight_matrix_prepacking_cpu);
  m.impl("torchao::fp6_weight_dequant_cpu", &weight_matrix_dequant_cpu);
}

}
#endif
