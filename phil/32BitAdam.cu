#include <stdio.h>
#include <iostream>

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, 
                              float eps, float weight_decay) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

int main() {
    long num_parameters = 4;

    float params_memory[num_parameters] = {0.8477, 0.3092, 0.2363, 0.2300};
    float * d_params_memory;
    cudaMalloc(&d_params_memory, 4 * sizeof(float));
    cudaMemcpy(d_params_memory, params_memory, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());

    // fake gradients
    float grads_memory[num_parameters] = {0.8530, 0.7153, 0.1018, 0.4003};;
    float * d_grads_memory;
    cudaMalloc(&d_grads_memory, sizeof(grads_memory));
    cudaMemcpy(d_grads_memory, grads_memory, sizeof(grads_memory), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());

    float m_memory[num_parameters] = {0, 0, 0, 0};
    float * d_m_memory;
    cudaMalloc(&d_m_memory, sizeof(m_memory));
    cudaMemcpy(d_m_memory, m_memory, sizeof(m_memory), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());
    
    float v_memory[num_parameters] = {0, 0, 0, 0};
    float * d_v_memory;
    cudaMalloc(&d_v_memory, sizeof(v_memory));
    cudaMemcpy(d_v_memory, v_memory, sizeof(v_memory), cudaMemcpyHostToDevice);
    cudaCheck(cudaGetLastError());
    
    // standard choices / https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    float learning_rate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    int t = 1;
    float beta1_correction = 1.0f - powf(beta1, t);;
    float beta2_correction = 1.0f - powf(beta2, t);;
    float eps = 1e-08;
    float weight_decay = 0.01;
    
    adamw_kernel2<<<1, num_parameters>>>(d_params_memory, d_grads_memory, d_m_memory, d_v_memory, num_parameters,
                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, 
                              eps, weight_decay);
    cudaCheck(cudaGetLastError());

    cudaMemcpy(params_memory, d_params_memory, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck(cudaGetLastError());

    // TODO - Free device memory
    
    for (int i = 0; i < num_parameters; i++) {
        std::cout << "Updated parameters: " << params_memory[i] << "\n";
    }
}