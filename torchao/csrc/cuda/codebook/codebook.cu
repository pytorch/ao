// modified from https://github.com/Vahe1994/AQLM/tree/ab272bfe09915f84bc4e2439055dd7d0e82e08ca/inference_lib/src/aqlm/inference_kernels
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>

namespace torchao {

template<bool use_bfloat16, size_t group_size>
__global__ void Code1x16MatVec(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / group_size;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  __shared__ int4 sh_b[32 * (group_size + 1)];
  float res = 0;

  int iters = (prob_k / group_size + group_size * 32 - 1) / (group_size * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * group_size; i += blockDim.x) {
      if (8 * (b_gl_rd + i) < prob_k)
        sh_b[(group_size + 1) * (i / group_size) + i % group_size] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * group_size;

    int b_sh_rd = (group_size + 1) * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[group_size / 2];
        // We bypass the L1 cache to avoid massive amounts of memory streaming that doesn't
        // actually help us; this brings > 2x speedup.
        asm volatile (
          "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
          : "l"((void*) &codebook[(group_size / 8) * enc[i]])
        );
        if constexpr (group_size == 16) {
          asm volatile (
            "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(dec[4]), "=r"(dec[5]), "=r"(dec[6]), "=r"(dec[7])
            : "l"((void*) &codebook[(group_size / 8) * enc[i] + 1])
          );
        }
        if constexpr (use_bfloat16) {
        #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
          nv_bfloat162* a = reinterpret_cast<nv_bfloat162*>(&dec);
          nv_bfloat162* b = reinterpret_cast<nv_bfloat162*>(&sh_b[b_sh_rd]);
          nv_bfloat162 res2 = {};
          #pragma unroll
          for (int j = 0; j < group_size / 2; j++)
            res2 = __hfma2(a[j], b[j], res2);
          res += __bfloat162float(res2.x) + __bfloat162float(res2.y);
        #endif
        } else {
          half2* a = reinterpret_cast<half2*>(&dec);
          half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
          half2 res2 = {};
          #pragma unroll
          for (int j = 0; j < group_size / 2; j++)
            res2 = __hfma2(a[j], b[j], res2);
          res += __half2float(res2.x) + __half2float(res2.y);
        }
        b_sh_rd += group_size / 8;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
    #pragma unroll
    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      if constexpr (use_bfloat16) {
        reinterpret_cast<__nv_bfloat16*>(C)[c_gl_wr] = __float2bfloat16(res);
      } else {
        reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
      }
    }
  }
}


template<size_t group_size>
__global__ void Code1x16Dequant(
  const int4* __restrict__ A,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / group_size;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  int iters = (prob_k / group_size + group_size * 32 - 1) / (group_size * 32);
  while (iters--) {
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[group_size / 2];
        // We bypass the L1 cache to avoid massive amounts of memory streaming that doesn't
        // actually help us; this brings > 2x speedup.
        asm volatile (
          "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
          : "l"((void*) &codebook[(group_size / 8) * enc[i]])
        );
        if constexpr (group_size == 16) {
          asm volatile (
            "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(dec[4]), "=r"(dec[5]), "=r"(dec[6]), "=r"(dec[7])
            : "l"((void*) &codebook[(group_size / 8) * enc[i] + 1])
          );
        }

        C[a_gl_rd * group_size + (group_size / 8) * i] = reinterpret_cast<int4*>(&dec)[0];
        if constexpr (group_size == 16) {
          C[a_gl_rd * group_size + (group_size / 8) * i + 1] = reinterpret_cast<int4*>(&dec)[1];
        }
      }
    }
    a_gl_rd += 32;
  }
}

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

const int THREAD_M = 16;

template <bool use_bfloat16, size_t group_size>
void  code1x16_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  if (cc_major < 8 && use_bfloat16) {
    throw c10::TypeError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "You're trying to run AQLM with bfloat16 on a GPU with low compute capability. Use torch.float16 instead."
      )
    );
  }

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16MatVec<use_bfloat16, group_size><<<blocks, threads, 16*32*(group_size + 1), stream>>>(
    (const int4*) A,
    (const int4*) B,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}

template void code1x16_matvec_cuda<false, 8>(const void*, const void*, void*, const void*, int, int);
template void code1x16_matvec_cuda<true, 8>(const void*, const void*, void*, const void*, int, int);
template void code1x16_matvec_cuda<false, 16>(const void*, const void*, void*, const void*, int, int);
template void code1x16_matvec_cuda<true, 16>(const void*, const void*, void*, const void*, int, int);

template <size_t group_size>
void  code1x16_dequant_cuda(
  const void* __restrict__ A,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16Dequant<group_size><<<blocks, threads, 0, stream>>>(
    (const int4*) A,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}

template void code1x16_dequant_cuda<8>(const void*, void*, const void*, int, int);
template void code1x16_dequant_cuda<16>(const void*, void*, const void*, int, int);




inline bool check_use_bfloat16(const torch::Tensor& input) {
  auto dtype = input.dtype();
  if (dtype == at::kHalf) {
    return false;
  } else if (dtype == at::kBFloat16) {
    return true;
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support float16 and bfloat16. Got ",
        dtype.name(),
        ". Please specify the correct `torch_dtype` when loading the model."
      )
    );
  }
}

inline torch::Tensor scale_bias_unflatten_output(
        torch::Tensor& flat_output,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias,
  const c10::IntArrayRef& input_sizes
) {
  flat_output *= scales.flatten().unsqueeze(0);
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(flat_output.size(-1));
  auto output = flat_output.reshape(output_sizes).clone();
  return output;
}

void code1x16_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  const bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  if (codebook.size(3) == 8) {
    if (use_bfloat16) {
      code1x16_matvec_cuda<true, 8>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    } else {
      code1x16_matvec_cuda<false, 8>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    }
  } else if (codebook.size(3) == 16) {
    if (use_bfloat16) {
      code1x16_matvec_cuda<true, 16>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    } else {
      code1x16_matvec_cuda<false, 16>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    }
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebook.size(3),
        "."
      )
    );
  }
}

torch::Tensor code1x16_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code1x16_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      use_bfloat16
    );
  }
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code1x16_dequant(
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales
) {
  check_use_bfloat16(codebooks);
  auto in_features = codes.size(1) * codebooks.size(3);
  auto out_features = scales.size(0);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }
  weight *= scales.index({"...", 0, 0});

  return weight;
}

int4 accumulate_sizes(const torch::Tensor& codebook_partition_sizes)
{
  int4 cumulative_sizes;
  auto cumulative_size = &cumulative_sizes.x;
  int i = 0;
  int last = 0;
  assert(codebook_partition_sizes.size(0) <= 4);
  for (; i <  codebook_partition_sizes.size(0); ++i, ++cumulative_size)
  {
    *cumulative_size = codebook_partition_sizes[i].item<int>() + last;
    last = *cumulative_size;
  }
  // fill in the rest with unreachable.
  for (; i < 4; ++i, ++cumulative_size)
  {
    *cumulative_size = last*10;
  }
  return cumulative_sizes;
}

torch::Tensor code1x16_matmat_dequant(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);

  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * codebooks.size(3);
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }

  auto flat_output = at::native::linear(flat_input, weight);
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}



TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("code1x16_matmat", &code1x16_matmat);
  m.impl("code1x16_matmat_dequant", &code1x16_matmat_dequant);
}

} // namespace torchao
