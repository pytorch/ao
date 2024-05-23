import os
import unittest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4
try:
    from torchao.prototype.fp8 import gemm_split_k, to_float8
    triton_available = True
except ImportError:
    triton_available = False

@unittest.skipIf(not triton_available, "Triton is required but not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFP8Gemm(TestCase):
    # @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_gemm_split_k(self):
        dtype = torch.float16
        qdtype = torch.float8_e4m3fn

        torch.cuda.manual_seed(0)

        m = 64
        n = 4096
        k = 4096

        # create test inputs
        x = torch.randn((m, k), dtype=dtype, device='cuda')
        w = torch.randn((n, k), dtype=dtype, device='cuda')

        x_fp8, x_inv_s = to_float8(x, dtype=qdtype)
        w_fp8, w_inv_s = to_float8(w, dtype=qdtype)

        y_torch, _ = torch._scaled_mm(x_fp8, w_fp8.t(), out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
        y_triton = gemm_split_k(x_fp8, w_fp8.t(), scale_a=x_inv_s.item(), scale_b=w_inv_s.item())
        y_fp16 = torch.nn.functional.linear(x, w)

        cos_sim_torch = torch.nn.functional.cosine_similarity(y_fp16.reshape(-1), y_torch.reshape(-1), dim=0)
        cos_sim_triton = torch.nn.functional.cosine_similarity(y_fp16.reshape(-1), y_triton.reshape(-1), dim=0)

        assert cos_sim_torch > 0.99, f"fp16 vs torch cos_sim is too low: {cos_sim_torch}"
        assert cos_sim_triton > 0.99, f"fp16 vs triton cos_sim is too low: {cos_sim_triton}"

    # https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
    @unittest.skip("fp8 kernel compilation does not work on a10g")
    def test_user_defined_triton_function(self):
        m, n, k = 256, 256, 512

        a = torch.randn((m, k), dtype=torch.float16, device="cuda")
        b = torch.randn((k, n), dtype=torch.float16, device="cuda")
        compiled_function = torch.compile(gemm_split_k, fullgraph=True)(a,b)

if __name__ == "__main__":
    run_tests()
