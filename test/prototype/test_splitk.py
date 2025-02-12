import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

try:
    from torchao.prototype.splitk import gemm_split_k, to_float8

    triton_available = True
except ImportError:
    triton_available = False


from torchao.utils import skip_if_compute_capability_less_than, skip_if_rocm


@unittest.skipIf(not triton_available, "Triton is required but not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFP8Gemm(TestCase):
    @skip_if_compute_capability_less_than(9.0)
    @skip_if_rocm("ROCm enablement in progress")
    def test_gemm_split_k(self):
        dtype = torch.float16
        qdtype = torch.float8_e4m3fn

        torch.cuda.manual_seed(0)

        m = 64
        n = 4096
        k = 4096

        # create test inputs
        x = torch.randn((m, k), dtype=dtype, device="cuda")
        w = torch.randn((n, k), dtype=dtype, device="cuda")

        x_fp8, x_inv_s = to_float8(x, dtype=qdtype)
        w_fp8, w_inv_s = to_float8(w, dtype=qdtype)

        y_torch = torch._scaled_mm(
            x_fp8, w_fp8.t(), out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s
        )
        y_triton = gemm_split_k(
            x_fp8, w_fp8.t(), scale_a=x_inv_s.item(), scale_b=w_inv_s.item()
        )
        y_fp16 = torch.nn.functional.linear(x, w)

        cos_sim_torch = torch.nn.functional.cosine_similarity(
            y_fp16.reshape(-1), y_torch.reshape(-1), dim=0
        )
        cos_sim_triton = torch.nn.functional.cosine_similarity(
            y_fp16.reshape(-1), y_triton.reshape(-1), dim=0
        )

        assert (
            cos_sim_torch > 0.99
        ), f"fp16 vs torch cos_sim is too low: {cos_sim_torch}"
        assert (
            cos_sim_triton > 0.99
        ), f"fp16 vs triton cos_sim is too low: {cos_sim_triton}"

    # https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
    @skip_if_compute_capability_less_than(9.0)
    @unittest.skip(
        "On H100: OutOfResources: out of resource: shared memory, Required: 393216, Hardware limit: 232448. Reducing block sizes or `num_stages` may help."
    )
    def test_user_defined_triton_function(self):
        m, n, k = 256, 256, 512

        a = torch.randn((m, k), dtype=torch.float16, device="cuda")
        b = torch.randn((k, n), dtype=torch.float16, device="cuda")
        torch.compile(gemm_split_k, fullgraph=True)(a, b)


if __name__ == "__main__":
    run_tests()
