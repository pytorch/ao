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
    from torchao.prototype.fp8 import gemm_split_k
    triton_available = True
except ImportError:
    triton_available = False

@unittest.skipIf(not triton_available, "Triton is required but not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFP8Gemm(TestCase):
    # @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_gemm_split_k(self):
        m, n, k = 256, 256, 512

        a = torch.randn((m, k), dtype=torch.float16, device="cuda")
        b = torch.randn((k, n), dtype=torch.float16, device="cuda")
        c = gemm_split_k(a, b)
        c_expected = torch.matmul(a, b)
        assert torch.allclose(c, c_expected, atol=0.07) # less than this and the accuracy check fails

    # https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "User defined triton functions are only supported in PyTorch 2.4 and above")
    def test_user_defined_triton_function(self):
        import torch._inductor.config
        torch._inductor.config.force_disable_caches = True
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        m, n, k = 256, 256, 512

        a = torch.randn((m, k), dtype=torch.float16, device="cuda")
        b = torch.randn((k, n), dtype=torch.float16, device="cuda")
        compiled_function = torch.compile(gemm_split_k, fullgraph=True)(a,b)



if __name__ == "__main__":
    run_tests()
