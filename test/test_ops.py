import torch
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
from torch.testing._internal.optests import opcheck
import torchao
from torchao.quantization.fp6_llm import from_tc_float6_e3m2
import unittest
from parameterized import parameterized
import pytest

try:
    import torchao.ops
except RuntimeError:
    pytest.skip("torchao.ops not available")


# torch.testing._internal.optests.generate_tests.OpCheckError: opcheck(op, ...):
# test_faketensor failed with module 'torch' has no attribute '_custom_ops' (scroll up for stack trace)
@pytest.mark.filterwarnings("ignore:create_unbacked_symint is deprecated, please use new_dynamic_size instead:UserWarning")
@unittest.skipIf(IS_FBCODE, "Skipping the test in fbcode since we don't have TARGET file for kernels")
class TestOps(TestCase):
    def _create_fp6_inputs(self, BS: int, OC: int, IC: int, device):
        # Randomly initialize each bytes. The highest value for randint() is set the the max value of uint32_t.
        fp6_weight = torch.randint(4294967295, (OC, IC // 16 * 3)).to(torch.int)
        fp16_scale = torch.rand(OC).half() + 0.5
        fp16_activation = torch.rand(BS, IC).half() + 0.5
        return fp6_weight.to(device), fp16_scale.to(device), fp16_activation.to(device)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp16act_fp6weight_linear(self):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC, "cuda")

        # smoke test
        torchao.ops.fp16act_fp6weight_linear(fp16_activation, fp6_weight, fp16_scale, splitK)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp16act_fp6weight_linear, (fp16_activation, fp6_weight, fp16_scale, splitK), test_utils=test_utils)

    # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/main/tests/python/kernel_test.py
    @parameterized.expand([(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp6_matmul_correctness(self, BS, OC, IC, splitK):
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC, "cuda")

        results_fp6 = torchao.ops.fp16act_fp6weight_linear(fp16_activation, fp6_weight, fp16_scale, splitK)

        fp16_weight = from_tc_float6_e3m2(fp6_weight.view(torch.uint8), dtype=torch.float16) * fp16_scale[:, None]
        results_fp16 = fp16_activation @ fp16_weight.T

        error = (results_fp6 - results_fp16).abs()
        relative_error = error / results_fp16.abs()
        assert relative_error.mean() < 1e-2


if __name__ == "__main__":
    unittest.main()
