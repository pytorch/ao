import itertools
import torch
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
from torch.testing._internal.optests import opcheck
import torchao
from torchao.prototype.fp6_llm.fp6_llm import from_tc_float6_e3m2
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
    def test_fp6_llm_linear(self):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC, "cuda")

        # smoke test
        torchao.ops.fp6_llm_linear(fp16_activation, fp6_weight, fp16_scale, splitK)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp6_llm_linear, (fp16_activation, fp6_weight, fp16_scale, splitK), test_utils=test_utils)

    # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/main/tests/python/kernel_test.py
    @parameterized.expand([(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp6_llm_linear_correctness(self, BS, OC, IC, splitK):
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC, "cuda")

        results_fp6 = torchao.ops.fp6_llm_linear(fp16_activation, fp6_weight, fp16_scale, splitK)

        fp16_weight = from_tc_float6_e3m2(fp6_weight.view(torch.uint8), dtype=torch.float16) * fp16_scale[:, None]
        results_fp16 = fp16_activation @ fp16_weight.T

        error = (results_fp6 - results_fp16).abs()
        relative_error = error / results_fp16.abs()
        assert relative_error.mean() < 1e-2

## Tests for `unpack_int4_packed`
kTileSizeN = 8
kTileSizeK = 16

SHAPES = [
    (4096, 4096),
    # Llama 2 GEMM shapes
    (4096, 11008),
    (11008, 4096),
    # Llama 3 GEMM shapes
    (4096, 14336),
    (14336, 4096),
]
INNERKTILES = [2, 4, 8]

TEST_CONFIGS = list(itertools.product(SHAPES, INNERKTILES))

@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS, ids=str)
def test_int4_unpack_correctness(shape, innerKTiles):
    N, K = shape
    assert K % (innerKTiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, innerKTiles)
    unpacked = torchao.ops.unpack_int4_to_int(packed_w, innerKTiles)
    assert torch.allclose(t, unpacked)


# TODO: Fix "test_aot_dispatch_dynamic" test failure
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS , ids=str)
def test_int4_unpack_op(shape, innerKTiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
      #  "test_aot_dispatch_dynamic",
    ]
    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, innerKTiles)

    opcheck(
        torch.ops.torchao.unpack_int4_to_int,
        (packed_w, innerKTiles),
        test_utils=test_utils,
    )


if __name__ == "__main__":
    unittest.main()
