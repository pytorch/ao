import itertools
import torch
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
from torch.testing._internal.optests import opcheck
import torchao
from torchao.prototype.fp6_llm.fp6_llm import from_tc_float6_e3m2
import unittest
from parameterized import parameterized
import pytest

import torchao.quantization

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
QGROUP_SIZES = [32, 64, 128, 256]
TEST_CONFIGS_UNPACK = list(itertools.product(SHAPES, INNERKTILES))
TEST_CONFIGS_DEQUANT = list(itertools.product(SHAPES, INNERKTILES, QGROUP_SIZES))

@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS_UNPACK, ids=str)
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
@pytest.mark.parametrize("shape, innerKTiles", TEST_CONFIGS_UNPACK , ids=str)
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

def dequant_ref(q, scales, zeros, group_size, dtype=torch.bfloat16):
    n, k = q.shape
    assert q.dtype == torch.int

    n_groups = k // group_size
    assert scales.shape[0] == n and scales.shape[1] == n_groups
    assert scales.shape == zeros.shape

    q_bf16 = q.to(dtype=dtype)
    q_bf16 = q_bf16.reshape(-1, group_size)
    dq = (q_bf16 - zeros.reshape(-1, 1)) * scales.reshape(-1, 1)
    return dq.reshape(n, k)

@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape, innerKTiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_int4_correctness(shape, innerKTiles, group_size):
    n, k = shape
    
    # tinygemm params
    nTileSize = 8
    kTileSize = 16
    nTiles = n // nTileSize
    kTiles = k // (innerKTiles * kTileSize)
    numThreads = 32

    device = "cuda"
    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
    packed_w = torch._convert_weight_to_int4pack(q, innerKTiles)
    # tinygemm params
    assert packed_w.shape == torch.Size([nTiles, kTiles, numThreads, innerKTiles // 2])

    # scales and zeros init
    q_groups = k // group_size
    scales = torch.randn(n, q_groups, dtype=torch.bfloat16, device=device)
    zeros = torch.randn_like(scales)
    
    scales_and_zeros = torchao.quantization.utils.pack_tinygemm_scales_and_zeros(scales, zeros)
    assert scales_and_zeros.shape == torch.Size([q_groups, n, 2])
    scales_unpacked, zeros_unpacked = torchao.quantization.utils.unpack_tinygemm_scales_and_zeros(scales_and_zeros)
    assert torch.allclose(scales_unpacked.reshape(scales.shape), scales)
    assert torch.allclose(zeros_unpacked.reshape(zeros.shape), zeros)

    dq_ref = dequant_ref(q, scales, zeros, group_size)
    dq = torchao.ops.dequantize_int4(packed_w, scales_and_zeros, group_size, innerKTiles)
    assert torch.allclose(dq, dq_ref, atol=1e-4, rtol=1e-4)
    
    # TODO: Figure out why this fails
    # This is how torchao.dtypes.affine_quantized_tensor recovers the original tensor
    # https://github.com/pytorch/ao/blob/9dc2c118f59ad4135a8c39166c4ceebda73c62a9/torchao/dtypes/affine_quantized_tensor.py#L505 
    # a_eye = torch.eye(k, device=device, dtype=torch.bfloat16)
    # dq_check = torch.ops.aten._weight_int4pack_mm(
    #     a_eye,
    #     packed_w,
    #     group_size,
    #     scales_and_zeros,
    # ).t()
    # assert torch.allclose(dq, dq_check, atol=1e-4, rtol=1e-4)
    
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape, innerKTiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_int4_op(shape, innerKTiles, group_size):
    n, k = shape
    
    device = "cuda"
    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
    packed_w = torch._convert_weight_to_int4pack(q, innerKTiles)
    print(packed_w.shape)
    q_groups = k // group_size
    scales = torch.randn(n, q_groups, dtype=torch.bfloat16, device=device)
    zeros = torch.randn_like(scales)
    scales_and_zeros = torchao.quantization.utils.pack_tinygemm_scales_and_zeros(scales, zeros)
    
    test_utils = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    #  "test_aot_dispatch_dynamic",
    ]
    opcheck(
        torch.ops.torchao.dequantize_int4,
        (packed_w, scales_and_zeros, group_size, innerKTiles),
        test_utils=test_utils,
    )

if __name__ == "__main__":
    unittest.main()
