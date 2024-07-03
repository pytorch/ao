import itertools
import unittest

import pytest
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import IS_FBCODE, TestCase
from torch.testing._internal.optests import opcheck

import torchao
import torchao.quantization
from torchao.prototype.fp6_llm.fp6_llm import from_tc_float6_e3m2
from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    pack_tinygemm_scales_and_zeros,
    unpack_tinygemm_scales_and_zeros,
)

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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK, ids=str)
def test_unpack_tensor_core_tiled_layout_correctness(shape, inner_k_tiles):
    N, K = shape
    assert K % (inner_k_tiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed_w, inner_k_tiles)
    assert torch.equal(t, unpacked)

# TODO: Fix "test_aot_dispatch_dynamic" test failure
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK , ids=str)
def test_unpack_tensor_core_tiled_layout_op(shape, inner_k_tiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    ]
    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)

    opcheck(
        torch.ops.torchao.unpack_tensor_core_tiled_layout,
        (packed_w, inner_k_tiles),
        test_utils=test_utils,
    )

def dequant_ref(q, scales, zeros, group_size, nbits=4, dtype=torch.bfloat16):
    n, k = q.shape
    assert q.dtype == torch.int

    n_groups = k // group_size
    assert scales.shape[0] == n and scales.shape[1] == n_groups
    assert scales.shape == zeros.shape

    midpoint = 2 ** (nbits - 1)
    
    #Convert fron u4 -> s4 and upcast to bfloat16
    q = q.sub(midpoint).to(dtype)
    
    # Dequantize
    q = q.reshape(-1, group_size)
    dq = q * scales.reshape(-1, 1) + zeros.reshape(-1, 1)

    return dq.reshape(n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_correctness_quant_dequant(shape, inner_k_tiles, group_size):
    n, k = shape
    dtype = torch.bfloat16    

    device = "cuda"

    t = torch.randn(n, k, dtype=dtype, device=device)
    scales, zeros = get_groupwise_affine_qparams(t, n_bit=4, groupsize=group_size, dtype=dtype)
    
    # Quantize
    q = groupwise_affine_quantize_tensor_from_qparams(
        t, scales, zeros, n_bit=4, groupsize=group_size
    )
    
    # Pack to tensor core layout
    packed = torch.ops.aten._convert_weight_to_int4pack(q, inner_k_tiles)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
    q_groups = k // group_size
    assert scales_and_zeros.shape == torch.Size([q_groups, n, 2])

    # Dequantize 'ao' ref
    dq_ao = groupwise_affine_dequantize_tensor_from_qparams(
        q, scales, zeros, n_bit=4, groupsize=group_size
    )
    
    # Dequantize by passing in an identity matrix as the activation
    a_eye = torch.eye(k, device=device, dtype=dtype)
    dq_id = torch.ops.aten._weight_int4pack_mm(
        a_eye,
        packed,
        group_size,
        scales_and_zeros,
    ).t()
    
    # Actual operation to test
    dq_op = torchao.ops.dequantize_tensor_core_tiled_layout(packed, scales_and_zeros, group_size, inner_k_tiles)
        
    # Compare results
    diff_ao_id = (dq_id - dq_ao).abs().max()
    diff_op_id = (dq_op - dq_id).abs().max()
    diff_op_ao = (dq_op - dq_ao).abs().max()
    
    # There are slight numerical differences when dequantizing with an identity matrix when compared to `groupwise_affine_dequantize`
    # Since the `dequantize_tensor_core_layout` kernel relies on the same underlying bit twiddling tricks for fast
    # conversion from u4 -> s4 -> bf16, the identity matrix dequant hack and `dequantize_tensor_core_layout` are
    # expected to give same results, while both will have similar numerical differences to `groupwise_affine_dequantize`.
    
    # Test that the `dequant` kernel gives same results as identity matrix-based dequant 
    assert diff_op_id == 0
    
    # Test that the `dequant` kernel gives same numerical diffs as the `groupwise_affine_dequantize` when compared against the identity matrix
    assert diff_op_ao == diff_ao_id

    assert diff_op_ao < 1e-1

# This test differs from one above in that it uses `unpack_tensor_core_tiled_layout` to unpack then dequantize
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_correctness_unpack_and_dequant(shape, inner_k_tiles, group_size):
    n, k = shape
    dtype = torch.bfloat16    
    device = "cuda"

    # Quantize and pack
    t = torch.randn(n, k, dtype=dtype, device=device)
    scales, zeros = get_groupwise_affine_qparams(t, n_bit=4, groupsize=group_size, dtype=dtype)
    q = groupwise_affine_quantize_tensor_from_qparams(
        t, scales, zeros, n_bit=4, groupsize=group_size
    )

    packed = torch.ops.aten._convert_weight_to_int4pack(q, inner_k_tiles)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
    
    # Unpack and dequantize
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed, inner_k_tiles)
    dq_ao = groupwise_affine_dequantize_tensor_from_qparams(
        unpacked, scales, zeros, n_bit=4, groupsize=group_size
    )
    
    # Dequantize by passing in an identity matrix as the activation
    a_eye = torch.eye(k, device=device, dtype=dtype)
    dq_id = torch.ops.aten._weight_int4pack_mm(
        a_eye,
        packed,
        group_size,
        scales_and_zeros,
    ).t()
    
    # Actual operation to test
    dq_op = torchao.ops.dequantize_tensor_core_tiled_layout(packed, scales_and_zeros, group_size, inner_k_tiles)
    
    # Compare results
    diff_ao_id = (dq_id - dq_ao).abs().max()
    diff_op_id = (dq_op - dq_id).abs().max()
    diff_op_ao = (dq_op - dq_ao).abs().max()
    
    # There are slight numerical differences when dequantizing with an identity matrix when compared to `groupwise_affine_dequantize`
    # Since the `dequantize_tensor_core_layout` kernel relies on the same underlying bit twiddling tricks for fast
    # conversion from u4 -> s4 -> bf16, the identity matrix dequant hack and `dequantize_tensor_core_layout` are
    # expected to give same results, while both will have similar numerical differences to `groupwise_affine_dequantize`.
    
    # Test that the `dequant` kernel gives same results as identity matrix-based dequant 
    assert diff_op_id == 0
    
    # Test that the `dequant` kernel gives same numerical diffs as the `groupwise_affine_dequantize` when compared against the identity matrix
    assert diff_op_ao == diff_ao_id

    assert diff_op_ao < 1e-1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(IS_FBCODE, reason="Skipping the test in fbcode since we don't have TARGET file for kernels")
@pytest.mark.parametrize("shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_op(shape, inner_k_tiles, group_size):
    n, k = shape
    device = "cuda"

    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
    packed_w = torch._convert_weight_to_int4pack(q, inner_k_tiles)
    q_groups = k // group_size
    scales = torch.randn(n, q_groups, dtype=torch.bfloat16, device=device)
    zeros = torch.randn_like(scales)
    scales_and_zeros = torchao.quantization.utils.pack_tinygemm_scales_and_zeros(scales, zeros)
    
    test_utils = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
    ]
    opcheck(
        torch.ops.torchao.dequantize_tensor_core_tiled_layout,
        (packed_w, scales_and_zeros, group_size, inner_k_tiles),
        test_utils=test_utils,
    )

if __name__ == "__main__":
    unittest.main()
