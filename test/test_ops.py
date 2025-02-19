import itertools
import sys

import pytest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.optests import opcheck

import torchao
from torchao.dtypes.floatx import from_scaled_tc_floatx
from torchao.quantization.marlin_qqq import (
    marlin_qqq_workspace,
    pack_to_marlin_qqq,
)
from torchao.quantization.quant_primitives import choose_qparams_and_quantize_affine_qqq
from torchao.sparsity.marlin import inject_24, marlin_24_workspace, pack_to_marlin_24
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, compute_max_diff

try:
    import torchao.ops
except RuntimeError:
    pytest.skip("torchao.ops not available")

from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    pack_tinygemm_scales_and_zeros,
)


class TestOps(TestCase):
    def _create_floatx_inputs(
        self, ebits: int, mbits: int, BS: int, OC: int, IC: int, device, dtype
    ):
        # Randomly initialize each byte
        nbits = 1 + ebits + mbits
        floatx_weight = torch.randint(256, (OC, IC // 8 * nbits), dtype=torch.uint8)
        scale = torch.rand(OC).to(dtype) + 0.5
        fp16_act = torch.rand(BS, IC).to(dtype) + 0.5
        return floatx_weight.to(device), scale.to(device), fp16_act.to(device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    @parametrize("dtype", [torch.half, torch.bfloat16])
    def test_quant_llm_linear(self, ebits, mbits, dtype):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        floatx_weight, scale, fp16_act = self._create_floatx_inputs(
            ebits, mbits, BS, OC, IC, "cuda", dtype
        )

        # smoke test
        torchao.ops.quant_llm_linear(
            ebits, mbits, fp16_act, floatx_weight, scale, splitK
        )

        # comprehensive testing
        test_utils = [
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ]
        opcheck(
            torch.ops.torchao.quant_llm_linear,
            (ebits, mbits, fp16_act, floatx_weight, scale, splitK),
            test_utils=test_utils,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("BS,OC,IC,splitK", [(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    @parametrize("dtype", [torch.half, torch.bfloat16])
    def test_quant_llm_linear_correctness(
        self, ebits, mbits, BS, OC, IC, splitK, dtype
    ):
        # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/tests/python/kernel_test_fpx.py
        floatx_weight, scale, fp16_act = self._create_floatx_inputs(
            ebits, mbits, BS, OC, IC, "cuda", dtype
        )

        results_floatx = torchao.ops.quant_llm_linear(
            ebits, mbits, fp16_act, floatx_weight, scale, splitK
        )

        fp16_weight = from_scaled_tc_floatx(floatx_weight, ebits, mbits, scale).to(
            dtype
        )
        results_fp16 = fp16_act @ fp16_weight.T

        error = (results_floatx - results_fp16).abs().mean()
        gt = results_fp16.abs().mean()
        relative_error = error / gt
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        assert relative_error < rtol


instantiate_parametrized_tests(TestOps)


## Tests for `tensor_core_layout`
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


def make_test_id(param):
    if isinstance(param, tuple) and len(param) == 2:  # This is a shape
        return f"shape_{param[0]}x{param[1]}"
    else:  # This is inner_k_tiles
        return f"tiles_{param}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK, ids=make_test_id)
def test_unpack_tensor_core_tiled_layout_correctness(shape, inner_k_tiles):
    N, K = shape
    assert K % (inner_k_tiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    if TORCH_VERSION_AT_LEAST_2_5:
        t = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed_w, inner_k_tiles)
    if TORCH_VERSION_AT_LEAST_2_5:
        unpacked = (unpacked[::, ::2] << 4 | unpacked[::, 1::2]).to(torch.uint8)
    assert torch.equal(t, unpacked)


# TODO: Fix "test_aot_dispatch_dynamic" test failure
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK, ids=make_test_id)
def test_unpack_tensor_core_tiled_layout_op(shape, inner_k_tiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
    ]

    # TODO: Figure out why test fails unless torch >= 2.5
    if TORCH_VERSION_AT_LEAST_2_5:
        test_utils.append("test_aot_dispatch_dynamic")

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    if TORCH_VERSION_AT_LEAST_2_5:
        t = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
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

    # Convert fron u4 -> s4 and upcast to bfloat16
    q = q.sub(midpoint).to(dtype)

    # Dequantize
    q = q.reshape(-1, group_size)
    dq = q * scales.reshape(-1, 1) + zeros.reshape(-1, 1)

    return dq.reshape(n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize(
    "shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str
)
def test_dequantize_tensor_core_tiled_layout_correctness_quant_dequant(
    shape, inner_k_tiles, group_size
):
    n, k = shape
    dtype = torch.bfloat16

    device = "cuda"

    t = torch.randn(n, k, dtype=dtype, device=device)
    scales, zeros = get_groupwise_affine_qparams(
        t, n_bit=4, groupsize=group_size, dtype=dtype
    )

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
    dq_op = torchao.ops.dequantize_tensor_core_tiled_layout(
        packed, scales_and_zeros, group_size, inner_k_tiles
    )

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
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize(
    "shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str
)
def test_dequantize_tensor_core_tiled_layout_correctness_unpack_and_dequant(
    shape, inner_k_tiles, group_size
):
    n, k = shape
    dtype = torch.bfloat16
    device = "cuda"

    # Quantize and pack
    t = torch.randn(n, k, dtype=dtype, device=device)
    scales, zeros = get_groupwise_affine_qparams(
        t, n_bit=4, groupsize=group_size, dtype=dtype
    )
    q = groupwise_affine_quantize_tensor_from_qparams(
        t, scales, zeros, n_bit=4, groupsize=group_size
    )

    packed = torch.ops.aten._convert_weight_to_int4pack(q, inner_k_tiles)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)

    # Unpack and dequantize
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed, inner_k_tiles)
    if TORCH_VERSION_AT_LEAST_2_5:
        unpacked = (unpacked[::, ::2] << 4 | unpacked[::, 1::2]).to(torch.uint8)

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
    dq_op = torchao.ops.dequantize_tensor_core_tiled_layout(
        packed, scales_and_zeros, group_size, inner_k_tiles
    )

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
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize(
    "shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str
)
def test_dequantize_tensor_core_tiled_layout_op(shape, inner_k_tiles, group_size):
    n, k = shape
    device = "cuda"

    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
    if TORCH_VERSION_AT_LEAST_2_5:
        q = (q[::, ::2] << 4 | q[::, 1::2]).to(torch.uint8)
    packed_w = torch._convert_weight_to_int4pack(q, inner_k_tiles)
    q_groups = k // group_size
    scales = torch.randn(n, q_groups, dtype=torch.bfloat16, device=device)
    zeros = torch.randn_like(scales)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)

    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
    ]
    # TODO: Figure out why test fails unless torch >= 2.5
    if TORCH_VERSION_AT_LEAST_2_5:
        test_utils.append("test_aot_dispatch_dynamic")
    opcheck(
        torch.ops.torchao.dequantize_tensor_core_tiled_layout,
        (packed_w, scales_and_zeros, group_size, inner_k_tiles),
        test_utils=test_utils,
    )


MARLIN_24_BATCH_SIZE = [1, 4, 8, 16, 32, 64]
MARLIN_24_K_CHUNKS = [128]
MARLIN_24_N_CHUNKS = [512]
MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
]
MARLIN_24_SUPPORTED_NUM_BITS = [4, 8]
MARLIN_24_SUPPORTED_GROUP_SIZES = [-1, 128]

MARLIN_TEST_PARAMS = list(
    itertools.product(
        MARLIN_24_BATCH_SIZE,
        MARLIN_24_K_CHUNKS,
        MARLIN_24_N_CHUNKS,
        MARLIN_24_SUPPORTED_NUM_BITS,
        MARLIN_24_SUPPORTED_GROUP_SIZES,
        MNK_FACTORS,
    )
)


def _symmetric_quantize_with_ref(w: torch.Tensor, num_bits: int, group_size: int):
    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2

    # Reshape to [groupsize, -1]
    if group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / max_q_val  # 2 => symmetric

    # Quantize
    q_w = torch.round(w / s).int()
    q_w += half_q_val
    q_w = torch.clamp(q_w, 0, max_q_val)

    # Compute ref (dequantized)
    w_ref = (q_w - half_q_val).half() * s

    # Restore original shapes
    if group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

    s = s.reshape((-1, size_n)).contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        s.to(device=orig_device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "batch_size, k_chunk, n_chunk, num_bits, group_size, mnk_factors",
    MARLIN_TEST_PARAMS,
    ids=str,
)
def test_marlin_24(batch_size, k_chunk, n_chunk, num_bits, group_size, mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    a_input = torch.randn(
        (batch_size, size_m, size_k), dtype=torch.float16, device="cuda"
    )
    b_weight = torch.rand((size_k, size_n), dtype=torch.float16, device="cuda")

    # Inject 2:4 sparsity
    w_24, _ = inject_24(b_weight, size_k, size_n)

    # Symmetric quantize
    w_24_ref, q_w_24, scale = _symmetric_quantize_with_ref(w_24, num_bits, group_size)

    # Reshape input into 2D tensor
    input_2d = a_input.view(-1, a_input.shape[-1])
    a_input_in, a_input_out = input_2d.shape

    # Obtains reference output
    output_ref = torch.matmul(input_2d, w_24_ref)
    output_ref = output_ref.reshape(a_input.shape[:-1] + (scale.shape[1],))

    # Packs to marlin 2:4
    marlin_24_q_w_comp, marlin_24_scale, meta = pack_to_marlin_24(
        q_w_24, scale, num_bits, group_size
    )
    workspace_24 = marlin_24_workspace(size_n)

    fn_inputs = (
        input_2d,
        marlin_24_q_w_comp,
        meta,
        marlin_24_scale,
        workspace_24,
        num_bits,
        a_input_in,
        marlin_24_scale.shape[1],
        a_input_out,
    )
    output = torchao.ops.marlin_24_gemm(*fn_inputs)
    output = output.reshape(a_input.shape[:-1] + (marlin_24_scale.shape[1],))

    max_diff = compute_max_diff(output, output_ref)
    assert max_diff < 0.04

    # Performs opcheck
    test_utils = ["test_schema", "test_autograd_registration", "test_faketensor"]
    opcheck(
        torch.ops.torchao.marlin_24_gemm,
        fn_inputs,
        test_utils=test_utils,
    )


MARLIN_QQQ_BATCH_SIZE = [1, 4, 8, 16, 32, 64]
MARLIN_QQQ_K_CHUNKS = [128]
MARLIN_QQQ_N_CHUNKS = [64, 128, 256]
MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
]
MARLIN_QQQ_SUPPORTED_NUM_BITS = [4]
MARLIN_QQQ_SUPPORTED_GROUP_SIZES = [-1, 128]

MARLIN_TEST_PARAMS = list(
    itertools.product(
        MARLIN_QQQ_BATCH_SIZE,
        MARLIN_QQQ_K_CHUNKS,
        MARLIN_QQQ_N_CHUNKS,
        MARLIN_QQQ_SUPPORTED_NUM_BITS,
        MARLIN_QQQ_SUPPORTED_GROUP_SIZES,
        MNK_FACTORS,
    )
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "batch_size, k_chunk, n_chunk, num_bits, group_size, mnk_factors",
    MARLIN_TEST_PARAMS,
    ids=str,
)
@pytest.mark.skip(reason="test outputs nan after cuda is upgraded to 12.4")
def test_marlin_qqq(batch_size, k_chunk, n_chunk, num_bits, group_size, mnk_factors):
    int8_traits = torch.iinfo(torch.int8)
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    a_input = torch.randn(
        (batch_size, size_m, size_k), dtype=torch.float16, device="cuda"
    )
    b_weight = torch.rand((size_n, size_k), dtype=torch.float16, device="cuda")

    # Reshape input into 2D tensor
    input_2d = a_input.view(-1, a_input.shape[-1])
    a_input_in, a_input_out = input_2d.shape

    # Quantize activations
    s_a = (
        input_2d.abs()
        .max(dim=-1, keepdim=True)[0]
        .div(int8_traits.max)
        .to(torch.float32)
    )
    q_a = (
        (input_2d / s_a).round().clamp(int8_traits.min, int8_traits.max).to(torch.int8)
    )

    # Quantize weights
    q_w, s_group, s_channel, w_ref = choose_qparams_and_quantize_affine_qqq(
        b_weight, num_bits, group_size
    )
    q_w = q_w.t()
    s_group = s_group.t()
    s_channel = s_channel.t()
    w_ref = w_ref.t()
    marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel = pack_to_marlin_qqq(
        q_w, s_group, s_channel, num_bits, group_size
    )

    workspace = marlin_qqq_workspace(size_n)

    # Obtains reference output
    output_ref = torch.matmul(q_a.half() * s_a.half(), w_ref)
    output_ref = output_ref.reshape(a_input.shape[:-1] + (size_n,))

    fn_inputs = (
        q_a,
        marlin_qqq_q_w,
        s_a,
        marlin_qqq_s_channel,
        marlin_qqq_s_group,
        workspace,
        a_input_in,
        size_n,
        a_input_out,
    )
    output = torchao.ops.marlin_qqq_gemm(*fn_inputs)
    output = output.reshape(a_input.shape[:-1] + (size_n,))

    max_diff = compute_max_diff(output, output_ref)
    assert max_diff < 0.04

    # Performs opcheck
    test_utils = ["test_schema", "test_autograd_registration", "test_faketensor"]
    opcheck(
        torch.ops.torchao.marlin_qqq_gemm,
        fn_inputs,
        test_utils=test_utils,
    )


if __name__ == "__main__":
    pytest.main(sys.argv)
