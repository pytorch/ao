# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import math
import sys

import pytest
import torch
from torch.testing._internal.common_utils import (
    IS_LINUX,
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
from torchao.quantization.quant_primitives import (
    _choose_qparams_and_quantize_affine_qqq,
)
from torchao.sparsity.marlin import inject_24, marlin_24_workspace, pack_to_marlin_24
from torchao.utils import (
    compute_max_diff,
    torch_version_at_least,
)

IS_CUDA = torch.cuda.is_available() and torch.version.cuda
IS_ROCM = torch.cuda.is_available() and torch.version.hip

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

    @pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
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

    @pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
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

    def _scaled_dot_product_int8_op_ref(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0,
        is_causal=False,
        q_scale=1.0,
        q_zp=0,
        k_scale=1.0,
        k_zp=0,
        v_scale=1.0,
        v_zp=0,
        a_scale=1.0,
        a_zp=0,
        o_scale=1.0,
        o_zp=0,
    ):
        q = (q.to(torch.float) - q_zp) * q_scale
        k = (k.to(torch.float) - k_zp) * k_scale
        v = (v.to(torch.float) - v_zp) * v_scale
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn = q @ k.transpose(-2, -1)
        attn = attn * scale_factor
        if attn_mask is not None:
            attn = attn + attn_mask.to(torch.float)
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = attn - attn_max
        attn = torch.exp(attn)
        attn_sum = torch.sum(attn, dim=-1, keepdim=True)
        attn = attn / attn_sum
        attn = torch.clamp(torch.round(attn / a_scale) + a_zp, min=0, max=255)
        attn = (attn - a_zp) * a_scale
        out = attn @ v
        out = torch.clamp(torch.round(out / o_scale) + o_zp, min=0, max=255)
        return out.to(torch.uint8)

    def _scaled_dot_product_fp8_op_ref(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0,
        is_causal=False,
        q_scale=1.0,
        k_scale=1.0,
        v_scale=1.0,
        a_scale=1.0,
        o_scale=1.0,
    ):
        q = q.to(torch.float) * q_scale
        k = k.to(torch.float) * k_scale
        v = v.to(torch.float) * v_scale
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn = q @ k.transpose(-2, -1)

        attn = attn * scale_factor
        if attn_mask is not None:
            attn = attn + attn_mask.to(torch.float)
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = attn - attn_max
        attn = torch.exp(attn)
        attn_sum = torch.sum(attn, dim=-1, keepdim=True)
        attn = attn / attn_sum
        attn = torch.clamp(attn / a_scale, min=-448, max=448)
        attn = attn.to(torch.float8_e4m3fn).to(torch.float)
        attn = attn * a_scale
        out = attn @ v
        out = torch.clamp(out / o_scale, min=-448, max=448)
        return out.to(torch.float8_e4m3fn)

    @pytest.mark.skipif(
        not torch_version_at_least("2.7.0"),
        reason="quantized sdpa requires torch 2.7 or later",
    )
    @pytest.mark.skipif(not IS_LINUX, reason="only support on linux")
    @pytest.mark.skipif(
        "CPU" not in torch._C._dispatch_dump("torchao::qscaled_dot_product"),
        reason="cpp kernels not built",
    )
    @parametrize("input_dtype", [torch.uint8, torch.float8_e4m3fn])
    @parametrize("batch_size", [56, 120])
    @parametrize("n_head", [2, 16])
    @parametrize("q_seq_len", [18, 89])
    @parametrize("kv_seq_len", [100, 253])
    @parametrize("head_dim", [32, 64])
    @parametrize("mask_dtype", [None, torch.float32, torch.bfloat16])
    def test_quantized_scaled_dot_product_op(
        self,
        input_dtype,
        batch_size,
        n_head,
        q_seq_len,
        kv_seq_len,
        head_dim,
        mask_dtype,
    ):
        torch.manual_seed(1234)
        device = "cpu"
        if input_dtype == torch.uint8:
            q_scale = float(1.7907238006591797)
            k_scale = float(1.8039721250534058)
            v_scale = float(1.839004635810852)
            a_scale = float(0.003919653594493866)
            o_scale = float(1.8191684484481812)
            q_zp = int(127)
            k_zp = int(125)
            v_zp = int(127)
            a_zp = int(120)
            o_zp = int(128)
            atol, rtol = 1.0, 5e-6
        else:
            q_scale = float(5.96875)
            k_scale = float(5.78125)
            v_scale = float(0.98046875)
            a_scale = float(4.84375)
            o_scale = float(3.171875)
            atol, rtol = 0.125, 5e-6
        q_shape = [batch_size, q_seq_len, n_head, head_dim]
        kv_shape = [batch_size, kv_seq_len, n_head, head_dim]
        mask_shape = [batch_size, 1, 1, kv_seq_len]
        q = torch.randn(q_shape, dtype=torch.float, device=device).transpose(1, 2)
        k = torch.randn(kv_shape, dtype=torch.float, device=device).transpose(1, 2)
        v = torch.randn(kv_shape, dtype=torch.float, device=device).transpose(1, 2)
        if input_dtype == torch.uint8:
            q *= 100
            k *= 100
            v *= 100
        q = q.to(input_dtype)
        k = k.to(input_dtype)
        v = v.to(input_dtype)
        attn_mask = (
            torch.randn(mask_shape, dtype=mask_dtype, device=device)
            if mask_dtype is not None
            else None
        )
        q2, k2, v2, attn_mask_2 = (
            q.clone(),
            k.clone(),
            v.clone(),
            attn_mask.clone() if mask_dtype is not None else None,
        )

        if input_dtype == torch.uint8:
            math_ref = self._scaled_dot_product_int8_op_ref(
                q2,
                k2,
                v2,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                q_scale=q_scale,
                q_zp=q_zp,
                k_scale=k_scale,
                k_zp=k_zp,
                v_scale=v_scale,
                v_zp=v_zp,
                a_scale=a_scale,
                a_zp=a_zp,
                o_scale=o_scale,
                o_zp=o_zp,
            )
            actual = torch.ops.torchao.qscaled_dot_product(
                q,
                k,
                v,
                attn_mask=attn_mask_2,
                dropout_p=0.0,
                is_causal=False,
                q_scale=q_scale,
                q_zp=q_zp,
                k_scale=k_scale,
                k_zp=k_zp,
                v_scale=v_scale,
                v_zp=v_zp,
                a_scale=a_scale,
                a_zp=a_zp,
                o_scale=o_scale,
                o_zp=o_zp,
            )
        else:
            math_ref = self._scaled_dot_product_fp8_op_ref(
                q2,
                k2,
                v2,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                a_scale=a_scale,
                o_scale=o_scale,
            )
            actual = torch.ops.torchao.qscaled_dot_product(
                q,
                k,
                v,
                attn_mask=attn_mask_2,
                dropout_p=0.0,
                is_causal=False,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                a_scale=a_scale,
                o_scale=o_scale,
            )
        self.assertEqual(actual.float(), math_ref.float(), atol=atol, rtol=rtol)


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


@pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK, ids=make_test_id)
def test_unpack_tensor_core_tiled_layout_correctness(shape, inner_k_tiles):
    N, K = shape
    assert K % (inner_k_tiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    t = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed_w, inner_k_tiles)
    unpacked = (unpacked[::, ::2] << 4 | unpacked[::, 1::2]).to(torch.uint8)
    assert torch.equal(t, unpacked)


# TODO: Fix "test_aot_dispatch_dynamic" test failure
@pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK, ids=make_test_id)
def test_unpack_tensor_core_tiled_layout_op(shape, inner_k_tiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
    ]

    test_utils.append("test_aot_dispatch_dynamic")

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
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


@pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
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
@pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
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


@pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
@pytest.mark.parametrize(
    "shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str
)
def test_dequantize_tensor_core_tiled_layout_op(shape, inner_k_tiles, group_size):
    n, k = shape
    device = "cuda"

    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
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


@pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
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


@pytest.mark.skipif(not IS_CUDA, reason="CUDA not available")
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
    q_w, s_group, s_channel, w_ref = _choose_qparams_and_quantize_affine_qqq(
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


@pytest.mark.skipif(not IS_ROCM, reason="ROCm not available")
def test_swizzle_mm():
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
    ]

    test_utils.append("test_aot_dispatch_dynamic")

    mat1 = torch.randint(0, 16, dtype=torch.float, size=(16, 32), device="cuda")
    mat2 = torch.randint(0, 16, dtype=torch.float, size=(32, 16), device="cuda")

    opcheck(
        torch.ops.torchao.swizzle_mm,
        (mat1, mat2, False, False),
        test_utils=test_utils,
    )


EMBEDINGBAG_MULTIHOT_SIZES = [1, 2, 3, 10]
EMBEDINGBAG_BAG_SIZES = [1, 2, 128, 1024]
EMBEDINGBAG_VECTOR_SIZES = [1, 128, 512]
EMBEDINGBAG_INDEX_DTYPES = [torch.int64, torch.int32]

EMBEDINGBAG_TEST_PARAMS = list(
    itertools.product(
        EMBEDINGBAG_MULTIHOT_SIZES,
        EMBEDINGBAG_BAG_SIZES,
        EMBEDINGBAG_VECTOR_SIZES,
        EMBEDINGBAG_INDEX_DTYPES,
    )
)


@pytest.mark.skipif(
    "CPU" not in torch._C._dispatch_dump("torchao::_scaled_embedding_bag"),
    reason="cpp kernels not built",
)
@pytest.mark.parametrize(
    "multi_hot, batch_size, vector_size, index_type",
    EMBEDINGBAG_TEST_PARAMS,
    ids=str,
)
def test_scaled_embedding_bag_cpu(multi_hot, batch_size, vector_size, index_type):
    qtype = torch.float8_e4m3fn
    dtype = torch.float32
    weight_scale = torch.tensor([2.0])
    include_last_offset = True
    mode = "sum"

    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2
    indices = torch.randint(1000, (batch_size * multi_hot,)).to(index_type)
    offsets = torch.arange(0, (batch_size + 1) * multi_hot, multi_hot).to(index_type)

    m = torch.nn.EmbeddingBag(
        1000,
        vector_size,
        mode=mode,
        dtype=dtype,
        include_last_offset=include_last_offset,
    )
    fp8_weight = m.weight.data.to(qtype)
    m.weight.data = fp8_weight.to(m.weight.dtype)

    with torch.no_grad():
        refe_out = m.forward(indices, offsets) * weight_scale
        test_out = torch.ops.torchao._scaled_embedding_bag(
            fp8_weight,
            indices,
            offsets,
            weight_scale,
            1.0,
            mode_enum,
            include_last_offset,
        ).to(dtype)
        torch.testing.assert_close(refe_out, test_out, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main(sys.argv)
