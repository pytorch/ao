import itertools

import torchao

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.optests import opcheck
from torchao.utils import is_fbcode, TORCH_VERSION_AFTER_2_5
from torchao.prototype.quant_llm import from_scaled_tc_fpx
import pytest

if is_fbcode():
    pytest.skip("Skipping the test in fbcode since we don't have TARGET file for kernels")

try:
    import torchao.ops
except RuntimeError:
    pytest.skip("torchao.ops not available")

from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    pack_tinygemm_scales_and_zeros,
    unpack_tinygemm_scales_and_zeros,
)


class TestOps(TestCase):
    def _create_fpx_inputs(self, ebits: int, mbits: int, BS: int, OC: int, IC: int, device):
        # Randomly initialize each byte
        nbits = 1 + ebits + mbits
        fpx_weight = torch.randint(256, (OC, IC // 8 * nbits), dtype=torch.uint8)
        scale = torch.rand(OC).half() + 0.5
        fp16_act = torch.rand(BS, IC).half() + 0.5
        return fpx_weight.to(device), scale.to(device), fp16_act.to(device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    def test_quant_llm_linear(self, ebits, mbits):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        fpx_weight, scale, fp16_act = self._create_fpx_inputs(ebits, mbits, BS, OC, IC, "cuda")

        # smoke test
        torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fpx_weight, scale, splitK)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.quant_llm_linear, (ebits, mbits, fp16_act, fpx_weight, scale, splitK), test_utils=test_utils)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("BS,OC,IC,splitK", [(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    def test_quant_llm_linear_correctness(self, ebits, mbits, BS, OC, IC, splitK):
        # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/tests/python/kernel_test_fpx.py
        fpx_weight, scale, fp16_act = self._create_fpx_inputs(ebits, mbits, BS, OC, IC, "cuda")

        results_fpx = torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fpx_weight, scale, splitK)

        fp16_weight = from_scaled_tc_fpx(fpx_weight, ebits, mbits, scale).half()
        results_fp16 = fp16_act @ fp16_weight.T

        error = (results_fpx - results_fp16).abs().mean()
        gt = results_fp16.abs().mean()
        relative_error = error / gt
        assert relative_error < 1e-3

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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AFTER_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK, ids=str)
def test_unpack_tensor_core_tiled_layout_correctness(shape, inner_k_tiles):
    N, K = shape
    assert K % (inner_k_tiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    if TORCH_VERSION_AFTER_2_5:
        t = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed_w, inner_k_tiles)
    if TORCH_VERSION_AFTER_2_5:
        unpacked = (unpacked[::, ::2] << 4 | unpacked[::, 1::2]).to(torch.uint8)
    assert torch.equal(t, unpacked)

# TODO: Fix "test_aot_dispatch_dynamic" test failure
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AFTER_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK , ids=str)
def test_unpack_tensor_core_tiled_layout_op(shape, inner_k_tiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
    ]

    # TODO: Figure out why test fails unless torch >= 2.5
    if TORCH_VERSION_AFTER_2_5:
        test_utils.append("test_aot_dispatch_dynamic")

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    if TORCH_VERSION_AFTER_2_5:
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

    #Convert fron u4 -> s4 and upcast to bfloat16
    q = q.sub(midpoint).to(dtype)

    # Dequantize
    q = q.reshape(-1, group_size)
    dq = q * scales.reshape(-1, 1) + zeros.reshape(-1, 1)

    return dq.reshape(n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AFTER_2_5, reason="weight packing is updated in 2.5+")
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
# @pytest.mark.skipif(TORCH_VERSION_AFTER_2_5, reason="weight packing is updated in 2.5+")
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
    if TORCH_VERSION_AFTER_2_5:
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
# @pytest.mark.skipif(TORCH_VERSION_AFTER_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_op(shape, inner_k_tiles, group_size):
    n, k = shape
    device = "cuda"

    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
    if TORCH_VERSION_AFTER_2_5:
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
    if TORCH_VERSION_AFTER_2_5:
        test_utils.append("test_aot_dispatch_dynamic")
    opcheck(
        torch.ops.torchao.dequantize_tensor_core_tiled_layout,
        (packed_w, scales_and_zeros, group_size, inner_k_tiles),
        test_utils=test_utils,
    )

###########################################################################################

DEV = torch.device("cuda:0")
import torch.nn as nn
import torch
import numpy as np


def _get_perms_2_4():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        col_o = col // 2
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col_o * 256 + 8 * (col % 2) + 4 * block)
        for j in range(4):
            perm.extend([p + 1 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i * 8 + j for j in [0, 4, 1, 5, 2, 6, 3, 7]])
    scale_perm_single = []
    for i in range(8):
        scale_perm_single.extend([8 * i + j for j in [0, 1, 2, 3, 4, 5, 6, 7]])
    return perm, scale_perm, scale_perm_single


# Precompute permutations for Marlin weight and scale shuffling
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()
_perm_2_4, _scale_perm_2_4, _scale_perm_single_2_4 = _get_perms_2_4()


def mask_creator(tensor):
    """
    Class for creating N:M sparsity masks.
    Masks will be created using the N:M ratio, where for every block of M weights,
    N will be pruned based on ranked weight value. Each mask will correspond to the given tensor.

    :param N: The number of weights in a group to keep
    :param M: The size of a weight group
    """
    N = 2
    M = 4

    mask = None
    # for i, tensor in enumerate(tensors):
    if tensor.numel() % M != 0:
        raise ValueError(
            f"Tensor of size {tensor.shape} can't be evenly divided into " f"{M} groups"
        )

    num_groups = tensor.numel() // M

    # N:M sparsity for linear layers
    tensor_temp = tensor.detach().abs().reshape(num_groups, M)
    index = torch.argsort(tensor_temp, dim=1)[:, : int(M - N)]

    w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
    mask = w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)

    return mask


# This function converts dense matrix into sparse semi-structured
# representation, producing "compressed" matrix, in the layout used by
# CUTLASS backend, and corresponding metadata matrix.
def sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float, torch.int32]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16"
            )
    else:
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32"
            )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    # Reorder meta tensor elements.
    meta_reordered = meta.new_empty((m * meta_ncols,))  # type: ignore[possibly-undefined]
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    return (sparse, meta_reordered.view(m, meta_ncols))


# This is PyTorch implementation of main part of reorder_meta()
# function, from tools/util/include/cutlass/util/host_reorder.h file
# of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
# GEMM decides upon layout of this matrix, and at the moment for the
# sparse GEMM executed on tensor cores, this is layout described by
# ColumnMajorInterleaved<2> data structure, in
# include/cutlass/layout/matrix.h of CUTLASS source tree.  The
# reordering of meta matrix into meta_reordered matrix calculated
# according to these segments of CUTLASS code is re-implemented here.
# Note that this calculation produces offsets for scattering metadata
# matrix elements into reordered metadata matrix elements (or,
# equivalently, for gathering reordered metadata matrix element back
# into metadata matrix elements).
def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)

    # Reorder the rows, then swizzle the 2x2 blocks.
    group_x = 64
    group_y = 32 if meta_dtype.itemsize == 2 else 16

    dst_rows = (
        dst_rows // group_x * group_x
        + (dst_rows % 2) * 2
        + (dst_rows % 8) // 4
        + ((dst_rows % group_y) % 4) // 2 * 32
        + ((dst_rows % group_x) // 8) * 4
    )

    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    # Assumed that meta tensor is to be stored in CUTLASS
    # InterleavedColumnMajor layout, and reverse engineered
    # corresponding code to store values into this tensor.
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave + cols_min).view(-1)


class Layer_2_4(nn.Module):
    """PyTorch compatible Marlin 2:4 layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError(
                "`infeatures` must be divisible by 64 and `outfeatures` by 256."
            )
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError("`infeatures` must be divisible by `groupsize`.")
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer(
            "B", torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int)
        )
        self.register_buffer(
            "meta", torch.empty((self.n, self.k // 16), dtype=torch.int16)
        )
        self.register_buffer(
            "s", torch.empty((self.k // groupsize, self.n), dtype=torch.half)
        )
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer(
            "workspace",
            torch.zeros(
                self.n // 128 * 16, dtype=torch.int32, device=torch.device("cuda:0")
            ),
            persistent=False,
        )

    def forward(self, A):
        # C = torch.empty(  NOTE(diogo): Remove this maybe?
        #     A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device
        # )

        C = torchao.ops.marlin_24_mm(
            A.view((-1, A.shape[-1])),
            self.B,
            self.meta,
            # C.view((-1, C.shape[-1])),  NOTE(diogo): Remove this maybe?
            self.s,
            self.workspace,
        )
        # mul_2_4(A, self.B, self.meta, C, self.s, self.workspace)
        return C

    def pack(self, linear, scales, trans=False):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        if trans:
            perm, scale_perm, scale_perm_single = (
                _perm_2_4,
                _scale_perm_2_4,
                _scale_perm_single_2_4,
            )
        else:
            perm, scale_perm, scale_perm_single = _perm, _scale_perm, _scale_perm_single
        tile = 16
        maxq = 2**4 - 1
        s = scales
        w = linear.weight.data
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)

        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
        else:
            s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]

        mask = mask_creator(w.T).cuda().bool()
        w = mask * w.T
        w, meta = sparse_semi_structured_from_dense_cutlass(w)
        w = w.t()
        self.k = self.k // 2
        self.groupsize = self.groupsize // 2

        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, perm.numel()))[:, perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i

        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
        self.meta[:, :] = meta.to(self.meta.device)


def gen_quant4_NT(m, k, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((m, k), dtype=torch.half, device=DEV)
    k_sp = k // 2

    w = w.t()
    if groupsize != -1:
        w = w.reshape((-1, groupsize, m))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, m))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, m)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    mask = mask_creator(w.T).cuda().bool()
    uncompress = (mask * ref.T).T

    s = s.reshape((-1, m)).contiguous()
    linear = nn.Linear(k, m)
    linear.weight.data = ref

    layer = Layer_2_4(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = m
    layer.groupsize = groupsize
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=DEV)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
    layer.s = torch.empty((k_sp // (groupsize // 2), m), dtype=torch.half, device=DEV)
    layer.pack(linear, s, True)
    q = layer.B
    s = layer.s
    meta = layer.meta

    return uncompress, q, s, meta


class Marlin24MM(TestCase):

    def _run_problem(self, m, n, k, thread_k, thread_m, groupsize=-1):
        A = torch.randn((n, k), dtype=torch.half, device=DEV)
        B_ref, B, s, meta = gen_quant4_NT(m, k, groupsize=groupsize)
        C_ref = torch.matmul(A, B_ref)

        workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)
        C = torchao.ops.marlin_24_mm(A, B, meta, s, workspace, thread_k, thread_m, -1)
        torch.cuda.synchronize()

        self.assertLess(
            torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.002
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_correctness(self):
        self._run_problem(256, 16, 256, 128, 128, -1)
        self._run_problem(21504, 16, 4096, 64, 256, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tiles(self):
        for m in [1, 2, 4, 8, 12, 16, 32, 64]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self._run_problem(2 * 256, m, 1024, thread_k, thread_n)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_k_stages_divisibility(self):
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self._run_problem(2 * 256, 16, k, 64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_very_few_stages(self):
        for k in [64, 128, 192]:
            self._run_problem(3 * 256, 16, k, 64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_llama_shapes(self):
        MODELS = {
            " 7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
            "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
            "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
            "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
        }

        try:
            for _, layers in MODELS.items():
                for layer in layers:
                    for thread_k, thread_m in [(128, 128)]:
                        for batch in [16]:
                            print(layer[1], batch, layer[0])
                            self._run_problem(layer[1], batch, layer[0], thread_k, thread_m)
        # If someone runs this on a GPU with less than 24GB of memory, it will run out of memory
        # but we don't want to fail the test
        except torch.OutOfMemoryError:
            pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_groups(self):
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        self._run_problem(n, m, k, *thread_shape, groupsize)

###########################################################################################

if __name__ == "__main__":
    run_tests()
