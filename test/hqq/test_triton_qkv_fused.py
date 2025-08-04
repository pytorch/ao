# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest

triton = pytest.importorskip(
    "triton", minversion="3.0.0", reason="Triton > 3.0.0 required to run this test"
)
hqq = pytest.importorskip("hqq", reason="hqq required to run this test")
hqq_quantize = pytest.importorskip(
    "hqq.core.quantize", reason="hqq required to run this test"
)
HQQLinear = hqq_quantize.HQQLinear
BaseQuantizeConfig = hqq_quantize.BaseQuantizeConfig

import itertools

import torch

from torchao.prototype.hqq import pack_2xint4, triton_mixed_mm
from torchao.utils import auto_detect_device

_DEVICE = auto_detect_device()

torch.manual_seed(0)
# N, K = shape
Q_SHAPES = [[4096, 4096]]
KV_SHAPES = [[4096, 4096], [1024, 4096]]
GROUP_SIZES = [64, 128]
AXES = [1]
DTYPES = [torch.bfloat16]

TRANSPOSED = [False, True]
TRITON_KERNEL_TYPE = ["compute_bound"]
TEST_CONFIGS = list(
    itertools.product(
        Q_SHAPES, KV_SHAPES, GROUP_SIZES, AXES, DTYPES, TRANSPOSED, TRITON_KERNEL_TYPE
    )
)


BASE_QUANT_CONFIG = {
    "optimize": True,
    "view_as_float": False,
    "nbits": 4,
    "bitpack": False,
    "axis": 1,
}


def _arg_to_id(arg):
    if isinstance(arg, list):
        return "x".join([str(x) for x in arg])
    return str(arg)


def quantize_helper(
    weight_shape, quant_config, dtype, device=_DEVICE, quant_dtype=torch.uint8
):
    N, K = weight_shape
    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)

    hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    W_q = W_q.to(dtype=quant_dtype)
    W_q = (
        W_q.reshape(meta["shape"])
        if quant_config["weight_quant_params"]["bitpack"] == False
        else W_q
    )

    scale, zero = meta["scale"], meta["zero"]
    scale = scale.reshape(N, -1)
    zero = zero.reshape(N, -1)

    return W_q, scale, zero


def fuse_qkv(W_qs, scales, zeros):
    """
    Args:
        W_qs (list[torch.Tensor]): len 3 list of tensors with shapes Nq x K, Nk x K, Nv x K where Nk == Nv
        scales (list[torch.Tensor]): each is N x (K // group_size), with same N requirements per W_qs
        zeros (list[torch.Tensor]): same as scales

    Returns:
        qkv (torch.Tensor): (N_qkv x K) where N_qkv = Nq + Nk + Nv
        scales (torch.Tensor): (N_qkv x (K // group_size))
        zeros (torch.Tensor): (N_qkv x (K // group_size))
    """
    qkv = torch.cat(W_qs, dim=0)  # Fuse along N
    fused_scales = torch.cat([s for s in scales], dim=0)
    fused_zeros = torch.cat([z for z in zeros], dim=0)
    return qkv, fused_scales, fused_zeros


def ref_proj(x, packed_w, scale, zero, group_size, kernel_type, transposed=False):
    return triton_mixed_mm(
        x,
        packed_w,
        scale.T,
        zero.T,
        transposed=transposed,
        group_size=group_size,
        fp8_fast_accum=False,
        kernel_type=kernel_type,
    )


@pytest.mark.parametrize(
    "q_shape, kv_shape, group_size, axis, dtype, transposed, kernel_type",
    TEST_CONFIGS,
    ids=_arg_to_id,
)
def test_mixed_mm(
    q_shape,
    kv_shape,
    group_size,
    axis,
    dtype,
    transposed,
    kernel_type,
    seqlen=16,
    device=_DEVICE,
    quant_dtype=torch.uint8,
):
    """
    Note we test with dtype float32 in the transposed case, since fused and non-fused ops are not exactly equivalent in this case.

    More specifically when running transposed matmul:
    - fused: we are reducing along fused N within the kernel
    - non-fused: we are launching 3 individual kernels and reducing along N within each of these kernels for q, k, v then post-hoc summing these three terms to simulate the fused op

    This gives rise to a number of numeric issues when testing equivalence, given how accumulation is treated within triton MAC loop.
    Using higher precision mitigates these issues for the purposes of this test.
    """

    # Override dtype per the above comment
    if transposed:
        dtype = torch.float32

    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})

    # Quantize q, k, v individually
    W_qs, packed_ws, scales, zeros = [], [], [], []
    for shape in [q_shape, kv_shape, kv_shape]:
        W_q, scale, zero = quantize_helper(
            shape, quant_config, dtype, device, quant_dtype
        )
        W_qs.append(W_q)
        packed_ws.append(pack_2xint4(W_q.T))
        scales.append(scale)
        zeros.append(zero)

    # Fuse q, k, v, scales, zeros
    qkv_fused, scales_fused, zeros_fused = fuse_qkv(W_qs, scales, zeros)
    qkv_fused_packed = pack_2xint4(qkv_fused.T)

    Ks = [shape[1] for shape in [q_shape, kv_shape]]

    K = Ks[0]

    # Check shapes
    assert all([k == K for k in Ks])
    assert qkv_fused_packed.shape[0] * 2 == qkv_fused.shape[1] == Ks[0]

    if transposed:
        Ns = [q_shape[0], kv_shape[0], kv_shape[0]]
        xs = [torch.randn(seqlen, n, dtype=dtype, device=device) for n in Ns]
        x_fused = torch.cat(xs, dim=1)
        q_ref, k_ref, v_ref = [
            ref_proj(x, p, s, z, group_size, kernel_type, transposed=True)
            for x, p, s, z in zip(xs, packed_ws, scales, zeros)
        ]
        tt_fused = triton_mixed_mm(
            x_fused,
            qkv_fused_packed,
            scales_fused.T,
            zeros_fused.T,
            transposed=True,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
        )
        tt_ref = q_ref + k_ref + v_ref
        assert torch.allclose(tt_ref, tt_fused, atol=1e-4)
    else:
        x = torch.randn(seqlen, K, dtype=dtype, device=device)

        q_ref, k_ref, v_ref = [
            ref_proj(x, p, s, z, group_size, kernel_type)
            for p, s, z in zip(packed_ws, scales, zeros)
        ]

        tt_fused = triton_mixed_mm(
            x,
            qkv_fused_packed,
            scales_fused.T,
            zeros_fused.T,
            transposed=False,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
        )
        qN, kN, vN = q_shape[0], kv_shape[0], kv_shape[0]
        q_fused, k_fused, v_fused = tt_fused.split([qN, kN, vN], dim=1)

        for ref, fused in zip([q_ref, k_ref, v_ref], [q_fused, k_fused, v_fused]):
            assert torch.allclose(ref, fused)
