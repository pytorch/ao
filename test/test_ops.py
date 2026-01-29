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

from torchao.utils import (
    torch_version_at_least,
)

IS_CUDA = torch.cuda.is_available() and torch.version.cuda
IS_ROCM = torch.cuda.is_available() and torch.version.hip

from torchao.quantization import PerGroup, PerRow, PerTensor
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _dequantize_affine_float8,
    _quantize_affine_float8,
)
from torchao.quantization.utils import (
    get_block_size,
)


class TestOps(TestCase):
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


def _test_scaled_embedding_bag_cpu_helper(
    multi_hot,
    batch_size,
    vector_size,
    index_type,
    qtype,
    out_dtype=torch.float,
):
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
        dtype=torch.float,
        include_last_offset=include_last_offset,
    )
    if qtype == torch.int8:
        weight_scale = 127.0 / m.weight.data.abs().max()
        qweight = (m.weight.data * weight_scale).to(qtype)
    else:
        weight_scale = torch.tensor([2.0])
        qweight = m.weight.data.to(qtype)
    m.weight.data = qweight.to(m.weight.dtype)

    out_scale = 1.0
    if out_dtype in [torch.int8, torch.float8_e4m3fn]:
        out_scale = 2.0

    with torch.no_grad():
        refe_out = m.forward(indices, offsets) * weight_scale
        if out_dtype == torch.int8:
            refe_out = torch.round(refe_out / out_scale).to(torch.int32)
            refe_out = torch.clamp(refe_out, -128, 127).to(out_dtype)
            atol = rtol = 1e-5
        if out_dtype == torch.float8_e4m3fn:
            refe_out = torch.clamp(refe_out / out_scale, -448, 448).to(out_dtype)
            # Rtol=1e-5 and atol=1e-5 are not supported for bitwise comparison of low dimensional floats.
            # Please use rtol=0.0 and atol=0.0.
            atol = rtol = 0.0
        test_out = torch.ops.torchao._scaled_embedding_bag(
            qweight,
            indices,
            offsets,
            weight_scale,
            out_scale,
            mode_enum,
            include_last_offset,
            out_dtype,
        )
        torch.testing.assert_close(refe_out, test_out, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    "CPU" not in torch._C._dispatch_dump("torchao::_scaled_embedding_bag"),
    reason="cpp kernels not built",
)
@pytest.mark.parametrize(
    "multi_hot, batch_size, vector_size, index_type",
    EMBEDINGBAG_TEST_PARAMS,
    ids=str,
)
def test_scaled_embedding_bag_int8_cpu(multi_hot, batch_size, vector_size, index_type):
    for out_dtype in [torch.float, torch.int8]:
        _test_scaled_embedding_bag_cpu_helper(
            multi_hot,
            batch_size,
            vector_size,
            index_type,
            torch.int8,
            out_dtype,
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
def test_scaled_embedding_bag_fp8_cpu(multi_hot, batch_size, vector_size, index_type):
    for out_dtype in [torch.float, torch.float8_e4m3fn]:
        _test_scaled_embedding_bag_cpu_helper(
            multi_hot, batch_size, vector_size, index_type, torch.float8_e4m3fn, out_dtype
        )


@pytest.mark.skipif(
    "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_prepack_cpu")
    or "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
    reason="cpp kernels not built",
)
@pytest.mark.skipif(
    not torch_version_at_least("2.6.0"), reason="Test only enabled for 2.6+"
)
@pytest.mark.parametrize("shape", [(64, 64), (256, 256)])
@pytest.mark.parametrize("bs", [1, 160])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16, torch.half])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("x_granularity", [PerTensor(), PerRow(), PerGroup(128)])
@pytest.mark.parametrize("w_granularity", [PerTensor(), PerRow(), PerGroup(128)])
def test_float8_linear_cpu(shape, bs, out_dtype, bias, x_granularity, w_granularity):
    in_feature, out_feature = shape
    if isinstance(x_granularity, PerGroup):
        if x_granularity.group_size >= in_feature:
            return
        if not isinstance(w_granularity, PerGroup):
            return
    if isinstance(w_granularity, PerGroup):
        if w_granularity.group_size >= in_feature:
            return
    m = torch.nn.Linear(in_feature, out_feature, bias=bias).eval()
    b = m.bias
    x = torch.randn(bs, in_feature)
    x_block_size = get_block_size(x.shape, x_granularity)
    x_scale = _choose_scale_float8(
        x,
        float8_dtype=torch.float8_e4m3fn,
        block_size=x_block_size,
    )
    x_fp8 = _quantize_affine_float8(x, x_scale, torch.float8_e4m3fn)

    w = m.weight.detach()
    w_block_size = get_block_size(w.shape, w_granularity)
    w_scale = _choose_scale_float8(
        w,
        float8_dtype=torch.float8_e4m3fn,
        block_size=w_block_size,
    )
    w_fp8 = _quantize_affine_float8(w, w_scale, torch.float8_e4m3fn)

    x_dq = _dequantize_affine_float8(x_fp8, x_scale)
    w_dq = _dequantize_affine_float8(w_fp8, w_scale)
    ref = torch.nn.functional.linear(x_dq, w_dq, b).to(out_dtype)

    packed_w, packed_scale = torch.ops.torchao.float8_linear_prepack_cpu(w_fp8, w_scale)
    y = torch.ops.torchao.float8_linear_cpu(
        x_fp8, x_scale, packed_w, packed_scale, b, out_dtype
    )

    torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main(sys.argv)
