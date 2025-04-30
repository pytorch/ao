# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the ops needed for our tensor subclass implementation
of `MXTensor` to work naturally in PyTorch programs.  For example, if
the modeling code is written as

  x_mx = MXTensor.to_mx(x, torch.float8_e4m3fn)
  w_mx = MXTensor.to_mx(w, torch.float8_e4m3fn)
  y = F.linear(x_mx, w_mx)

then the ops in this file are used under the hood to properly route
the underlying data fields to the MX matmul.
"""

from typing import Any, Dict

import torch
from torch.utils._pytree import tree_map

import torchao.ops
from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
)
from torchao.prototype.mx_formats.mx_tensor import (  # noqa: E501
    MXTensor,
    tensor_size_hp_to_fp4x2,
    tensor_size_hpx3_to_fp6x4,
)
from torchao.prototype.mx_formats.utils import to_blocked

aten = torch.ops.aten

MX_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the mx op table"""

    def decorator(func):
        for op in aten_ops:
            MX_OPS_TABLE[op] = func
        return func

    return decorator


@implements([aten.detach.default])
def mx_desugar_op(aten_op, args, kwargs=None):
    old = args[0]
    new_data = aten_op(old._data, *args[1:], **kwargs)
    new = MXTensor(
        old._scale_e8m0,
        new_data,
        old._elem_dtype,
        old._block_size,
        old._orig_dtype,
        old._use_fp4_custom_triton_dequant_kernel,
        old._gemm_kernel_choice,
        old._pack_fp6,
    )
    return new


@implements([aten.mm.default, aten.matmul.default])
def mx_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)
    assert a._gemm_kernel_choice == b._gemm_kernel_choice, "unsupported"
    if a._gemm_kernel_choice in (MXGemmKernelChoice.CUBLAS, MXGemmKernelChoice.CUTLASS):
        # real MX gemm backed by torchao's CUTLASS kernels
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        assert a._data.is_contiguous()
        assert b._data.t().is_contiguous()

        # TODO(future PR): use block_size instead of hardcoding 32
        a_scale = a._scale_e8m0.view(M, K // 32)
        b_scale = b._scale_e8m0.view(N, K // 32)
        a_scale_block = to_blocked(a_scale)
        b_scale_block = to_blocked(b_scale)
        if a._elem_dtype == torch.float8_e4m3fn:
            assert b._elem_dtype == torch.float8_e4m3fn
            if a._gemm_kernel_choice is MXGemmKernelChoice.CUBLAS:
                res = torch._scaled_mm(
                    a._data,
                    b._data,
                    a_scale_block.view(torch.float8_e8m0fnu),
                    b_scale_block.view(torch.float8_e8m0fnu),
                    out_dtype=torch.bfloat16,
                )
            else:
                res = torchao.ops.mx_fp8_bf16(
                    a._data, b._data, a_scale_block, b_scale_block
                )
        else:
            assert a._elem_dtype == DTYPE_FP4
            assert b._elem_dtype == DTYPE_FP4
            assert a._gemm_kernel_choice is MXGemmKernelChoice.CUTLASS, "unsupported"
            res = torchao.ops.mx_fp4_bf16(
                a._data, b._data, a_scale_block, b_scale_block
            )
    else:
        # emulated MX gemm
        a_hp = a.to_dtype(a._orig_dtype)
        b_hp = b.to_dtype(b._orig_dtype)
        # assert memory layout we expect to be required in hardware
        assert a_hp.is_contiguous()
        assert b_hp.t().is_contiguous()
        res = aten_op(a_hp, b_hp)
    return res


@implements([aten.t.default])
def mx_t(aten_op, args, kwargs=None):
    # For now, only transpose(input, 0, 1) is supported.
    old = args[0]
    new = MXTensor(
        old._scale_e8m0,
        old._data.t(),
        old._elem_dtype,
        old._block_size,
        old._orig_dtype,
        old._use_fp4_custom_triton_dequant_kernel,
        old._gemm_kernel_choice,
        old._pack_fp6,
    )
    return new


@implements([aten.sum.dim_IntList])
def mx_cast_up_op(aten_op, args, kwargs=None):
    """Be careful with this function, this is a "fallback" op that
    casts the output of the op to the original precision. And performs the op.

    We currently need this to support the backward for admmm bias.
    "addmm" -> out
    "hp_gradBias" <-"sum" <- "identity" <- gradOut <- "hp_gradOut"
    """

    def unwrap(x):
        if isinstance(x, MXTensor):
            return x.to_dtype(x._orig_dtype)
        return x

    new_args = tree_map(unwrap, args)
    new_kwargs = tree_map(unwrap, kwargs)
    return aten_op(*new_args, **new_kwargs)


@implements([aten.view.default])
def mx_view_op(aten_op, args, kwargs=None):
    data = args[0]._data
    new_size = args[1]
    if args[0]._elem_dtype == DTYPE_FP4:
        # special case fp4 as we pack two elements per byte
        new_size = tensor_size_hp_to_fp4x2(new_size, data.is_contiguous())
    elif args[0]._elem_dtype in [DTYPE_FP6_E3M2, DTYPE_FP6_E2M3] and args[0]._pack_fp6:
        # special case fp6 as we pack 4 elements in 3 bytes
        new_size = tensor_size_hpx3_to_fp6x4(new_size, data.is_contiguous())
    new_data = aten_op(data, new_size, *args[2:], **kwargs)
    return MXTensor(
        args[0]._scale_e8m0,
        new_data,
        args[0]._elem_dtype,
        args[0]._block_size,
        args[0]._orig_dtype,
        args[0]._use_fp4_custom_triton_dequant_kernel,
        args[0]._gemm_kernel_choice,
        args[0]._pack_fp6,
    )


@implements([aten._to_copy.default])
def autocast_to_copy(aten_op, args, kwargs=None):
    """This gets called when running matmul under autocast
    when the input is a MXTensor, presenting as a fp32
    tensor.
    """
    assert isinstance(args[0], MXTensor)
    assert len(kwargs) == 1 and "dtype" in kwargs, (
        "Only support dtype kwarg for autocast"
    )
    assert kwargs["dtype"] in {
        torch.float16,
        torch.bfloat16,
    }, "Only support floating point conversion for autocast w/ MXTensor"
    res = MXTensor(
        args[0]._scale_e8m0,
        args[0]._data,
        args[0]._elem_dtype,
        args[0]._block_size,
        kwargs["dtype"],
        args[0]._use_fp4_custom_triton_dequant_kernel,
        args[0]._gemm_kernel_choice,
        args[0]._pack_fp6,
    )
    return res

@implements([aten.linear.default])
def mx_linear(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]
    if len(args) > 2:
        c = args[2]
    else:
        c = None
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)
    a_hp = a.to_dtype(a._orig_dtype)
    b_hp = b.to_dtype(b._orig_dtype)
    res = aten_op(a_hp, b_hp, c)
    return res
