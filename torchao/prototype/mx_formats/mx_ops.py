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

from typing import Any, Dict, Optional

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)
from torch.utils._pytree import tree_map

import torchao.ops
from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
)
from torchao.prototype.mx_formats.mx_tensor import (  # noqa: E501
    MXTensor,
    tensor_size_hp_to_fp4x2,
    tensor_size_hpx3_to_fp6x4,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import fill_defaults

aten = torch.ops.aten

MX_OPS_TABLE: Dict[Any, Any] = {}
MX_FUNCTION_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the mx op table"""

    def decorator(func):
        for op in aten_ops:
            MX_OPS_TABLE[op] = func
        return func

    return decorator


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(func)
    )


def _get_gemm_choice(
    choice_a: Optional[MXGemmKernelChoice], choice_b: Optional[MXGemmKernelChoice]
) -> MXGemmKernelChoice:
    if choice_a is not None and choice_b is not None:
        assert choice_a == choice_b, (
            "Both MXTensor inputs must have the same gemm config if specified"
        )
        return choice_a

    # Assert that at least one is set and return that one
    assert choice_a is not None or choice_b is not None, (
        "At least one gemm choice must be specified"
    )
    return choice_a if choice_a is not None else choice_b


def _addmm_mx_dispatch(
    a: MXTensor, b: MXTensor, aten_op, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Core implementation shared between mx_mm and mx_addmm.
    The only difference is whether bias is None or not.
    """
    gemm_choice = _get_gemm_choice(a._gemm_kernel_choice, b._gemm_kernel_choice)

    if gemm_choice in (
        MXGemmKernelChoice.CUBLAS,
        MXGemmKernelChoice.CUTLASS,
        MXGemmKernelChoice.HIPBLASLT,
    ):
        # real MX gemm backed by torchao's CUTLASS kernels
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        assert a._data.is_contiguous()
        assert b._data.t().is_contiguous()
        assert a._block_size == 32, f"Invalid block size {a._block_size}"
        assert b._block_size == 32, f"Invalid block size {b._block_size}"

        a_scale = a._scale_e8m0.view(M, K // a._block_size)
        b_scale = b._scale_e8m0.view(N, K // b._block_size)
        a_scale_block = to_blocked(a_scale)
        b_scale_block = to_blocked(b_scale)

        if a._elem_dtype == torch.float8_e4m3fn:
            assert b._elem_dtype == torch.float8_e4m3fn
            assert gemm_choice in (
                MXGemmKernelChoice.CUBLAS,
                MXGemmKernelChoice.HIPBLASLT,
            ), "CUBLAS is the only supported kernel choice for MX FP8 operations"

            res = torch._scaled_mm(
                a._data,
                b._data,
                a_scale_block.view(torch.float8_e8m0fnu),
                b_scale_block.view(torch.float8_e8m0fnu),
                bias=bias,
                out_dtype=torch.bfloat16,
            )

        else:
            assert a._elem_dtype == torch.float4_e2m1fn_x2
            assert b._elem_dtype == torch.float4_e2m1fn_x2
            assert gemm_choice is MXGemmKernelChoice.CUTLASS, "unsupported"
            # FP4 operations
            res = torchao.ops.mx_fp4_bf16(
                a._data, b._data, a_scale_block, b_scale_block
            )
            # TODO add optional bias to kernel
            if bias is not None:
                res = res + bias

    else:
        # emulated MX gemm
        a_hp = a.to_dtype(a._orig_dtype)
        b_hp = b.to_dtype(b._orig_dtype)
        # assert memory layout we expect to be required in hardware
        assert a_hp.is_contiguous()
        assert b_hp.t().is_contiguous()

        # Call appropriate aten_op based on whether bias is provided
        if bias is not None:
            res = aten_op(bias, a_hp, b_hp)  # addmm
        else:
            res = aten_op(a_hp, b_hp)  # mm

    return res


@implements([aten.mm.default, aten.matmul.default])
def mx_mm(func, types, args, kwargs):
    a = args[0]
    b = args[1]
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)

    return _addmm_mx_dispatch(a, b, func)


@implements([aten.addmm.default])
def mx_addmm(func, types, args, kwargs):
    assert (
        isinstance(args[0], torch.Tensor)
        and isinstance(args[1], MXTensor)
        and isinstance(args[2], MXTensor)
    )
    bias = args[0]
    a = args[1]
    b = args[2]
    return _addmm_mx_dispatch(a, b, func, bias=bias)


@implements([aten.t.default])
def mx_t(func, types, args, kwargs):
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
def mx_cast_up_op(func, types, args, kwargs):
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
    return func(*new_args, **new_kwargs)


@implements([aten.view.default])
def mx_view_op(func, types, args, kwargs):
    data = args[0]._data
    new_size = args[1]
    if args[0]._elem_dtype == torch.float4_e2m1fn_x2:
        # special case fp4 as we pack two elements per byte
        new_size = tensor_size_hp_to_fp4x2(new_size, data.is_contiguous())
    elif args[0]._elem_dtype in [DTYPE_FP6_E3M2, DTYPE_FP6_E2M3] and args[0]._pack_fp6:
        # special case fp6 as we pack 4 elements in 3 bytes
        new_size = tensor_size_hpx3_to_fp6x4(new_size, data.is_contiguous())
    new_data = func(data, new_size, *args[2:], **kwargs)
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


@implements([aten.slice.Tensor])
def mx_slice(func, types, args, kwargs):
    x, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    M, K = x.shape[0], x.shape[1]

    # TODO why doesn't scale have shape?
    scale_shaped = x._scale_e8m0.view(M, K // x._block_size)

    if dim == 0:
        # Slicing along the first dimension (rows) TODO assuming that dim 1 is reduciton dim for now
        sliced_scale = aten.slice.Tensor(scale_shaped, dim, start, end, step).flatten()
        sliced_data = aten.slice.Tensor(x._data, dim, start, end, step)
    elif dim == 1:
        # Slicing along reduciton dim
        if start is not None:
            # Assert start is a multiple of block_size
            assert start % x._block_size == 0, (
                f"Start index {start} must be a multiple of block_size {x._block_size}"
            )

        if end is not None:
            # Assert end is a multiple of block_size
            assert end % x._block_size == 0, (
                f"End index {end} must be a multiple of block_size {x._block_size}"
            )

        sliced_data = aten.slice.Tensor(x._data, dim, start, end, step)

        # Calculate which scale elements to keep
        start_block = 0 if start is None else start // x._block_size
        end_block = -1 if end is None else end // x._block_size

        # Slice the scale tensor accordingly
        sliced_scale = aten.slice.Tensor(
            scale_shaped, 1, start_block, end_block, step
        ).flatten()
    else:
        raise ValueError(
            f"MXTensor only supports slicing along dimensions 0 and 1, got dim={dim}"
        )

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        MXTensor(
            sliced_scale,
            sliced_data,
            x._elem_dtype,
            x._block_size,
            x._orig_dtype,
            x._use_fp4_custom_triton_dequant_kernel,
            x._gemm_kernel_choice,
            x._pack_fp6,
        ),
    )


@implements([aten.copy_.default])
def mx_copy_(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if MXTensor._same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return
    raise ValueError(
        f"Not supported args for copy_ due to metadata mistach: {args[0], args[1]}"
    )


@implements([aten._to_copy.default])
def autocast_to_copy(func, types, args, kwargs):
    """Autocast + device movement"""
    assert isinstance(args[0], MXTensor)

    # Handle dtype parameter
    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        assert dtype in {
            torch.float16,
            torch.bfloat16,
        }, "Only support floating point conversion for autocast w/ MXTensor"

    # Handle device parameter
    device = kwargs.pop("device", None)
    if device is not None:
        # Apply device change using _apply_fn_to_data
        tensor = args[0]._apply_fn_to_data(lambda x: func(x, device=device))
        tensor = return_and_correct_aliasing(func, args, {}, tensor)
    else:
        tensor = args[0]

    # Verify no other kwargs remain
    assert len(kwargs) == 0, "Only support dtype and device kwargs for autocast"

    # If dtype is specified, create a new MXTensor with the requested dtype
    if dtype is not None:
        res = MXTensor(
            tensor._scale_e8m0,
            tensor._data,
            tensor._elem_dtype,
            tensor._block_size,
            dtype,
            tensor._use_fp4_custom_triton_dequant_kernel,
            tensor._gemm_kernel_choice,
            tensor._pack_fp6,
        )
        return res

    # If only device was changed, return the device-changed tensor
    return tensor
