# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)
from torchao.dtypes.floatx.cutlass_semi_sparse_layout import (
    _linear_fp8_act_fp8_weight_sparse_cutlass_check,
    _linear_fp8_act_fp8_weight_sparse_cutlass_impl,
)
from torchao.dtypes.floatx.float8_layout import (
    _linear_fp8_act_fp8_weight_check,
    _linear_fp8_act_fp8_weight_impl,
    _linear_fp_act_fp8_weight_check,
    _linear_fp_act_fp8_weight_impl,
)
from torchao.dtypes.floatx.floatx_tensor_core_layout import (
    _linear_f16_bf16_act_floatx_weight_check,
    _linear_f16_bf16_act_floatx_weight_impl,
)
from torchao.dtypes.uintx.block_sparse_layout import (
    _linear_int8_act_int8_weight_block_sparse_check,
    _linear_int8_act_int8_weight_block_sparse_impl,
)
from torchao.dtypes.uintx.cutlass_int4_packed_layout import (
    _linear_int4_act_int4_weight_cutlass_check,
    _linear_int4_act_int4_weight_cutlass_impl,
    _linear_int8_act_int4_weight_cutlass_check,
    _linear_int8_act_int4_weight_cutlass_impl,
)
from torchao.dtypes.uintx.dyn_int8_act_int4_wei_cpu_layout import (
    _linear_int8_act_int4_weight_cpu_check,
    _linear_int8_act_int4_weight_cpu_impl,
)
from torchao.dtypes.uintx.gemlite_layout import (
    _linear_fp_act_int4_weight_gemlite_check,
    _linear_fp_act_int4_weight_gemlite_impl,
)
from torchao.dtypes.uintx.int4_cpu_layout import (
    _linear_fp_act_uint4_weight_cpu_check,
    _linear_fp_act_uint4_weight_cpu_impl,
)
from torchao.dtypes.uintx.int4_xpu_layout import (
    _linear_bf16_act_uint4_weight_float_zero_check,
    _linear_bf16_act_uint4_weight_float_zero_impl,
    _linear_fp_act_uint4_weight_int8_zero_check,
    _linear_fp_act_uint4_weight_int8_zero_impl,
)
from torchao.dtypes.uintx.marlin_qqq_tensor import (
    _linear_int8_act_int4_weight_marlin_qqq_check,
    _linear_int8_act_int4_weight_marlin_qqq_impl,
)
from torchao.dtypes.uintx.marlin_sparse_layout import (
    _linear_fp_act_int4_weight_sparse_marlin_check,
    _linear_fp_act_int4_weight_sparse_marlin_impl,
)
from torchao.dtypes.uintx.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    _linear_check as _linear_int8_act_intx_weight_packed_check,
)
from torchao.dtypes.uintx.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    _linear_impl as _linear_int8_act_intx_weight_packed_impl,
)
from torchao.dtypes.uintx.plain_layout import (
    PlainAQTTensorImpl,
    _linear_fp_act_int8_weight_check,
    _linear_fp_act_int8_weight_impl,
    _linear_int8_act_int8_weight_check,
    _linear_int8_act_int8_weight_impl,
)
from torchao.dtypes.uintx.q_dq_layout import (
    _embedding_check as _embedding_q_dq_check,
)
from torchao.dtypes.uintx.q_dq_layout import (
    _embedding_impl as _embedding_q_dq_impl,
)
from torchao.dtypes.uintx.q_dq_layout import (
    _linear_check as _linear_q_dq_check,
)
from torchao.dtypes.uintx.q_dq_layout import (
    _linear_impl as _linear_q_dq_impl,
)
from torchao.dtypes.uintx.semi_sparse_layout import (
    _linear_int8_act_int8_weight_semi_structured_sparse_check,
    _linear_int8_act_int8_weight_semi_structured_sparse_impl,
)
from torchao.dtypes.uintx.tensor_core_tiled_layout import (
    _linear_bf16_act_uint4_weight_check,
    _linear_bf16_act_uint4_weight_impl,
)
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    _dequantize_affine_no_zero_point,
    _dequantize_affine_tinygemm,
    dequantize_affine,
)
from torchao.utils import (
    fill_defaults,
)

logger = logging.getLogger(__name__)


aten = torch.ops.aten


_AQT_QLINEAR_DISPATCH_TABLE = {}


def register_aqt_quantized_linear_dispatch(dispatch_condition, impl):
    """Register a dispatch for quantized linear op with dispatch_condition function and impl function
    both takes three arguments:
      input_tensor: dimension is (M1, M2, ..., in_features)
      weight_tensor: dimension is (out_features, in_features)
      bias: dimension is (out_features,)
      so that these can be shared by F.linear, aten.mm, aten.addmm dispatches

    Args:
        `dispatch_condition` (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], bool]: the dispatch
            condition for a specialized quantized linear implementation, e.g. bfloat16 activation + uint4 weight
        `impl` (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]: the specialized
            quantized linear implementation
    """
    _AQT_QLINEAR_DISPATCH_TABLE[dispatch_condition] = impl


def deregister_aqt_quantized_linear_dispatch(dispatch_condition):
    if dispatch_condition in _AQT_QLINEAR_DISPATCH_TABLE:
        del _AQT_QLINEAR_DISPATCH_TABLE[dispatch_condition]
    else:
        logger.warning(
            f"Attempting to remove non-existent dispatch condition {dispatch_condition}"
        )


def _same_metadata(self: AffineQuantizedTensor, src: AffineQuantizedTensor):
    return (
        isinstance(self, AffineQuantizedTensor)
        and isinstance(src, AffineQuantizedTensor)
        and all(
            [
                getattr(self, attr) == getattr(src, attr)
                for attr in [
                    "block_size",
                    "shape",
                    "quant_min",
                    "quant_max",
                    "zero_point_domain",
                    "dtype",
                ]
            ]
        )
        and isinstance(self.tensor_impl, type(src.tensor_impl))
    )


class QuantizedLinearNotImplementedError(NotImplementedError):
    """Thin wrapper around NotImplementedError to make it easier to catch this error in the dispatch table"""

    pass


# input_tensor: dimension is (M1, M2, ..., in_features)
# weight_tensor: dimension is (out_features, in_features)
# bias: dimension is (out_features,)
@staticmethod
def _quantized_linear_op(input_tensor, weight_tensor, bias):
    for dispatch_condition, impl in _AQT_QLINEAR_DISPATCH_TABLE.items():
        if dispatch_condition(input_tensor, weight_tensor, bias):
            return impl(input_tensor, weight_tensor, bias)
    raise QuantizedLinearNotImplementedError(
        "No specialized dispatch found for quantized linear op"
    )


# Attach the _quantized_linear_op to the AffineQuantizedTensor class
AffineQuantizedTensor._quantized_linear_op = _quantized_linear_op


# _register_aqt_quantized_linear_dispatches function has a list of (dispatch_condition, implementation) functions, defined in their dtype layout classes, that takes the following args:
# input_tensor: dimension is (M1, M2, ..., in_features)
# weight_tensor: dimension is (out_features, in_features)
# bias: dimension is (out_features,)
# so that these can be shared by F.linear, aten.mm, aten.addmm dispatches
def _register_aqt_quantized_linear_dispatches():
    for dispatch_condition, impl in [
        (_linear_int8_act_int8_weight_check, _linear_int8_act_int8_weight_impl),
        (
            _linear_int8_act_int8_weight_semi_structured_sparse_check,
            _linear_int8_act_int8_weight_semi_structured_sparse_impl,
        ),
        (
            _linear_int8_act_int8_weight_block_sparse_check,
            _linear_int8_act_int8_weight_block_sparse_impl,
        ),
        (_linear_fp8_act_fp8_weight_check, _linear_fp8_act_fp8_weight_impl),
        (_linear_fp_act_fp8_weight_check, _linear_fp_act_fp8_weight_impl),
        (_linear_bf16_act_uint4_weight_check, _linear_bf16_act_uint4_weight_impl),
        (_linear_fp_act_int8_weight_check, _linear_fp_act_int8_weight_impl),
        (
            _linear_f16_bf16_act_floatx_weight_check,
            _linear_f16_bf16_act_floatx_weight_impl,
        ),
        (
            _linear_fp_act_int4_weight_sparse_marlin_check,
            _linear_fp_act_int4_weight_sparse_marlin_impl,
        ),
        (
            _linear_int8_act_int4_weight_marlin_qqq_check,
            _linear_int8_act_int4_weight_marlin_qqq_impl,
        ),
        (
            _linear_fp_act_int4_weight_gemlite_check,
            _linear_fp_act_int4_weight_gemlite_impl,
        ),
        (
            _linear_int8_act_int4_weight_cutlass_check,
            _linear_int8_act_int4_weight_cutlass_impl,
        ),
        (
            _linear_int4_act_int4_weight_cutlass_check,
            _linear_int4_act_int4_weight_cutlass_impl,
        ),
        (
            _linear_fp8_act_fp8_weight_sparse_cutlass_check,
            _linear_fp8_act_fp8_weight_sparse_cutlass_impl,
        ),
        (
            _linear_fp_act_uint4_weight_cpu_check,
            _linear_fp_act_uint4_weight_cpu_impl,
        ),
        (
            _linear_int8_act_intx_weight_packed_check,
            _linear_int8_act_intx_weight_packed_impl,
        ),
        (
            _linear_q_dq_check,
            _linear_q_dq_impl,
        ),
        (
            _linear_fp_act_uint4_weight_int8_zero_check,
            _linear_fp_act_uint4_weight_int8_zero_impl,
        ),
        (
            _linear_bf16_act_uint4_weight_float_zero_check,
            _linear_bf16_act_uint4_weight_float_zero_impl,
        ),
        (
            _linear_int8_act_int4_weight_cpu_check,
            _linear_int8_act_int4_weight_cpu_impl,
        ),
    ]:
        register_aqt_quantized_linear_dispatch(dispatch_condition, impl)


_register_aqt_quantized_linear_dispatches()

implements = AffineQuantizedTensor.implements
implements_torch_function = AffineQuantizedTensor.implements_torch_function


@implements([aten.linear.default])
@implements_torch_function([torch.nn.functional.linear])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(
            f"{func} is not implemented for non floating point input"
        )
    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    try:
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)
    except QuantizedLinearNotImplementedError as e:
        # fallback path is only called when user did not specify a specific quantized linear implementation with `_layout.quantized_linear_impl`
        if (
            isinstance(weight_tensor, AffineQuantizedTensor)
            and hasattr(weight_tensor._layout, "quantized_linear_impl")
            and weight_tensor._layout.quantized_linear_impl is not None
        ):
            raise e

        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, AffineQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements_torch_function(torch.nn.functional.embedding)
def _(func, types, args, kwargs):
    if _embedding_q_dq_check(args, kwargs):
        return _embedding_q_dq_impl(args, kwargs)

    # new_arg1 = args[1].dequantize()
    # return torch.nn.embedding(args[0], new_arg1, *args[2:], **kwargs)
    assert isinstance(args[1].tensor_impl, PlainAQTTensorImpl), (
        f"embedding only works with PlainAQTTensorImpl but got {type(args[1].tensor_impl)}"
    )
    assert (
        kwargs["padding_idx"] is None
        and kwargs["max_norm"] is None
        and not kwargs["scale_grad_by_freq"]
        and not kwargs["sparse"]
        and kwargs["norm_type"] == 2.0
    )
    idx = args[0]
    int_data, scale, zero_point = args[1].tensor_impl.get_plain()

    sliced_data, sliced_scale, sliced_zero_point = (
        int_data[idx],
        scale[idx],
        zero_point[idx],
    )
    # Block size is expecting 2 dimensions [1, group size] but
    # batchsize or other dims gets added to sliced_data, sliced_scale and sliced_zero_point so
    # we need to increase block size to correct dim
    new_blocks = idx.dim() - 1
    if args[1].zero_point_domain == ZeroPointDomain.FLOAT:
        _dequantize_affine = _dequantize_affine_tinygemm
    elif args[1].zero_point_domain == ZeroPointDomain.NONE:
        _dequantize_affine = _dequantize_affine_no_zero_point
    else:
        _dequantize_affine = dequantize_affine

    return _dequantize_affine(
        sliced_data,
        new_blocks * [1] + list(args[1].block_size),
        sliced_scale,
        sliced_zero_point,
        sliced_data.dtype,
        args[1].quant_min,
        args[1].quant_max,
        output_dtype=sliced_scale.dtype,
    )


@implements(aten.addmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[1],
        args[2],
        args[0],
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(
            f"{func} is not implemented for non floating point input"
        )

    assert input_tensor.shape[-1] == weight_tensor.shape[0], (
        f"need mat1 shape: {input_tensor.shape} final dim"
        f"to match mat2 shape: {weight_tensor.shape} first dim"
    )

    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    try:
        transposed_weight_tensor = weight_tensor.t()
        return weight_tensor._quantized_linear_op(
            input_tensor, transposed_weight_tensor, bias
        )
    except QuantizedLinearNotImplementedError as e:
        # fallback path is only called when user did not specify a specific quantized linear implementation with `_layout.quantized_linear_impl`
        if (
            isinstance(weight_tensor, AffineQuantizedTensor)
            and hasattr(weight_tensor._layout, "quantized_linear_impl")
            and weight_tensor._layout.quantized_linear_impl is not None
        ):
            raise e

        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, AffineQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return func(bias, input_tensor, weight_tensor)


@implements(aten.mm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (args[0], args[1], None)
    if not input_tensor.is_floating_point():
        raise NotImplementedError(
            f"{func} is not implemented for non floating point input"
        )

    assert input_tensor.shape[-1] == weight_tensor.shape[0], (
        f"need mat1 shape: {input_tensor.shape} final dim"
        f"to match mat2 shape: {weight_tensor.shape} first dim"
    )

    try:
        transposed_weight_tensor = weight_tensor.t()
        return weight_tensor._quantized_linear_op(
            input_tensor, transposed_weight_tensor, bias
        )
    except QuantizedLinearNotImplementedError as e:
        # fallback path is only called when user did not specify a specific quantized linear implementation with `_layout.quantized_linear_impl`
        if (
            isinstance(weight_tensor, AffineQuantizedTensor)
            and hasattr(weight_tensor._layout, "quantized_linear_impl")
            and weight_tensor._layout.quantized_linear_impl is not None
        ):
            raise e

        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, AffineQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return func(input_tensor, weight_tensor)


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[1]._apply_fn_to_data(torch.clone)
    )


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if _same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return
    raise ValueError(
        f"Not supported args for copy_ due to metadata mistach: {args[0], args[1]}"
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    block_size = args[0].block_size
    assert len(block_size) == 2
    transposed_block_size = (block_size[1], block_size[0])
    tensor = args[0]
    shape = tensor.shape[::-1]
    new = tensor.__class__(
        tensor.tensor_impl.t(),
        transposed_block_size,
        shape,
        tensor.quant_min,
        tensor.quant_max,
        tensor.zero_point_domain,
        dtype=tensor.dtype,
        strides=tensor.stride(),
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]
    shape = list(self.shape)
    shape[dim] = end - start
    block_size = self.block_size
    assert len(block_size) in [
        2,
        3,
    ], f"Slice only works for 2 and 3d block_size right now, got: {block_size}"
    # with slice, some shape dimension might be smaller than block_size dimension, so
    # we need to make sure there is no overflow
    if len(block_size) == 2:
        block_size = (min(shape[0], block_size[0]), min(shape[1], block_size[1]))

    new = self.__class__(
        aten.slice.Tensor(self.tensor_impl, dim, start, end, step),
        block_size,
        shape,
        self.quant_min,
        self.quant_max,
        self.zero_point_domain,
        dtype=self.dtype,
        strides=self.stride() if len(block_size) == 2 else None,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.index.Tensor)
def _(func, types, args, kwargs):
    self, indices = args
    assert len(indices) == 1, (
        f"op {func} currently only implemented for single dimensional indexing but got indices: {indices}"
    )
    new_tensor_impl = aten.index.Tensor(self.tensor_impl, indices)
    shape = tuple([indices[0].numel(), *self.shape[1:]])

    block_size = self.block_size
    new = self.__class__(
        new_tensor_impl,
        block_size,
        shape,
        self.quant_min,
        self.quant_max,
        self.zero_point_domain,
        dtype=self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.select.int)
def _(func, types, args, kwargs):
    self, dim, index = fill_defaults(args, 3, [0, 0])
    assert dim == 0, f"op {func} currently only implemented for dim=0 but got dim={dim}"
    assert self.dim() == 3, (
        f"op {func} currently only implemented for 3 dimensional tensors but got shape={self.shape}"
    )

    new_tensor_impl = aten.select.int(self.tensor_impl, dim, index)

    shape = self.shape[1:]
    block_size = self.block_size[1:]
    new = self.__class__(
        new_tensor_impl,
        block_size,
        shape,
        self.quant_min,
        self.quant_max,
        self.zero_point_domain,
        dtype=self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


# this is needed for DTensor.from_local() and for flattening tensor
@implements(aten.view.default)
def _(func, types, args, kwargs):
    self, shape = args

    if tuple(self.shape) == tuple(shape):
        return self.__class__(
            self.tensor_impl,
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            dtype=self.dtype,
            strides=self.stride(),
        )

    if len(shape) == 1 and shape[0] == -1:
        assert len(self.block_size) == 2 and self.block_size[0] == 1
        block_size = (self.block_size[1],)
        return self.__class__(
            self.tensor_impl,
            block_size,
            (self.numel(),),
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            dtype=self.dtype,
            strides=self.stride(),
        )

    raise ValueError(
        f"{self.__class__.__name__} only supports .view() with same shape or shape=[-1]"
    )
