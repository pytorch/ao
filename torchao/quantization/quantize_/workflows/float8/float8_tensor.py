# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.utils import get_out_shape
from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _is_rowwise_scaled,
    addmm_float8_unwrapped_inference,
    preprocess_data,
)
from torchao.quantization.granularity import PerRow
from torchao.quantization.observer import get_block_size
from torchao.quantization.quant_primitives import (
    _choose_qparams_affine_float8,
    _dequantize_affine_float8,
    _quantize_affine_float8,
)
from torchao.quantization.quantize_.common import QuantizeTensorKwargs
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
    fill_defaults,
)

from .._utils import _choose_quant_func_and_quantize_tensor

__all__ = [
    "Float8Tensor",
    "QuantizeTensorToFloat8Kwargs",
]

aten = torch.ops.aten


def _is_per_tensor_quantized(tensor):
    block_size = tensor.block_size
    return all(x == -1 or x == y for x, y in zip(block_size, tensor.shape))


def _is_per_row_quantized(tensor):
    block_size = tensor.block_size
    return all(x == 1 for x in block_size[:-1]) and block_size[-1] == tensor.shape[-1]


def preprocess_scale(input_scale: torch.Tensor, input_shape: Tuple[int, ...]):
    """Ensures input tensor is correctly formatted for _scaled_mm"""

    # For PerTensor quantization, scale should be a scalar or have shape [1]
    if input_scale.numel() == 1:
        # Already a scalar, ensure it has the right shape for _scaled_mm
        return input_scale.reshape(1, 1)

    # For per-row/block quantization, we need to handle the reshaping
    input_scale = input_scale.unsqueeze(-1)

    # Match: #input_data.reshape(-1, input_data.shape[-1])
    if input_scale.dim() > 2:
        input_scale = input_scale.reshape(-1, input_scale.shape[-1])

    return input_scale


def _slice_scale_for_dimension(
    scale: torch.Tensor,
    data_shape: List[int],
    dim: int,
    start: int,
    end: int,
    step: int,
) -> torch.Tensor:
    """
    Slice the scale tensor appropriately based on the data tensor slicing.

    This function calculates how the scale should be sliced when the data tensor
    is sliced along a given dimension, taking into account the block structure.
    """
    # Unsupported case for now, this would be 1 scale per data element
    if scale.shape == data_shape:
        return aten.slice.Tensor(scale, dim, start, end, step)

    # Reconstruct block sizes based on data shape and scale shape
    block_sizes = tuple(data_shape[i] // scale.shape[i] for i in range(len(data_shape)))

    if dim >= len(block_sizes):
        # Slicing beyond the dimensions we care about
        return scale

    block_size_for_dim = block_sizes[dim]

    if block_size_for_dim == 1:
        # Scale is per-element along this dimension
        # Slice away as normal
        return aten.slice.Tensor(scale, dim, start, end, step)
    else:
        # There is blocking in this dimension
        # Calculate which scale elements correspond to the sliced data
        scale_start = start // block_size_for_dim if start is not None else None
        scale_end = (
            (end + block_size_for_dim - 1) // block_size_for_dim
            if end is not None
            else None
        )

        # Error on Step > 1
        if step > 1:
            raise NotImplementedError(
                "Slicing with step > 1 is not implemented for scale tensors."
            )

        return aten.slice.Tensor(scale, dim, scale_start, scale_end, 1)


@dataclass
class QuantizeTensorToFloat8Kwargs(QuantizeTensorKwargs):
    """Tensor kwargs for creating float8 tensor (either activation or weight)

    Args:
       dtype (torch.dtype): the dtype for float8 Tensor
       granularity (FP8Granularity): the granularity for the Tensor, currently either PerRow() or PerTensor()
    """

    dtype: torch.dtype = torch.float8_e4m3fn
    granularity: FP8Granularity = PerRow()
    mm_config: Float8MMConfig = Float8MMConfig(use_fast_accum=True)


class Float8Tensor(TorchAOBaseTensor):
    """
    Float8 Quantized (weight) Tensor, with float8 dynamic quantization for activation or bfloat16 activation.

    TODO: needs padding for cutlass kernels

    Tensor Attributes:
        _data: float8 raw data
        scale: the scale for float8 Tensor

    Non-Tensor Attributes:
        block_size (List[int]): the block size for float8 quantization, meaning the shape of the elements
        sharing the same set of quantization parameters (scale), have the same rank as _data or
        is an empty list (representing per tensor quantization)
        act_quant_kwargs (QuantizeTensorToFloat8Kwargs): the kwargs for Float8Tensor.to_float8
        dtype: Original Tensor dtype
    """

    tensor_data_attrs = ["_data", "scale"]
    tensor_attributes = ["block_size", "mm_config", "act_quant_kwargs", "dtype"]

    def __new__(
        cls,
        _data,
        scale,
        block_size,
        mm_config,
        act_quant_kwargs,
        dtype,
    ):
        shape = _data.shape
        kwargs = {}
        kwargs["device"] = _data.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        _data: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        mm_config: Float8MMConfig = Float8MMConfig(use_fast_accum=True),
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._data = _data
        self.scale = scale
        self.block_size = block_size
        self.mm_config = mm_config
        self.act_quant_kwargs = act_quant_kwargs

    def __tensor_flatten__(self):
        return self.tensor_data_attrs, [
            getattr(self, attr) for attr in self.tensor_attributes
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        tensors = [tensor_data_dict[name] for name in cls.tensor_data_attrs]
        return cls(
            *tensors,
            *tensor_attributes,
        )

    def _apply_fn_to_data(self, fn):
        tensors = [fn(getattr(self, attr)) for attr in self.tensor_data_attrs]
        return self.__class__(
            *tensors,
            *[getattr(self, attr) for attr in self.tensor_attributes],
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self._data=}, {self.scale=}, "
            f"{self.block_size=}, {self.mm_config=}"
            f"{self.shape=}, {self.device=}, {self.dtype=}, "
            f"{self.requires_grad=})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        t = self.__class__(
            self._data.to(device),
            self.scale.to(device),
            self.block_size,
            self.mm_config,
            self.act_quant_kwargs,
            self.dtype,
        )
        return t

    def _transpose_and_reshape(self):
        """This is added for resharding support, since the resharding logic for the model we are
        working with only support 2D

        High level goal is to match the shape of the original unquantized Tensor and reshape
        it to 2D since resharding logic only supports 2D Tensor

        * transpose(1, 2) since we did a transpose initially to quantize the weight
        * reshape to 2D
        """
        assert len(self.shape) == 3, (
            f"Only expected to be used when the Tensor is 3D, got {len(self.shape)}"
        )
        dim0, dim1, dim2 = self.shape
        # because we first transpose the weight before quantization, we'll recover the original shape
        # by swapping dim1 and dim2
        original_shape = (dim0, dim2, dim1)
        # we must save this as 2D in the state dict, since loading code expects 2D weights
        new_shape = (-1, original_shape[-1])
        _data = self._data
        _data = _data.transpose(1, 2).reshape(*new_shape).contiguous()
        scale = self.scale.transpose(1, 2).reshape(*new_shape).contiguous()
        block_size = self.block_size.copy()
        block_size[1], block_size[2] = block_size[2], block_size[1]
        block_size = [block_size[0] * block_size[1], block_size[2]]

        return self.__class__(
            _data,
            scale,
            block_size,
            self.mm_config,
            self.act_quant_kwargs,
            self.dtype,
        )

    def _unflatten(self, num_experts):
        """This is added for resharding support, since the resharding logic for the model we are
        working with only support 2D

        This is called after resharding logic, and it reverses the reshape to 2D in `_transpose_and_reshape`
        and gives a 3D tensor with `num_experts` as the first dimension
        """
        _data = self._data
        scale = self.scale
        _data = _data.unflatten(0, (num_experts, -1)).squeeze(dim=0)
        scale = scale.unflatten(0, (num_experts, -1)).squeeze(dim=0)
        block_size = self.block_size.copy()
        block_size = [1, block_size[0], block_size[1]]

        return self.__class__(
            _data,
            scale,
            block_size,
            self.mm_config,
            self.act_quant_kwargs,
            self.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        _data, scale = self._data, self.scale
        return _dequantize_affine_float8(_data, scale, output_dtype)

    @classmethod
    def to_float8(
        cls,
        hp_tensor: torch.Tensor,
        quant_kwargs: QuantizeTensorToFloat8Kwargs,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        lp_dtype = quant_kwargs.dtype
        granularity = quant_kwargs.granularity
        mm_config = quant_kwargs.mm_config
        block_size = get_block_size(hp_tensor.shape, granularity)
        block_size = list(block_size)

        # TODO: https://github.com/pytorch/ao/issues/2511
        # NOTE: fbgemm quant primitives like `torch.ops.fbgemm.quantize_fp8_per_row`
        # can't pass some of the numerics tests, so we use torchao ones
        scale = _choose_qparams_affine_float8(
            hp_tensor, float8_dtype=lp_dtype, block_size=block_size
        )
        data = _quantize_affine_float8(hp_tensor, scale, lp_dtype)
        hp_dtype = hp_tensor.dtype
        return Float8Tensor(
            data,
            scale,
            block_size=block_size,
            mm_config=mm_config,
            act_quant_kwargs=act_quant_kwargs,
            dtype=hp_dtype,
        )


implements = Float8Tensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if not isinstance(weight_tensor, Float8Tensor):
        raise NotImplementedError("Only support handling Float8 weight")

    # NOTE: we are using torchao quant primitive ops instead of fbgemm ones due to various numerical
    # issues with fbgemm quant primitive ops
    # for example fbgemm per tensor quant op is not as accurate as the quant primitives in torchao and won't
    # pass the accuracy check in
    # TestAffineQuantizedFloat8Compile.test_float8_tensor_slicing_functional_correctness
    # https://github.com/pytorch/ao/blob/20d45036603c96873b8a8a8391fdbf2a2771ab7e/test/dtypes/test_affine_quantized_float.py#L631
    # a_data, a_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(input_tensor)
    act_quant_kwargs = weight_tensor.act_quant_kwargs
    if act_quant_kwargs is not None:
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

    if isinstance(input_tensor, Float8Tensor):
        scaled_mm_config = weight_tensor.mm_config
        assert scaled_mm_config is not None
        out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)

        # Extract tensor data and scales
        inpt_data = input_tensor._data.reshape(-1, input_tensor._data.shape[-1])
        w_data = weight_tensor._data
        input_scale = input_tensor.scale
        w_scale = weight_tensor.scale.contiguous()

        # Handle rowwise scaling
        if _is_rowwise_scaled(weight_tensor):
            assert _is_rowwise_scaled(input_tensor), (
                "Input tensor must be rowwise block size"
            )
            w_scale = w_scale.transpose(-1, -2)

        input_scale = preprocess_scale(input_scale, input_tensor.shape)
        inpt_data, w_data = preprocess_data(inpt_data, w_data.T, scaled_mm_config)

        return addmm_float8_unwrapped_inference(
            inpt_data,
            input_scale,
            w_data,
            w_scale,
            output_dtype=input_tensor.dtype,
            bias=bias,
            use_fast_accum=scaled_mm_config.use_fast_accum,
        ).reshape(out_shape)
    else:
        return torch.nn.functional.linear(
            input_tensor, weight_tensor.dequantize(), bias
        )


@implements(torch.bmm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    if not isinstance(weight_tensor, Float8Tensor):
        raise NotImplementedError("Only support handling Float8 weight")

    orig_act_size = input_tensor.size()
    act_quant_kwargs = weight_tensor.act_quant_kwargs
    if act_quant_kwargs is not None:
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

    if isinstance(input_tensor, Float8Tensor):
        a_data = input_tensor._data
        a_scale = input_tensor.scale

        b_data = weight_tensor._data
        b_scale = weight_tensor.scale.squeeze(-1)
        assert b_data.is_contiguous(), "weight for bmm must be contiguous"

        assert (
            all(x == 1 for x in weight_tensor.block_size[:-1])
            and weight_tensor.block_size[-1] == weight_tensor.shape[-1]
        ), "bmm only works for per row weight quantization"
        assert (
            all(x == 1 for x in input_tensor.block_size[:-1])
            and input_tensor.block_size[-1] == input_tensor.shape[-1]
        ), "bmm only works for per row activation quantization"

        orig_out_features = b_data.shape[-2]

        res = torch.ops.fbgemm.f8f8bf16_rowwise_batched(
            a_data,
            b_data,
            a_scale,
            b_scale,
        )
        res = res.reshape(*orig_act_size[:-1], orig_out_features)
    else:
        raise NotImplementedError(
            "bmm only support float8 dynamic activation + float8 weight"
        )

    return res


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


def _same_metadata(self: "Float8Tensor", src: "Float8Tensor") -> bool:
    return (
        isinstance(self, Float8Tensor)
        and isinstance(src, Float8Tensor)
        and self.shape == src.shape
        and self._data.shape == src._data.shape
        and self.scale.shape == src.scale.shape
        and self.act_quant_kwargs == src.act_quant_kwargs
        and self.dtype == src.dtype
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
        f"Not supported args for copy_ due to metadata mismatch: {args[0], args[1]}"
    )


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Only supports slicing for dim == 1 and dim == 2
    original tensor shape has dimension (N, K)
    _data has dimension (N, K)
    scale (per row quantization) has dimension: (N,)

    since _data has the same dimension as original tensor, we can directly slice that
    for scale, we'll do a slice when dim is 0, and don't need to do anything for dim 1

    Note that we need to call slice on the _data and scale directly because slice
    is an operation that need to preserve aliasing
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self._data.ndim == 2, (
        f"Expected packed weight to have dim 2, got {self._data.dim}"
    )

    # Always slice the _data
    sliced_data = aten.slice.Tensor(self._data, dim, start, end, step)

    if self.scale.numel() == 1:
        # Per-tensor quantization - scale doesn't change
        sliced_scale = self.scale
    else:
        # Block-wise quantization - need to slice the scale appropriately
        sliced_scale = _slice_scale_for_dimension(
            self.scale, self._data.shape, dim, start, end, step
        )

    # adjust block_size for rowwise quantization
    block_size = self.block_size.copy()
    if _is_rowwise_scaled(self):
        for i in range(len(self.block_size)):
            block_size[i] = min(block_size[i], sliced_data.shape[i])

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Float8Tensor(
            sliced_data,
            sliced_scale,
            block_size,
            self.mm_config,
            self.act_quant_kwargs,
            dtype=self.dtype,
        ),
    )


@implements(aten.cat.default)
def _(func, types, args, kwargs):
    """Concatenate multiple float8 quantized tensors
    (scale and _data has the same rank)
    If the concatenation dimension is not the same as block_size, then we can just concatenate the
    _data and scale directly
    If the concatention dimension is the same as block_size, theoretically we should either
      (1) check that scales from all tensors are equal and use the first scale
      (2) dequantize and requantize
    but for now we just use the first scale directly, which might have slight implication on accuaracy
    we can improve upon this a bit later
    """

    tensors, dim = fill_defaults(args, 2, [[], 0])
    tensor_0 = tensors[0]
    dim = dim % tensor_0.ndim

    for i in range(1, len(tensors)):
        assert tensor_0._data.ndim == tensors[i]._data.ndim
        assert tensor_0.scale.ndim == tensors[i].scale.ndim
        assert tensor_0.block_size == tensors[i].block_size
        assert tensor_0.mm_config == tensors[i].mm_config

    _datas = [t._data for t in tensors]
    scales = [t.scale for t in tensors]

    cat_data = aten.cat.default(_datas, dim=dim)
    if tensor_0.block_size[dim] == 1:
        cat_scale = aten.cat.default(scales, dim=dim)
    else:
        # TODO: this is not exactly correct, we'll need to
        # figure out how to do this properly in the future
        cat_scale = scales[0]

    new = tensor_0.__class__(
        cat_data,
        cat_scale,
        tensor_0.block_size,
        tensor_0.mm_config,
        tensor_0.act_quant_kwargs,
        tensor_0.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args
    _data = self._data.transpose(dim0, dim1).contiguous()
    scale = self.scale.transpose(dim0, dim1).contiguous()
    block_size = self.block_size.copy()

    block_size[dim0], block_size[dim1] = block_size[dim1], block_size[dim0]

    new = self.__class__(
        _data,
        scale,
        block_size,
        self.mm_config,
        self.act_quant_kwargs,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


Float8Tensor.__module__ = "torchao.quantization"

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with Float8Tensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([Float8Tensor, QuantizeTensorToFloat8Kwargs])
