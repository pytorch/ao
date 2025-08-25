# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout, get_out_shape
from torchao.float8.inference import (
    Float8MMConfig,
    _is_rowwise_scaled,
    addmm_float8_unwrapped_inference,
    preprocess_data,
)
from torchao.utils import _is_float8_type, fill_defaults

aten = torch.ops.aten
FLOAT8_IMPL_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops: List[Any]):
    """Register aten ops to the float8 op table"""

    def decorator(func):
        for op in aten_ops:
            FLOAT8_IMPL_OPS_TABLE[op] = func
        return func

    return decorator


def _same_metadata(self: "Float8AQTTensorImpl", src: "Float8AQTTensorImpl") -> bool:
    # Special handling for transposed attribute
    transposed_match = (self.transposed == src.transposed) or (
        self.transposed is False and src.transposed is None
    )
    return (
        isinstance(self, Float8AQTTensorImpl)
        and isinstance(src, Float8AQTTensorImpl)
        and self.shape == src.shape
        and self.float8_data.shape == src.float8_data.shape
        and self.scale.shape == src.scale.shape
        and transposed_match
        and type(self._layout) == type(src._layout)
    )


@dataclass(frozen=True)
class Float8Layout(Layout):
    """Represents the layout configuration for Float8 affine quantized tensors.

    Attributes:
        mm_config (Optional[Float8MMConfig]): Configuration for matrix multiplication operations involving Float8 tensors. If None, default settings are used.
    """

    mm_config: Optional[Float8MMConfig] = None


_fallback_warning_shown = False


@register_layout(Float8Layout)
class Float8AQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for float8 layout affine quantized tensor

    Note: technically we should not create a new layout for float8 we should merge this into
    plain layout
    """

    float8_data: torch.Tensor
    scale: torch.Tensor
    transposed: bool

    def __new__(
        cls,
        float8_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = float8_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else float8_data.layout
        )
        kwargs["dtype"] = float8_data.dtype
        kwargs["requires_grad"] = False
        shape = float8_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        float8_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        self.float8_data = float8_data
        self.scale = scale
        self.transposed = transposed
        self._layout = _layout

    def _apply_fn_to_data(self, fn):
        """Applys a fn to all tensor components stored on this class"""
        global _fallback_warning_shown

        try:
            return self.__class__(
                fn(self.float8_data),
                fn(self.scale),
                self.transposed,
                self._layout,
            )
        except RuntimeError as e:
            if '"index_cuda" not implemented for ' in str(e):
                if not _fallback_warning_shown:
                    import warnings

                    warnings.warn(
                        f"When trying to index Float8AQTTensorImpl, got known error {e}, will use slower fallback but "
                        + "note: You can torch.compile the model to avoid this problem.",
                        UserWarning,
                    )
                    _fallback_warning_shown = True

                return self.__class__(  # do indexing in bfloat16 then convert back
                    fn(self.float8_data.to(torch.bfloat16)).to(self.float8_data.dtype),
                    fn(self.scale),
                    self.transposed,
                    self._layout,
                )
            else:
                raise e

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.float8_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.transposed,
            self._layout,
        )

    def __tensor_flatten__(self):
        return ["float8_data", "scale"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        float8_data, scale = tensor_data_dict["float8_data"], tensor_data_dict["scale"]
        (
            transposed,
            _layout,
        ) = tensor_attributes
        return cls(float8_data, scale, transposed, _layout)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented

        if func in FLOAT8_IMPL_OPS_TABLE:
            return FLOAT8_IMPL_OPS_TABLE[func](func, types, args, kwargs)

        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.float8_data, self.scale, None

    def get_layout(self) -> Layout:
        return self._layout

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """Main entrypoint for constructing Float8TensorImpl"""
        assert _is_float8_type(data.dtype), (
            f"Float8 TensorImpl must be constructed from float8 dtype but got {data.dtype}"
        )
        assert isinstance(_layout, Float8Layout), (
            f"Float8 TensorImpl must be constructed from Float8Layout but got {_layout}"
        )
        return cls(data, scale, False, _layout)

    def __repr__(self):
        float8_data, scale, _ = self.get_plain()
        _layout = self.get_layout()
        return (
            f"{self.__class__.__name__}(\n"
            f"float8_data={float8_data},\n"
            f"scale={scale},\n"
            f"transposed={self.transposed}, "
            f"_layout={_layout})"
        )


##########################
# Regsiter FP8 Ops
##########################


@implements([aten.detach.default, aten.alias.default, aten.clone.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(func)
    )


@implements([aten.t.default])
def _(func, types, args, kwargs):
    """we don't need to repack the weight and just rely on external
    shape being changed and record the status of transpose/no-transpose
    """
    args[0].transposed = not args[0].transposed
    return return_and_correct_aliasing(func, args, kwargs, args[0])


@implements([aten.copy_.default])
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


@implements([aten.select.int, aten.index.Tensor])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
    )


@implements([aten.slice.Tensor])
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    # Always slice the float8_data
    sliced_data = aten.slice.Tensor(self.float8_data, dim, start, end, step)

    if self.scale.numel() == 1:
        # Per-tensor quantization - scale doesn't change
        sliced_scale = self.scale
    else:
        # Block-wise quantization - need to slice the scale appropriately
        sliced_scale = _slice_scale_for_dimension(
            self.scale, self.float8_data.shape, dim, start, end, step
        )

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Float8AQTTensorImpl(
            sliced_data,
            sliced_scale,
            self.transposed,
            self._layout,
        ),
    )


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


##########################
# Float8 Dispatch Kernels
##########################


def _linear_fp8_act_fp8_weight_check(
    input_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    weight_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    bias: Optional[torch.Tensor],
) -> bool:
    def check_aqt(aqt: Union[torch.Tensor, AffineQuantizedTensor]) -> bool:
        return (
            isinstance(aqt, AffineQuantizedTensor)
            and isinstance(aqt._layout, Float8Layout)
            and _is_float8_type(aqt.tensor_impl.dtype)
            and (aqt.shape == aqt.block_size or _is_rowwise_scaled(aqt))
        )

    return check_aqt(input_tensor) and check_aqt(weight_tensor)


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


def _linear_fp8_act_fp8_weight_impl(
    input_tensor: "AffineQuantizedTensor",
    weight_tensor: "AffineQuantizedTensor",
    bias: Optional[torch.Tensor],
):
    """Implements matmul between FP8 input and FP8 weight with compute using _scaled_mm"""
    scaled_mm_config = weight_tensor._layout.mm_config
    assert scaled_mm_config is not None
    assert not weight_tensor.tensor_impl.transposed, "Weight tensor must be contiguous"

    out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)

    # Extract tensor data and scales
    inpt_data = input_tensor.tensor_impl.float8_data.reshape(
        -1, input_tensor.tensor_impl.float8_data.shape[-1]
    )
    w_data = weight_tensor.tensor_impl.float8_data
    input_scale = input_tensor.tensor_impl.scale
    w_scale = weight_tensor.tensor_impl.scale

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


def _linear_fp_act_fp8_weight_check(
    input_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    weight_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    bias: Optional[torch.Tensor],
) -> bool:
    return (
        # input is native float tensor
        not is_traceable_wrapper_subclass(input_tensor)
        and input_tensor.is_floating_point()
        and
        # weight is float8 quantized affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor)
        and isinstance(weight_tensor._layout, Float8Layout)
        and _is_float8_type(weight_tensor.tensor_impl.dtype)
        and (
            weight_tensor.shape == weight_tensor.block_size
            or _is_rowwise_scaled(weight_tensor)
        )
    )


def _linear_fp_act_fp8_weight_impl(
    input_tensor: torch.Tensor,
    weight_tensor: "AffineQuantizedTensor",
    bias: Optional[torch.Tensor],
):
    return torch.nn.functional.linear(input_tensor, weight_tensor.dequantize(), bias)
