# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
        return self.__class__(
            fn(self.float8_data),
            fn(self.scale),
            self.transposed,
            self._layout,
        )

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

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        elif func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )
        elif func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            args[0].transposed = not args[0].transposed
            return return_and_correct_aliasing(func, args, kwargs, args[0])
        elif func is aten.copy_.default:
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
        elif func in [aten.select.int, func is aten.index.Tensor]:
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
            )
        elif func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                # TODO: scale replecation should be dependent on block size
                if self.scale.ndim == 1:
                    return return_and_correct_aliasing(
                        func,
                        args,
                        kwargs,
                        args[0]._apply_fn_to_data(
                            lambda x: aten.slice.Tensor(x, dim, start, end, step)
                        ),
                    )
                elif self.scale.ndim == 0:
                    return return_and_correct_aliasing(
                        func,
                        args,
                        kwargs,
                        Float8AQTTensorImpl(
                            aten.slice.Tensor(self.float8_data, dim, start, end, step),
                            self.scale,
                            None,
                            self._layout,
                        ),
                    )
                else:
                    raise NotImplementedError(
                        f"Float8AQTTensorImpl dispatch: attempting to run {func}, with scale ndim={dim}, that is not supported"
                    )
            elif dim == 1:
                return return_and_correct_aliasing(
                    func,
                    args,
                    kwargs,
                    Float8AQTTensorImpl(
                        aten.slice.Tensor(
                            self.float8_data, dim, start, end, step
                        ).contiguous(),
                        self.scale,
                        None,
                        self._layout,
                    ),
                )
            else:
                raise NotImplementedError(
                    f"Float8AQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )
        else:
            raise NotImplementedError(
                f"Float8AQTTensorImpl dispatch: attempting to run {func}, this is not supported"
            )

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
            and aqt.tensor_impl.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            and (aqt.shape == aqt.block_size or _is_rowwise_scaled(aqt))
        )

    return check_aqt(input_tensor) and check_aqt(weight_tensor)


def preprocess_scale(input_scale: torch.Tensor, input_shape: Tuple[int]):
    """Ensures input tensor is correctly formated for _scaled_mm"""
    input_scale = input_scale.unsqueeze(-1)

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
    out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)

    # Weight tensor preprocessing
    w_tensor_impl = weight_tensor.tensor_impl
    assert not w_tensor_impl.transposed, "Weight tensor must be contiguous"
    w_data = w_tensor_impl.float8_data
    w_scale = w_tensor_impl.scale

    # Input tensor preprocessing
    inpt_data = input_tensor.tensor_impl.float8_data
    input_scale = input_tensor.tensor_impl.scale
    # Handle case where input tensor is more than 2D
    inpt_data = inpt_data.reshape(-1, inpt_data.shape[-1])

    # Handle rowwise case
    if _is_rowwise_scaled(weight_tensor):
        assert _is_rowwise_scaled(input_tensor), (
            "Input tensor must be rowwise block size"
        )
        w_scale = w_scale.unsqueeze(-1).T
        input_scale = preprocess_scale(input_scale, input_tensor.shape)

    # Preprocess data
    inpt_data, w_data = preprocess_data(inpt_data, w_data.T, scaled_mm_config)

    # Perform the computation
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
        and weight_tensor.tensor_impl.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
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
