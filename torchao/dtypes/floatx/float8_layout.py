from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math

import torch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.nf4tensor import implements
from torchao.dtypes.utils import AQTTensorImpl, Layout, get_out_shape
from torchao.float8.inference import (
    Float8MMConfig,
    _is_rowwise_scaled,
    addmm_float8_unwrapped_inference,
    preprocess_data,
)
from torchao.utils import _is_float8_type, fill_defaults, TorchAOBaseTensor
from torchao.quantization.quant_primitives import (
    FP8_TYPES,
    MappingType,
    choose_qparams_affine_float8,
    quantize_affine_float8,
)
aten = torch.ops.aten


@dataclass(frozen=True)
class Float8Layout(Layout):
    """Represents the layout configuration for Float8 affine quantized tensors.

    Attributes:
        mm_config (Optional[Float8MMConfig]): Configuration for matrix multiplication operations involving Float8 tensors. If None, default settings are used.
    """

    mm_config: Optional[Float8MMConfig] = None


class Float8Tensor(TorchAOBaseTensor):
    """
    Float8 Tensor is a subclass of torch.Tensor that supports float8 data types.
    It is used to represent the data in a float8 tensor.

    Attributes:
        float8_data (torch.Tensor): The float8 data tensor.
        scale (torch.Tensor): The scale tensor.
        transposed (bool): Whether the tensor is transposed or not.
        _layout (Layout): The layout of the tensor.
    """

    float8_data: torch.Tensor
    scale: torch.Tensor
    transposed: bool

    def __new__(
        cls,
        float8_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        _layout: Layout = Float8Layout(),
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
        _layout: Layout = Float8Layout(),
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

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.float8_data, self.scale, None

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """Main entrypoint for constructing Float8TensorImpl"""
        assert _is_float8_type(
            data.dtype
        ), f"Float8 TensorImpl must be constructed from float8 dtype but got {data.dtype}"
        assert isinstance(
            _layout, Float8Layout
        ), f"Float8 TensorImpl must be constructed from Float8Layout but got {_layout}"
        return cls(data, scale, False, _layout)

    @classmethod
    def from_hp_to_floatx(
        cls,
        input_float: torch.Tensor,
        target_dtype: torch.dtype,
        _layout: Layout = Float8Layout(),
    ):
        """Convert a high precision tensor to a float8 quantized tensor."""
        if target_dtype not in FP8_TYPES:
            raise NotImplementedError(
                f"Unsupported dtype {target_dtype} for from_hp_to_floatx"
            )
        scale = choose_qparams_affine_float8(
            input_float,
            target_dtype,
        )
        float_data = quantize_affine_float8(
            input_float,
            scale,
            target_dtype,
        )

        return cls(
            float_data,
            scale,
            False,
            _layout,
        )

    @classmethod
    def from_hp_to_floatx_static(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype,
        _layout: Layout,
    ):
        """Create a float8 AffineQuantizedTensor from a high precision tensor using static parameters."""
        if target_dtype not in FP8_TYPES:
            raise NotImplementedError(
                f"Unsupported dtype {target_dtype} for from_hp_to_floatx_static"
            )
        float_data = quantize_affine_float8(
            input_float,
            scale,
            target_dtype,
        )

        return cls(
            float_data,
            scale,
            False,
            _layout,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


@implements(aten.t.default)
def _(func, types, args, kwargs):
    """we don't need to repack the weight and just rely on external
    shape being changed and record the status of transpose/no-transpose
    """
    args[0].transposed = not args[0].transposed
    return return_and_correct_aliasing(func, args, kwargs, args[0])


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
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
                Float8Tensor(
                    aten.slice.Tensor(self.float8_data, dim, start, end, step),
                    self.scale,
                    self.transposed,
                    self._layout,
                ),
            )
        else:
            raise NotImplementedError(
                f"Float8Tensor dispatch: attempting to run {func}, with scale ndim={dim}, that is not supported"
            )
    elif dim == 1:
        return return_and_correct_aliasing(
            func,
            args,
            kwargs,
            Float8Tensor(
                aten.slice.Tensor(
                    self.float8_data, dim, start, end, step
                ).contiguous(),
                self.scale,
                self.transposed,
                self._layout,
            ),
        )
    else:
        raise NotImplementedError(
            f"Float8Tensor dispatch: attempting to run {func}, with dim={dim}, that is not supported"
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
        assert _is_rowwise_scaled(
            input_tensor
        ), "Input tensor must be rowwise block size"
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


to_quantized_float8 = Float8Tensor.from_hp_to_floatx
to_quantized_float8_static = Float8Tensor.from_hp_to_float8_static
