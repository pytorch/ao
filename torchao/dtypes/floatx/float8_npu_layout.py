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
from torchao.dtypes.utils import AQTTensorImpl, Layout, is_device
from torchao.float8.inference import Float8MMConfig
from torchao.utils import _is_float8_type, fill_defaults

aten = torch.ops.aten


def _linear_fp8_act_fp8_weight_npu_check(
    input_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    weight_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    bias: Optional[torch.Tensor],
) -> bool:
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _is_float8_type(input_tensor.dtype)
        and is_device(input_tensor.device.type, "npu")
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _is_float8_type(weight_tensor.dtype)
        and isinstance(weight_tensor._layout, Float8NPULayout)
    )


def _linear_fp8_act_fp8_weight_npu_impl(
    input_tensor: "AffineQuantizedTensor",
    weight_tensor: "AffineQuantizedTensor",
    bias: Optional[torch.Tensor],
):
    input_data = input_tensor.tensor_impl.float8_data
    weight_data = weight_tensor.tensor_impl.float8_data
    x_scale = input_tensor.tensor_impl.scale
    w_scale = weight_tensor.tensor_impl.scale

    try:
        import torch_npu

        # NOTE: torch_npu op name and signature need to be verified against
        # the actual torch_npu documentation / release version being used.
        y = torch_npu.npu_quant_matmul(
            input_data,
            weight_data,
            x_scale,
            w_scale,
            bias=bias,
            output_dtype=torch.bfloat16,
        )
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "torch_npu is required for NPU Float8 quantized linear. "
            f"Original error: {e}"
        )

    return y


def _linear_fp_act_fp8_weight_npu_check(
    input_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    weight_tensor: Union[torch.Tensor, "AffineQuantizedTensor"],
    bias: Optional[torch.Tensor],
) -> bool:
    return (
        # input is a native floating point tensor (not quantized)
        not is_traceable_wrapper_subclass(input_tensor)
        and input_tensor.is_floating_point()
        and is_device(input_tensor.device.type, "npu")
        and
        # weight is float8 quantized affine quantized tensor with NPU layout
        isinstance(weight_tensor, AffineQuantizedTensor)
        and isinstance(weight_tensor._layout, Float8NPULayout)
        and _is_float8_type(weight_tensor.tensor_impl.dtype)
    )


def _linear_fp_act_fp8_weight_npu_impl(
    input_tensor: torch.Tensor,
    weight_tensor: "AffineQuantizedTensor",
    bias: Optional[torch.Tensor],
):
    # Dequantize weight to bfloat16, then perform standard matmul
    return torch.nn.functional.linear(input_tensor, weight_tensor.dequantize(), bias)


@dataclass(frozen=True)
class Float8NPULayout(Layout):
    """Float8 layout for Huawei Ascend NPU via torch_npu."""

    mm_config: Optional[Float8MMConfig] = None


@register_layout(Float8NPULayout)
class Float8NPUAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for float8 NPU layout for affine quantized tensor, targeting
    Huawei Ascend NPU via torch_npu.

    fields:
      float8_data (torch.Tensor): the quantized data in float8 dtype
      scale (torch.Tensor): the scale tensor (row-wise or per-tensor)
      transposed (bool): whether the weight tensor is transposed
      _layout (Layout): the Float8NPULayout instance
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
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else float8_data.layout
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
        return self.__class__(
            fn(self.float8_data),
            fn(self.scale),
            self.transposed,
            self._layout,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device(torch.device(self.device).type, device):
            raise ValueError(
                f"Float8NPUAQTTensorImpl does not support conversion from {self.device} to {device}"
            )
        return self.__class__(
            self.float8_data.to(device),
            self.scale.to(device),
            self.transposed,
            self._layout,
        )

    def __tensor_flatten__(self):
        return ["float8_data", "scale"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        float8_data = tensor_data_dict["float8_data"]
        scale = tensor_data_dict["scale"]
        (
            transposed,
            _layout,
        ) = tensor_attributes
        return cls(float8_data, scale, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert _is_float8_type(data.dtype), (
            f"Float8NPUAQTTensorImpl must be constructed from float8 dtype but got {data.dtype}"
        )
        assert isinstance(_layout, Float8NPULayout), (
            f"Float8NPUAQTTensorImpl must be constructed from Float8NPULayout but got {_layout}"
        )
        return cls(data, scale, False, _layout)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            args[0].transposed = not args[0].transposed
            return return_and_correct_aliasing(func, args, kwargs, args[0])

        if func is aten.copy_.default:
            self = args[0]
            src = args[1]
            if (
                isinstance(src, Float8NPUAQTTensorImpl)
                and self.shape == src.shape
                and self.float8_data.shape == src.float8_data.shape
                and self.scale.shape == src.scale.shape
            ):
                self_tensors = self.__tensor_flatten__()[0]
                for tensor_name in self_tensors:
                    getattr(self, tensor_name).copy_(getattr(src, tensor_name))
                return
            raise ValueError(
                f"Not supported args for copy_ due to metadata mismatch: {args[0], args[1]}"
            )

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            sliced_data = aten.slice.Tensor(self.float8_data, dim, start, end, step)
            if self.scale.numel() == 1:
                sliced_scale = self.scale
            else:
                sliced_scale = aten.slice.Tensor(self.scale, dim, start, end, step)
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                Float8NPUAQTTensorImpl(
                    sliced_data,
                    sliced_scale,
                    self.transposed,
                    self._layout,
                ),
            )

        raise NotImplementedError(
            f"Float8NPUAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.float8_data, self.scale, None

    def get_layout(self) -> Layout:
        return self._layout

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
