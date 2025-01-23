import logging
import math
from typing import Optional, Tuple, Union

import torch

from torchao.dtypes.utils import (
    AQTTensorImpl,
    Layout,
    PlainLayout,
)
from torchao.quantization.quant_primitives import (
    FP8_TYPES,
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    choose_qparams_and_quantize_affine_hqq,
    dequantize_affine,
    quantize_affine,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

logger = logging.getLogger(__name__)
aten = torch.ops.aten

__all__ = [
    "AffineQuantizedTensor",
    "register_layout",
    "to_affine_quantized_intx",
    "to_affine_quantized_floatx",
    "to_affine_quantized_intx_static",
    "to_affine_quantized_floatx_static",
]


##############################
# Tensor Subclass Definition #
##############################
class AffineQuantizedTensor(TorchAOBaseTensor):
    """
    Affine quantized tensor subclass. Affine quantization means we quantize the floating point tensor with an affine transformation:
       quantized_tensor = float_tensor / scale + zero_point

    To see what happens during choose_qparams, quantization and dequantization for affine quantization,
    please checkout https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py
    and check the three quant primitive ops: choose_qparams_affine, quantize_affine qand dequantize_affine

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    fields:
      tensor_impl (AQTTensorImpl): tensor that serves as a general tensor impl storage for the quantized data,
         e.g. storing plain tensors (int_data, scale, zero_point) or packed formats depending on device
         and operator/kernel
      block_size (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
         e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      shape (torch.Size): the shape for the original high precision Tensor
      quant_min (Optional[int]): minimum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      quant_max (Optional[int]): maximum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT
      dtype: dtype for original high precision tensor, e.g. torch.float32
    """

    @staticmethod
    def __new__(
        cls,
        tensor_impl: AQTTensorImpl,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[Union[int, float]] = None,
        quant_max: Optional[Union[int, float]] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        kwargs = {}
        kwargs["device"] = tensor_impl.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else tensor_impl.layout
        )
        kwargs["dtype"] = dtype
        if strides is not None:
            kwargs["strides"] = strides
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        tensor_impl: AQTTensorImpl,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[Union[int, float]] = None,
        quant_max: Optional[Union[int, float]] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        self.tensor_impl = tensor_impl
        self.block_size = block_size
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.zero_point_domain = zero_point_domain

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(tensor_impl={self.tensor_impl}, block_size={self.block_size}, "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}, _layout={self._layout}, tensor_impl_dtype={self.tensor_impl.dtype}, quant_min={self.quant_min}, quant_max={self.quant_max}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        data, scale, zero_point = self.tensor_impl.get_plain()
        dq = dequantize_affine(
            data,
            self.block_size,
            scale,
            zero_point,
            data.dtype,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            output_dtype=output_dtype,
        )
        from torchao.dtypes.uintx import TensorCoreTiledLayout

        if isinstance(self._layout, TensorCoreTiledLayout):
            # need to return to original shape if tensor was padded
            # in preprocessing
            # TODO: we could add an API for this if there are more use cases
            # (e.g. dequant_post_process) in TensorImpl or Layout
            for dim, dim_size in enumerate(self.shape):
                dq = dq.narrow(dim, 0, dim_size)
        return dq

    def __tensor_flatten__(self):
        return ["tensor_impl"], [
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            self.dtype,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        tensor_impl = tensor_data_dict["tensor_impl"]
        block_size, shape, quant_min, quant_max, zero_point_domain, dtype = (
            tensor_attributes
        )
        return cls(
            tensor_impl,
            block_size,
            shape if outer_size is None else outer_size,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_hp_to_intx(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
        _layout: Layout = PlainLayout(),
        use_hqq: bool = False,
    ):
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)

        scale, zero_point = choose_qparams_affine(
            input_float,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
            preserve_zero,
            zero_point_domain,
        )
        # choose_qparams_affine is a custom op that does support returning optional Tensors. We thus set the zero_point to None if its domain is None
        # TODO should probably consolidate ZeroPointDomain.NONE and None
        if zero_point_domain is None or zero_point_domain == ZeroPointDomain.NONE:
            zero_point = None
        data = quantize_affine(
            input_float,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min,
            quant_max,
            zero_point_domain,
        )
        # Note: output will be uint8 tensor for sub byte tensors for now

        data = _layout.post_process(data)
        tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(data, scale, zero_point, _layout)
        return cls(
            tensor_impl,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype,
        )

    @classmethod
    def from_hp_to_intx_static(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
        _layout: Layout = PlainLayout(),
    ):
        if target_dtype not in FP8_TYPES:
            assert (
                zero_point_domain is not None
            ), "zero_point_domain must be specified for non-fp8 types"
            assert (
                zero_point is not None
            ), "zero_point must be specified for non-fp8 types"
        original_shape = input_float.shape
        input_float, scale, zero_point = _layout.pre_process_static(
            input_float, scale, zero_point, block_size
        )

        int_data = quantize_affine(
            input_float,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min,
            quant_max,
            zero_point_domain,
        )

        int_data = _layout.post_process(int_data)

        tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(int_data, scale, zero_point, _layout)
        return cls(
            tensor_impl,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype,
        )

    @classmethod
    def from_hp_to_floatx(
        cls,
        input_float: torch.Tensor,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        _layout: Layout,
        scale_dtype: Optional[torch.dtype] = None,
    ):
        if target_dtype in FP8_TYPES:
            return cls.from_hp_to_intx(
                input_float=input_float,
                mapping_type=MappingType.SYMMETRIC,
                block_size=block_size,
                target_dtype=target_dtype,
                quant_min=math.ceil(torch.finfo(target_dtype).min),
                quant_max=math.ceil(torch.finfo(target_dtype).max),
                eps=torch.finfo(torch.float32).eps,
                scale_dtype=scale_dtype,
                zero_point_dtype=None,
                preserve_zero=True,
                zero_point_domain=None,
                _layout=_layout,
                use_hqq=False,
            )
        else:
            raise NotImplementedError(
                f"Unsupported dtype {target_dtype} for from_hp_to_floatx"
            )

    @classmethod
    def from_hp_to_floatx_static(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        _layout: Layout,
    ):
        if target_dtype in FP8_TYPES:
            return cls.from_hp_to_intx_static(
                input_float=input_float,
                scale=scale,
                zero_point=None,
                block_size=block_size,
                target_dtype=target_dtype,
                quant_min=math.ceil(torch.finfo(target_dtype).min),
                quant_max=math.ceil(torch.finfo(target_dtype).max),
                zero_point_domain=None,
                _layout=_layout,
            )
        else:
            raise NotImplementedError(
                f"Unsupported dtype {target_dtype} for from_hp_to_floatx_static"
            )

    @property
    def _layout(self) -> Layout:
        return self.tensor_impl._layout

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.tensor_impl.to(device),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.tensor_impl),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            dtype=self.dtype,
            strides=self.stride(),
        )

    # following are the comments for __torch_function__/__torch_dispatch__, -> this is defined in affine_quantized_tensor_ops.py
    # a bit later
    # Note: we only added cpu path here for 8da4w, this is for executorch, in the future
    # 1. we'll add cpu/cuda version (int4mm etc.)
    # 2. we'll need to hide the 8da4w executorch version under things like layouts (we also have multiple impl for cpu kernel as Michael mentioned), so it will be something like
    #   cpu device + et laytout --> gives current 8da4w executorch representation
    #   cpu device + avx layout --> gives optimized kernel for 8da4w in avx cpu etc.
    #   cuda device + some layout --> gives cuda kernel

    # two scenarios where we currently fall back to vanilla mm:
    # 1 - when tensor is on CUDA: we'll add this later, we'll also enable dispatching to optimized
    #     kernels in CPU as well, see the note above
    # 2 - we're given non-floats - quantizing long to int8 is crazy


######################################################
# Layout and TensorImpl Subclass Registration #
######################################################
register_layout = AffineQuantizedTensor.register_layout
get_tensor_impl_constructor = AffineQuantizedTensor.get_tensor_impl_constructor


to_affine_quantized_intx = AffineQuantizedTensor.from_hp_to_intx
to_affine_quantized_intx_static = AffineQuantizedTensor.from_hp_to_intx_static
to_affine_quantized_floatx = AffineQuantizedTensor.from_hp_to_floatx
to_affine_quantized_floatx_static = AffineQuantizedTensor.from_hp_to_floatx_static

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with AffineQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([AffineQuantizedTensor])
