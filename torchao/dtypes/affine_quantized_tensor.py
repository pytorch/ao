# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
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
    choose_qparams_affine_dont_preserve_zero,
    choose_qparams_affine_float8,
    choose_qparams_affine_floatx,
    choose_qparams_affine_tinygemm,
    choose_qparams_and_quantize_affine_hqq,
    dequantize_affine,
    dequantize_affine_float8,
    dequantize_affine_floatx,
    quantize_affine,
    quantize_affine_float8,
    quantize_affine_floatx,
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
    "to_affine_quantized_fpx",
]


##############################
# Tensor Subclass Definition #
##############################
class AffineQuantizedTensor(TorchAOBaseTensor):
    """Affine quantized tensor subclass. Affine quantization means we quantize the floating point tensor with an affine transformation:
    quantized_tensor = float_tensor / scale + zero_point

    To see what happens during choose_qparams, quantization and dequantization for affine quantization,
    please checkout https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py
    and check the three quant primitive ops: choose_qparams_affine, quantize_affine qand dequantize_affine

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    fields:
        - tensor_impl (AQTTensorImpl): tensor that serves as a general tensor impl storage for the quantized data,
            e.g. storing plain tensors (int_data, scale, zero_point) or packed formats depending on device and operator/kernel
        - block_size (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
            e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
        - shape (torch.Size): the shape for the original high precision Tensor
        - quant_min (Optional[int]): minimum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
        - quant_max (Optional[int]): maximum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
        - zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
            if zero_point is in integer domain, zero point is added to the quantized integer value during quantization
            if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized) value during quantization
            default is ZeroPointDomain.INT
        - dtype: dtype for original high precision tensor, e.g. torch.float32
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
        if zero_point_domain is None:
            raise ValueError("please use ZeroPointDomain.NONE instead of None")
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

        from torchao.dtypes.floatx import Float8Layout, FloatxTensorCoreLayout

        if isinstance(self._layout, FloatxTensorCoreLayout):
            int_data, scale = self.tensor_impl.get_plain()
            return dequantize_affine_floatx(
                int_data,
                scale,
                self._layout.ebits,
                self._layout.mbits,
                output_dtype=output_dtype,
            )
        elif isinstance(self._layout, Float8Layout):
            data, scale, _ = self.tensor_impl.get_plain()
            return dequantize_affine_float8(data, scale, output_dtype)
        else:
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
        # This is used in rumtime to unwrap AffineQuantizedTensor activations.
        # AffineQuantizedTensor has __torch_function__ override:
        # Each getattr will go through it, which is up to 10x slower than default attribute access.
        with torch._C.DisableTorchFunctionSubclass():
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
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        _layout: Layout = PlainLayout(),
        use_hqq: bool = False,
    ):
        """Convert a high precision tensor to an integer affine quantized tensor."""
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)

        if use_hqq:
            assert (
                zero_point_domain == ZeroPointDomain.FLOAT
                and mapping_type == MappingType.ASYMMETRIC
                and quant_min == 0
            ), "Invalid input parameters for HQQ quantization."
            nbits = int(math.log2(quant_max + 1))
            axis = 1 if (block_size[0] == 1) else 0
            group_size = max(block_size)
            compute_dtype = (
                zero_point_dtype
                if (zero_point_dtype is not None)
                else input_float.dtype
            )
            device = input_float.device
            from torchao.dtypes import Int4CPULayout
            from torchao.dtypes.uintx import TensorCoreTiledLayout

            data, scale, zero_point, _ = choose_qparams_and_quantize_affine_hqq(
                input_float,
                nbits=nbits,
                group_size=group_size,
                axis=axis,
                compute_dtype=compute_dtype,
                device=device,
                verbose=False,
                raw_output=not isinstance(
                    _layout, (TensorCoreTiledLayout, PlainLayout, Int4CPULayout)
                ),
                # raw_output=False is basically the 'convert to TensorCoreTiledLayout zero_point version' option (add scale*midpoint)
                # note in choose_qparams_affine, preserve_zero = False does this same thing while also controlling whether
                # zero is preserved.
                # TODO uncouple preserve_zero and conversion of zero_point to TensorCoreTiledLayout version
                # TODO move the conversion of zero_point out of quant_primitives and into TensorCoreTiledLayout.from_plain
                # TODO change PlainLayout to use raw_output.
            )
            data = data.to(target_dtype)
        else:
            if zero_point_domain == ZeroPointDomain.FLOAT and not preserve_zero:
                scale, zero_point = choose_qparams_affine_tinygemm(
                    input_float,
                    mapping_type,
                    block_size,
                    target_dtype,
                    quant_min,
                    quant_max,
                    eps,
                    scale_dtype,
                    zero_point_dtype,
                )
            elif zero_point_domain == ZeroPointDomain.INT and not preserve_zero:
                scale, zero_point = choose_qparams_affine_dont_preserve_zero(
                    input_float,
                    mapping_type,
                    block_size,
                    target_dtype,
                    quant_min,
                    quant_max,
                    eps,
                    scale_dtype,
                    zero_point_dtype,
                )
            else:  # Default case: zero_point_domain == ZeroPointDomain.INT/NONE and preserve_zero
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
                )
            # choose_qparams_affine is a custom op that does support returning optional Tensors. We thus set the zero_point to None if its domain is None
            if zero_point_domain == ZeroPointDomain.NONE:
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

        data, scale, zero_point = _layout.post_process(
            data, scale, zero_point, block_size
        )
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
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        _layout: Layout = PlainLayout(),
    ):
        """Create an integer AffineQuantizedTensor from a high precision tensor using static parameters."""
        if zero_point_domain is None:
            raise ValueError("please use ZeroPointDomain.NONE instead of None")
        elif zero_point_domain is ZeroPointDomain.NONE and zero_point is not None:
            raise ValueError("zero_point should be None when zero_point_domain is NONE")
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

        int_data, scale, zero_point = _layout.post_process(
            int_data,
            scale,
            zero_point,
            block_size,
        )

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
    ):
        """Convert a high precision tensor to a float8 quantized tensor."""
        if target_dtype in FP8_TYPES:
            original_shape = input_float.shape
            input_float = _layout.pre_process(input_float)

            scale = choose_qparams_affine_float8(input_float, float8_dtype=target_dtype)
            data = quantize_affine_float8(input_float, scale, target_dtype)

            data, scale, zero_point = _layout.post_process(
                data, scale, None, block_size
            )
            tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
            tensor_impl = tensor_impl_ctr(data, scale, zero_point, _layout)
            return cls(
                tensor_impl,
                block_size,
                original_shape,
                dtype=input_float.dtype,
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
        """Create a float8 AffineQuantizedTensor from a high precision tensor using static parameters."""
        if target_dtype in FP8_TYPES:
            original_shape = input_float.shape
            input_float, scale, zero_point = _layout.pre_process_static(
                input_float, scale, ZeroPointDomain.NONE, block_size
            )

            data = quantize_affine_float8(
                input_float,
                scale,
                target_dtype,
            )

            data, scale, zero_point = _layout.post_process(
                data,
                scale,
                zero_point,
                block_size,
            )

            tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
            tensor_impl = tensor_impl_ctr(data, scale, zero_point, _layout)
            return cls(
                tensor_impl,
                block_size,
                original_shape,
                dtype=input_float.dtype,
            )
        else:
            raise NotImplementedError(
                f"Unsupported dtype {target_dtype} for from_hp_to_floatx_static"
            )

    @classmethod
    def from_hp_to_fpx(
        cls,
        input_float: torch.Tensor,
        _layout: Layout,
    ):
        """Create a floatx AffineQuantizedTensor from a high precision tensor. Floatx is represented as ebits and mbits, and supports the representation of float1-float7."""
        from torchao.dtypes.floatx import FloatxTensorCoreLayout

        assert isinstance(_layout, FloatxTensorCoreLayout), (
            f"Only FloatxTensorCoreLayout is supported for floatx, got {_layout}"
        )
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)
        # per axis quantization, where axis = 1
        block_size = list(input_float.shape)
        block_size[1] = 1

        ebits, mbits = _layout.ebits, _layout.mbits
        # Note: these ops are hardcoded to have per axis quantization (axis=1) right now
        scale = choose_qparams_affine_floatx(input_float, ebits, mbits)
        floatx_unpacked = quantize_affine_floatx(input_float, scale, ebits, mbits)
        floatx_packed, scale, _ = _layout.post_process(
            floatx_unpacked, scale, None, block_size
        )

        tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(floatx_packed, scale, None, _layout)
        return cls(tensor_impl, block_size, original_shape, dtype=input_float.dtype)

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
# experimental will be merged in to floatx
to_affine_quantized_fpx = AffineQuantizedTensor.from_hp_to_fpx

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with AffineQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([AffineQuantizedTensor])
