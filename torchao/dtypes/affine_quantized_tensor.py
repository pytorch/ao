import torch
from typing import Dict, Callable, Any, Tuple, Optional
from collections import defaultdict
import functools
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    ZeroPointDomain,
    MappingType,
    int_scaled_matmul,
)
from torchao.quantization.utils import (
    pack_tinygemm_scales_and_zeros,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import find_multiple
from torchao.dtypes.utils import (
    _implements,
    _dispatch__torch_function__,
    _dispatch__torch_dispatch__,
    _register_layout_cls,
    _get_layout_tensor_constructor,
    LayoutType,
    PlainLayoutType,
    is_device,
)
from dataclasses import dataclass
from torchao.utils import TORCH_VERSION_AFTER_2_5

aten = torch.ops.aten

###############################
# Base Layout Tensor Subclass #
###############################
class AQTLayout(torch.Tensor):
    """
    Base class for the layout tensor for `AffineQuantizedTensor`
    """
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def get_layout_type(self) -> LayoutType:
        pass

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        pass

    def __repr__(self):
        int_data, scale, zero_point = self.get_plain()
        layout_type = self.get_layout_type()
        return f"{self.__class__.__name__}(int_data={int_data}, scale={scale}, zero_point={zero_point}, layout_type={layout_type})"

    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = (
            memory_format if memory_format is not None else torch.preserve_format
        )
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        return kwargs

##############################
# Tensor Subclass Definition #
##############################

class AffineQuantizedTensor(torch.Tensor):
    """
    Affine quantized tensor subclass. Affine quantization means we quantize the floating point tensor with an affine transformation:
       quantized_tensor = float_tensor / scale + zero_point

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    fields:
      layout_tensor (AQTLayout): tensor that serves as a general layout storage for the quantized data,
         e.g. storing plain tensors (int_data, scale, zero_point) or packed formats depending on device
         and operator/kernel
      block_size (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
         e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      shape (torch.Size): the shape for the Tensor
      quant_min (Optional[int]): minimum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      quant_max (Optional[int]): maximum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT
      input_quant_func (Optional[Callable]): function for quantizing the input float Tensor to a quantized tensor subclass object, that takes float Tensor as input and outputs an AffineQuantizedTensor object
      dtype: dtype for external representation of the tensor, e.g. torch.float32
    """

    @staticmethod
    def __new__(
        cls,
        layout_tensor: AQTLayout,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        kwargs = {}
        kwargs["device"] = layout_tensor.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else layout_tensor.layout
        )
        kwargs["dtype"] = dtype
        if strides is not None:
            kwargs["strides"] = strides
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        layout_tensor: AQTLayout,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        self.layout_tensor = layout_tensor
        self.block_size = block_size
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.zero_point_domain = zero_point_domain

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def dequantize(self, output_dtype=None):
        if output_dtype is None:
            output_dtype = self.dtype
        int_data, scale, zero_point = self.layout_tensor.get_plain()
        return dequantize_affine(int_data, self.block_size, scale, zero_point, int_data.dtype, self.quant_min, self.quant_max, self.zero_point_domain, output_dtype=output_dtype)

    def __tensor_flatten__(self):
        return ["layout_tensor"], [self.block_size, self.shape, self.quant_min, self.quant_max, self.zero_point_domain, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        layout_tensor = tensor_data_dict["layout_tensor"]
        block_size, shape, quant_min, quant_max, zero_point_domain, dtype = tensor_attributes
        return cls(
            layout_tensor,
            block_size,
            shape if outer_size is None else outer_size,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int]  = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        layout_type: LayoutType = PlainLayoutType(),
    ):
        original_shape = input_float.shape
        input_float = layout_type.pre_process(input_float)

        scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype, preserve_zero, zero_point_domain)
        int_data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max, zero_point_domain)
        int_data = layout_type.post_process(int_data)

        layout_tensor_ctr = get_layout_tensor_constructor(type(layout_type))
        layout_tensor = layout_tensor_ctr(int_data, scale, zero_point, layout_type)
        return cls(
            layout_tensor,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype
        )

    @classmethod
    def from_float_static(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int]  = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        layout_type: LayoutType = PlainLayoutType(),
    ):
        original_shape = input_float.shape
        input_float = layout_type.pre_process(input_float)

        int_data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max, zero_point_domain)

        int_data = layout_type.post_process(int_data)

        layout_tensor_ctr = get_layout_tensor_constructor(type(layout_type))
        layout_tensor = layout_tensor_ctr(int_data, scale, zero_point, layout_type)
        return cls(
            layout_tensor,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype,
        )

    @property
    def layout_type(self) -> LayoutType:
        return self.layout_tensor.layout_type

    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = (
            memory_format if memory_format is not None else torch.preserve_format
        )
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        return kwargs

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        # not supported yet
        kwargs.pop("memory_format")
        return self.__class__(
            self.layout_tensor.to(device),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.layout_tensor),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            dtype=self.dtype,
            strides=self.stride(),
        )

    implements = classmethod(_implements)
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
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    __torch_function__ = classmethod(_dispatch__torch_function__)


######################################################
# LayoutType and Layout Tensor Subclass Registration #
######################################################

def register_layout_cls(layout_type_class: type(LayoutType)):
    return _register_layout_cls(AffineQuantizedTensor, layout_type_class)

def get_layout_tensor_constructor(layout_type_class: type(LayoutType)):
    return _get_layout_tensor_constructor(AffineQuantizedTensor, layout_type_class)

@dataclass(frozen=True)
class SemiSparseLayoutType(LayoutType):

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        # prune to 2:4 if not already
        temp = input.detach()
        pruning_inds = temp.abs().view(-1, 4).argsort(dim=1)[:, :2]
        temp.view(-1, 4).scatter_(1, pruning_inds, value=0)
        return temp


@dataclass(frozen=True)
class TensorCoreTiledLayoutType(LayoutType):
    inner_k_tiles: int = 8

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        orig_out_features, orig_in_features = input.shape
        in_features = find_multiple(orig_in_features, 1024)
        out_features = find_multiple(orig_out_features, 8)
        input = torch.nn.functional.pad(
            input,
            (0, in_features - orig_in_features, 0, out_features - orig_out_features),
        )
        return input

    def extra_repr(self):
        return f"inner_k_tiles={self.inner_k_tiles}"


@register_layout_cls(PlainLayoutType)
class PlainAQTLayout(AQTLayout):
    """
    Layout storage class for plain layout for affine quantized tensor, it stores int_data, scale, zero_point
    tensors directly as plain tensors.

    fields:
      int_data (torch.Tensor): the quantized integer data Tensor
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
    """
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.layout_type = layout_type

    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero_point"], [self.layout_type]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = tensor_data_dict["int_data"], tensor_data_dict["scale"], tensor_data_dict["zero_point"]
        layout_type, = tensor_attributes
        return cls(int_data, scale, zero_point, layout_type)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"]),
            self.layout_type,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point),
            self.layout_type,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.t.default:
            tensor = args[0]
            new = tensor.__class__(
                tensor.int_data.view(tensor.shape[::-1]), tensor.scale, tensor.zero_point
            )
            return return_and_correct_aliasing(func, args, kwargs, new)

        raise NotImplementedError(
            f"PlainAQTLayout dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.int_data, self.scale, self.zero_point

    def get_layout_type(self) -> LayoutType:
        return self.layout_type

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, PlainLayoutType)
        return cls(int_data, scale, zero_point, layout_type)

@register_layout_cls(SemiSparseLayoutType)
class SemiSparseAQTLayout(PlainAQTLayout):
    """
    Layout storage class for semi_sparse_cusparselt layout for affine quantized tensor
    """
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"SparseAQTLayout dispatch: attempting to run {func}, this is not supported"
        )

    def get_plain(self):
        # Currently we don't have cuSPARSELt expansion routines, so we matmul by
        # the identity matrix to get the original dense matrix. This is slow though.
        cols = self.int_data.numel() * 16 // (10 * self.scale.shape[0])
        int_data_expanded = torch._cslt_sparse_mm(self.int_data,
                                                  torch.eye(cols,
                                                            dtype=self.int_data.dtype,
                                                            device=self.int_data.device).t())
        return int_data_expanded, self.scale, self.zero_point

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, SemiSparseLayoutType)
        int_data_compressed = torch._cslt_compress(int_data)
        return cls(int_data_compressed, scale, zero_point, layout_type)


@register_layout_cls(TensorCoreTiledLayoutType)
class TensorCoreTiledAQTLayout(AQTLayout):
    """
    Layout storage class for tensor_core_tiled layout for affine quantized tensor, this is for int4 only,
    it stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 4-d tensor of
    dimension: [n / 8][k / (inner_k_tiles * 16)][32][inner_k_tiles / 2]

    fields:
      packed_weight (torch.Tensor): the 4-d packed tensor in a tensor_core_tiled layout
      scale_and_zero (torch.Tensor): the combined scale Tensor used to map between floating point tensor to quantized tensor and zero_point Tensor
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
        layout_type: LayoutType,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else packed_weight.layout
        )
        kwargs["dtype"] = packed_weight.dtype
        kwargs["requires_grad"] = False
        shape = packed_weight.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
        layout_type: LayoutType,
    ):
        self.packed_weight = packed_weight
        self.scale_and_zero = scale_and_zero
        self.transposed = False
        self.layout_type = layout_type

    def __tensor_flatten__(self):
        return ["packed_weight", "scale_and_zero"], [self.transposed, self.layout_type]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale_and_zero = tensor_data_dict["packed_weight"], tensor_data_dict["scale_and_zero"]
        transposed, layout_type, = tensor_attributes
        return cls(packed_weight, scale_and_zero, transposed, layout_type)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType
    ):
        assert isinstance(layout_type, TensorCoreTiledLayoutType)
        if TORCH_VERSION_AFTER_2_5:
            int_data = (int_data[::, ::2] << 4 | int_data[::, 1::2]).to(torch.uint8)
            assert int_data.dtype == torch.uint8, "torch.ops.aten._convert_weight_to_int4pack in torch 2.5 expects `uint8` dtype"
        else:
            assert int_data.dtype == torch.int32, "torch.ops.aten._convert_weight_to_int4pack in torch 2.4 expects `int32` dtype"
        packed_weight = torch.ops.aten._convert_weight_to_int4pack(int_data, layout_type.inner_k_tiles)
        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point)
        return cls(packed_weight, scale_and_zero, False, layout_type)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device("cuda", device):
            raise ValueError(f"TensorCoreTiledAQTLayout is only available for cuda device, can't convert to {device}")
        return self.__class__(
            self.packed_weight.to(device),
            self.scale_and_zero.to(device),
            self.transposed,
            self.layout_type,
        )

    def _apply_fn_to_data(self, fn):
        self.packed_weight = fn(self.packed_weight)
        self.scale_and_zero = fn(self.scale_and_zero)
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            args[0].transposed = not args[0].transposed
            return return_and_correct_aliasing(func, args, kwargs, args[0])

        raise NotImplementedError(
            f"TensorCoreTiledAQTLayout dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchao.quantization.quant_primitives import (
            ZeroPointDomain,
            quantize_affine,
        )
        from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros
        scale, zero = unpack_tinygemm_scales_and_zeros(self.scale_and_zero)

        cur_shape = self.shape
        assert len(cur_shape) == 4
        inner_k_tiles = cur_shape[-1] * 2
        original_shape = (cur_shape[0] * 8, cur_shape[1] * (inner_k_tiles * 16))
        eye_shape = original_shape[1]
        groupsize = int(original_shape[1] / scale.shape[-2])
        block_size = (1, groupsize)
        device = self.device
        original_dtype = torch.bfloat16
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        zero_point_domain = ZeroPointDomain.FLOAT
        assert len(block_size) == 2 and block_size[0] == 1
        dequantized = torch.ops.aten._weight_int4pack_mm(torch.eye(eye_shape, device=device, dtype=original_dtype), self.packed_weight, groupsize, self.scale_and_zero)
        dequantized = dequantized.t().contiguous()
        # TODO: move this to `unpack_tinygemm_scales_and_zeros`?
        scale = scale.reshape(scale.shape[:-1]).contiguous()
        zero = zero.reshape(zero.shape[:-1]).contiguous()
        int_data = quantize_affine(dequantized, block_size, scale, zero, target_dtype, quant_min, quant_max, zero_point_domain)
        return int_data, scale, zero

    def get_layout_type(self) -> LayoutType:
        return self.layout_type

#####################################################
# torch functional and aten operator implementation #
#####################################################

def _aqt_is_int8(aqt):
    """Check if an AffineQuantizedTensor is int8 quantized Tensor"""
    return (
        aqt.layout_tensor.dtype == torch.int8 and
        aqt.quant_min is None or aqt.quant_min == -128 and
        aqt.quant_max is None or aqt.quant_max == 127
    )

def _aqt_is_int8_reduced_range(aqt):
    return (
        aqt.layout_tensor.dtype == torch.int8 and
        aqt.quant_min == -127 and
        aqt.quant_max is None or aqt.quant_max == 127
    )

def _aqt_is_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    # TODO: use torch.uint4
    return (
        aqt.layout_tensor.dtype == torch.int32 and
        aqt.quant_min is None or aqt.quant_min == 0 and
        aqt.quant_max is None or aqt.quant_max == 15
    )

def _quantized_linear_op(input_tensor, weight_qtensor, bias):
    """
    Quantized version of F.linear operator

    Args:
        input_tensor: dimension is (batch_size, in_features)
        weight_tensor: dimension is (out_features, in_features)
        bias: dimension is (out_features,)
    """
    is_cuda = weight_qtensor.is_cuda
    is_cpu = weight_qtensor.device == torch.device("cpu")
    if isinstance(weight_qtensor, AffineQuantizedTensor):
        weight_is_int8 = _aqt_is_int8(weight_qtensor)
        weight_is_uint4 = _aqt_is_uint4(weight_qtensor)

        if isinstance(input_tensor, AffineQuantizedTensor):
            # if input tensor is quantized, either dispatch to the int8 mm kernel
            # or just dequantize the input tensor
            input_is_int8 = _aqt_is_int8_reduced_range(input_tensor)
            if (
                is_cuda and
                input_is_int8 and
                input_tensor.dtype == weight_qtensor.dtype and
                isinstance(input_tensor.layout_type, PlainLayoutType) and
                isinstance(weight_qtensor.layout_type, PlainLayoutType)
            ):
                #
                # 1. do the matrix form of dot(X_i, W_j)
                #
                #
                # 2. rescale the output
                #
                # in cases with large matrices, y_dot_int32 can grow sufficiently
                # large that y_dot_int32 * a float16 scale is greater than the maximum
                # value of a float 16, (which results in a value of inf even if multiplying
                # by the other scale would bring it within the expected range)

                x_vals_int8 = input_tensor.layout_tensor.int_data
                x_scales = input_tensor.layout_tensor.scale
                w_vals_int8_t = weight_qtensor.layout_tensor.int_data.contiguous().t()
                w_scales = weight_qtensor.layout_tensor.scale
                tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
                y_dot_scaled = int_scaled_matmul(tmp, w_vals_int8_t, x_scales.reshape(-1, 1))

                y = (y_dot_scaled * w_scales).reshape(
                    *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
                )

                # can downcast only at the very end
                output_dtype = input_tensor.dtype
                y = y.to(output_dtype)
                if bias is not None:
                    y += bias
                return y
            # handle int8 dynamic_quant + semi_structured_sparse
            elif(
                is_cuda and
                input_is_int8 and
                input_tensor.dtype == weight_qtensor.dtype and
                isinstance(input_tensor.layout_type, PlainLayoutType) and
                isinstance(weight_qtensor.layout_type, SemiSparseLayoutType)
            ):
                x_vals_int8 = input_tensor.layout_tensor.int_data
                x_scales = input_tensor.layout_tensor.scale
                w_vals_int8 = weight_qtensor.layout_tensor.int_data
                w_scales = weight_qtensor.layout_tensor.scale
                tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
                # we fuse one of the scalar matrix multiplications (w_scales) into the sparse mm
                y_dot_bf16_w_scales_fused = torch._cslt_sparse_mm(
                    w_vals_int8, tmp.t(), alpha=w_scales.to(torch.float32), out_dtype=torch.bfloat16
                ).t()
                y = (y_dot_bf16_w_scales_fused * x_scales.reshape(-1, 1)).reshape(
                    *x_vals_int8.shape[:-1], y_dot_bf16_w_scales_fused.shape[-1]
                )
                output_dtype = input_tensor.dtype
                y = y.to(output_dtype)
                if bias is not None:
                    y += bias
                return y
            else:
                input_tensor = input_tensor.dequantize()

        # weight only quantization
        # TODO: enable cpu and mps path as well
        # TODO: make sure weight dimension matches the expectation of the int4mm kernel
        # TODO: cpu/cuda are sharing the same code now, may need some special handling for cpu
        if (
            weight_is_uint4 and
            weight_qtensor.dtype == torch.bfloat16 and
            len(weight_qtensor.shape) == 2 and
            weight_qtensor.zero_point_domain == ZeroPointDomain.FLOAT and
            isinstance(weight_qtensor.layout_type, TensorCoreTiledLayoutType)
        ):
            assert weight_qtensor.block_size[0] == 1, f"Requires groupwise quantization, got block_size: {block_size}"
            assert input_tensor.shape[-1] == weight_qtensor.shape[1], (
                f"need input_tensor shape: {input_tensor.shape} final"
                f"dim to match weight_tensor shape: {weight_qtensor.shape} second dim "
            )

            # TODO: check groupsize quantization
            # avoid circular dep, TODO: move this to a common util.py
            act_mat = input_tensor
            # weight is packed from padded (out_features, in_features) weight tensor
            # (same dimension requirement as F.linear weight)
            packed_weight = weight_qtensor.layout_tensor.packed_weight
            scale_and_zero = weight_qtensor.layout_tensor.scale_and_zero

            orig_act_size = act_mat.size()
            orig_dtype = act_mat.dtype

            # reshape and pad activation
            act_mat = act_mat.reshape(-1, act_mat.shape[-1]).to(torch.bfloat16)
            pad_size = find_multiple(act_mat.shape[-1], 1024)
            act_mat = torch.nn.functional.pad(act_mat, (0, pad_size - act_mat.shape[-1]))

            # groupwise int4 quantization
            groupsize = weight_qtensor.block_size[1]
            y = torch.ops.aten._weight_int4pack_mm(act_mat.contiguous(), packed_weight, groupsize, scale_and_zero)

            # remove out_feature padding
            orig_out_features = weight_qtensor.shape[-2]
            y = y[:, :orig_out_features]
            y = y.reshape(*orig_act_size[:-1], orig_out_features)

            if bias is not None:
                y += bias
            return y.to(orig_dtype)
        elif (
            weight_is_int8 and
            len(weight_qtensor.shape) == 2 and
            len(weight_qtensor.block_size) == 2 and
            weight_qtensor.block_size[0] == 1 and
            weight_qtensor.block_size[1] == weight_qtensor.shape[1] and
            weight_qtensor.zero_point_domain == ZeroPointDomain.INT and
            isinstance(weight_qtensor.layout_type, PlainLayoutType)
        ):
            # TODO: enable cpu and mps efficient path
            # per channel int8 weight only quantizated mm
            w_vals_int8_t = weight_qtensor.layout_tensor.int_data.t()
            scale = weight_qtensor.layout_tensor.scale
            orig_dtype = input_tensor.dtype
            m = torch.mm(
                    input_tensor.reshape(-1, input_tensor.shape[-1]),
                    w_vals_int8_t.to(input_tensor.dtype),
                )
            y = m * scale.to(m.dtype)
            y = y.reshape(*input_tensor.shape[:-1], y.shape[-1])
            if bias is not None:
                y += bias.to(m.dtype)
            return y

            # is_cpu and is_mps only, some issue with is_contiguous() currently
            # return torch.ops.aten._weight_int8pack_mm(input_tensor.contiguous(), w_vals_int8_t, weight_qtensor.layout_tensor.scale)

    raise NotImplementedError("No specialized dispatch found for quantized linear op")


implements = AffineQuantizedTensor.implements

@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    try:
        return _quantized_linear_op(input_tensor, weight_tensor, bias)
    except:
        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, AffineQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

@implements([aten.mm.default, aten.addmm.default])
def _(func, types, args, kwargs):
    if not args[0].is_floating_point():
        raise NotImplementedError(f"{func} is not implemented for non floating point input")

    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    if func == aten.addmm.default:
        input_tensor, weight_tensor, bias = (
            args[1],
            args[2],
            args[0],
        )
        try:
            weight_tensor = weight_tensor.t()
            return _quantized_linear_op(input_tensor, weight_tensor, bias)
        except:
            if isinstance(input_tensor, AffineQuantizedTensor):
                input_tensor = input_tensor.dequantize()
            if isinstance(weight_tensor, AffineQuantizedTensor):
                weight_tensor = weight_tensor.dequantize()
            return func(bias, input_tensor, weight_tensor)
    else:
        input_tensor, weight_tensor, bias = (
            args[0],
            args[1],
            None
        )
        try:
            weight_tensor = weight_tensor.t()
            return _quantized_linear_op(input_tensor, weight_tensor, bias)
        except:
            if isinstance(input_tensor, AffineQuantizedTensor):
                input_tensor = input_tensor.dequantize()
            if isinstance(weight_tensor, AffineQuantizedTensor):
                weight_tensor = weight_tensor.dequantize()
            return func(input_tensor, weight_tensor)

@implements([aten.detach.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements([aten.clone.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements([aten._to_copy.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements([aten.t.default])
def _(func, types, args, kwargs):
    block_size = args[0].block_size
    assert len(block_size) == 2
    transposed_block_size = (block_size[1], block_size[0])
    tensor = args[0]
    shape = tensor.shape[::-1]
    new = tensor.__class__(
        tensor.layout_tensor.t(), transposed_block_size, shape, tensor.quant_min, tensor.quant_max, tensor.zero_point_domain, dtype=tensor.dtype, strides=tensor.stride()
    )
    return return_and_correct_aliasing(func, args, kwargs, new)

to_affine_quantized = AffineQuantizedTensor.from_float
to_affine_quantized_static = AffineQuantizedTensor.from_float_static
