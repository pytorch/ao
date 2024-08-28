import torch
from typing import Dict, Callable, Any, Tuple, Optional
from collections import defaultdict
import functools
import math
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    ZeroPointDomain,
    MappingType,
    int_scaled_matmul,
    quantize_affine_hqq,
    FP8_TYPES,
)
from torchao.quantization.utils import (
    pack_tinygemm_scales_and_zeros,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
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
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from dataclasses import dataclass
from torchao.utils import (
    find_multiple,
    TorchAOBaseTensor,
    TORCH_VERSION_AT_LEAST_2_5,
)

aten = torch.ops.aten

###############################
# Base Layout Tensor Subclass #
###############################
class AQTLayout(TorchAOBaseTensor):
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


##############################
# Tensor Subclass Definition #
##############################


class QuantizedLinearNotImplementedError(NotImplementedError):
    """ Thin wrapper around NotImplementedError to make it easier to catch this error in the dispatch table """
    pass


_QLINEAR_DISPATCH_TABLE = {}
def _register_quantized_linear_dispatch(dispatch_condition, impl):
    _QLINEAR_DISPATCH_TABLE[dispatch_condition] = impl

class AffineQuantizedTensor(TorchAOBaseTensor):
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
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
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

    @staticmethod
    def _quantized_linear_op(input_tensor, weight_tensor, bias):
        for dispatch_condition, impl in _QLINEAR_DISPATCH_TABLE.items():
            if dispatch_condition(input_tensor, weight_tensor, bias):
                return impl(input_tensor, weight_tensor, bias)
        raise QuantizedLinearNotImplementedError("No specialized dispatch found for quantized linear op")

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
    def from_hp_to_intx(
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
        use_hqq: bool = False,
    ):
        original_shape = input_float.shape
        input_float = layout_type.pre_process(input_float)

        if use_hqq:
            assert zero_point_domain == ZeroPointDomain.FLOAT and mapping_type == MappingType.ASYMMETRIC and quant_min==0, "Invalid input parameters for HQQ quantization."
            nbits = int(math.log2(quant_max + 1))
            axis  = 1 if (block_size[0]==1) else 0
            group_size = max(block_size)
            compute_dtype = zero_point_dtype if (zero_point_dtype is not None) else input_float.dtype
            device = input_float.device
            data, scale, zero_point, _ = quantize_affine_hqq(input_float, nbits=nbits, group_size=group_size, axis=axis, compute_dtype=compute_dtype, device=device, verbose=False, raw_output=False)
            data = data.to(target_dtype)
        else:
            scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype, preserve_zero, zero_point_domain)
            data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max, zero_point_domain)
            # Note: output will be uint8 tensor for sub byte tensors for now

        data = layout_type.post_process(data)
        layout_tensor_ctr = get_layout_tensor_constructor(type(layout_type))
        layout_tensor = layout_tensor_ctr(data, scale, zero_point, layout_type)
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
    def from_hp_to_intx_static(
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

    @classmethod
    def from_hp_to_floatx(
        cls,
        input_float: torch.Tensor,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype = torch.float8_e4m3fn,
        layout_type: LayoutType = PlainLayoutType(),
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
                scale_dtype=None,
                zero_point_dtype=None,
                preserve_zero=True,
                zero_point_domain=ZeroPointDomain.INT,
                layout_type=PlainLayoutType(),
                use_hqq=False,
            )
        else:
            raise NotImplementedError(f"Unsupported dtype {target_dtype} for from_float_to_floatx")

    @property
    def layout_type(self) -> LayoutType:
        return self.layout_tensor.layout_type

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
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

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten.t.default:
            tensor = args[0]
            new = tensor.__class__(
                tensor.int_data.view(tensor.shape[::-1]), tensor.scale, tensor.zero_point, tensor.layout_type
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

        if TORCH_VERSION_AT_LEAST_2_5:
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

implements = AffineQuantizedTensor.implements

# following are a list of (dispatch_condition, implementation) functions that takes the following args:
# input_tensor: dimension is (batch_size, in_features)
# weight_tensor: dimension is (out_features, in_features)
# bias: dimension is (out_features,)
# so that these can be shared by F.linear, aten.mm, aten.addmm dispatches

def _linear_int8_act_int8_weight_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor) and
        _aqt_is_int8_reduced_range(input_tensor) and
        isinstance(weight_tensor, AffineQuantizedTensor) and
        weight_tensor.is_cuda and
        input_tensor.dtype == weight_tensor.dtype and
        isinstance(input_tensor.layout_type, PlainLayoutType) and
        isinstance(weight_tensor.layout_type, PlainLayoutType)
    )

def _linear_int8_act_int8_weight_impl(input_tensor, weight_tensor, bias):
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
    w_vals_int8_t = weight_tensor.layout_tensor.int_data.contiguous().t()
    w_scales = weight_tensor.layout_tensor.scale
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


def _linear_int8_act_int8_weight_semi_structured_sparse_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor) and
        _aqt_is_int8_reduced_range(input_tensor) and
        isinstance(weight_tensor, AffineQuantizedTensor) and
        weight_tensor.is_cuda and
        input_tensor.dtype == weight_tensor.dtype and
        isinstance(input_tensor.layout_type, PlainLayoutType) and
        isinstance(weight_tensor.layout_type, SemiSparseLayoutType)
    )

def _linear_int8_act_int8_weight_semi_structured_sparse_impl(input_tensor, weight_tensor, bias):
    x_vals_int8 = input_tensor.layout_tensor.int_data
    x_scales = input_tensor.layout_tensor.scale
    w_vals_int8 = weight_tensor.layout_tensor.int_data
    w_scales = weight_tensor.layout_tensor.scale
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    # we fuse one of the scalar matrix multiplications (w_scales) into the sparse mm
    y_dot_bf16_w_scales_fused = torch._cslt_sparse_mm(
        w_vals_int8, tmp.t(), alpha=w_scales.to(torch.float32), out_dtype=torch.bfloat16
    ).t()
    y = (y_dot_bf16_w_scales_fused * x_scales.reshape(-1, 1)).reshape(
        *x_vals_int8.shape[:-1], y_dot_bf16_w_scales_fused.shape[-1]
    )
    output_dtype = input_tensor.dtype
    # TODO: waiting for jesse's test/fix
    y = y.to(output_dtype).contiguous()
    if bias is not None:
        y += bias
    return y

# this is for the case when linear activation is quantized, but is not caught by the previous
# conditions that expects a quantized activation, we just dequantize the activation so that
# it can continue with the weight only quantization dispatches
# NOTE: this is a fallback path that must be registered after all the implementations that expects
# input tensor to be quantized
def _linear_quantized_act_fallback_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
    )

def _linear_quantized_act_fallback_impl(input_tensor, weight_tensor, bias):
    input_tensor = input_tensor.dequantize()
    # dequantize activation and redispatch to F.linear
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

def _linear_bf16_act_uint4_weight_check(input_tensor, weight_tensor, bias):
    return (
        # input is native bfloat16 tensor
        not is_traceable_wrapper_subclass(input_tensor) and
        input_tensor.dtype == torch.bfloat16 and
        # weight is uint4, group quantized tensor_core_tiled layout affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor) and
        _aqt_is_uint4(weight_tensor) and
        weight_tensor.dtype == torch.bfloat16 and
        len(weight_tensor.shape) == 2 and
        weight_tensor.zero_point_domain == ZeroPointDomain.FLOAT and
        isinstance(weight_tensor.layout_type, TensorCoreTiledLayoutType)
    )


def _linear_bf16_act_uint4_weight_impl(input_tensor, weight_tensor, bias):
    assert weight_tensor.block_size[0] == 1, f"Requires groupwise quantization, got block_size: {block_size}"
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    # TODO: check groupsize quantization
    # avoid circular dep, TODO: move this to a common util.py
    act_mat = input_tensor
    # weight is packed from padded (out_features, in_features) weight tensor
    # (same dimension requirement as F.linear weight)
    packed_weight = weight_tensor.layout_tensor.packed_weight
    scale_and_zero = weight_tensor.layout_tensor.scale_and_zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape and pad activation
    act_mat = act_mat.reshape(-1, act_mat.shape[-1]).to(torch.bfloat16)
    pad_size = find_multiple(act_mat.shape[-1], 1024)
    act_mat = torch.nn.functional.pad(act_mat, (0, pad_size - act_mat.shape[-1]))

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm(act_mat.contiguous(), packed_weight, groupsize, scale_and_zero)

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


def _linear_fp_act_int8_weight_check(input_tensor, weight_tensor, bias):
    return (
        # input is native float tensor
        not is_traceable_wrapper_subclass(input_tensor) and
        input_tensor.is_floating_point() and
        # weight is int8 per channel quantized affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor) and
        _aqt_is_int8(weight_tensor) and
        len(weight_tensor.shape) == 2 and
        len(weight_tensor.block_size) == 2 and
        weight_tensor.block_size[0] == 1 and
        weight_tensor.block_size[1] == weight_tensor.shape[1] and
        weight_tensor.zero_point_domain == ZeroPointDomain.INT and
        isinstance(weight_tensor.layout_type, PlainLayoutType)
    )

def _linear_fp_act_int8_weight_impl(input_tensor, weight_tensor, bias):
    # TODO: enable cpu and mps efficient path
    # is_cpu and is_mps only, some issue with is_contiguous() currently
    # return torch.ops.aten._weight_int8pack_mm(input_tensor.contiguous(), w_vals_int8_t, weight_tensor.layout_tensor.scale)

    # per channel int8 weight only quantizated mm
    w_vals_int8_t = weight_tensor.layout_tensor.int_data.t()
    scale = weight_tensor.layout_tensor.scale
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


def _register_quantized_linear_dispatches():
    for dispatch_condition, impl in [
        (_linear_int8_act_int8_weight_check, _linear_int8_act_int8_weight_impl),
        (_linear_int8_act_int8_weight_semi_structured_sparse_check, _linear_int8_act_int8_weight_semi_structured_sparse_impl),
        (_linear_quantized_act_fallback_check, _linear_quantized_act_fallback_impl),
        (_linear_bf16_act_uint4_weight_check, _linear_bf16_act_uint4_weight_impl),
        (_linear_fp_act_int8_weight_check, _linear_fp_act_int8_weight_impl),
    ]:
        _register_quantized_linear_dispatch(dispatch_condition, impl)

_register_quantized_linear_dispatches()

@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(f"{func} is not implemented for non floating point input")

    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    try:
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)
    except QuantizedLinearNotImplementedError:
        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, AffineQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

@implements(aten.addmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[1],
        args[2],
        args[0],
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(f"{func} is not implemented for non floating point input")

    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    try:
        weight_tensor = weight_tensor.t()
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)
    except QuantizedLinearNotImplementedError:
        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, AffineQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return func(bias, input_tensor, weight_tensor)

@implements(aten.mm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        None
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(f"{func} is not implemented for non floating point input")

    try:
        weight_tensor = weight_tensor.t()
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)
    except QuantizedLinearNotImplementedError:
        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, AffineQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return func(input_tensor, weight_tensor)

@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements(aten.t.default)
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

to_affine_quantized_intx = AffineQuantizedTensor.from_hp_to_intx
to_affine_quantized_intx_static = AffineQuantizedTensor.from_hp_to_intx_static
to_affine_quantized_floatx = AffineQuantizedTensor.from_hp_to_floatx

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with AffineQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([AffineQuantizedTensor])
