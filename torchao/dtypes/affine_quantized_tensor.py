import torch
from typing import Tuple, Optional, Union
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
    choose_qparams_and_quantize_affine_hqq,
    FP8_TYPES,
    choose_qparams_affine_fpx,
    quantize_affine_fpx,
    dequantize_affine_fpx,
)
from torchao.quantization.utils import (
    pack_tinygemm_scales_and_zeros,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.utils import (
    LayoutType,
    PlainLayoutType,
    is_device,
    get_out_shape,
)
from torchao.float8.inference import (
    preprocess_data,
    Float8MMConfig,
    addmm_float8_unwrapped_inference,
    _is_rowwise_scaled
)
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from dataclasses import dataclass
from torchao.utils import (
    find_multiple,
    TorchAOBaseTensor,
    TORCH_VERSION_AT_LEAST_2_5,
    _is_float8_type
)
import logging

logger = logging.getLogger(__name__)

from torchao.float8.inference import Float8MMConfig
aten = torch.ops.aten


###############################
# Base Layout Tensor Subclass #
###############################
class AQTLayout(TorchAOBaseTensor):
    """
    Base class for the layout tensor for `AffineQuantizedTensor`
    """
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the plain (unpacked) Tensor for the layout Tensor

        Returns data, scale and zero_point
        Can be overwritten if other types of AQTLayout Tensor has different numbers of plain tensors
        """
        pass

    def get_layout_type(self) -> LayoutType:
        pass

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        """ Construct a Layout from data, scale, zero_point and the layout_type"""
        pass

    def __repr__(self):
        data, scale, zero_point = self.get_plain()
        layout_type = self.get_layout_type()
        return f"{self.__class__.__name__}(data={str(data)}... , scale={str(scale)}... , zero_point={str(zero_point)}... , layout_type={layout_type})"


##############################
# Tensor Subclass Definition #
##############################


class QuantizedLinearNotImplementedError(NotImplementedError):
    """ Thin wrapper around NotImplementedError to make it easier to catch this error in the dispatch table """
    pass


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
        logger.warn(f"Attempting to remove non-existant dispatch condition {dispatch_condition}")

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
      layout_tensor (AQTLayout): tensor that serves as a general layout storage for the quantized data,
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
        layout_tensor: AQTLayout,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[Union[int, float]] = None,
        quant_max: Optional[Union[int, float]] = None,
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
        quant_min: Optional[Union[int, float]] = None,
        quant_max: Optional[Union[int, float]] = None,
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
            f"{self.__class__.__name__}(data={str(self.dequantize())}..., shape={self.shape}, block_size={self.block_size}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad}, layout_tensor={self.layout_tensor})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}, layout_type={self.layout_type}, layout_tensor_dtype={self.layout_tensor.dtype}, quant_min={self.quant_min}, quant_max={self.quant_max}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        from torchao.dtypes.fpx import FpxTensorCoreLayoutType
        if isinstance(self.layout_type, FpxTensorCoreLayoutType):
            int_data, scale = self.layout_tensor.get_plain()
            return dequantize_affine_fpx(int_data, scale, self.layout_type.ebits, self.layout_type.mbits, output_dtype=output_dtype)
        else:
            data, scale, zero_point = self.layout_tensor.get_plain()
            return dequantize_affine(
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

    @staticmethod
    def _quantized_linear_op(input_tensor, weight_tensor, bias):
        for dispatch_condition, impl in _AQT_QLINEAR_DISPATCH_TABLE.items():
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
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
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
            data, scale, zero_point, _ = choose_qparams_and_quantize_affine_hqq(input_float, nbits=nbits, group_size=group_size, axis=axis, compute_dtype=compute_dtype, device=device, verbose=False, raw_output=False)
            data = data.to(target_dtype)
        else:
            scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype, preserve_zero, zero_point_domain)
            # choose_qparams_affine is a custom op that does support returning optional Tensors. We thus set the zero_point to None if its domain is None
            if zero_point_domain is None:
                zero_point = None
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
        zero_point: Optional[torch.Tensor],
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
        layout_type: LayoutType = PlainLayoutType(),
    ):
        if target_dtype not in FP8_TYPES:
            assert zero_point_domain is not None, "zero_point_domain must be specified for non-fp8 types"
            assert zero_point is not None, "zero_point must be specified for non-fp8 types"
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
        target_dtype: torch.dtype,
        scale_dtype: Optional[torch.dtype],
        layout_type: LayoutType,
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
                layout_type=layout_type,
                use_hqq=False,
            )
        else:
            raise NotImplementedError(f"Unsupported dtype {target_dtype} for from_hp_to_floatx")

    @classmethod
    def from_hp_to_floatx_static(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        layout_type: LayoutType,
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
                layout_type=layout_type,
            )
        else:
            raise NotImplementedError(f"Unsupported dtype {target_dtype} for from_hp_to_floatx_static")

    @classmethod
    def from_hp_to_fpx(
        cls,
        input_float: torch.Tensor,
        layout_type: LayoutType,
    ):
        from torchao.dtypes.fpx import FpxTensorCoreLayoutType
        assert isinstance(layout_type, FpxTensorCoreLayoutType), f"Only FpxTensorCoreLayoutType is supported for fpx, got {layout_type}"
        original_shape = input_float.shape
        input_float = layout_type.pre_process(input_float)
        # per axis quantization, where axis = 1
        block_size = list(input_float.shape)
        block_size[1] = 1

        ebits, mbits = layout_type.ebits, layout_type.mbits
        # Note: these ops are hardcoded to have per axis quantization (axis=1) right now
        scale = choose_qparams_affine_fpx(input_float, ebits, mbits)
        fpx_unpacked = quantize_affine_fpx(input_float, scale, ebits, mbits)
        fpx_packed = layout_type.post_process(fpx_unpacked)

        layout_tensor_ctr = get_layout_tensor_constructor(type(layout_type))
        layout_tensor = layout_tensor_ctr(fpx_packed, scale, None, layout_type)
        return cls(
            layout_tensor,
            block_size,
            original_shape,
            dtype=input_float.dtype
        )

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

    # following are the comments for __torch_function__/__torch_dispatch__, we can clean this up
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
# LayoutType and Layout Tensor Subclass Registration #
######################################################
register_layout_cls = AffineQuantizedTensor.register_layout_cls
get_layout_tensor_constructor = AffineQuantizedTensor.get_layout_tensor_constructor

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


@dataclass(frozen=True)
class Float8LayoutType(LayoutType):
    mm_config: Optional[Float8MMConfig] = None


@dataclass(frozen=True)
class MarlinSparseLayoutType(LayoutType):

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor to be in the correct format for the Marlin sparse kernel.
            - 1º: the input tensor is transposed since the linear layer keeps the weights in a transposed format
            - 2º: tensor is injected with 2:4 sparsity 
            - 3º: transposes it again because the quantization process will compute the scales for dim=-1

        Args:
            input (torch.Tensor): the input tensor to preprocess

        Returns:
            torch.Tensor: the preprocessed tensor
        """
        from torchao.sparsity.marlin import inject_24  # avoid circular import
        input_t = input.t()
        w_24, _ = inject_24(input_t, *input_t.shape)
        return w_24.t()


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
        zero_point: Optional[torch.Tensor],
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
        zero_point: Optional[torch.Tensor],
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, SemiSparseLayoutType)
        int_data_compressed = torch._cslt_compress(int_data)
        return cls(int_data_compressed, scale, zero_point, layout_type)


@register_layout_cls(MarlinSparseLayoutType)
class MarlinSparseAQTLayout(AQTLayout):
    """
    Layout storage class for sparse_marlin_24 layout for affine quantized tensor. 
    
    Can be used with 4 bits and 8 bits quantization.

    Original marlin documentation and information:
    https://github.com/IST-DASLab/marlin/tree/master

    Sparse marlin documentation and information:
    https://github.com/IST-DASLab/Sparse-Marlin?tab=readme-ov-file

    fields:
        original_shape (torch.Size): the original shape of the tensor. used to unpack the tensor to the original shape
        group_size (int): the group size used to pack the tensor
        num_bits (int): the number of bits used to quantize the tensor
    """
    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        meta: torch.Tensor,
        layout_type: LayoutType,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
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
        meta: torch.Tensor,
        layout_type: LayoutType,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.meta = meta
        self.layout_type = layout_type
        self.original_shape = original_shape
        self.group_size = group_size
        self.num_bits = num_bits

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"MarlinSparseAQTLayout dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero_point", "meta"], [self.layout_type, self.original_shape, self.group_size, self.num_bits]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        zero_point = tensor_data_dict["zero_point"]
        meta = tensor_data_dict["meta"]
        layout_type, original_shape, group_size, num_bits = tensor_attributes
        return cls(int_data, scale, zero_point, meta, layout_type, original_shape, group_size, num_bits)

    def get_plain(self):
        from torchao.sparsity.marlin import unpack_from_marlin_24  # avoid circular import
        int_data_expanded, scales_expanded = unpack_from_marlin_24(
            self.int_data, 
            self.scale, 
            self.meta, 
            self.original_shape,
            self.group_size,
            self.num_bits,
        )
        int_data_expanded_t = int_data_expanded.t()
        scales_expanded_t = scales_expanded.t()
        return int_data_expanded_t, scales_expanded_t, self.zero_point

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        from torchao.sparsity.marlin import pack_to_marlin_24, const  # avoid circular import
        assert isinstance(layout_type, MarlinSparseLayoutType)

        # Linear layers are (in_features, out_features) but the int_data that is reaching this point
        # is (out_features, in_features). We need to transpose it to match the expected shape in the marlin code.
        q_w_24 = int_data.t()
        scale_t = scale.t()

        if not torch.cuda.get_device_capability()[0] >= 8:
            raise ValueError(
                f'Can not use Sparse Marlin 2:4 int4*fp16 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for Marlin kernel.'
            )

        if q_w_24.dtype != torch.int32:
            raise ValueError("Only `torch.int32` weights are supported.")
        
        in_features, out_features = q_w_24.shape
        if in_features % 128 != 0 or out_features != 256 == 0:
            raise ValueError(
                "`in_features` must be divisible by 64 and `out_features` by 256."
            )

        # NOTE: The current marlin 2:4 kernel supports both 4 and 8 bits quantization but fp8
        # will require a bit more work to get our current quantization flow to work with it.
        # Check the link for a reference: https://github.com/neuralmagic/nm-vllm/tree/main
        num_bits = 4 if torch.max(q_w_24) < 16 else -1
        if num_bits not in [4]:
            raise ValueError(
                f"Only {[4]} bits are supported, got {num_bits}."
            )

        group_size = in_features // scale_t.shape[0]
        if group_size == 0:
            group_size = in_features
        assert group_size <= in_features, "Group size must be less than or equal to in_features."

        if group_size not in const.SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Only {const.SUPPORTED_GROUP_SIZES} group sizes are supported, got {group_size}."
            )

        # Compress quantized weight to marlin 2:4 format
        marlin_24_q_w_comp, marlin_24_s, meta = pack_to_marlin_24(q_w_24, scale_t, num_bits, group_size)

        return cls(
            marlin_24_q_w_comp, marlin_24_s, zero_point, 
            meta, layout_type, q_w_24.shape,
            group_size, num_bits
        )
    
    def get_layout_type(self) -> LayoutType:
        return self.layout_type

    def _apply_fn_to_data(self, fn):
        self.int_data = fn(self.int_data)
        self.scale = fn(self.scale)
        self.zero_point = fn(self.zero_point)
        self.meta = fn(self.meta)
        return self


@register_layout_cls(Float8LayoutType)
class Float8AQTLayout(AQTLayout):
    """
    Layout storage class for float8 layout for affine quantized tensor
    """
    float8_data: torch.Tensor
    scale: torch.Tensor
    transposed: bool

    def __new__(
        cls,
        float8_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        layout_type: LayoutType,
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
        layout_type: LayoutType,
    ):
        self.float8_data = float8_data
        self.scale = scale
        self.transposed = transposed
        self.layout_type = layout_type

    def _apply_fn_to_data(self, fn):
        """ Applys a fn to all tensor components stored on this class"""
        fn(self.float8_data)
        fn(self.scale)
        return self

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.float8_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.transposed,
            self.layout_type,
        )

    def __tensor_flatten__(self):
        return ["float8_data", "scale"], [self.transposed, self.layout_type]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        float8_data, scale = tensor_data_dict["float8_data"], tensor_data_dict["scale"]
        transposed, layout_type, = tensor_attributes
        return cls(float8_data, scale, transposed, layout_type)

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
            f"Float8AQTLayout dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.float8_data, self.scale, None

    def get_layout_type(self) -> LayoutType:
        return self.layout_type

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        layout_type: LayoutType,
    ):
        """ Main entrypoint for constructing Float8Layout Tensor"""
        assert _is_float8_type(data.dtype), f"Float8 Layout must be constructed from float8 dtype but got {data.dtype}"
        assert isinstance(layout_type, Float8LayoutType), f"Float8 Layout must be constructed from Float8LayoutType but got {layout_type}"
        return cls(data, scale, False, layout_type)

    def __repr__(self):
        float8_data, scale, _ = self.get_plain()
        layout_type = self.get_layout_type()
        return (f"{self.__class__.__name__}(\n"
                f"float8_data={float8_data},\n"
                f"scale={scale},\n"
                f"transposed={self.transposed}, "
                f"layout_type={layout_type})")
    

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
        zero_point: Optional[torch.Tensor],
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
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
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
# input_tensor: dimension is (M1, M2, ..., in_features)
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
        w_vals_int8, tmp.t(), alpha=w_scales.to(torch.float32), out_dtype=torch.bfloat16,
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

def _linear_f16_act_fpx_weight_check(input_tensor, weight_tensor, bias):
    from torchao.dtypes.fpx import FpxTensorCoreLayoutType
    return (
        # input is native float32 tensor
        not is_traceable_wrapper_subclass(input_tensor) and
        input_tensor.is_floating_point() and
        input_tensor.dtype == torch.float16 and
        # weight is fpx Tensor
        isinstance(weight_tensor, AffineQuantizedTensor) and
        isinstance(weight_tensor.layout_type, FpxTensorCoreLayoutType) and
        (
            # weight is using fp6 quantization
            (weight_tensor.layout_type.ebits == 3 and
             weight_tensor.layout_type.mbits == 2) or
            (weight_tensor.layout_type.ebits == 2 and
             weight_tensor.layout_type.mbits == 3) or
            # weight is using fp5 quantization
            (weight_tensor.layout_type.ebits == 2 and
             weight_tensor.layout_type.mbits == 2) or
            (weight_tensor.layout_type.ebits == 3 and
             weight_tensor.layout_type.mbits == 1)
        )
    )

def _linear_f16_act_fpx_weight_impl(input_tensor, weight_tensor, bias):
    from torchao.dtypes.fpx import _SPLIT_K_MAP
    from torchao.ops import quant_llm_linear

    act = input_tensor
    weight = weight_tensor

    out_dim, in_dim = weight.shape
    act_reshaped = act.view(-1, in_dim).half()

    # https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
    bsize = act_reshaped.shape[0]
    splitK = _SPLIT_K_MAP[(bsize - 1) // 64].get(out_dim, 1) if bsize <= 768 else 1

    out = quant_llm_linear(
        weight.layout_type.ebits,
        weight.layout_type.mbits,
        act_reshaped,
        weight.layout_tensor.packed_fpx_data,
        weight.layout_tensor.scale,
        splitK=splitK,
    )

    if bias is not None:
        out += bias

    return out.view(*act.shape[:-1], out_dim).to(act.dtype)

def _linear_fp_act_fp8_weight_check(
    input_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    weight_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    bias: Optional[torch.Tensor],
) -> bool:
    def check_aqt(aqt: Union[torch.Tensor, AffineQuantizedTensor]) -> bool:
        return (
            isinstance(aqt, AffineQuantizedTensor) and
            isinstance(aqt.layout_type, Float8LayoutType)
            and aqt.layout_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            and (aqt.shape == aqt.block_size or _is_rowwise_scaled(aqt))
        )
    return check_aqt(input_tensor) and check_aqt(weight_tensor)


def preprocess_scale(input_scale: torch.Tensor, input_shape: Tuple[int]):
    """ Ensures input tensor is correctly formated for _scaled_mm """
    input_scale = input_scale.unsqueeze(-1)
    
    if input_scale.dim() > 2:
        input_scale = input_scale.reshape(-1, input_scale.shape[-1])
    
    return input_scale

def _linear_fp_act_fp8_weight_impl(
    input_tensor: AffineQuantizedTensor,
    weight_tensor: AffineQuantizedTensor,
    bias: Optional[torch.Tensor],
):
    """Implements matmul between FP8 input and FP8 weight with compute using _scaled_mm"""
    scaled_mm_config = weight_tensor.layout_type.mm_config
    out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)

    # Weight tensor preprocessing
    w_layout = weight_tensor.layout_tensor
    assert not w_layout.transposed, "Weight tensor must be contiguous"
    w_data = w_layout.float8_data
    w_scale = w_layout.scale

    # Input tensor preprocessing
    inpt_data = input_tensor.layout_tensor.float8_data
    input_scale = input_tensor.layout_tensor.scale
    # Handle case where input tensor is more than 2D
    inpt_data = inpt_data.reshape(-1, inpt_data.shape[-1])

    # Handle rowwise case
    if _is_rowwise_scaled(weight_tensor):
        assert _is_rowwise_scaled(input_tensor), "Input tensor must be rowwise block size"
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


def _linear_fp_act_int4_weight_sparse_marlin_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(weight_tensor, AffineQuantizedTensor) and
        _aqt_is_uint4(weight_tensor) and
        input_tensor.dtype == torch.float16 and
        len(weight_tensor.shape) == 2 and
        weight_tensor.zero_point_domain == ZeroPointDomain.INT and
        isinstance(weight_tensor.layout_type, MarlinSparseLayoutType)
    )

def _linear_fp_act_int4_weight_sparse_marlin_impl(input_tensor, weight_tensor, bias):
    from torchao.sparsity.marlin import marlin_24_workspace, const
    from torchao.ops import marlin_24_gemm

    assert isinstance(weight_tensor, AffineQuantizedTensor)

    sparse_w_int4 = weight_tensor.layout_tensor.int_data
    scale = weight_tensor.layout_tensor.scale
    meta = weight_tensor.layout_tensor.meta
    original_shape = weight_tensor.layout_tensor.original_shape
    num_bits = weight_tensor.layout_tensor.num_bits

    # Folds batch dimension into the first dimension
    input_2d = input_tensor.view(-1, input_tensor.shape[-1])

    size_m = input_2d.shape[0]
    size_n = scale.shape[1]
    size_k = input_2d.shape[1]
    workspace_24 = marlin_24_workspace(original_shape[1])

    out = marlin_24_gemm(
        input_2d, sparse_w_int4, meta, scale, 
        workspace_24, num_bits, size_m, size_n, size_k
    )

    # Unfold the batch dimension
    out = out.reshape(input_tensor.shape[:-1] + (scale.shape[1],))

    if bias is not None:
        out += bias.to(out.dtype)
    return out


def _register_aqt_quantized_linear_dispatches():
    for dispatch_condition, impl in [
        (_linear_int8_act_int8_weight_check, _linear_int8_act_int8_weight_impl),
        (_linear_int8_act_int8_weight_semi_structured_sparse_check, _linear_int8_act_int8_weight_semi_structured_sparse_impl),
        (_linear_fp_act_fp8_weight_check, _linear_fp_act_fp8_weight_impl),
        (_linear_bf16_act_uint4_weight_check, _linear_bf16_act_uint4_weight_impl),
        (_linear_fp_act_int8_weight_check, _linear_fp_act_int8_weight_impl),
        (_linear_f16_act_fpx_weight_check, _linear_f16_act_fpx_weight_impl),
        (_linear_fp_act_int4_weight_sparse_marlin_check, _linear_fp_act_int4_weight_sparse_marlin_impl),
    ]:
        register_aqt_quantized_linear_dispatch(dispatch_condition, impl)

_register_aqt_quantized_linear_dispatches()

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
    except QuantizedLinearNotImplementedError as e:
        # fallback path is only called when user did not specify a specfic quantized linear implementation with `layout_type.quantized_linear_impl`
        if isinstance(weight_tensor, AffineQuantizedTensor) and hasattr(weight_tensor.layout_type, "quantized_linear_impl") and weight_tensor.layout_type.quantized_linear_impl is not None:
            raise e

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
    except QuantizedLinearNotImplementedError as e:
        # fallback path is only called when user did not specify a specfic quantized linear implementation with `layout_type.quantized_linear_impl`
        if isinstance(weight_tensor, AffineQuantizedTensor) and hasattr(weight_tensor.layout_type, "quantized_linear_impl") and weight_tensor.layout_type.quantized_linear_impl is not None:
            raise e

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
    except QuantizedLinearNotImplementedError as e:
        # fallback path is only called when user did not specify a specfic quantized linear implementation with `layout_type.quantized_linear_impl`
        if isinstance(weight_tensor, AffineQuantizedTensor) and hasattr(weight_tensor.layout_type, "quantized_linear_impl") and weight_tensor.layout_type.quantized_linear_impl is not None:
            raise e

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
to_affine_quantized_floatx_static = AffineQuantizedTensor.from_hp_to_floatx_static
# experimental will be merged in to floatx
to_affine_quantized_fpx = AffineQuantizedTensor.from_hp_to_fpx

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with AffineQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([AffineQuantizedTensor])
