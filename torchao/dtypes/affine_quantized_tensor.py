import torch
from typing import Tuple, Optional, Union, List
import torchao.ops
from collections import defaultdict
import functools
import math
from torchao.quantization.quant_primitives import (
    _get_reduction_params,
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    ZeroPointDomain,
    MappingType,
    int_scaled_matmul,
    choose_qparams_and_quantize_affine_hqq,
    FP8_TYPES,
    choose_qparams_affine_floatx,
    quantize_affine_floatx,
    dequantize_affine_floatx,
)
from torchao.quantization.utils import (
    pack_tinygemm_scales_and_zeros,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.utils import (
    Layout,
    PlainLayout,
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
    _is_float8_type,
    fill_defaults,
)
import logging

logger = logging.getLogger(__name__)

from torchao.float8.inference import Float8MMConfig
aten = torch.ops.aten

###############################
# Base Tensor Impl Subclass #
###############################
class AQTTensorImpl(TorchAOBaseTensor):
    """
    Base class for the tensor impl for `AffineQuantizedTensor`

    Note: This is not a user facing API, it's used by AffineQuantizedTensor to construct
    the underlying implementation of a AQT based on layout
    """
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the plain (unpacked) Tensor for the tensor impl

        Returns data, scale and zero_point
        Can be overwritten if other types of AQTTensorImpl has different numbers of plain tensors
        """
        pass

    def get_layout(self) -> Layout:
        pass

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):
        """ Construct a TensorImpl from data, scale, zero_point and the _layout"""
        pass

    def __repr__(self):
        data, scale, zero_point = self.get_plain()
        _layout = self.get_layout()
        return f"{self.__class__.__name__}(data={str(data)}... , scale={str(scale)}... , zero_point={str(zero_point)}... , _layout={_layout})"


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

        from torchao.dtypes.floatx import FloatxTensorCoreLayout
        if isinstance(self._layout, FloatxTensorCoreLayout):
            int_data, scale = self.tensor_impl.get_plain()
            return dequantize_affine_floatx(int_data, scale, self._layout.ebits, self._layout.mbits, output_dtype=output_dtype)
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
            # need to return to original shape if tensor was padded
            # in preprocessing
            for dim, dim_size in enumerate(self.shape):
                dq = dq.narrow(dim, 0, dim_size)
            return dq

    @staticmethod
    def _quantized_linear_op(input_tensor, weight_tensor, bias):
        for dispatch_condition, impl in _AQT_QLINEAR_DISPATCH_TABLE.items():
            if dispatch_condition(input_tensor, weight_tensor, bias):
                return impl(input_tensor, weight_tensor, bias)
        raise QuantizedLinearNotImplementedError("No specialized dispatch found for quantized linear op")

    def __tensor_flatten__(self):
        return ["tensor_impl"], [self.block_size, self.shape, self.quant_min, self.quant_max, self.zero_point_domain, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        tensor_impl = tensor_data_dict["tensor_impl"]
        block_size, shape, quant_min, quant_max, zero_point_domain, dtype = tensor_attributes
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
        _layout: Layout = PlainLayout(),
    ):
        if target_dtype not in FP8_TYPES:
            assert zero_point_domain is not None, "zero_point_domain must be specified for non-fp8 types"
            assert zero_point is not None, "zero_point must be specified for non-fp8 types"
        original_shape = input_float.shape
        input_float, scale, zero_point = _layout.pre_process_static(input_float, scale, zero_point, block_size)

        int_data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max, zero_point_domain)

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
            raise NotImplementedError(f"Unsupported dtype {target_dtype} for from_hp_to_floatx")

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
            raise NotImplementedError(f"Unsupported dtype {target_dtype} for from_hp_to_floatx_static")

    @classmethod
    def from_hp_to_fpx(
        cls,
        input_float: torch.Tensor,
        _layout: Layout,
    ):
        from torchao.dtypes.floatx import FloatxTensorCoreLayout
        assert isinstance(_layout, FloatxTensorCoreLayout), f"Only FloatxTensorCoreLayout is supported for floatx, got {_layout}"
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)
        # per axis quantization, where axis = 1
        block_size = list(input_float.shape)
        block_size[1] = 1

        ebits, mbits = _layout.ebits, _layout.mbits
        # Note: these ops are hardcoded to have per axis quantization (axis=1) right now
        scale = choose_qparams_affine_floatx(input_float, ebits, mbits)
        floatx_unpacked = quantize_affine_floatx(input_float, scale, ebits, mbits)
        floatx_packed = _layout.post_process(floatx_unpacked)

        tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(floatx_packed, scale, None, _layout)
        return cls(
            tensor_impl,
            block_size,
            original_shape,
            dtype=input_float.dtype
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
# Layout and TensorImpl Subclass Registration #
######################################################
register_layout = AffineQuantizedTensor.register_layout
get_tensor_impl_constructor = AffineQuantizedTensor.get_tensor_impl_constructor

@dataclass(frozen=True)
class SemiSparseLayout(Layout):

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        # prune to 2:4 if not already
        temp = input.detach()
        pruning_inds = temp.abs().view(-1, 4).argsort(dim=1)[:, :2]
        temp.view(-1, 4).scatter_(1, pruning_inds, value=0)
        return temp


@dataclass(frozen=True)
class BlockSparseLayout(Layout):
    blocksize: int = 64


@dataclass(frozen=True)
class TensorCoreTiledLayout(Layout):
    """
    inner_k_tiles is an internal argument for packing function of tensor core tiled layout
    that can affect the performance of the matmul kernel
    """
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

    def pre_process_static(self, input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, block_size: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = self.pre_process(input)
        orig_qparam_shape = scale.shape
        new_qparam_shape, reduction_dims = _get_reduction_params(block_size, input.size())
        for dim in reduction_dims:
            new_qparam_shape.pop(dim)
        change_in_qparam_shape = [new_dim_size-orig_dim_size for new_dim_size, orig_dim_size in zip(new_qparam_shape, orig_qparam_shape)]
        padding_changes=[]
        for dim_change in change_in_qparam_shape:
            padding_changes = [0, dim_change] + padding_changes
        scale = torch.nn.functional.pad(scale, padding_changes)
        zero_point = torch.nn.functional.pad(zero_point, padding_changes)
        return input, scale, zero_point

    def post_process(self, input: torch.Tensor) -> torch.Tensor:
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
class Float8Layout(Layout):
    mm_config: Optional[Float8MMConfig] = None


@dataclass(frozen=True)
class MarlinSparseLayout(Layout):

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


@register_layout(PlainLayout)
class PlainAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for plain layout for affine quantized tensor, it stores int_data, scale, zero_point
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
        _layout: Layout,
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
        _layout: Layout,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero_point"], [self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = tensor_data_dict["int_data"], tensor_data_dict["scale"], tensor_data_dict["zero_point"]
        _layout, = tensor_attributes
        return cls(int_data, scale, zero_point, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"]),
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point),
            self._layout,
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

        elif func is aten.t.default:
            tensor = args[0]
            new = tensor.__class__(
                tensor.int_data.t(), tensor.scale, tensor.zero_point, tensor._layout
            )
            return return_and_correct_aliasing(func, args, kwargs, new)

        elif func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                return return_and_correct_aliasing(
                    func, args, kwargs, args[0]._apply_fn_to_data(lambda x: aten.slice.Tensor(x, dim, start, end, step))
                )
            elif dim == 1:
                assert len(self.scale.shape) == 1, f"slice dim==1 only works when len(scale.shape) == 1 currently, got: {self.scale.shape}"
                return PlainAQTTensorImpl(aten.slice.Tensor(self.int_data, dim, start, end, step), self.scale.view(-1), self.zero_point.view(-1), self._layout)
            else:
                raise NotImplementedError(f"PlainAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported")

        raise NotImplementedError(
            f"PlainAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.int_data, self.scale, self.zero_point

    def get_layout(self) -> Layout:
        return self._layout

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, PlainLayout)
        return cls(int_data, scale, zero_point, _layout)

@register_layout(SemiSparseLayout)
class SemiSparseAQTTensorImpl(PlainAQTTensorImpl):
    """
    TensorImpl for semi_sparse_cusparselt layout for affine quantized tensor
    """
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"SparseAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
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
        _layout: Layout,
    ):
        assert isinstance(_layout, SemiSparseLayout)
        int_data_compressed = torch._cslt_compress(int_data)
        return cls(int_data_compressed, scale, zero_point, _layout)

@register_layout(BlockSparseLayout)
class BlockSparseAQTTensorImpl(PlainAQTTensorImpl):
    bsr_crow_indices: Optional[torch.Tensor]
    bsr_col_indices: Optional[torch.Tensor]
    bsr_values: Optional[torch.Tensor]
    scale: Optional[torch.Tensor]
    zero_point: Optional[torch.Tensor]

    __slots__ = ["bsr_crow_indices", "bsr_col_indices", "bsr_values", "scale", "zero_point"]

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        bsr_crow_indices: Optional[torch.Tensor],
        bsr_col_indices: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
        requires_grad: bool = False,
    ):
        if bsr_values is None:
            raise ValueError("bsr values must be provided!")
        else:
            previous_tensor = bsr_values

        kwargs = {
            "device": previous_tensor.device,
            "dtype": previous_tensor.dtype,
            "layout": previous_tensor.layout,
            "requires_grad": requires_grad,
        }
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(  # noqa: PYI034
        self,
        shape: torch.Size,
        bsr_crow_indices: Optional[torch.Tensor],
        bsr_col_indices: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
        requires_grad: bool = False,
    ):
        self.bsr_crow_indices = bsr_crow_indices
        self.bsr_col_indices = bsr_col_indices
        self.bsr_values = bsr_values
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __tensor_flatten__(self):
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (self.shape, self._layout, self.requires_grad)
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: Tuple[torch.Size, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape, _layout, requires_grad = tensor_meta
        return cls(
            shape=shape,
            bsr_crow_indices=inner_tensors.get("bsr_crow_indices", None),
            bsr_col_indices=inner_tensors.get("bsr_col_indices", None),
            bsr_values=inner_tensors.get("bsr_values", None),
            scale=inner_tensors.get("scale", None),
            zero_point=inner_tensors.get("zero_point", None),
            _layout=_layout,
            requires_grad=requires_grad,
        )

    @classmethod
    def from_plain(cls, int_data, scale, zero_point, _layout):
        bsr_tensor = int_data.to_sparse_bsr(_layout.blocksize)
        return cls(
            shape=int_data.shape,
            bsr_crow_indices=bsr_tensor.crow_indices(),
            bsr_col_indices=bsr_tensor.col_indices(),
            bsr_values=bsr_tensor.values(),
            scale=scale,
            zero_point=zero_point,
            _layout = _layout,
            requires_grad=False,
        )

    def get_plain(self):
        int_data_expanded = torch.ops.blocksparse.bsr_to_dense(self.crow_indices(), self.col_indices(), self.values(), self.shape[0], self.shape[1])
        return int_data_expanded, self.scale, self.zero_point

    def _apply_fn_to_data(self, func):
        return self.__class__(
            shape = self.shape,
            bsr_crow_indices=func(self.bsr_crow_indices),
            bsr_col_indices=func(self.bsr_col_indices),
            bsr_values=func(self.bsr_values),
            scale=self.scale,
            zero_point=self.zero_point,
            _layout=self._layout,
            requires_grad=self.requires_grad,
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

        # Need the following for bsr specific functions
        if func is aten.crow_indices.default:
            return args[0].bsr_crow_indices.detach()

        if func is aten.col_indices.default:
            return args[0].bsr_col_indices.detach()

        if func is aten.values.default:
            return args[0].bsr_values.detach()

        if func is aten._nnz.default:
            return args[0].bsr_values.shape[0]

        raise NotImplementedError(
            f"BlockSparseAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

@register_layout(MarlinSparseLayout)
class MarlinSparseAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for sparse_marlin_24 layout for affine quantized tensor.

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
        _layout: Layout,
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
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.meta = meta
        self._layout = _layout
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
            f"MarlinSparseAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero_point", "meta"], [self._layout, self.original_shape, self.group_size, self.num_bits]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        zero_point = tensor_data_dict["zero_point"]
        meta = tensor_data_dict["meta"]
        _layout, original_shape, group_size, num_bits = tensor_attributes
        return cls(int_data, scale, zero_point, meta, _layout, original_shape, group_size, num_bits)

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
        _layout: Layout,
    ):
        from torchao.sparsity.marlin import pack_to_marlin_24, const  # avoid circular import
        assert isinstance(_layout, MarlinSparseLayout)

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
            meta, _layout, q_w_24.shape,
            group_size, num_bits
        )

    def get_layout(self) -> Layout:
        return self._layout

    def _apply_fn_to_data(self, fn):
        self.int_data = fn(self.int_data)
        self.scale = fn(self.scale)
        self.zero_point = fn(self.zero_point)
        self.meta = fn(self.meta)
        return self


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
        """ Applys a fn to all tensor components stored on this class"""
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
        transposed, _layout, = tensor_attributes
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
        elif func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                #TODO: scale replecation should be dependent on block size
                if self.scale.ndim == 1:
                    return return_and_correct_aliasing(
                        func, args, kwargs, args[0]._apply_fn_to_data(lambda x: aten.slice.Tensor(x, dim, start, end, step))
                    )
                elif self.scale.ndim == 0:
                    return return_and_correct_aliasing(
                        func, args, kwargs, Float8AQTTensorImpl(aten.slice.Tensor(self.float8_data, dim, start, end, step), self.scale, None, self._layout)
                    )
                else:
                    raise NotImplementedError(f"Float8AQTTensorImpl dispatch: attempting to run {func}, with scale ndim={dim}, that is not supported")
            elif dim == 1:
                return return_and_correct_aliasing(
                        func, args, kwargs, Float8AQTTensorImpl(aten.slice.Tensor(self.float8_data, dim, start, end, step).contiguous(), self.scale, None, self._layout)
                )
            else:
                raise NotImplementedError(f"Float8AQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported")
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
        """ Main entrypoint for constructing Float8TensorImpl"""
        assert _is_float8_type(data.dtype), f"Float8 TensorImpl must be constructed from float8 dtype but got {data.dtype}"
        assert isinstance(_layout, Float8Layout), f"Float8 TensorImpl must be constructed from Float8Layout but got {_layout}"
        return cls(data, scale, False, _layout)

    def __repr__(self):
        float8_data, scale, _ = self.get_plain()
        _layout = self.get_layout()
        return (f"{self.__class__.__name__}(\n"
                f"float8_data={float8_data},\n"
                f"scale={scale},\n"
                f"transposed={self.transposed}, "
                f"_layout={_layout})")


@register_layout(TensorCoreTiledLayout)
class TensorCoreTiledAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for tensor_core_tiled layout for affine quantized tensor, this is for int4 only,
    used by tinygemm kernels `_weight_int4pack_mm`

    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 4-d tensor of
    dimension: [n / 8][k / (inner_k_tiles * 16)][32][inner_k_tiles / 2]
    (unpacked Tensor shape is n * k)
    where inner_k_tiles is an internal argument for packing function of tensor core tiled layout
    that can affect the performance of the matmul kernel (defaults to 8)

    Note: we also pack scale and zero point together here for tinygemm kernel

    Note: technically tensor core tiled layout should be the layout for the underlying packed weight
    (int Tensor) but since the scale and zero_point are also packed into the same tensor here which is not used
    in plain layout, we just created a layout for AQT right now, this could be improved if we split out
    int4 aqt into a separate tensor subclass

    fields:
      packed_weight (torch.Tensor): the 4-d packed tensor in a tensor_core_tiled layout
      scale_and_zero (torch.Tensor): the combined scale Tensor used to map between floating point tensor to quantized tensor and zero_point Tensor
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
        _layout: Layout,
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
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scale_and_zero = scale_and_zero
        self.transposed = False
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["packed_weight", "scale_and_zero"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale_and_zero = tensor_data_dict["packed_weight"], tensor_data_dict["scale_and_zero"]
        transposed, _layout, = tensor_attributes
        return cls(packed_weight, scale_and_zero, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout
    ):

        assert isinstance(_layout, TensorCoreTiledLayout)

        if TORCH_VERSION_AT_LEAST_2_5:
            int_data = (int_data[::, ::2] << 4 | int_data[::, 1::2]).to(torch.uint8)
            assert int_data.dtype == torch.uint8, "torch.ops.aten._convert_weight_to_int4pack in torch 2.5 expects `uint8` dtype"
        else:
            assert int_data.dtype == torch.int32, "torch.ops.aten._convert_weight_to_int4pack in torch 2.4 expects `int32` dtype"
        packed_weight = torch.ops.aten._convert_weight_to_int4pack(int_data, _layout.inner_k_tiles)
        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point)
        return cls(packed_weight, scale_and_zero, False, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device("cuda", device):
            raise ValueError(f"TensorCoreTiledAQTTensorImpl is only available for cuda device, can't convert to {device}")
        return self.__class__(
            self.packed_weight.to(device),
            self.scale_and_zero.to(device),
            self.transposed,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        # self.packed_weight = fn(self.packed_weight)
        # self.scale_and_zero = fn(self.scale_and_zero)
        # return self
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale_and_zero),
            self.transposed,
            self._layout,
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
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            transposed = TensorCoreTiledAQTTensorImpl(args[0].packed_weight, args[0].scale_and_zero, not args[0].transposed, args[0]._layout)
            return return_and_correct_aliasing(func, args, kwargs, transposed)

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                int_data, scale, zero_point = self.get_plain()
                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                # this is to handle padding
                int_data = self._layout.post_process(int_data)
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return return_and_correct_aliasing(func, args, kwargs, sliced)
            elif dim == 1:
                int_data, scale, zero_point = self.get_plain()
                assert step == 1, "Only step == 1 is supported in slicing right now"
                data_len = int_data.shape[dim]
                scale_len = scale.shape[dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                # this is to handle padding
                int_data = self._layout.post_process(int_data)
                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                zero_point = aten.slice.Tensor(zero_point, dim, start_scale, end_scale, step)
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return sliced
            else:
                raise NotImplementedError(f"TensorCoreTiledAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported")

        raise NotImplementedError(
            f"TensorCoreTiledAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
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

    def get_layout(self) -> Layout:
        return self._layout


#####################################################
# torch functional and aten operator implementation #
#####################################################

def _aqt_is_int8(aqt):
    """Check if an AffineQuantizedTensor is int8 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.int8 and
        (aqt.quant_min is None or aqt.quant_min == -128) and
        (aqt.quant_max is None or aqt.quant_max == 127)
    )

def _aqt_is_int8_reduced_range(aqt):
    return (
        aqt.tensor_impl.dtype == torch.int8 and
        aqt.quant_min == -127 and
        (aqt.quant_max is None or aqt.quant_max == 127)
    )

def _aqt_is_tensor_core_tile_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    # TODO: use torch.uint4
    return (
        aqt.tensor_impl.dtype == torch.int32 and
        aqt.quant_min == 0 and
        aqt.quant_max == 15
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
        input_tensor.dtype == weight_tensor.dtype and
        isinstance(input_tensor._layout, PlainLayout) and
        isinstance(weight_tensor._layout, PlainLayout)
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

    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scales = input_tensor.tensor_impl.scale
    w_vals_int8_t = weight_tensor.tensor_impl.int_data.contiguous().t()
    w_scales = weight_tensor.tensor_impl.scale
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    x_scales_dtype = x_scales.dtype
    # Cast fp16 scale to float to avoid overflow in int_scaled_matmul
    intermediate_dtype = torch.float if x_scales_dtype == torch.half else x_scales_dtype
    y_dot_scaled = int_scaled_matmul(tmp, w_vals_int8_t, x_scales.reshape(-1, 1).to(intermediate_dtype))
    y_dot_scaled = y_dot_scaled.to(x_scales_dtype)

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
        isinstance(input_tensor._layout, PlainLayout) and
        isinstance(weight_tensor._layout, SemiSparseLayout)
    )

def _linear_int8_act_int8_weight_semi_structured_sparse_impl(input_tensor, weight_tensor, bias):
    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scales = input_tensor.tensor_impl.scale
    w_vals_int8 = weight_tensor.tensor_impl.int_data
    w_scales = weight_tensor.tensor_impl.scale
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

def _linear_int8_act_int8_weight_block_sparse_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor) and
        _aqt_is_int8_reduced_range(input_tensor) and
        isinstance(weight_tensor, AffineQuantizedTensor) and
        weight_tensor.is_cuda and
        input_tensor.dtype == weight_tensor.dtype and
        isinstance(input_tensor._layout, PlainLayout) and
        isinstance(weight_tensor._layout, BlockSparseLayout)
    )


def _linear_int8_act_int8_weight_block_sparse_impl(input_tensor, weight_tensor, bias):
    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scales = input_tensor.tensor_impl.scale
    w_vals = weight_tensor.tensor_impl
    w_scales = weight_tensor.tensor_impl.scale
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    tmp_t = tmp.t()

    y = torch.ops.blocksparse.int_addmm(w_vals.crow_indices(),
                                        w_vals.col_indices(),
                                        w_vals.values(),
                                        tmp_t,
                                        w_scales,
                                        x_scales.reshape(-1))
    y_shape = (*x_vals_int8.shape[:-1], w_scales.shape[-1])
    y = y.reshape(*y_shape)

    # can downcast only at the very end
    output_dtype = input_tensor.dtype
    y = y.to(output_dtype)
    if bias is not None:
        y += bias
    return y


def _linear_bf16_act_uint4_weight_check(input_tensor, weight_tensor, bias):
    return (
        # input is native bfloat16 tensor
        not is_traceable_wrapper_subclass(input_tensor) and
        input_tensor.dtype == torch.bfloat16 and
        # weight is uint4, group quantized tensor_core_tiled tensor impl affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor) and
        _aqt_is_tensor_core_tile_uint4(weight_tensor) and
        weight_tensor.dtype == torch.bfloat16 and
        len(weight_tensor.shape) == 2 and
        weight_tensor.zero_point_domain == ZeroPointDomain.FLOAT and
        isinstance(weight_tensor._layout, TensorCoreTiledLayout)
    )


def _linear_bf16_act_uint4_weight_impl(input_tensor, weight_tensor, bias):
    assert weight_tensor.block_size[0] == 1, f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    # TODO: check groupsize quantization
    # avoid circular dep, TODO: move this to a common util.py
    act_mat = input_tensor
    # weight is packed from padded (out_features, in_features) weight tensor
    # (same dimension requirement as F.linear weight)
    packed_weight = weight_tensor.tensor_impl.packed_weight
    scale_and_zero = weight_tensor.tensor_impl.scale_and_zero

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
        isinstance(weight_tensor._layout, PlainLayout)
    )

def _linear_fp_act_int8_weight_impl(input_tensor, weight_tensor, bias):
    # TODO: enable cpu and mps efficient path
    # is_cpu and is_mps only, some issue with is_contiguous() currently
    # return torch.ops.aten._weight_int8pack_mm(input_tensor.contiguous(), w_vals_int8_t, weight_tensor.tensor_impl.scale)

    # per channel int8 weight only quantizated mm
    w_vals_int8_t = weight_tensor.tensor_impl.int_data.t()
    scale = weight_tensor.tensor_impl.scale
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

def _linear_f16_act_floatx_weight_check(input_tensor, weight_tensor, bias):
    from torchao.dtypes.floatx import FloatxTensorCoreLayout
    return (
        # input is native float32 tensor
        not is_traceable_wrapper_subclass(input_tensor) and
        input_tensor.is_floating_point() and
        input_tensor.dtype == torch.float16 and
        # weight is floatx Tensor
        isinstance(weight_tensor, AffineQuantizedTensor) and
        isinstance(weight_tensor._layout, FloatxTensorCoreLayout) and
        (
            # weight is using fp6 quantization
            (weight_tensor._layout.ebits == 3 and
             weight_tensor._layout.mbits == 2) or
            (weight_tensor._layout.ebits == 2 and
             weight_tensor._layout.mbits == 3) or
            # weight is using fp5 quantization
            (weight_tensor._layout.ebits == 2 and
             weight_tensor._layout.mbits == 2) or
            (weight_tensor._layout.ebits == 3 and
             weight_tensor._layout.mbits == 1)
        )
    )

def _linear_f16_act_floatx_weight_impl(input_tensor, weight_tensor, bias):
    from torchao.dtypes.floatx import _SPLIT_K_MAP
    from torchao.ops import quant_llm_linear

    act = input_tensor
    weight = weight_tensor

    out_dim, in_dim = weight.shape
    act_reshaped = act.view(-1, in_dim).half()

    # https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
    bsize = act_reshaped.shape[0]
    splitK = _SPLIT_K_MAP[(bsize - 1) // 64].get(out_dim, 1) if bsize <= 768 else 1

    out = quant_llm_linear(
        weight._layout.ebits,
        weight._layout.mbits,
        act_reshaped,
        weight.tensor_impl.packed_floatx_data,
        weight.tensor_impl.scale,
        splitK=splitK,
    )

    if bias is not None:
        out += bias

    return out.view(*act.shape[:-1], out_dim).to(act.dtype)

def _linear_fp8_act_fp8_weight_check(
    input_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    weight_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    bias: Optional[torch.Tensor],
) -> bool:
    def check_aqt(aqt: Union[torch.Tensor, AffineQuantizedTensor]) -> bool:
        return (
            isinstance(aqt, AffineQuantizedTensor) and
            isinstance(aqt._layout, Float8Layout)
            and aqt.tensor_impl.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            and (aqt.shape == aqt.block_size or _is_rowwise_scaled(aqt))
        )
    return check_aqt(input_tensor) and check_aqt(weight_tensor)


def preprocess_scale(input_scale: torch.Tensor, input_shape: Tuple[int]):
    """ Ensures input tensor is correctly formated for _scaled_mm """
    input_scale = input_scale.unsqueeze(-1)

    if input_scale.dim() > 2:
        input_scale = input_scale.reshape(-1, input_scale.shape[-1])

    return input_scale

def _linear_fp8_act_fp8_weight_impl(
    input_tensor: AffineQuantizedTensor,
    weight_tensor: AffineQuantizedTensor,
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

def _linear_fp_act_fp8_weight_check(
    input_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    weight_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    bias: Optional[torch.Tensor],
) -> bool:
    return (
        # input is native float tensor
        not is_traceable_wrapper_subclass(input_tensor) and
        input_tensor.is_floating_point() and
        # weight is float8 quantized affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor) and
        isinstance(weight_tensor._layout, Float8Layout)
        and weight_tensor.tensor_impl.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        and (weight_tensor.shape == weight_tensor.block_size or _is_rowwise_scaled(weight_tensor))
    )

def _linear_fp_act_fp8_weight_impl(
    input_tensor: torch.Tensor,
    weight_tensor: AffineQuantizedTensor,
    bias: Optional[torch.Tensor],
):
    return torch.nn.functional.linear(input_tensor, weight_tensor.dequantize(), bias)

def _linear_fp_act_int4_weight_sparse_marlin_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(weight_tensor, AffineQuantizedTensor) and
        _aqt_is_tensor_core_tile_uint4(weight_tensor) and
        input_tensor.dtype == torch.float16 and
        len(weight_tensor.shape) == 2 and
        weight_tensor.zero_point_domain == ZeroPointDomain.INT and
        isinstance(weight_tensor._layout, MarlinSparseLayout)
    )

def _linear_fp_act_int4_weight_sparse_marlin_impl(input_tensor, weight_tensor, bias):
    from torchao.sparsity.marlin import marlin_24_workspace, const
    from torchao.ops import marlin_24_gemm

    assert isinstance(weight_tensor, AffineQuantizedTensor)

    sparse_w_int4 = weight_tensor.tensor_impl.int_data
    scale = weight_tensor.tensor_impl.scale
    meta = weight_tensor.tensor_impl.meta
    original_shape = weight_tensor.tensor_impl.original_shape
    num_bits = weight_tensor.tensor_impl.num_bits

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
        (_linear_int8_act_int8_weight_block_sparse_check, _linear_int8_act_int8_weight_block_sparse_impl),
        (_linear_fp8_act_fp8_weight_check, _linear_fp8_act_fp8_weight_impl),
        (_linear_fp_act_fp8_weight_check, _linear_fp_act_fp8_weight_impl),
        (_linear_bf16_act_uint4_weight_check, _linear_bf16_act_uint4_weight_impl),
        (_linear_fp_act_int8_weight_check, _linear_fp_act_int8_weight_impl),
        (_linear_f16_act_floatx_weight_check, _linear_f16_act_floatx_weight_impl),
        (_linear_fp_act_int4_weight_sparse_marlin_check, _linear_fp_act_int4_weight_sparse_marlin_impl),
    ]:
        register_aqt_quantized_linear_dispatch(dispatch_condition, impl)

_register_aqt_quantized_linear_dispatches()

@implements([torch.nn.functional.linear, aten.linear.default])
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
        # fallback path is only called when user did not specify a specfic quantized linear implementation with `_layout.quantized_linear_impl`
        if isinstance(weight_tensor, AffineQuantizedTensor) and hasattr(weight_tensor._layout, "quantized_linear_impl") and weight_tensor._layout.quantized_linear_impl is not None:
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
        # fallback path is only called when user did not specify a specfic quantized linear implementation with `_layout.quantized_linear_impl`
        if isinstance(weight_tensor, AffineQuantizedTensor) and hasattr(weight_tensor._layout, "quantized_linear_impl") and weight_tensor._layout.quantized_linear_impl is not None:
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
        # fallback path is only called when user did not specify a specfic quantized linear implementation with `_layout.quantized_linear_impl`
        if isinstance(weight_tensor, AffineQuantizedTensor) and hasattr(weight_tensor._layout, "quantized_linear_impl") and weight_tensor._layout.quantized_linear_impl is not None:
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
        tensor.tensor_impl.t(), transposed_block_size, shape, tensor.quant_min, tensor.quant_max, tensor.zero_point_domain, dtype=tensor.dtype, strides=tensor.stride()
    )
    return return_and_correct_aliasing(func, args, kwargs, new)

@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]
    shape = list(self.shape)
    shape[dim] = end - start
    block_size = self.block_size
    assert len(block_size) == 2, f"Slice only works for 2d block_size right now, got: {block_size}"
    # with slice, some shape dimension might be smaller than block_size dimension, so
    # we need to make sure there is no overflow
    block_size = (min(shape[0], block_size[0]), min(shape[1], block_size[1]))
    new = self.__class__(aten.slice.Tensor(self.tensor_impl, dim, start, end, step), block_size, shape, self.quant_min, self.quant_max, self.zero_point_domain, dtype=self.dtype, strides=self.stride())
    return return_and_correct_aliasing(func, args, kwargs, new)

# this is needed for DTensor.from_local() and for flattening tensor
@implements(aten.view.default)
def _(func, types, args, kwargs):
    self, shape = args

    if tuple(self.shape) == tuple(shape):
        return self.__class__(self.tensor_impl, self.block_size, self.shape, self.quant_min, self.quant_max, self.zero_point_domain, dtype=self.dtype, strides=self.stride())

    if len(shape) == 1 and shape[0] == -1:
        assert len(self.block_size) == 2 and self.block_size[0] == 1
        block_size = (self.block_size[1],)
        return self.__class__(self.tensor_impl, block_size, (self.numel(),), self.quant_min, self.quant_max, self.zero_point_domain, dtype=self.dtype, strides=self.stride())

    raise ValueError(f"{self.__class__.__name__} only supports .view() with same shape or shape=[-1]")


to_affine_quantized_intx = AffineQuantizedTensor.from_hp_to_intx
to_affine_quantized_intx_static = AffineQuantizedTensor.from_hp_to_intx_static
to_affine_quantized_floatx = AffineQuantizedTensor.from_hp_to_floatx
to_affine_quantized_floatx_static = AffineQuantizedTensor.from_hp_to_floatx_static
# experimental will be merged in to floatx
to_affine_quantized_fpx = AffineQuantizedTensor.from_hp_to_fpx

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with AffineQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([AffineQuantizedTensor])
