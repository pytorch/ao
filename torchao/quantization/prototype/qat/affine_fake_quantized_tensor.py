import torch
import torch.utils._pytree as pytree
from typing import Callable, Optional, Tuple
from torchao.quantization.quant_primitives import (
    _get_and_check_qmin_qmax,
    choose_qparams_affine,
    fake_quantize_affine,
    ZeroPointDomain,
    MappingType,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.utils import (
    _implements,
    _dispatch__torch_function__,
    _dispatch__torch_dispatch__,
)
from .utils import _GenericFakeQuantize

aten = torch.ops.aten


class AffineFakeQuantizedTensor(torch.Tensor):
    """
    Affine fake quantized tensor subclass. Affine quantization means we quantize the floating point tensor
    with an affine transformation:
       quantized_tensor = float_tensor / scale + zero_point

    Fake quantization refers to performing the quantization math without actually casting the floating point
    tensor into lower bit-width dtypes. It is commonly used for quantization-aware training (QAT).

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    fields:
      original_tensor (torch.Tensor): tensor holding the original float values, needed for actual quantization later
      apply_fake_quant_fn (Callable): function that transforms `original_tensor` to fake quantized values
    """

    @staticmethod
    def __new__(
        cls,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool = True,
        requires_grad: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("dtype", original_tensor.dtype)
        kwargs.setdefault("device", original_tensor.device)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            original_tensor.shape,
            requires_grad=requires_grad,
            **kwargs,
        )

    def __init__(
        self,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool = True,
        requires_grad: bool = False,
        **kwargs,
    ):
        self.original_tensor = original_tensor
        self.apply_fake_quant_fn = apply_fake_quant_fn
        self.fake_quant_enabled = fake_quant_enabled
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"AffineFakeQuantizedTensor({self.original_tensor})"

    def __tensor_flatten__(self):
        return ["original_tensor"], [
            self.apply_fake_quant_fn,
            self.fake_quant_enabled,
            self.requires_grad,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride,
    ):
        original_tensor = tensor_data_dict["original_tensor"]
        (apply_fake_quant_fn, fake_quant_enabled, requires_grad) = tensor_attributes
        return cls(
            original_tensor,
            apply_fake_quant_fn,
            fake_quant_enabled,
            requires_grad,
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
    ):
        def apply_fake_quant_fn(t: AffineFakeQuantizedTensor):
            qmin, qmax = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
            scale, zero_point = choose_qparams_affine(
                t.original_tensor,
                mapping_type,
                block_size,
                target_dtype,
                qmin,
                qmax,
                eps,
                scale_dtype,
                zero_point_dtype,
                preserve_zero,
                zero_point_domain,
            )
            fq = _GenericFakeQuantize.apply(
                t,
                block_size,
                scale,
                zero_point,
                qmin,
                qmax,
                zero_point_domain,
            )
            return fq
        fake_quant_enabled = True
        requires_grad = True
        return cls(
            input_float,
            apply_fake_quant_fn,
            fake_quant_enabled,
            requires_grad,
        )

    def to_fake_quantized(self) -> torch.Tensor:
        return self.apply_fake_quant_fn(self).original_tensor

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
            self.original_tensor.to(device),
            self.apply_fake_quant_fn,
            self.fake_quant_enabled,
            self.requires_grad,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        """
        Create a new subclass with `fn` applied to the original tensor,
        to be called within __torch_dispatch__.
        """
        return self.__class__(
            fn(self.original_tensor),
            self.apply_fake_quant_fn,
            self.fake_quant_enabled,
        )

    implements = classmethod(_implements)
    __torch_function__ = classmethod(_dispatch__torch_function__)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)

implements = AffineFakeQuantizedTensor.implements


@implements(torch.nn.functional.linear)
def _(func, types, *args, **kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(input_tensor, AffineFakeQuantizedTensor):
        input_tensor = input_tensor.to_fake_quantized()
    if isinstance(weight_tensor, AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.to_fake_quantized()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

@implements([aten.mm.default, aten.addmm.default])
def _(func, types, *args, **kwargs):
    if func == aten.addmm.default:
        bias = args[0]
        input_index = 1
    else:
        bias = None
        input_index = 0
    input_tensor = args[input_index]
    weight_tensor = args[input_index + 1]
    if isinstance(input_tensor, AffineFakeQuantizedTensor):
        input_tensor = input_tensor.to_fake_quantized()
    if isinstance(weight_tensor, AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.to_fake_quantized()
    if bias is not None:
        return func(bias, input_tensor, weight_tensor)
    else:
        return func(input_tensor, weight_tensor)

@implements([aten.detach.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach),
    )

@implements([aten.clone.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone),
    )

@implements([aten._to_copy.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements([aten.t.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.t),
    )

# TODO: merge all these?
#@implements([aten.view.default])
#def _(func, types, *args, **kwargs):
#    fn = lambda x: x.view(*args[1:], **kwargs)
#    return return_and_correct_aliasing(
#        func, args, kwargs, args[0]._apply_fn_to_data(fn),
#    )
#
#@implements([aten.amin.default])
#def _(func, types, *args, **kwargs):
#    fn = lambda x: x.amin(*args[1:], **kwargs)
#    return return_and_correct_aliasing(
#        func, args, kwargs, args[0]._apply_fn_to_data(fn),
#    )
#
#@implements([aten.amax.default])
#def _(func, types, *args, **kwargs):
#    fn = lambda x: x.amax(*args[1:], **kwargs)
#    return return_and_correct_aliasing(
#        func, args, kwargs, args[0]._apply_fn_to_data(fn),
#    )
#
#@implements([aten.clamp.default])
#def _(func, types, *args, **kwargs):
#    clamp_fn = lambda x: x.clamp(*args[1:], **kwargs)
#    return return_and_correct_aliasing(
#        func, args, kwargs, args[0]._apply_fn_to_data(clamp_fn),
#    )
#
#@implements([aten.round.default])
#def _(func, types, *args, **kwargs):
#    return return_and_correct_aliasing(
#        func, args, kwargs, args[0]._apply_fn_to_data(torch.round),
#    )

#@implements([
#    aten.sub.Tensor,
#    aten.add.Tensor,
#    aten.add_.Tensor,
#    aten.div.Tensor,
#    aten.mul.Tensor,
#    aten.mul_.Tensor,
#    aten.ge.Scalar,
#    aten.le.Scalar,
#    aten.logical_and.default,
#])
#def _(func, types, *args, **kwargs):
#    assert len(args) == 2, f"dispatched the wrong op to the binary handler: {func}"
#    new_args = pytree.tree_map_only(AffineFakeQuantizedTensor, lambda x: x.original_tensor, args)
#    new_data = func(*new_args, **kwargs)
#    first_afqt = args[0] if isinstance(args[0], AffineFakeQuantizedTensor) else args[1]
#    out = AffineFakeQuantizedTensor(
#        new_data,
#        first_afqt.apply_fake_quant_fn,
#        first_afqt.fake_quant_enabled,
#    )
#    return return_and_correct_aliasing(func, args, kwargs, out)


to_affine_fake_quantized = AffineFakeQuantizedTensor.from_float
