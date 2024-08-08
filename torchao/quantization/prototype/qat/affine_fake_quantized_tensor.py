import torch
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


class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool,
    ) -> "AffineFakeQuantizedTensor":
        return AffineFakeQuantizedTensor(
            original_tensor,
            apply_fake_quant_fn,
            fake_quant_enabled,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _ToTorchTensor.apply(grad_output), None, None

class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fq_tensor: "AffineFakeQuantizedTensor") -> torch.Tensor:
        ctx.apply_fake_quant_fn = fq_tensor.apply_fake_quant_fn
        ctx.fake_quant_enabled = fq_tensor.fake_quant_enabled
        return fq_tensor.original_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> "AffineFakeQuantizedTensor":
        apply_fake_quant_fn = ctx.apply_fake_quant_fn
        fake_quant_enabled = ctx.fake_quant_enabled
        return AffineFakeQuantizedTensor(
            grad_output,
            apply_fake_quant_fn,
            fake_quant_enabled,
        )

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
    ):
        kwargs = {}
        kwargs["device"] = original_tensor.device
        kwargs["dtype"] = original_tensor.dtype
        kwargs["requires_grad"] = True
        return torch.Tensor._make_wrapper_subclass(cls, original_tensor.shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool = True,
    ):
        # TODO: original_tensor is not getting updated!
        original_tensor.requires_grad_(True)
        self.original_tensor = original_tensor
        self.apply_fake_quant_fn = apply_fake_quant_fn
        self.fake_quant_enabled = fake_quant_enabled

    def __tensor_flatten__(self):
        return ["original_tensor"], [self.apply_fake_quant_fn, self.fake_quant_enabled]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride,
    ):
        original_tensor = tensor_data_dict["original_tensor"]
        (apply_fake_quant_fn, fake_quant_enabled) = tensor_attributes
        return cls(
            original_tensor,
            apply_fake_quant_fn,
            fake_quant_enabled,
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
        def apply_fake_quant_fn(t: torch.Tensor):
            qmin, qmax = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
            scale, zero_point = choose_qparams_affine(
                t,
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
        return _FromTorchTensor.apply(
            input_float,
            apply_fake_quant_fn,
            fake_quant_enabled,
        )

    def to_fake_quantized(self) -> torch.Tensor:
        return self.apply_fake_quant_fn(self.original_tensor)

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
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
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
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )

@implements([aten.clone.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
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
        func, args, kwargs, args[0]._apply_fn_to_data(torch.t)
    )

to_affine_fake_quantized = AffineFakeQuantizedTensor.from_float
