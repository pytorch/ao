# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    _get_and_check_qmin_qmax,
    choose_qparams_affine,
    choose_qparams_affine_dont_preserve_zero,
    choose_qparams_affine_tiny_gemm,
)
from torchao.utils import TorchAOBaseTensor

from .utils import (
    _GenericFakeQuantize,
    _UnwrapAffineFakeQuantizedTensor,
)

aten = torch.ops.aten


class _ToAffineFakeQuantized(torch.autograd.Function):
    """
    Differentiable constructor for `AffineFakeQuantizedTensor`,
    needed for input activation fake quantization.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        original_tensor: torch.Tensor,
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
    ) -> "AffineFakeQuantizedTensor":
        if zero_point_domain is None:
            raise ValueError("Please use ZeroPointDomain.NONE instead of None")

        def apply_fake_quant_fn(t: torch.Tensor):
            assert isinstance(t, AffineFakeQuantizedTensor)
            qmin, qmax = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
            if zero_point_domain == ZeroPointDomain.FLOAT and not preserve_zero:
                scale, zero_point = choose_qparams_affine_tiny_gemm(
                    t.original_tensor,
                    mapping_type,
                    block_size,
                    target_dtype,
                    qmin,
                    qmax,
                    eps,
                    scale_dtype,
                    zero_point_dtype,
                )
            elif zero_point_domain == ZeroPointDomain.INT and not preserve_zero:
                scale, zero_point = choose_qparams_affine_dont_preserve_zero(
                    t.original_tensor,
                    mapping_type,
                    block_size,
                    target_dtype,
                    qmin,
                    qmax,
                    eps,
                    scale_dtype,
                    zero_point_dtype,
                )
            else:  # Default case: zero_point_domain == ZeroPointDomain.INT and preserve_zero
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

        return AffineFakeQuantizedTensor(
            original_tensor,
            apply_fake_quant_fn,
            fake_quant_enabled=True,
        )

    @staticmethod
    def backward(ctx, gy):
        return gy, None, None, None, None, None, None, None, None, None, None


class AffineFakeQuantizedTensor(TorchAOBaseTensor):
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
        **kwargs,
    ):
        kwargs.setdefault("dtype", original_tensor.dtype)
        kwargs.setdefault("device", original_tensor.device)
        kwargs.setdefault("requires_grad", original_tensor.requires_grad)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            original_tensor.shape,
            **kwargs,
        )

    def __init__(
        self,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool = True,
        **kwargs,
    ):
        self.original_tensor = original_tensor
        self.apply_fake_quant_fn = apply_fake_quant_fn
        self.fake_quant_enabled = fake_quant_enabled

    def __tensor_flatten__(self):
        return ["original_tensor"], [self.apply_fake_quant_fn, self.fake_quant_enabled]

    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict,
        tensor_attributes,
        outer_size,
        outer_stride,
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
        original_input: torch.Tensor,
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
    ):
        if zero_point_domain is None:
            raise ValueError("Please use ZeroPointDomain.NONE instead of None")
        return _ToAffineFakeQuantized.apply(
            original_input,
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

    def get_value(self) -> torch.Tensor:
        if self.fake_quant_enabled:
            return self.apply_fake_quant_fn(self)
        else:
            return _UnwrapAffineFakeQuantizedTensor.apply(self)

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
            "requires_grad": self.requires_grad,
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

    def _apply_fn_to_data(self, fn: Callable):
        """
        Create a new `AffineFakeQuantizedTensor` with `fn` applied to the
        original tensor, to be called within __torch_dispatch__.
        """
        return self._create_new(fn(self.original_tensor))

    def _create_new(self, new_value: torch.Tensor):
        """
        Create a new `AffineFakeQuantizedTensor` with a new value,
        to be called within __torch_dispatch__.

        Note: `requires_grad` must be False here because tensors created
        in `__torch_dispatch__` cannot produce gradients, since autograd
        will try to attach autograd metadata to these tensors when we exit
        `__torch_dispatch__`, but if these tensors already have metadata
        attached then autograd will throw an error.
        """
        return self.__class__(
            new_value,
            self.apply_fake_quant_fn,
            self.fake_quant_enabled,
            requires_grad=False,
        )


implements = AffineFakeQuantizedTensor.implements


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(input_tensor, AffineFakeQuantizedTensor):
        input_tensor = input_tensor.get_value()
    if isinstance(weight_tensor, AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.get_value()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements(aten.mm.default)
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    if isinstance(input_tensor, AffineFakeQuantizedTensor):
        input_tensor = input_tensor.get_value()
    if isinstance(weight_tensor, AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.get_value()
    return func(input_tensor, weight_tensor)


@implements(aten.addmm.default)
def _(func, types, args, kwargs):
    bias = args[0]
    input_tensor = args[1]
    weight_tensor = args[2]
    if isinstance(input_tensor, AffineFakeQuantizedTensor):
        input_tensor = input_tensor.get_value()
    if isinstance(weight_tensor, AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.get_value()
    return func(bias, input_tensor, weight_tensor)


@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0]._apply_fn_to_data(torch.detach),
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0]._apply_fn_to_data(torch.clone),
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0]._apply_fn_to_data(torch.t),
    )


@implements(
    [
        aten.add.Tensor,
        aten.add_.Tensor,
        aten.mul_.Tensor,
        aten.copy_.default,
    ]
)
def _(func, types, args, kwargs):
    assert len(args) == 2, f"dispatched the wrong op to the binary handler: {func}"
    new_args = pytree.tree_map_only(
        AffineFakeQuantizedTensor, lambda x: x.original_tensor, args
    )
    first_afq_tensor = (
        args[0] if isinstance(args[0], AffineFakeQuantizedTensor) else args[1]
    )
    new_value = func(*new_args, **kwargs)
    out = first_afq_tensor._create_new(new_value)
    return return_and_correct_aliasing(func, args, kwargs, out)


# Needed by FSDP:


@implements(aten.empty_like.default)
def _(func, types, args, kwargs):
    out = torch.empty_like(args[0].original_tensor, **kwargs)
    return return_and_correct_aliasing(func, args, kwargs, out)


@implements(aten.split.Tensor)
def _(func, types, args, kwargs):
    new_values = torch.split(args[0].original_tensor, *args[1:], **kwargs)

    def make_new_tensor(value):
        out = args[0]._create_new(value)
        return return_and_correct_aliasing(func, args, kwargs, out)

    return list(map(make_new_tensor, new_values))


@implements(aten.new_zeros.default)
def _(func, types, args, kwargs):
    out = args[0].original_tensor.new_zeros(*args[1:], **kwargs)
    return return_and_correct_aliasing(func, args, kwargs, out)


to_affine_fake_quantized = AffineFakeQuantizedTensor.from_float
