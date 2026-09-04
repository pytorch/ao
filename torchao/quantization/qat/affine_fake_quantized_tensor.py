# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    _choose_qparams_affine_dont_preserve_zero,
    _choose_qparams_affine_tinygemm,
    _dequantize_affine_no_dtype_check,
    _dequantize_affine_no_zero_point_no_dtype_check,
    _dequantize_affine_tinygemm_no_dtype_check,
    _fake_quantize_affine,
    _get_and_check_qmin_qmax,
    _quantize_affine_no_dtype_cast,
    _quantize_affine_no_zero_point_no_dtype_cast,
    _quantize_affine_tinygemm_no_dtype_cast,
    choose_qparams_affine,
)
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten


@dataclass(frozen=True)
class _AffineFakeQuantizedConfig:
    mapping_type: MappingType
    block_size: Tuple[int, ...]
    target_dtype: torch.dtype
    quant_min: Optional[int]
    quant_max: Optional[int]
    eps: Optional[float]
    scale_dtype: Optional[torch.dtype]
    zero_point_dtype: Optional[torch.dtype]
    preserve_zero: bool
    zero_point_domain: ZeroPointDomain
    is_per_tensor: bool


def _get_effective_block_size(
    config: _AffineFakeQuantizedConfig, input_tensor: torch.Tensor
) -> Tuple[int, ...]:
    return tuple(input_tensor.shape) if config.is_per_tensor else config.block_size


def _choose_qparams_for_config(
    input_tensor: torch.Tensor, config: _AffineFakeQuantizedConfig
) -> Tuple[Tuple[int, ...], torch.Tensor, Optional[torch.Tensor], int, int]:
    block_size = _get_effective_block_size(config, input_tensor)
    quant_min, quant_max = _get_and_check_qmin_qmax(
        config.target_dtype, config.quant_min, config.quant_max
    )

    if config.zero_point_domain == ZeroPointDomain.FLOAT and not config.preserve_zero:
        scale, zero_point = _choose_qparams_affine_tinygemm(
            input_tensor,
            config.mapping_type,
            block_size,
            config.target_dtype,
            quant_min,
            quant_max,
            config.eps,
            config.scale_dtype,
            config.zero_point_dtype,
        )
    elif config.zero_point_domain == ZeroPointDomain.INT and not config.preserve_zero:
        scale, zero_point = _choose_qparams_affine_dont_preserve_zero(
            input_tensor,
            config.mapping_type,
            block_size,
            config.target_dtype,
            quant_min,
            quant_max,
            config.eps,
            config.scale_dtype,
            config.zero_point_dtype,
        )
    else:
        scale, zero_point = choose_qparams_affine(
            input_tensor,
            config.mapping_type,
            block_size,
            config.target_dtype,
            quant_min,
            quant_max,
            config.eps,
            config.scale_dtype,
            config.zero_point_dtype,
        )
        if config.zero_point_domain == ZeroPointDomain.NONE:
            zero_point = None

    return block_size, scale, zero_point, quant_min, quant_max


def _get_quantized_comm_dtype(quant_min: int, quant_max: int) -> torch.dtype:
    if quant_min >= 0 and quant_max <= torch.iinfo(torch.uint8).max:
        return torch.uint8
    if (
        quant_min >= torch.iinfo(torch.int8).min
        and quant_max <= torch.iinfo(torch.int8).max
    ):
        return torch.int8
    raise RuntimeError(
        f"Expected 8-bit quantization range, but got quant_min={quant_min}, quant_max={quant_max}"
    )


def _quantize_for_fsdp_all_gather(
    input_tensor: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: int,
    quant_max: int,
    zero_point_domain: ZeroPointDomain,
) -> torch.Tensor:
    comm_dtype = _get_quantized_comm_dtype(quant_min, quant_max)
    if zero_point_domain == ZeroPointDomain.INT:
        quantized = _quantize_affine_no_dtype_cast(
            input_tensor, block_size, scale, zero_point, quant_min, quant_max
        )
    elif zero_point_domain == ZeroPointDomain.FLOAT:
        quantized = _quantize_affine_tinygemm_no_dtype_cast(
            input_tensor, block_size, scale, zero_point, quant_min, quant_max
        )
    elif zero_point_domain == ZeroPointDomain.NONE:
        quantized = _quantize_affine_no_zero_point_no_dtype_cast(
            input_tensor, block_size, scale, None, quant_min, quant_max
        )
    else:
        raise ValueError(f"Unrecognized zero_point_domain: {zero_point_domain}")
    return quantized.to(comm_dtype)


def _dequantize_from_fsdp_all_gather(
    quantized_tensor: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: int,
    quant_max: int,
    zero_point_domain: ZeroPointDomain,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if zero_point_domain == ZeroPointDomain.INT:
        return _dequantize_affine_no_dtype_check(
            quantized_tensor,
            block_size,
            scale,
            zero_point,
            quant_min,
            quant_max,
            output_dtype,
        )
    if zero_point_domain == ZeroPointDomain.FLOAT:
        return _dequantize_affine_tinygemm_no_dtype_check(
            quantized_tensor,
            block_size,
            scale,
            zero_point,
            quant_min,
            quant_max,
            output_dtype,
        )
    if zero_point_domain == ZeroPointDomain.NONE:
        return _dequantize_affine_no_zero_point_no_dtype_check(
            quantized_tensor,
            block_size,
            scale,
            None,
            quant_min,
            quant_max,
            output_dtype,
        )
    raise ValueError(f"Unrecognized zero_point_domain: {zero_point_domain}")


class _ToAffineFakeQuantized(torch.autograd.Function):
    """
    Differentiable constructor for `_AffineFakeQuantizedTensor`,
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
    ) -> "_AffineFakeQuantizedTensor":
        if zero_point_domain is None:
            raise ValueError("Please use ZeroPointDomain.NONE instead of None")

        quantization_config = _AffineFakeQuantizedConfig(
            mapping_type=mapping_type,
            block_size=block_size,
            target_dtype=target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            scale_dtype=scale_dtype,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            is_per_tensor=tuple(block_size) == tuple(original_tensor.shape),
        )

        def apply_fake_quant_fn(t: torch.Tensor):
            assert isinstance(t, _AffineFakeQuantizedTensor)
            config = (
                t.quantization_config
                if t.quantization_config is not None
                else quantization_config
            )
            (
                effective_block_size,
                scale,
                zero_point,
                qmin,
                qmax,
            ) = _choose_qparams_for_config(t.original_tensor, config)
            fq = _fake_quantize_affine(
                t.original_tensor,
                effective_block_size,
                scale,
                zero_point,
                quant_dtype=torch.int32,
                quant_min=qmin,
                quant_max=qmax,
                zero_point_domain=config.zero_point_domain,
            )
            return fq

        return _AffineFakeQuantizedTensor(
            original_tensor,
            apply_fake_quant_fn,
            fake_quant_enabled=True,
            quantization_config=quantization_config,
        )

    @staticmethod
    def backward(ctx, gy):
        return gy, None, None, None, None, None, None, None, None, None, None


class _AffineFakeQuantizedTensor(TorchAOBaseTensor):
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
      quantization_config (Optional[_AffineFakeQuantizedConfig]): quantization metadata used for
          reproducing fake quant numerics and for FSDP low-bit all-gather hooks
    """

    @staticmethod
    def __new__(
        cls,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool = True,
        quantization_config: Optional[_AffineFakeQuantizedConfig] = None,
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
        quantization_config: Optional[_AffineFakeQuantizedConfig] = None,
        **kwargs,
    ):
        self.original_tensor = original_tensor
        self.apply_fake_quant_fn = apply_fake_quant_fn
        self.fake_quant_enabled = fake_quant_enabled
        self.quantization_config = quantization_config

    def __tensor_flatten__(self):
        return ["original_tensor"], [
            self.apply_fake_quant_fn,
            self.fake_quant_enabled,
            self.quantization_config,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict,
        tensor_attributes,
        outer_size,
        outer_stride,
    ):
        original_tensor = tensor_data_dict["original_tensor"]
        if len(tensor_attributes) == 2:
            (apply_fake_quant_fn, fake_quant_enabled) = tensor_attributes
            quantization_config = None
        else:
            (apply_fake_quant_fn, fake_quant_enabled, quantization_config) = (
                tensor_attributes
            )
        return cls(
            original_tensor,
            apply_fake_quant_fn,
            fake_quant_enabled,
            quantization_config,
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
            return self.original_tensor

    def _require_quantization_config(self) -> _AffineFakeQuantizedConfig:
        if self.quantization_config is None:
            raise RuntimeError(
                "Missing quantization metadata on _AffineFakeQuantizedTensor. "
                "Please construct this tensor via `_to_affine_fake_quantized`."
            )
        return self.quantization_config

    # FSDP all-gather extension v2
    # https://github.com/pytorch/pytorch/pull/137005
    # default values keep compatibility with older torch versions
    def fsdp_pre_all_gather(
        self,
        mesh,
        outer_size=None,
        outer_stride=None,
        module=None,
        mp_policy=None,
    ):
        input_tensor = self.original_tensor
        if mp_policy is not None and mp_policy.param_dtype is not None:
            input_tensor = input_tensor.to(mp_policy.param_dtype)
        config = self._require_quantization_config()
        block_size, scale, zero_point, quant_min, quant_max = _choose_qparams_for_config(
            input_tensor, config
        )
        quantized_tensor = _quantize_for_fsdp_all_gather(
            input_tensor,
            block_size,
            scale,
            zero_point,
            quant_min,
            quant_max,
            config.zero_point_domain,
        )
        metadata = (
            scale,
            zero_point,
            block_size,
            quant_min,
            quant_max,
            config.zero_point_domain,
        )
        return (quantized_tensor,), metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (quantized_tensor,) = all_gather_outputs
        (
            scale,
            zero_point,
            block_size,
            quant_min,
            quant_max,
            zero_point_domain,
        ) = metadata

        dequantized = _dequantize_from_fsdp_all_gather(
            quantized_tensor,
            block_size,
            scale,
            zero_point,
            quant_min,
            quant_max,
            zero_point_domain,
            param_dtype,
        )

        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, DTensor):
                out._local_tensor.copy_(dequantized)
            elif isinstance(out, _AffineFakeQuantizedTensor):
                out.original_tensor.copy_(dequantized)
            else:
                out.copy_(dequantized)
            return
        return dequantized, (dequantized,)

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
            self.quantization_config,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn: Callable):
        """
        Create a new `_AffineFakeQuantizedTensor` with `fn` applied to the
        original tensor, to be called within __torch_dispatch__.
        """
        return self._create_new(fn(self.original_tensor))

    def _create_new(self, new_value: torch.Tensor):
        """
        Create a new `_AffineFakeQuantizedTensor` with a new value,
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
            self.quantization_config,
            requires_grad=False,
        )


implements = _AffineFakeQuantizedTensor.implements
implements_torch_function = _AffineFakeQuantizedTensor.implements_torch_function


@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(input_tensor, _AffineFakeQuantizedTensor):
        input_tensor = input_tensor.get_value()
    if isinstance(weight_tensor, _AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.get_value()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements(aten.mm.default)
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    if isinstance(input_tensor, _AffineFakeQuantizedTensor):
        input_tensor = input_tensor.get_value()
    if isinstance(weight_tensor, _AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.get_value()
    return func(input_tensor, weight_tensor)


@implements(aten.addmm.default)
def _(func, types, args, kwargs):
    bias = args[0]
    input_tensor = args[1]
    weight_tensor = args[2]
    if isinstance(input_tensor, _AffineFakeQuantizedTensor):
        input_tensor = input_tensor.get_value()
    if isinstance(weight_tensor, _AffineFakeQuantizedTensor):
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
        _AffineFakeQuantizedTensor, lambda x: x.original_tensor, args
    )
    first_afq_tensor = (
        args[0] if isinstance(args[0], _AffineFakeQuantizedTensor) else args[1]
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


_to_affine_fake_quantized = _AffineFakeQuantizedTensor.from_float
