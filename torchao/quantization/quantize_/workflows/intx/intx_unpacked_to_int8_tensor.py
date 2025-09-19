# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import enum
from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.quantization.utils import _get_per_token_block_size
from torchao.utils import (
    TorchAOBaseTensor,
    fill_defaults,
)

__all__ = [
    "IntxUnpackedToInt8Tensor",
]

aten = torch.ops.aten

_FLOAT_TYPES: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]


class IntxUnpackedToInt8TensorActivationQuantization(str, enum.Enum):
    """
    This applies int8 asymmetric activation quantization per token.
    """

    INT8_ASYM_PER_TOKEN = "int8_asym_per_token"


class IntxUnpackedToInt8Tensor(TorchAOBaseTensor):
    """
    intx quantization with unpacked format.  Subbyte quantized data is represented as int8.
    The range of the quantized values are restricted to the quant_min and quant_max of the target_dtype, e.g.,
    if target_dtype=torch.int4, qdata will be an int8 tensor with values in [-8, 7].
    Quantization is represented in a decomposed way.
    This format is inteded for torch.export use cases.

    Tensor Attributes:
        qdata: int data for quantization.
                dtype is int8, but the range of the qdata is determined by target_dtype
                Shape is the same as original Tensor: (n, k) for 2D tensor
        scale: block scales for quantization
               dtype is the same as the original Tensor dtype.
               Shape is (n // block_size[0], k // block_size[1]) for 2D tensor
        zero_point: block zero points for quantization
               dtype is the same as the original Tensor dtype or int8
               Shape is (n // block_size[0], k // block_size[1]) for 2D tensor

    Non-Tensor Attributes:
        target_dtype: this determines the quant_min/quant_max of the qdata (can be torch.int1, ..., torch.int8)
        block_size: the block size for quantization, representing the granularity, for example groupwise quantization will have block_size (1, group_size)
        dtype: the dtype of the dequantized Tensor
        activation_quantization: Optional[IntxUnpackedToInt8TensorActivationQuantization] = None, kind of activation quantization to apply.  Default is None, which means weight-only quantization
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = [
        "target_dtype",
        "block_size",
        "dtype",
        "activation_quantization",
    ]

    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        target_dtype,
        block_size,
        dtype,
        activation_quantization,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        shape = qdata.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata,
        scale,
        zero_point,
        target_dtype,
        block_size,
        dtype,
        activation_quantization,
    ):
        super().__init__()
        assert qdata.dtype == torch.int8, (
            f"qdata dtype must be int8, but got {qdata.dtype}"
        )
        assert scale.dtype in _FLOAT_TYPES, (
            f"scale dtype must be one of {_FLOAT_TYPES}, but got {scale.dtype}"
        )
        assert zero_point.dtype in _FLOAT_TYPES or zero_point.dtype == torch.int8, (
            f"zero_point dtype must be {torch.int8} or one of {_FLOAT_TYPES}, but got {zero_point.dtype}"
        )

        assert target_dtype in [
            getattr(torch, f"int{bit_width}") for bit_width in range(1, 9)
        ]

        assert len(block_size) == qdata.ndim
        n_blocks = []
        for i in range(len(block_size)):
            assert qdata.shape[i] % block_size[i] == 0
            n_blocks.append(qdata.shape[i] // block_size[i])

        # Assert shapes
        assert scale.shape == tuple(n_blocks), (
            f"Expected scale to have shape {n_blocks} (inferred from block_size={block_size}), but got {scale.shape}"
        )
        assert zero_point.shape == tuple(n_blocks), (
            f"Expected zero_point to have shape {n_blocks} (inferred from block_size={block_size}), but got {zero_point.shape}"
        )

        assert dtype in _FLOAT_TYPES, (
            f"dtype must be one of {_FLOAT_TYPES}, but got {dtype}"
        )

        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point

        self.target_dtype = target_dtype
        self.block_size = block_size
        self.activation_quantization = activation_quantization

    def _quantization_type(self):
        return f"target_dtype={self.target_dtype}, block_size={self.block_size}, shape={self.shape}, dtype={self.dtype}, device={self.device}, activation_quantization={self.activation_quantization}"

    def _has_float_zero_point(self) -> bool:
        return self.zero_point.dtype in _FLOAT_TYPES

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        dtype = kwargs.pop("dtype")
        assert dtype in _FLOAT_TYPES
        return IntxUnpackedToInt8Tensor(
            self.qdata.to(device),
            self.scale.to(device=device, dtype=dtype),
            self.zero_point.to(device=device, dtype=dtype)
            if self._has_float_zero_point()
            else self.zero_point.to(device),
            self.target_dtype,
            self.block_size,
            dtype,
            self.activation_quantization,
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        block_size: Tuple[int],
        target_dtype: torch.dtype,
        *,
        mapping_type: MappingType = MappingType.SYMMETRIC,
        activation_quantization: Optional[
            IntxUnpackedToInt8TensorActivationQuantization
        ] = None,
        custom_scale: Optional[torch.Tensor] = None,
        custom_zero_point: Optional[torch.Tensor] = None,
    ):
        """
        Create an IntxUnpackedToInt8Tensor from a high-precision tensor
        """
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[target_dtype]
        if custom_scale is not None and custom_zero_point is not None:
            scale, zero_point = custom_scale, custom_zero_point
        elif custom_scale is None and custom_zero_point is None:
            scale, zero_point = choose_qparams_affine(
                hp_tensor,
                mapping_type,
                block_size,
                target_dtype=torch.int8,
                quant_min=qmin,
                quant_max=qmax,
                zero_point_dtype=torch.int8,
            )
        else:
            raise ValueError(
                "`custom_scale` and `custom_zero_point` must be both defined or both None"
            )
        qdata = quantize_affine(
            hp_tensor,
            block_size,
            scale,
            zero_point,
            output_dtype=torch.int8,
            quant_min=qmin,
            quant_max=qmax,
        )

        # Reshape scale and zero_point to be compatible with block_size
        # This is asserted in IntxUnpackedToInt8Tensor's __init__
        n_blocks = []
        for i in range(len(block_size)):
            assert qdata.shape[i] % block_size[i] == 0
            n_blocks.append(qdata.shape[i] // block_size[i])
        scale = scale.reshape(*n_blocks)
        zero_point = zero_point.reshape(*n_blocks)

        return IntxUnpackedToInt8Tensor(
            qdata=qdata,
            scale=scale,
            zero_point=zero_point,
            target_dtype=target_dtype,
            block_size=block_size,
            dtype=hp_tensor.dtype,
            activation_quantization=activation_quantization,
        )

    def dequantize(self):
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.target_dtype]
        return dequantize_affine(
            self.qdata,
            self.block_size,
            self.scale,
            self.zero_point,
            torch.int8,
            qmin,
            qmax,
            output_dtype=self.dtype,
        )


def _apply_int8_act_asym_per_token_quant_dequant(hp_tensor):
    target_dtype = torch.int8
    mapping_type = MappingType.ASYMMETRIC
    block_size = _get_per_token_block_size(hp_tensor)
    qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[target_dtype]
    scale, zero_point = choose_qparams_affine(
        hp_tensor,
        mapping_type,
        block_size,
        target_dtype=target_dtype,
        quant_min=qmin,
        quant_max=qmax,
        zero_point_dtype=torch.int8,
    )
    qdata = quantize_affine(
        hp_tensor,
        block_size,
        scale,
        zero_point,
        output_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
    )
    dequantized_affine = dequantize_affine(
        qdata,
        block_size,
        scale,
        zero_point,
        torch.int8,
        qmin,
        qmax,
        output_dtype=hp_tensor.dtype,
    )
    return dequantized_affine


implements = IntxUnpackedToInt8Tensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert isinstance(weight_tensor, IntxUnpackedToInt8Tensor)

    # Apply dynamic activation quant
    if weight_tensor.activation_quantization is not None:
        if (
            weight_tensor.activation_quantization
            == IntxUnpackedToInt8TensorActivationQuantization.INT8_ASYM_PER_TOKEN
        ):
            input_tensor = _apply_int8_act_asym_per_token_quant_dequant(input_tensor)
        else:
            raise NotImplementedError(
                f"Unsupported activation quantization: {weight_tensor.activation_quantization}"
            )

    weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements([torch.nn.functional.embedding, aten.embedding.default])
def _(func, types, args, kwargs):
    assert len(args) == 2
    indices, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.embedding(indices, weight_tensor, **kwargs)


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1

    # Slicing must be compatible with the block size to make sense on the quantized tensor
    # In particular both start and end must be a multiple of block_size[dim]
    # Otherwise the sliced tensor cannot be represented as a IntxUnpackedToInt8Tensor
    # For example, if block_size = 4, we might have:
    #
    # qdata: i i i i | i i i i
    #    scale: s s
    #
    # If we set start = 2 and end = 8, then the qdata slice is:
    #
    # qdata_slice: i i (i i | i i i i)
    #
    # But then the block_size for the first two qdata in the slice is 2
    # and remaining blocks have size 4.  This cannot be represented
    # with the metadata we store in an IntxUnpackedToInt8Tensor, which requires uniform blocking

    assert start % self.block_size[dim] == 0, (
        f"slice args are incompatible with blocking: start={start} must be divisible by block_size[dim]={self.block_size[dim]}"
    )
    start_scale = start // self.block_size[dim]

    assert end % self.block_size[dim] == 0, (
        f"slice args are incompatible with blocking: end={end} must be divisible by block_size[dim]={self.block_size[dim]}"
    )
    end_scale = end // self.block_size[dim]

    qdata = aten.slice.Tensor(self.qdata, dim, start, end, step)
    scale = aten.slice.Tensor(self.scale, dim, start_scale, end_scale, step)
    zero_point = aten.slice.Tensor(self.zero_point, dim, start_scale, end_scale, step)

    new_block_size = []
    for i in range(qdata.ndim):
        assert scale.shape[i] == zero_point.shape[i]
        n_blocks = scale.shape[i]
        assert qdata.shape[i] % n_blocks == 0
        new_block_size.append(qdata.shape[i] // n_blocks)
    new_block_size = tuple(new_block_size)

    new = IntxUnpackedToInt8Tensor(
        qdata,
        scale,
        zero_point,
        self.target_dtype,
        new_block_size,
        self.dtype,
        self.activation_quantization,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


IntxUnpackedToInt8Tensor.__module__ = "torchao.quantization"

# Allow a model with IntxUnpackedToInt8Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals(
    [IntxUnpackedToInt8Tensor, IntxUnpackedToInt8TensorActivationQuantization]
)
