# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
    fill_defaults,
)

__all__ = [
    "IntxTilePackedTensor",
]

aten = torch.ops.aten

_FLOAT_TYPES: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]

 bit_width: Optional[int]
    group_size: Optional[int]
    has_weight_zeros: Optional[bool]
    has_bias: Optional[bool]
    target: Optional[Target]


class IntxTilePackedTensor(TorchAOBaseTensor):
    """
    intx quantization with unpacked format.  Subbyte quantized data is represented as int8.
    Quantization is represented in a decomposed way.
    This format is inteded for torch.export use cases.

    Tensor Attributes:
        _data: int data for
        scale: (K/group_size, N) for 2D Tensor, (B, N, K/group_size) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype
        zero_point: (K/group_size, N) for 2D Tensor, (B, N, K/group_size) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity, for example groupwise quantization will have block_size (1, group_size)
        shape: the shape of the original Tensor
    """

    tensor_data_attrs = ["packed_data"]
    tensor_attributes = ["bit_width", "block_size", "shape", "dtype", "has_weight_zeros", "has_bias", "target"]

    def __new__(cls, packed_data, bit_width, block_size, shape, dtype, has_weight_zeros, has_bias, target):
        kwargs = {}
        kwargs["device"] = packed_data.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self, packed_data, bit_width, block_size, shape, dtype, has_weight_zeros, has_bias, target
    ):

        assert packed_data.device == torch.device("cpu")
        self.packed_data = packed_data
        self.bit_width = bit_width
        self.block_size = block_size
        self.has_weight_zeros = has_weight_zeros
        self.has_bias = has_bias
        self.target = target


    def __tensor_flatten__(self):
        return self.tensor_data_attrs, [
            getattr(self, attr) for attr in self.tensor_attributes
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_data_attrs],
            *tensor_attributes,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            *[fn(getattr(self, attr)) for attr in self.tensor_data_attrs],
            *[getattr(self, attr) for attr in self.tensor_attributes],
        )

    def __repr__(self):
        repr_fields = (
            self.tensor_data_attrs
            + self.tensor_attributes
            + ["shape", "device", "dtype", "require_grad"]
        )
        inner_repr = [f"{attr}={getattr(self, attr)}" for attr in repr_fields]
        inner_repr = ", ".join(inner_repr)
        return f"{self.__class__.__name__}({inner_repr}))"

    def _quantization_type(self):
        return f"bit_width={self.bit_width}, block_size={self.block_size}, shape={self.shape}, dtype={self.dtype}, device={self.device}"

    def to(self, *args, **kwargs):
        raise NotImplementedError("to() is not implemented for IntxTilePackedTensor")
    
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int],
        bit_width: int,
        bias: Optional[torch.Tensor] = None):


        zero_point.dtype


        zero_point_min = zero_point.min().item()
        zero_point_max = zero_point.max().item()
        assert zero_point.min().item() >= qmin
        assert zero_point.max().item() <= qmax
        has_weight_zeros = True





     if layout.target != Target.ATEN:
            _check_torchao_ops_loaded()
        else:
            assert TORCH_VERSION_AT_LEAST_2_6, (
                "aten target is requires torch version > 2.6.0"
            )
            assert torch.backends.kleidiai.is_available(), (
                "ATEN target requires torch.backends.kleidiai.is_available()"
            )
            layout.bit_width == 4, "ATEN target only supports torch.int4"
            assert not layout.has_weight_zeros, "ATEN target does not support zeros"

        data_dtype = getattr(torch, f"int{layout.bit_width}")
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[data_dtype]

        int_types = [torch.int8, torch.int16, torch.int32, torch.int64]

        # Check int_data
        assert int_data.device == torch.device("cpu")
        assert int_data.dtype in int_types
        n, k = int_data.shape
        assert k % layout.group_size == 0, "k must be divisible by group_size"
        if validate_inputs:
            assert int_data.min().item() >= qmin
            assert int_data.max().item() <= qmax
        int_data = int_data.to(torch.int8)

        # Check scale
        assert scale.device == torch.device("cpu")
        if scale.dtype != torch.float32:
            logging.info(f"scale has dtype {scale.dtype}, converting to torch.float32")
            scale = scale.to(torch.float32)
        n_, _ = scale.shape
        assert n_ == n
        assert scale.numel() * layout.group_size == int_data.numel(), (
            "must have 1 scale per group"
        )
        if validate_inputs:
            assert scale.min().item() > 0
            # Some targets round scales to bfloat16, give warning if scales are at higher precision
            scale_is_rounded_to_bf16 = torch.allclose(
                scale, scale.to(torch.bfloat16).to(torch.float32)
            )
            if not scale_is_rounded_to_bf16:
                if layout.target == Target.ATEN and (layout.group_size < k):
                    logging.warning(
                        "When using Target.ATEN with group_size < k, scales will be rounded to bfloat16"
                    )
                if layout.target in [Target.AUTO, Target.KLEIDIAI]:
                    logging.warning(
                        "When using [Target.AUTO, Target.KLEIDIAI], scales will be rounded to bfloat16"
                    )

        # Check zero_point
        if zero_point is None:
            assert not layout.has_weight_zeros, (
                "zero_point must be provided if has_weight_zeros=True"
            )
        else:
            assert zero_point.device == torch.device("cpu")
            assert zero_point.shape == scale.shape
            assert zero_point.dtype in int_types
            assert zero_point.numel() * layout.group_size == int_data.numel(), (
                "must have 1 zero_point per group"
            )
            if validate_inputs:
                zero_point_min = zero_point.min().item()
                zero_point_max = zero_point.max().item()
                assert zero_point.min().item() >= qmin
                assert zero_point.max().item() <= qmax
                has_weight_zeros = True
                if zero_point_min == 0 and zero_point_max == 0:
                    has_weight_zeros = False
                assert has_weight_zeros == layout.has_weight_zeros, (
                    "zero_point being all zeros must be consistent with layout.has_weight_zeros"
                )
            zero_point = zero_point.to(torch.int8)

        # Check bias
        has_bias = bias is not None
        assert has_bias == layout.has_bias, (
            "bias being None must be consistent with layout.has_bias"
        )
        if has_bias:
            assert bias.device == torch.device("cpu")
            if bias.dtype != torch.float32:
                logging.info(
                    f"bias has dtype {bias.dtype}, converting to torch.float32"
                )
                bias = bias.to(torch.float32)
            assert bias.shape == (n,)

        # Construct packed_weight
        if layout.target == Target.ATEN:
            int_data = int_data.add(8)
            int_data = (int_data[::, 1::2] << 4 | int_data[::, ::2]).to(torch.uint8)

            # If group_size < k, convert scales to bfloat16
            # to call optimized kernel
            if layout.group_size < k:
                scale = scale.to(torch.bfloat16)
            packed_weight = torch.ops.aten._dyn_quant_pack_4bit_weight(
                int_data, scale, bias, layout.group_size, k, n
            )
            return cls(packed_weight, layout)

        args = [
            int_data,
            scale.reshape(-1),
            zero_point.reshape(-1) if layout.has_weight_zeros else None,
            layout.group_size,
            bias,
            target_to_str(layout.target) if layout.target != Target.AUTO else None,
        ]
        packed_weight = getattr(
            torch.ops.torchao,
            f"_pack_8bit_act_{layout.bit_width}bit_weight",
        )(*args)



    

    def get_plain(self):
        return self.int_data, self.scale, self.zero_point

    def dequantize(self):
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[getattr(torch, f"int{self.bit_width}")]
        return dequantize_affine(
            self.int_data,
            self.block_size,
            self.scale,
            self.zero_point,
            torch.int8,
            qmin,
            qmax,
            output_dtype=self.dtype,
        )


implements = IntxUnpackedTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


IntxUnpackedTensor.__module__ = "torchao.quantization"

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with IntxUnpackedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([IntxUnpackedTensor])
