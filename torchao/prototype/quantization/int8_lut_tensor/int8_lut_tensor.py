# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
)
from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
    _is_kernel_library_loaded,
)
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    IntxUnpackedToInt8Tensor,
    IntxUnpackedToInt8TensorActivationQuantization,
)
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten


class Int8LutTensor(TorchAOBaseTensor):
    """
    Tensor subclass that does int8 dynamic activation quantization with lookup table quantization
    """

    tensor_data_names = ["packed_weights"]
    tensor_attribute_names = [
        "bit_width",
        "block_size",
        "shape",
        "dtype",
        "packed_weights_has_bias",
    ]

    def __new__(
        cls,
        packed_weights,
        bit_width,
        block_size,
        shape,
        dtype,
        packed_weights_has_bias,
    ):
        kwargs = {}
        kwargs["device"] = packed_weights.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weights,
        bit_width,
        block_size,
        shape,
        dtype,
        packed_weights_has_bias,
    ):
        super().__init__()
        assert packed_weights.device == torch.device("cpu")
        self.packed_weights = packed_weights
        self.bit_width = bit_width
        self.block_size = block_size
        self.packed_weights_has_bias = packed_weights_has_bias

    def _quantization_type(self):
        return f"bit_width={self.bit_width}, block_size={self.block_size}, shape={self.shape}, dtype={self.dtype}, device={self.device}"

    def to(self, *args, **kwargs):
        raise NotImplementedError("to() is not implemented for IntxOpaqueTensor")

    @classmethod
    def _get_lut_params(cls, tensor: IntxUnpackedToInt8Tensor):
        assert isinstance(tensor, IntxUnpackedToInt8Tensor)
        assert tensor.target_dtype in [torch.int1, torch.int2, torch.int3, torch.int4]

        qdata = tensor.qdata
        scale = tensor.scale
        zero_point = tensor.zero_point

        if tensor._has_float_zero_point():
            # Stretched tensors from PARQ should have -0.5 has zero_point
            assert torch.all(zero_point == -0.5)
            is_stretched_tensor = True
        else:
            assert torch.all(zero_point == 0)
            is_stretched_tensor = False

        quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[tensor.target_dtype]
        lut_indices = qdata - quant_min
        lut = torch.arange(quant_min, quant_max + 1)

        # Construct LUT as 2 * ([q_min, q_max] - 0.5)
        if is_stretched_tensor:
            lut = 2 * lut + 1
            scale = 0.5 * scale

        # LUT must be 2D and int8
        lut = lut.reshape(1, -1).to(torch.int8)

        # Scale must be 1D and float32
        scale = scale.reshape(-1).to(torch.float32)

        return lut, lut_indices, scale

    @classmethod
    def from_intx_unpacked_to_int8_tensor(
        cls,
        tensor: IntxUnpackedToInt8Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Constructs a Int8LutTensor from an IntxUnpackedToInt8Tensor.
        If bias is passed, bias is packed into the tensor.
        """

        assert _is_kernel_library_loaded(), "TorchAO kernel library is not loaded"
        assert (
            tensor.activation_quantization
            == IntxUnpackedToInt8TensorActivationQuantization.INT8_ASYM_PER_TOKEN
        ), (
            "IntxUnpackedToInt8Tensor must have INT8_ASYM_PER_TOKEN activation quantization"
        )

        assert len(tensor.block_size) == 2
        assert tensor.block_size[0] == 1
        scale_group_size = tensor.block_size[1]

        packed_weights_has_bias = bias is not None
        if packed_weights_has_bias:
            n, k = tensor.shape
            assert bias.shape == (n,)
            bias = bias.to(torch.float32)

        lut, lut_indices, scale = cls._get_lut_params(tensor)
        bit_width = _DTYPE_TO_BIT_WIDTH[tensor.target_dtype]
        packed_weights = getattr(
            torch.ops.torchao, f"_pack_8bit_act_{bit_width}bit_weight_with_lut"
        )(
            lut_indices,
            lut,
            scale,
            scale_group_size,
            bias,
            None,
        )

        block_size = [b for b in tensor.block_size]
        shape = tensor.shape
        bit_width = _DTYPE_TO_BIT_WIDTH[tensor.target_dtype]
        return cls(
            packed_weights,
            bit_width,
            block_size,
            shape,
            tensor.dtype,
            packed_weights_has_bias,
        )


implements = Int8LutTensor.implements


def _linear_impl_2d(
    input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
):
    assert isinstance(weight_tensor, Int8LutTensor)
    assert input_tensor.dim() == 2
    assert weight_tensor.dim() == 2
    assert weight_tensor.block_size[0] == 1
    group_size = weight_tensor.block_size[1]

    m, k = input_tensor.shape
    n, k_ = weight_tensor.shape
    assert k_ == k

    packed_weights = weight_tensor.packed_weights
    bit_width = weight_tensor.bit_width

    if weight_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(torch.float32)

    res = getattr(
        torch.ops.torchao,
        f"_linear_8bit_act_{bit_width}bit_weight",
    )(
        input_tensor,
        packed_weights,
        group_size,
        n,
        k,
    )
    if weight_tensor.dtype != torch.float32:
        res = res.to(weight_tensor.dtype)

    return res


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    # TODO: why was this added https://github.com/pytorch/ao/pull/2043
    if input_tensor.numel() == 0:
        return input_tensor

    if input_tensor.dim() == 1:
        k = input_tensor.shape[0]
        input_tensor = input_tensor.reshape(1, k)
        res = _linear_impl_2d(input_tensor, weight_tensor, bias)
        res = res.reshape(-1)
    elif input_tensor.dim() == 2:
        res = _linear_impl_2d(input_tensor, weight_tensor, bias)
    else:
        assert input_tensor.dim() >= 3
        lead_shape = input_tensor.shape[0:-2]
        m, k = input_tensor.shape[-2], input_tensor.shape[-1]
        n, k_ = weight_tensor.shape
        assert k_ == k
        res = _linear_impl_2d(input_tensor.reshape(-1, k), weight_tensor, bias)
        res = res.reshape(*lead_shape, m, n)

    if bias is not None:
        assert not weight_tensor.packed_weights_has_bias
        res = res + bias

    return res


# Allow a model with Int8LutTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int8LutTensor])
