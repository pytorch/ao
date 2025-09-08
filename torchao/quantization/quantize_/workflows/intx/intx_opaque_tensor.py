# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import enum
import logging
from typing import Optional, Union

import torch

from torchao.experimental.op_lib_utils import _check_torchao_ops_loaded
from torchao.quantization.quant_primitives import _DTYPE_TO_BIT_WIDTH
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    IntxUnpackedToInt8Tensor,
    IntxUnpackedToInt8TensorActivationQuantization,
)
from torchao.utils import (
    TorchAOBaseTensor,
    torch_version_at_least,
)

__all__ = [
    "IntxOpaqueTensor",
]

aten = torch.ops.aten


def is_kernel_library_loaded():
    loaded = False
    try:
        _check_torchao_ops_loaded()
        loaded = True
    except Exception:
        pass
    return loaded


class ComputeTarget(enum.Enum):
    """
    This packs the tensor for PyTorch CPU kernels in ATen.
    It does not require installing torchao C++ kernels.
    """

    ATEN = "aten"

    """
    This packs the tensor for TorchAO CPU kernels by selecting the best available kernel
    based on the quantization scheme, either using KlediAI kernels or lowbit kernels.
    It requires TorchAO C++ kernels to be installed.
    """
    TORCHAO_AUTO = "torchao_auto"

    """
    This packs the tensor for TorchAO CPU kernels using KlediAI kernels.
    It requires TorchAO C++ kernels to be installed.
    """
    TORCHAO_KLEIDIAI = "torchao_kleidiai"

    """
    This packs the tensor for TorchAO CPU kernels using lowbit kernels.
    It requires TorchAO C++ kernels to be installed.
    """
    TORCHAO_LOWBIT = "torchao_lowbit"


class IntxOpaqueTensor(TorchAOBaseTensor):
    """
    intx quantization with tile packed format for CPUs

    Tensor Attributes:
        packed_weights: packed bytes.  Only interpretable by kernel

    Non-Tensor Attributes:
        bit_width: the bit width for quantization (can be 1 - 8)
        block_size: the block size for quantization, representing the granularity, for example groupwise quantization will have block_size (1, group_size)
        shape: the shape of the original Tensor
        dtype: dtype for activations/outputs
        packed_weights_has_zeros: whether zeros are present in packed_weights
        packed_weights_has_bias: whether bias is present in packed_weights
        compute_target: the compute target for the packed data.  Compute targets may pack the data differently.  See ComputeTarget enum for details.
    """

    tensor_data_names = ["packed_weights"]
    tensor_attribute_names = [
        "bit_width",
        "block_size",
        "shape",
        "dtype",
        "packed_weights_has_zeros",
        "packed_weights_has_bias",
        "compute_target",
    ]

    def __new__(
        cls,
        packed_weights,
        bit_width,
        block_size,
        shape,
        dtype,
        packed_weights_has_zeros,
        packed_weights_has_bias,
        compute_target,
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
        packed_weights_has_zeros,
        packed_weights_has_bias,
        compute_target,
    ):
        super().__init__()
        assert packed_weights.device == torch.device("cpu")
        self.packed_weights = packed_weights
        self.bit_width = bit_width
        self.block_size = block_size
        self.packed_weights_has_zeros = packed_weights_has_zeros
        self.packed_weights_has_bias = packed_weights_has_bias
        self.compute_target = compute_target

    def _quantization_type(self):
        return f"bit_width={self.bit_width}, block_size={self.block_size}, shape={self.shape}, dtype={self.dtype}, device={self.device} compute_target={self.compute_target}"

    def to(self, *args, **kwargs):
        raise NotImplementedError("to() is not implemented for IntxOpaqueTensor")

    @classmethod
    def from_intx_unpacked_to_int8_tensor(
        cls,
        tensor: IntxUnpackedToInt8Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        compute_target: Union[ComputeTarget, str] = ComputeTarget.TORCHAO_AUTO,
    ):
        """
        Constructs a IntxOpaqueTensor from an IntxUnpackedToInt8Tensor.
        If bias is passed, bias is packed into the tensor.
        The compute_target indicates how the data is packed.
        """
        if isinstance(compute_target, str):
            compute_target = ComputeTarget[compute_target.upper()]

        # Extract data from IntxUnpackedToInt8Tensor
        assert (
            tensor.activation_quantization
            == IntxUnpackedToInt8TensorActivationQuantization.INT8_ASYM_PER_TOKEN
        )
        qdata, scale, zero_point = tensor.qdata, tensor.scale, tensor.zero_point
        bit_width = _DTYPE_TO_BIT_WIDTH[tensor.target_dtype]
        dtype = tensor.dtype
        shape = tensor.shape

        block_size = tensor.block_size
        assert len(block_size) == 2, "only 2D block_size is supported"
        assert block_size[0] == 1, (
            "only per group or per channel quantization is supported"
        )
        group_size = block_size[1]
        is_per_channel = group_size == shape[1]

        packed_weights_has_bias = bias is not None
        packed_weights_has_zeros = not torch.all(zero_point == 0.0).item()

        assert scale.dtype in [torch.bfloat16, torch.float32]
        scale_is_bfloat16_or_is_rounded_to_bf16 = (
            scale.dtype == torch.bfloat16
        ) or torch.allclose(scale, scale.to(torch.bfloat16).to(torch.float32))

        # Handle ATEN
        if compute_target == ComputeTarget.ATEN:
            assert torch_version_at_least("2.6.0"), (
                "ATEN target requires torch version > 2.6.0"
            )
            assert torch.backends.kleidiai.is_available(), (
                "ATEN target requires torch.backends.kleidiai.is_available()"
            )
            assert bit_width == 4, "ATEN target only supports 4-bit"
            assert not packed_weights_has_zeros, "ATEN target does not support zeros"
            qdata = qdata.add(8)
            qdata = (qdata[::, 1::2] << 4 | qdata[::, ::2]).to(torch.uint8)

            # If per-group, convert scales to bfloat16 to call optimized kernel
            if not is_per_channel:
                if not scale_is_bfloat16_or_is_rounded_to_bf16:
                    logging.info(
                        f"scale has dtype {scale.dtype}, converting to torch.bfloat16"
                    )
                scale = scale.to(torch.bfloat16)

            packed_weight = torch.ops.aten._dyn_quant_pack_4bit_weight(
                qdata, scale, bias, group_size, shape[1], shape[0]
            )
            return cls(
                packed_weight,
                bit_width,
                block_size,
                shape,
                dtype,
                packed_weights_has_zeros,
                packed_weights_has_bias,
                compute_target,
            )

        # Handle TORCHAO
        assert is_kernel_library_loaded(), "TorchAO kernel library is not loaded"
        compute_target_map = {
            ComputeTarget.TORCHAO_AUTO: None,
            ComputeTarget.TORCHAO_KLEIDIAI: "kleidiai",
            ComputeTarget.TORCHAO_LOWBIT: "universal",
        }
        assert compute_target in compute_target_map, (
            f"compute_target {compute_target} not supported"
        )

        if not scale_is_bfloat16_or_is_rounded_to_bf16 and compute_target in [
            ComputeTarget.TORCHAO_AUTO,
            ComputeTarget.TORCHAO_KLEIDIAI,
        ]:
            logging.info("scale may be rounded to bf16 in the kernel")
        if scale.dtype != torch.float32:
            logging.info(f"scale has dtype {scale.dtype}, converting to torch.float32")
            scale = scale.to(torch.float32)
        if bias is not None and bias.dtype != torch.float32:
            logging.info(f"bias has dtype {bias.dtype}, converting to torch.float32")
            bias = bias.to(torch.float32)
        if packed_weights_has_zeros and not tensor._has_float_zero_point():
            zero_point = zero_point.to(torch.int8)

        packed_weights = getattr(
            torch.ops.torchao,
            f"_pack_8bit_act_{bit_width}bit_weight",
        )(
            qdata,
            scale.reshape(-1),
            zero_point.reshape(-1) if packed_weights_has_zeros else None,
            group_size,
            bias,
            compute_target_map[compute_target],
        )
        return cls(
            packed_weights,
            bit_width,
            block_size,
            shape,
            dtype,
            packed_weights_has_zeros,
            packed_weights_has_bias,
            compute_target,
        )


implements = IntxOpaqueTensor.implements


def _linear_impl_2d_aten(input_tensor, weight_tensor):
    assert isinstance(weight_tensor, IntxOpaqueTensor)
    assert weight_tensor.compute_target == ComputeTarget.ATEN
    assert input_tensor.dim() == 2
    assert weight_tensor.dim() == 2
    assert weight_tensor.block_size[0] == 1
    assert weight_tensor.bit_width == 4
    group_size = weight_tensor.block_size[1]

    m, k = input_tensor.shape
    n, k_ = weight_tensor.shape
    assert k_ == k

    packed_weights = weight_tensor.packed_weights

    return torch.ops.aten._dyn_quant_matmul_4bit(
        input_tensor, packed_weights, group_size, k, n
    )


def _linear_impl_2d_torchao(input_tensor, weight_tensor):
    assert weight_tensor.compute_target != ComputeTarget.ATEN
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
    res = getattr(torch.ops.torchao, f"_linear_8bit_act_{bit_width}bit_weight")(
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

    if weight_tensor.compute_target == ComputeTarget.ATEN:
        _impl_2d = _linear_impl_2d_aten
    else:
        _impl_2d = _linear_impl_2d_torchao

    # TODO: why was this added https://github.com/pytorch/ao/pull/2043
    if input_tensor.numel() == 0:
        return input_tensor

    if input_tensor.dim() == 1:
        k = input_tensor.shape[0]
        input_tensor = input_tensor.reshape(1, k)
        res = _impl_2d(input_tensor, weight_tensor)
        res = res.reshape(-1)
    elif input_tensor.dim() == 2:
        res = _impl_2d(input_tensor, weight_tensor)
    else:
        assert input_tensor.dim() >= 3
        lead_shape = input_tensor.shape[0:-2]
        m, k = input_tensor.shape[-2], input_tensor.shape[-1]
        n, k_ = weight_tensor.shape
        assert k_ == k
        res = _impl_2d(input_tensor.reshape(-1, k), weight_tensor)
        res = res.reshape(*lead_shape, m, n)

    if bias is not None:
        assert not weight_tensor.packed_weights_has_bias
        res = res + bias

    return res


IntxOpaqueTensor.__module__ = "torchao.quantization"

torch.serialization.add_safe_globals([IntxOpaqueTensor, ComputeTarget])
