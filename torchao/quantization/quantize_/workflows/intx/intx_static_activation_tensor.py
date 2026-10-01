# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    _choose_qparams_and_quantize_scale_only_hqq,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.quantization.quantize_.workflows.int8.kernels import _int_scaled_matmul
from torchao.quantization.quantize_.workflows.intx.intx_choose_qparams_algorithm import (
    IntxChooseQParamsAlgorithm,
)
from torchao.utils import (
    TorchAOBaseTensor,
    fill_defaults,
)

__all__ = [
    "IntxStaticActivationTensor",
]

aten = torch.ops.aten

_FLOAT_TYPES: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]


class IntxStaticActivationTensor(TorchAOBaseTensor):
    """Intx weight quantization with static int8 activation and output scaling.

    This tensor subclass is designed for models with pre-calibrated static
    activation scales (e.g., from static range quantization / SRQ). On CUDA
    with per-channel weights, the linear dispatch uses the optimized
    `int_scaled_matmul` kernel.

    Tensor Data:
        qdata: int8 quantized weight data (range restricted by target_dtype)
        scale: block scales for weight quantization
        zero_point: block zero points for weight quantization
        act_quant_scale: static activation quantization scale
        act_quant_zero_point: static activation quantization zero point (optional)
        output_quant_scale: output quantization scale for quant-dequant (optional)

    Non-Tensor Attributes:
        target_dtype: determines qmin/qmax of qdata (torch.int1 .. torch.int8)
        block_size: block size for weight quantization granularity
        dtype: dtype of the dequantized tensor
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    optional_tensor_data_names = [
        "act_quant_scale",
        "act_quant_zero_point",
        "output_quant_scale",
    ]
    tensor_attribute_names = [
        "target_dtype",
        "block_size",
        "dtype",
    ]

    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        target_dtype,
        block_size,
        dtype,
        act_quant_scale=None,
        act_quant_zero_point=None,
        output_quant_scale=None,
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
        act_quant_scale=None,
        act_quant_zero_point=None,
        output_quant_scale=None,
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

        self.act_quant_scale = act_quant_scale
        self.act_quant_zero_point = act_quant_zero_point
        self.output_quant_scale = output_quant_scale

    def _quantization_type(self):
        return (
            f"target_dtype={self.target_dtype}, block_size={self.block_size}, "
            f"shape={self.shape}, dtype={self.dtype}, device={self.device}"
        )

    def _has_float_zero_point(self) -> bool:
        return self.zero_point.dtype in _FLOAT_TYPES

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        dtype = kwargs.pop("dtype")
        assert dtype in _FLOAT_TYPES

        act_quant_scale = (
            self.act_quant_scale.to(device=device, dtype=dtype)
            if self.act_quant_scale is not None
            else None
        )

        act_quant_zero_point = None
        if self.act_quant_zero_point is not None:
            if self.act_quant_zero_point.dtype in _FLOAT_TYPES:
                act_quant_zero_point = self.act_quant_zero_point.to(
                    device=device, dtype=dtype
                )
            else:
                act_quant_zero_point = self.act_quant_zero_point.to(device=device)

        output_quant_scale = (
            self.output_quant_scale.to(device=device, dtype=dtype)
            if self.output_quant_scale is not None
            else None
        )

        return IntxStaticActivationTensor(
            self.qdata.to(device),
            self.scale.to(device=device, dtype=dtype),
            self.zero_point.to(device=device, dtype=dtype)
            if self._has_float_zero_point()
            else self.zero_point.to(device),
            self.target_dtype,
            self.block_size,
            dtype,
            act_quant_scale,
            act_quant_zero_point,
            output_quant_scale,
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        block_size: Tuple[int],
        target_dtype: torch.dtype,
        *,
        mapping_type: MappingType = MappingType.SYMMETRIC,
        intx_choose_qparams_algorithm: Optional[
            IntxChooseQParamsAlgorithm
        ] = IntxChooseQParamsAlgorithm.AFFINE,
        custom_scale: Optional[torch.Tensor] = None,
        custom_zero_point: Optional[torch.Tensor] = None,
        act_quant_scale: Optional[torch.Tensor] = None,
        act_quant_zero_point: Optional[torch.Tensor] = None,
        output_quant_scale: Optional[torch.Tensor] = None,
    ):
        """Create an IntxStaticActivationTensor from a high-precision tensor."""
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[target_dtype]

        if intx_choose_qparams_algorithm is not None:
            assert custom_scale is None, (
                "custom_scale is not supported with intx_choose_qparams_algorithm"
            )
            assert custom_zero_point is None, (
                "custom_zero_point is not supported with intx_choose_qparams_algorithm"
            )

        if intx_choose_qparams_algorithm is None:
            assert custom_scale is not None, "custom_scale must be given"
            assert custom_zero_point is not None, "custom_zero_point must be given"
            scale = custom_scale
            zero_point = custom_zero_point
            qdata = quantize_affine(
                hp_tensor,
                block_size,
                scale,
                zero_point,
                output_dtype=torch.int8,
                quant_min=qmin,
                quant_max=qmax,
            )
        elif intx_choose_qparams_algorithm == IntxChooseQParamsAlgorithm.HQQ_SCALE_ONLY:
            qdata, scale = _choose_qparams_and_quantize_scale_only_hqq(
                hp_tensor, block_size, qmin, qmax
            )
            qdata = qdata.to(torch.int8)
            zero_point = torch.zeros_like(scale, dtype=torch.int8)
        elif intx_choose_qparams_algorithm == IntxChooseQParamsAlgorithm.AFFINE:
            scale, zero_point = choose_qparams_affine(
                hp_tensor,
                mapping_type,
                block_size,
                target_dtype=torch.int8,
                quant_min=qmin,
                quant_max=qmax,
                zero_point_dtype=torch.int8,
                keepdim=True,
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
        else:
            raise ValueError(
                f"Unsupported IntxChooseQParamsAlgorithm: {intx_choose_qparams_algorithm}"
            )

        return IntxStaticActivationTensor(
            qdata=qdata,
            scale=scale,
            zero_point=zero_point,
            target_dtype=target_dtype,
            block_size=block_size,
            dtype=hp_tensor.dtype,
            act_quant_scale=act_quant_scale,
            act_quant_zero_point=act_quant_zero_point,
            output_quant_scale=output_quant_scale,
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


implements = IntxStaticActivationTensor.implements
implements_torch_function = IntxStaticActivationTensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert isinstance(weight_tensor, IntxStaticActivationTensor)

    output_quant_scale = getattr(weight_tensor, "output_quant_scale", None)
    output_dtype = input_tensor.dtype

    is_cuda = input_tensor.is_cuda
    is_per_channel = weight_tensor.block_size[-1] == weight_tensor.shape[-1]

    if is_cuda and is_per_channel:
        act_quant_scale = weight_tensor.act_quant_scale
        if act_quant_scale is not None and act_quant_scale.ndim == 0:
            act_quant_scale = act_quant_scale.view((1,) * input_tensor.ndim)

        if act_quant_scale is not None:
            input_block_size = list(input_tensor.shape)
            input_q = quantize_affine(
                input_tensor,
                block_size=input_block_size,
                scale=act_quant_scale,
                zero_point=torch.zeros_like(act_quant_scale, dtype=torch.int8),
                output_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
            )

            tmp = input_q.reshape(-1, input_q.shape[-1])
            w_vals_int8_t = weight_tensor.qdata.t()

            x_scales_expanded = (
                act_quant_scale.view(-1, 1).expand(tmp.shape[0], 1).contiguous()
            )
            intermediate_dtype = (
                torch.float
                if act_quant_scale.dtype == torch.half
                else act_quant_scale.dtype
            )

            y_dot_scaled = _int_scaled_matmul(
                tmp, w_vals_int8_t, x_scales_expanded.to(intermediate_dtype)
            ).to(output_dtype)

            w_scales = weight_tensor.scale
            y = (y_dot_scaled * w_scales.flatten()).reshape(
                *input_tensor.shape[:-1], y_dot_scaled.shape[-1]
            )

            if bias is not None:
                y += bias
        else:
            w_dequant = weight_tensor.dequantize()
            y = torch.nn.functional.linear(input_tensor, w_dequant, bias)
    else:
        w_dequant = weight_tensor.dequantize()
        y = torch.nn.functional.linear(input_tensor, w_dequant, bias)

    if output_quant_scale is not None:
        block_size = list(y.shape)
        zp = torch.zeros_like(output_quant_scale, dtype=torch.int8)
        y_quant = quantize_affine(
            y,
            block_size=block_size,
            scale=output_quant_scale,
            zero_point=zp,
            output_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        )
        y = dequantize_affine(
            input=y_quant,
            block_size=block_size,
            scale=output_quant_scale,
            zero_point=zp,
            input_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            output_dtype=output_dtype,
        )

    return y.to(output_dtype)


@implements(aten.embedding.default)
@implements_torch_function(torch.nn.functional.embedding)
def _(func, types, args, kwargs):
    assert len(args) == 2
    indices, weight_tensor = (
        args[0],
        args[1],
    )
    assert isinstance(weight_tensor, IntxStaticActivationTensor)

    padding_idx = kwargs.get("padding_idx", None)

    if weight_tensor.block_size[0] == 1:
        sliced_qdata = weight_tensor.qdata[indices]
        sliced_scale = weight_tensor.scale[indices]
        sliced_zero_point = weight_tensor.zero_point[indices]

        new_block_size = [1] * indices.ndim + list(weight_tensor.block_size[1:])
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[weight_tensor.target_dtype]

        weight_sliced_dequant = dequantize_affine(
            sliced_qdata,
            new_block_size,
            sliced_scale,
            sliced_zero_point,
            torch.int8,
            qmin,
            qmax,
            output_dtype=weight_tensor.dtype,
        )

        if padding_idx is not None:
            mask = indices == padding_idx
            weight_sliced_dequant = torch.where(
                mask.unsqueeze(-1),
                torch.zeros_like(weight_sliced_dequant),
                weight_sliced_dequant,
            )

        return weight_sliced_dequant
    else:
        weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.embedding(indices, weight_tensor, **kwargs)


@implements(aten.select.int)
def _(func, types, args, kwargs):
    self, dim, index = args[0], args[1], args[2]
    assert isinstance(self, IntxStaticActivationTensor)
    if dim == 0 and self.block_size[0] == 1:
        sliced_qdata = self.qdata[index]
        sliced_scale = self.scale[index]
        sliced_zero_point = self.zero_point[index]

        new_block_size = list(self.block_size[1:])
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.target_dtype]

        dequantized_row = dequantize_affine(
            sliced_qdata,
            new_block_size,
            sliced_scale,
            sliced_zero_point,
            torch.int8,
            qmin,
            qmax,
            output_dtype=self.dtype,
        )
        return dequantized_row
    else:
        dequantized = self.dequantize()
        return aten.select.int(dequantized, dim, index)


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1

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

    new = IntxStaticActivationTensor(
        qdata,
        scale,
        zero_point,
        self.target_dtype,
        new_block_size,
        self.dtype,
        self.act_quant_scale,
        self.act_quant_zero_point,
        self.output_quant_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


IntxStaticActivationTensor.__module__ = "torchao.quantization"

# Allow a model with IntxStaticActivationTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([IntxStaticActivationTensor])
