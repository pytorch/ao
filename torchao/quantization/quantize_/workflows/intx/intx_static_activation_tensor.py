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
from torchao.quantization.quantize_.workflows.int8.kernels import safe_int_mm
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


def _check_static_scale_shape(t: torch.Tensor, num_channels: int, name: str) -> None:
    if t.numel() == 1:
        return
    assert t.numel() == num_channels and t.shape[-1] == num_channels, (
        f"{name} must be per-tensor (numel() == 1) or per-channel with shape[-1] == "
        f"{num_channels} and numel() == {num_channels}, but got shape {tuple(t.shape)}"
    )


def _broadcast_static_scale(
    t: torch.Tensor, num_channels: int, target_ndim: int
) -> Tuple[torch.Tensor, bool]:
    if t.numel() == 1:
        return t.reshape((1,) * target_ndim), True
    return t.reshape((1,) * (target_ndim - 1) + (num_channels,)), False


class IntxStaticActivationTensor(TorchAOBaseTensor):
    """Intx weight quantization with static int8 activation and output scaling.

    This tensor subclass is designed for models with pre-calibrated static
    activation scales (e.g., from static range quantization / SRQ). On CUDA
    with per-channel weights, the linear dispatch uses the optimized
    `safe_int_mm` kernel.

    Tensor Data:
        qdata: int8 quantized weight data (range restricted by target_dtype)
        scale: block scales for weight quantization
        zero_point: block zero points for weight quantization
        act_quant_scale: static activation quantization scale (optional); per-tensor
            (numel() == 1) or per-input-channel (shape[-1] == in_features)
        act_quant_zero_point: static int8 activation zero point (optional); per-tensor or
            matching act_quant_scale
        output_quant_scale: output quantize-dequantize scale (optional); per-tensor or
            per-output-channel (shape[-1] == out_features)

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
            f"zero_point dtype must be int8 or one of {_FLOAT_TYPES}, but got {zero_point.dtype}"
        )

        assert target_dtype in [
            getattr(torch, f"int{bit_width}") for bit_width in range(1, 9)
        ]

        block_size = tuple(block_size)
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

        if act_quant_scale is not None:
            _check_static_scale_shape(
                act_quant_scale, qdata.shape[-1], "act_quant_scale"
            )
        if act_quant_zero_point is not None:
            _check_static_scale_shape(
                act_quant_zero_point, qdata.shape[-1], "act_quant_zero_point"
            )
        if output_quant_scale is not None:
            _check_static_scale_shape(
                output_quant_scale, qdata.shape[0], "output_quant_scale"
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

        scale_dtype = self.scale.dtype if dtype == self.dtype else dtype
        act_scale_dtype = (
            self.act_quant_scale.dtype
            if (self.act_quant_scale is not None and dtype == self.dtype)
            else dtype
        )
        out_scale_dtype = (
            self.output_quant_scale.dtype
            if (self.output_quant_scale is not None and dtype == self.dtype)
            else dtype
        )

        act_quant_scale = (
            self.act_quant_scale.to(device=device, dtype=act_scale_dtype)
            if self.act_quant_scale is not None
            else None
        )
        act_quant_zero_point = (
            self.act_quant_zero_point.to(device=device)
            if self.act_quant_zero_point is not None
            else None
        )
        output_quant_scale = (
            self.output_quant_scale.to(device=device, dtype=out_scale_dtype)
            if self.output_quant_scale is not None
            else None
        )

        return IntxStaticActivationTensor(
            self.qdata.to(device),
            self.scale.to(device=device, dtype=scale_dtype),
            self.zero_point.to(device=device, dtype=scale_dtype)
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
        block_size: Tuple[int, ...],
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
            scale = custom_scale.to(device=hp_tensor.device)
            if scale.dtype not in _FLOAT_TYPES:
                scale = scale.to(dtype=hp_tensor.dtype)
            if custom_zero_point is None:
                zero_point = torch.zeros_like(scale, dtype=torch.int8)
            else:
                zero_point = custom_zero_point.to(device=hp_tensor.device)
                if zero_point.dtype in (torch.int32, torch.int64):
                    assert torch.all((zero_point >= -128) & (zero_point <= 127)), (
                        "custom_zero_point values must be in int8 range [-128, 127]"
                    )
                    zero_point = zero_point.to(torch.int8)
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

        if act_quant_scale is not None:
            act_quant_scale = act_quant_scale.to(device=hp_tensor.device)
            if (
                act_quant_scale.is_floating_point()
                and act_quant_scale.dtype not in _FLOAT_TYPES
            ):
                act_quant_scale = act_quant_scale.to(dtype=torch.float32)
        if act_quant_zero_point is not None:
            act_quant_zero_point = act_quant_zero_point.to(device=hp_tensor.device)
            if act_quant_zero_point.dtype in (torch.int32, torch.int64):
                assert torch.all(
                    (act_quant_zero_point >= -128) & (act_quant_zero_point <= 127)
                ), "act_quant_zero_point values must be in int8 range [-128, 127]"
                act_quant_zero_point = act_quant_zero_point.to(torch.int8)
        if output_quant_scale is not None:
            output_quant_scale = output_quant_scale.to(device=hp_tensor.device)
            if (
                output_quant_scale.is_floating_point()
                and output_quant_scale.dtype not in _FLOAT_TYPES
            ):
                output_quant_scale = output_quant_scale.to(dtype=torch.float32)

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

    output_quant_scale = weight_tensor.output_quant_scale
    output_dtype = input_tensor.dtype

    act_quant_scale = weight_tensor.act_quant_scale
    act_quant_zero_point = weight_tensor.act_quant_zero_point

    is_cuda = input_tensor.is_cuda
    is_per_channel = (
        weight_tensor.block_size[0] == 1
        and weight_tensor.block_size[-1] == weight_tensor.shape[-1]
    )
    K = input_tensor.shape[-1]
    N = weight_tensor.shape[0]

    if act_quant_scale is not None:
        act_scale, act_scale_is_per_tensor = _broadcast_static_scale(
            act_quant_scale.to(device=input_tensor.device), K, input_tensor.ndim
        )
        if act_scale_is_per_tensor:
            input_block_size = list(input_tensor.shape)
        else:
            input_block_size = list(input_tensor.shape[:-1]) + [1]

        if act_quant_zero_point is not None:
            act_zp, _ = _broadcast_static_scale(
                act_quant_zero_point.to(device=input_tensor.device),
                K,
                input_tensor.ndim,
            )
            act_zp = act_zp.expand_as(act_scale)
        else:
            act_zp = torch.zeros_like(act_scale, dtype=torch.int8)

        input_q = quantize_affine(
            input_tensor,
            block_size=input_block_size,
            scale=act_scale,
            zero_point=act_zp,
            output_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        )

        # aten._int_mm requires M > 16 and K, N multiples of 8.
        num_tokens = input_tensor.numel() // K if K > 0 else 0
        use_int_mm = (
            is_cuda
            and is_per_channel
            and act_scale_is_per_tensor
            and num_tokens > 16
            and K % 8 == 0
            and N % 8 == 0
        )
        if use_int_mm:
            tmp = input_q.reshape(-1, K)
            w_vals_int8_t = weight_tensor.qdata.contiguous().t()
            acc = safe_int_mm(tmp, w_vals_int8_t)

            if act_quant_zero_point is not None:
                w_row_sums = weight_tensor.qdata.sum(dim=-1, dtype=torch.int32)
                zp_x = act_zp.reshape(1, 1).to(torch.int32)
                acc = acc - zp_x * w_row_sums

            x_row_sums = tmp.sum(dim=-1, keepdim=True, dtype=torch.int32)
            if act_quant_zero_point is not None:
                x_row_sums = x_row_sums - K * zp_x

            if weight_tensor._has_float_zero_point():
                zp_w = weight_tensor.zero_point.flatten().to(torch.float32)
                acc_fp32 = acc.to(torch.float32) - x_row_sums.to(torch.float32) * zp_w
            else:
                zp_w = weight_tensor.zero_point.flatten().to(torch.int32)
                acc_fp32 = (acc - x_row_sums * zp_w).to(torch.float32)

            s_x = act_scale.reshape(1, 1).to(torch.float32)
            w_scales = weight_tensor.scale.flatten().to(torch.float32)
            y = (
                (acc_fp32 * (s_x * w_scales))
                .to(output_dtype)
                .reshape(*input_tensor.shape[:-1], acc_fp32.shape[-1])
            )

            if bias is not None:
                y = y + bias
        else:
            input_dequant = dequantize_affine(
                input_q,
                block_size=input_block_size,
                scale=act_scale,
                zero_point=act_zp,
                input_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                output_dtype=output_dtype,
            )
            w_dequant = weight_tensor.dequantize()
            y = torch.nn.functional.linear(input_dequant, w_dequant, bias)
    else:
        w_dequant = weight_tensor.dequantize()
        y = torch.nn.functional.linear(input_tensor, w_dequant, bias)

    if output_quant_scale is not None:
        out_scale, out_scale_is_per_tensor = _broadcast_static_scale(
            output_quant_scale.to(device=y.device), N, y.ndim
        )
        if out_scale_is_per_tensor:
            out_block_size = list(y.shape)
        else:
            out_block_size = list(y.shape[:-1]) + [1]
        zp = torch.zeros_like(out_scale, dtype=torch.int8)
        y_quant = quantize_affine(
            y,
            block_size=out_block_size,
            scale=out_scale,
            zero_point=zp,
            output_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        )
        y = dequantize_affine(
            input=y_quant,
            block_size=out_block_size,
            scale=out_scale,
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
    if func is aten.embedding.default:
        weight_tensor, indices = args[0], args[1]
        extra_args = args[2:]
        max_norm = None
    else:
        assert len(args) >= 2
        indices, weight_tensor = args[0], args[1]
        extra_args = args[2:]
        max_norm = kwargs.get("max_norm", args[3] if len(args) > 3 else None)

    assert isinstance(weight_tensor, IntxStaticActivationTensor)

    if weight_tensor.block_size[0] == 1 and max_norm is None:
        sliced_qdata = weight_tensor.qdata[indices]
        sliced_scale = weight_tensor.scale[indices]
        sliced_zero_point = weight_tensor.zero_point[indices]

        new_block_size = [1] * indices.ndim + list(weight_tensor.block_size[1:])
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[weight_tensor.target_dtype]

        return dequantize_affine(
            sliced_qdata,
            new_block_size,
            sliced_scale,
            sliced_zero_point,
            torch.int8,
            qmin,
            qmax,
            output_dtype=weight_tensor.dtype,
        )
    else:
        dequantized = weight_tensor.dequantize()
        if func is aten.embedding.default:
            return aten.embedding.default(dequantized, indices, *extra_args, **kwargs)
        return torch.nn.functional.embedding(
            indices, dequantized, *extra_args, **kwargs
        )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    self, dim, index = args[0], args[1], args[2]
    assert isinstance(self, IntxStaticActivationTensor)
    if dim < 0:
        dim += self.ndim
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
    if step != 1:
        raise NotImplementedError(
            f"Slicing with step != 1 is not supported, got step={step}"
        )
    if dim < 0:
        dim += self.ndim
    if start is None or start < 0:
        start = 0 if start is None else max(0, self.shape[dim] + start)
    if end is None or end > self.shape[dim]:
        end = self.shape[dim]
    elif end < 0:
        end = max(0, self.shape[dim] + end)

    qdata = aten.slice.Tensor(self.qdata, dim, start, end, step)
    if self.scale.shape[dim] == 1:
        scale = self.scale
        zero_point = self.zero_point
    else:
        assert start % self.block_size[dim] == 0, (
            f"slice args are incompatible with blocking: start={start} must be divisible by block_size[dim]={self.block_size[dim]}"
        )
        start_scale = start // self.block_size[dim]

        assert end % self.block_size[dim] == 0, (
            f"slice args are incompatible with blocking: end={end} must be divisible by block_size[dim]={self.block_size[dim]}"
        )
        end_scale = end // self.block_size[dim]

        scale = aten.slice.Tensor(self.scale, dim, start_scale, end_scale, step)
        zero_point = aten.slice.Tensor(
            self.zero_point, dim, start_scale, end_scale, step
        )

    new_block_size = []
    for i in range(qdata.ndim):
        assert scale.shape[i] == zero_point.shape[i]
        n_blocks = scale.shape[i]
        assert qdata.shape[i] % n_blocks == 0
        new_block_size.append(qdata.shape[i] // n_blocks)
    new_block_size = tuple(new_block_size)

    act_quant_scale = self.act_quant_scale
    if dim == 1 and act_quant_scale is not None and act_quant_scale.numel() > 1:
        act_quant_scale = aten.slice.Tensor(
            act_quant_scale, act_quant_scale.ndim - 1, start, end, step
        )

    act_quant_zero_point = self.act_quant_zero_point
    if (
        dim == 1
        and act_quant_zero_point is not None
        and act_quant_zero_point.numel() > 1
    ):
        act_quant_zero_point = aten.slice.Tensor(
            act_quant_zero_point, act_quant_zero_point.ndim - 1, start, end, step
        )

    output_quant_scale = self.output_quant_scale
    if dim == 0 and output_quant_scale is not None and output_quant_scale.numel() > 1:
        output_quant_scale = aten.slice.Tensor(
            output_quant_scale, output_quant_scale.ndim - 1, start, end, step
        )

    new = IntxStaticActivationTensor(
        qdata,
        scale,
        zero_point,
        self.target_dtype,
        new_block_size,
        self.dtype,
        act_quant_scale,
        act_quant_zero_point,
        output_quant_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


IntxStaticActivationTensor.__module__ = "torchao.quantization"

# Allow a model with IntxStaticActivationTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([IntxStaticActivationTensor])
