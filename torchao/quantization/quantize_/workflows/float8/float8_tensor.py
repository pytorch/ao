# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _is_1_128_scaled,
    _is_128_128_scaled,
    _is_rowwise_scaled,
    _is_tensorwise_scaled,
    _slice_scale_for_dimension,
    addmm_float8_unwrapped_inference,
    preprocess_data,
    preprocess_scale,
)
from torchao.kernel.blockwise_quantization import (
    blockwise_fp8_gemm,
)
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _dequantize_affine_float8,
    _quantize_affine_float8,
)
from torchao.quantization.quantize_.common import (
    KernelPreference,
    QuantizeTensorKwargs,
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.utils import get_block_size
from torchao.utils import (
    TorchAOBaseTensor,
    _is_mslk_available,
    fill_defaults,
    is_sm_at_least_90,
    is_sm_at_least_100,
)

if _is_mslk_available():
    import mslk.conv  # noqa: F401

__all__ = [
    "Float8Tensor",
    "QuantizeTensorToFloat8Kwargs",
]

aten = torch.ops.aten


@dataclass
class QuantizeTensorToFloat8Kwargs(QuantizeTensorKwargs):
    """Tensor kwargs for creating float8 tensor (either activation or weight)

    Args:
       dtype (torch.dtype): the dtype for float8 Tensor
       granularity (FP8Granularity): the granularity for the Tensor, currently either PerRow() or PerTensor()
       mm_config (Float8MMConfig): Configuration for the scaled_mm in the forward and backward pass.
       hp_value_lb (Optional[float]): the lower bound for high precision floating point value for calculating scale
       hp_value_ub (Optional[float]): the upper bound for high precision floating point value for calculating scale
       kernel_preference (KernelPreference): kernel preference for ops like matmul, grouped matmul etc. by defalut (None) it will be chosen for user based on hardware or other information
    """

    float8_dtype: torch.dtype = torch.float8_e4m3fn
    granularity: FP8Granularity = PerRow()
    mm_config: Optional[Float8MMConfig] = None
    hp_value_lb: Optional[float] = None
    hp_value_ub: Optional[float] = None
    kernel_preference: KernelPreference = KernelPreference.AUTO


class Float8Tensor(TorchAOBaseTensor):
    """
    Float8 Quantized (weight) Tensor, with float8 dynamic quantization for activation or bfloat16 activation.

    TODO: needs padding for cutlass kernels

    Tensor Attributes:
        qdata: float8 raw data
        scale: the scale for float8 Tensor

    Non-Tensor Attributes:
        block_size (List[int]): the block size for float8 quantization, meaning the shape of the elements
        sharing the same set of quantization parameters (scale), have the same rank as qdata or
        is an empty list (representing per tensor quantization)
        mm_config (Float8MMConfig): Configuration for the matrix multiplication. Default uses fast accumulation.
        act_quant_kwargs (QuantizeTensorToFloat8Kwargs): the kwargs for Float8Tensor.from_hp
        kernel_preference (KernelPreference): the preference for quantize, mm etc. kernel to use,
        by default, this will be chosen for user based on hardware, library availabilities etc.
        dtype: Original Tensor dtype
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = []
    optional_tensor_attribute_names = [
        "block_size",
        "mm_config",
        "act_quant_kwargs",
        "kernel_preference",
        "dtype",
    ]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        mm_config: Optional[Float8MMConfig] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        dtype: Optional[torch.dtype] = None,
    ):
        shape = qdata.shape
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        mm_config: Optional[Float8MMConfig] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        self.mm_config = mm_config
        self.act_quant_kwargs = act_quant_kwargs
        self.kernel_preference = kernel_preference

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata=}, {self.scale=}, "
            f"{self.block_size=}, {self.mm_config=}, {self.kernel_preference=} "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    def _quantization_type(self):
        return f"{self.act_quant_kwargs=}, {self.block_size=}, {self.mm_config=}, {self.scale.shape=}, {self.kernel_preference=}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        qdata, scale = self.qdata, self.scale
        return _dequantize_affine_float8(qdata, scale, output_dtype)

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
        granularity: FP8Granularity = PerRow(),
        mm_config: Optional[Float8MMConfig] = None,
        hp_value_lb: Optional[float] = None,
        hp_value_ub: Optional[float] = None,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        block_size = get_block_size(hp_tensor.shape, granularity)
        block_size = list(block_size)

        kernel_choice = None
        if (
            kernel_preference == KernelPreference.AUTO
            and _is_mslk_available()
            and is_sm_at_least_90()
            and isinstance(granularity, PerRow)
            # mslk path only supports quantizing along the last dim
            and granularity.dim in (-1, len(hp_tensor.shape) - 1)
            and float8_dtype == torch.float8_e4m3fn
            and hp_value_lb is None
        ):
            # if kernel_preference is AUTO and per row quantization
            # we'll use mslk quantize kernel for best performance
            kernel_choice = "mslk"
        elif kernel_preference == KernelPreference.MSLK:
            # if user explicitly chose MSLK kernel preference, we'll also use mslk kernel
            assert _is_mslk_available() and is_sm_at_least_90(), (
                "Specified mslk but mslk is not installed or hardware is not >= SM 9.0 (>= H100)"
            )
            assert hp_value_lb is None, (
                "hp_value_lb should not be specified if with KernelPreference.MSLK"
            )
            kernel_choice = "mslk"
        else:
            # fallback quantize kernel for everything else will be torch
            kernel_choice = "torch"

        if kernel_choice == "mslk":
            assert hp_value_lb is None, f"{hp_value_lb=} is not supported"
            if hp_value_ub is not None:
                maybe_hp_value_ub_tensor = torch.tensor(
                    hp_value_ub, dtype=torch.float, device=hp_tensor.device
                )
            else:
                maybe_hp_value_ub_tensor = None
            if isinstance(granularity, PerRow):
                data, scale = torch.ops.triton.quantize_fp8_row(
                    hp_tensor, scale_ub=maybe_hp_value_ub_tensor
                )
            else:
                assert isinstance(granularity, PerTensor), (
                    f"Expected per tensor, got {granularity}"
                )
                data, scale = torch.ops.triton.quantize_fp8_tensor(hp_tensor)

            # Reshape scale to match expected shape for quantization
            scale_shape = []
            for i in range(hp_tensor.ndim):
                scale_shape.append(hp_tensor.shape[i] // block_size[i])
            scale = scale.reshape(*scale_shape)

        else:
            assert kernel_choice == "torch", f"Expected torch, got {kernel_choice}"
            scale = _choose_scale_float8(
                hp_tensor,
                float8_dtype=float8_dtype,
                block_size=block_size,
                hp_value_lb=hp_value_lb,
                hp_value_ub=hp_value_ub,
            )
            data = _quantize_affine_float8(hp_tensor, scale, float8_dtype)

        hp_dtype = hp_tensor.dtype
        return Float8Tensor(
            data,
            scale,
            block_size=block_size,
            mm_config=mm_config,
            act_quant_kwargs=act_quant_kwargs,
            kernel_preference=kernel_preference,
            dtype=hp_dtype,
        )


implements = Float8Tensor.implements
implements_torch_function = Float8Tensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    return _float8_addmm_impl(input_tensor, weight_tensor.t(), bias)


@implements(aten.matmul.default)
@implements_torch_function(torch.matmul)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = args[0], args[1]
    return _float8_addmm_impl(input_tensor, weight_tensor)


@implements(aten.mm.default)
@implements_torch_function(torch.mm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = args[0], args[1]
    return _float8_addmm_impl(input_tensor, weight_tensor)


@implements(aten.addmm_.default)
def _(func, types, args, kwargs):
    bias_tensor, input_tensor, weight_tensor = (
        args[0],
        args[1],
        args[2],
    )
    assert kwargs.get("alpha", 1) == 1, "only alpha=1 is supported"
    assert kwargs.get("beta", 1) == 1, "only beta=1 is supported"
    out = _float8_addmm_impl(input_tensor, weight_tensor)
    return bias_tensor.add_(out)


@implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    is_pinned = args[0].qdata.is_pinned() and args[0].scale.is_pinned()
    return is_pinned


@implements(aten._pin_memory.default)
def _(func, types, args, kwargs):
    pinned_qdata = args[0].qdata.pin_memory()
    pinned_scale = args[0].scale.pin_memory()

    return Float8Tensor(
        pinned_qdata,
        pinned_scale,
        args[0].block_size,
        args[0].mm_config,
        act_quant_kwargs=args[0].act_quant_kwargs,
        kernel_preference=args[0].kernel_preference,
        dtype=args[0].dtype,
    )


def _float8_addmm_impl(
    input_tensor: Float8Tensor,
    weight_tensor: Float8Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert isinstance(weight_tensor, Float8Tensor), (
        f"Don't expect to reach here with an override other than weight currently, {type(input_tensor)} {type(weight_tensor)}"
    )

    act_quant_kwargs = weight_tensor.act_quant_kwargs
    # quantize activation, if `act_quant_kwargs` is specified
    if act_quant_kwargs is not None:
        assert not isinstance(input_tensor, TorchAOBaseTensor), (
            "input tensor was already quantized"
        )
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

    # TODO: technically addmm and mm don't support broadcasting,
    # would be more correct to enforce x and w to be 2d here and
    # move 3d support to matmul and linear
    out_shape = (*input_tensor.shape[:-1], weight_tensor.shape[1])

    if isinstance(input_tensor, Float8Tensor):
        kernel_choice = None

        if weight_tensor.kernel_preference == KernelPreference.AUTO:
            kernel_choice = "torch"
            if (
                _is_mslk_available()
                and is_sm_at_least_90()
                and (not _is_128_128_scaled(weight_tensor))
            ):
                kernel_choice = "mslk"
        elif weight_tensor.kernel_preference == KernelPreference.MSLK:
            kernel_choice = "mslk"
        else:
            assert weight_tensor.kernel_preference == KernelPreference.TORCH, (
                f"{weight_tensor.kernel_preference=} not handled"
            )
            kernel_choice = "torch"

        if kernel_choice == "mslk":
            assert _is_mslk_available(), "Expected mslk package to be installed"
            assert is_sm_at_least_90(), "Expected SM90+ for mslk"
            mm_config = weight_tensor.mm_config
            assert mm_config is not None
            assert not _is_128_128_scaled(weight_tensor), "unimplemented"

            xq = input_tensor.qdata.reshape(-1, input_tensor.qdata.shape[-1])
            x_scale = input_tensor.scale
            if _is_rowwise_scaled(weight_tensor.t()):
                assert _is_rowwise_scaled(input_tensor), (
                    "Input tensor must be rowwise block size"
                )
                res = torch.ops.mslk.f8f8bf16_rowwise(
                    xq,
                    weight_tensor.qdata.t(),
                    input_tensor.scale,
                    weight_tensor.scale.t(),
                    bias=bias,
                    use_fast_accum=mm_config.use_fast_accum,
                ).reshape(out_shape)
            else:
                assert _is_tensorwise_scaled(weight_tensor)
                assert _is_tensorwise_scaled(input_tensor)
                res = torch.ops.mslk.f8f8bf16(
                    xq,
                    weight_tensor.qdata.t(),
                    x_scale * weight_tensor.scale.t(),
                    use_fast_accum=mm_config.use_fast_accum,
                ).reshape(out_shape)
                if bias is not None:
                    res = res + bias
            return res
        else:
            assert kernel_choice == "torch"
            scaled_mm_config = weight_tensor.mm_config
            assert scaled_mm_config is not None

            # Extract tensor data and scales
            inpt_data = input_tensor.qdata.reshape(-1, input_tensor.qdata.shape[-1])
            w_data = weight_tensor.qdata
            input_scale = input_tensor.scale
            w_scale = weight_tensor.scale

            if _is_rowwise_scaled(weight_tensor):
                assert _is_rowwise_scaled(input_tensor), (
                    "Input tensor must be rowwise block size"
                )
            elif _is_128_128_scaled(weight_tensor):
                assert _is_1_128_scaled(input_tensor), (
                    "input_tensor must be 1x128 scaled"
                )

            input_scale = preprocess_scale(input_scale, input_tensor.shape)
            inpt_data, w_data = preprocess_data(inpt_data, w_data, scaled_mm_config)

            if _is_128_128_scaled(weight_tensor):
                # TODO(future PR): add testing for torch._scaled_mm with
                # blockwise scaling on CUDA 12.9
                # TODO(future PR): add mslk path if available
                # TODO(future PR): proper out_dtype handling
                assert _is_1_128_scaled(input_tensor), "unsupported"
                res = blockwise_fp8_gemm(
                    inpt_data,
                    input_scale,
                    w_data.t(),
                    w_scale.t(),
                    block_size=128,
                )
                if bias is not None:
                    res = res + bias
            else:
                res = addmm_float8_unwrapped_inference(
                    inpt_data,
                    input_scale,
                    w_data,
                    w_scale,
                    output_dtype=input_tensor.dtype,
                    bias=bias,
                    use_fast_accum=scaled_mm_config.use_fast_accum,
                )
            return res.reshape(out_shape)
    else:
        assert not isinstance(input_tensor, TorchAOBaseTensor), (
            "Expecting input_tensor to be unquantized"
        )
        # when input is not `Float8Tensor`, we expect that it is not quantized
        # so this is float8 weight only quantization
        out = torch.matmul(input_tensor, weight_tensor.dequantize())
        if bias is not None:
            return out + bias
        else:
            return out


@implements_torch_function(torch.bmm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    assert isinstance(weight_tensor, Float8Tensor), (
        f"Don't expect to reach here with an override other than weight currently, {type(input_tensor)} {type(weight_tensor)}"
    )

    kernel_preference = weight_tensor.kernel_preference
    assert kernel_preference != KernelPreference.TORCH, "bmm is not supported for TORCH"
    assert _is_mslk_available(), "bmm is not supported when mslk is not installed"

    orig_act_size = input_tensor.size()
    act_quant_kwargs = weight_tensor.act_quant_kwargs
    if act_quant_kwargs is not None:
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

    if isinstance(input_tensor, Float8Tensor):
        a_data = input_tensor.qdata
        a_scale = input_tensor.scale

        b_data = weight_tensor.qdata
        b_scale = weight_tensor.scale

        assert (
            weight_tensor.block_size[0] == 1
            and weight_tensor.block_size[1] == weight_tensor.shape[1]
            and weight_tensor.block_size[2] == 1
        ), "bmm only works for per row weight quantization"
        assert (
            all(x == 1 for x in input_tensor.block_size[:-1])
            and input_tensor.block_size[-1] == input_tensor.shape[-1]
        ), "bmm only works for per row activation quantization"

        orig_out_features = b_data.shape[-1]

        res = torch.ops.mslk.f8f8bf16_rowwise_batched(
            a_data,
            b_data.transpose(-2, -1).contiguous(),
            a_scale,
            b_scale.transpose(-2, -1),
            b_scale,
        )
        res = res.reshape(*orig_act_size[:-1], orig_out_features)
    else:
        raise NotImplementedError(
            "bmm only support float8 dynamic activation + float8 weight"
        )

    return res


def _quantize_and_scaled_conv3d(
    input_tensor,
    weight_tensor,
    bias,
    stride,
    padding,
    dilation,
):
    assert isinstance(weight_tensor, Float8Tensor), (
        f"Don't expect to reach here with an override other than weight currently, {type(input_tensor)} {type(weight_tensor)}"
    )

    assert input_tensor.dim() == 5 and weight_tensor.dim() == 5, (
        "Only support 3D conv currently"
    )
    assert _is_mslk_available(), "quantized fp8 conv3d requires mslk to be available"
    act_quant_kwargs = weight_tensor.act_quant_kwargs
    # quantize activation, if `act_quant_kwargs` is specified
    if act_quant_kwargs is not None:
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

    if isinstance(input_tensor, Float8Tensor):
        kernel_choice = None
        if weight_tensor.kernel_preference == KernelPreference.AUTO:
            if _is_mslk_available() and is_sm_at_least_100():
                kernel_choice = "mslk"
            else:
                raise NotImplementedError(
                    f"No available kernel choice for {weight_tensor.kernel_preference}"
                )
        elif weight_tensor.kernel_preference == KernelPreference.MSLK:
            kernel_choice = "mslk"
        else:
            raise NotImplementedError(
                f"No available kernel choice for {weight_tensor.kernel_preference}"
            )

    assert kernel_choice == "mslk", "Only mslk kernel choice is supported currently"
    input_qdata = input_tensor.qdata
    weight_qdata = weight_tensor.qdata

    is_input_channels_last = input_qdata.is_contiguous(
        memory_format=torch.channels_last_3d
    )
    is_weight_channels_last = weight_qdata.is_contiguous(
        memory_format=torch.channels_last_3d
    )

    # convert the input/weight to channels_last_3d memory_format here
    # to make sure we can call the mslk conv
    # kernel, it should be a no-op if both activation and weight are in
    # channels_last_3d memory_format
    input_qdata = input_qdata.contiguous(memory_format=torch.channels_last_3d)
    weight_qdata = weight_qdata.contiguous(memory_format=torch.channels_last_3d)

    input_scale = input_tensor.scale
    weight_scale = weight_tensor.scale

    # input: (N, C_in, D, H, W)
    # weight: (C_out, C_in, K1, K2, K3)
    # output: (N, C_out, D_out, H_out, W_out)
    # all in channels_last_3d memory_format

    output = torch.ops.mslk.f8f8bf16_conv(
        input_qdata,
        weight_qdata,
        input_scale * weight_scale,
        padding,
        stride,
        dilation,
    )

    # aligning the semantics with bfloat16 conv ops, the
    # output should use contiguous_format if none of the input/weight
    # are in channels_last format, otherwise, the output is already
    # in channels_last format (from mslk kernel)
    if not (is_input_channels_last or is_weight_channels_last):
        output = output.contiguous()
    return output


@implements(aten.convolution.default)
def _(func, types, args, kwargs):
    """The semantics of memory_format will match high precision counterparts
    i.e. if any of input or weight are in channels_last_3d format
    the output will be in channels_last_3d format, otherwise the output
    will be contiguous
    """
    (
        input_tensor,
        weight_tensor,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ) = args
    assert not transposed, "transposed conv is not supported currently"
    dim = len(output_padding)
    assert dim in [2, 3], "Only 2d or 3d convs are supported"
    assert groups == 1, f"Only 1 is supported for `groups`, got: {groups}"

    if dim == 2:
        # (N, C, H, W) --> (N, C, 1, H, W)
        input_tensor = input_tensor.unsqueeze(2)
        weight_tensor = weight_tensor.unsqueeze(2)
        assert tuple(output_padding) == (0, 0), (
            f"Only (0, 0) is supported for `output_padding`, got: f{output_padding}"
        )
        padding = [0, *padding]
        stride = [1, *stride]
        dilation = [1, *dilation]
        res = _quantize_and_scaled_conv3d(
            input_tensor,
            weight_tensor,
            bias,
            stride,
            padding,
            dilation,
        )
        assert res.shape[2] == 1
        res = res.squeeze(2)
        return res
    else:
        assert tuple(output_padding) == (0, 0, 0), (
            f"Only (0, 0, 0) is supported for `output_padding`, got: f{output_padding}"
        )
        return _quantize_and_scaled_conv3d(
            input_tensor,
            weight_tensor,
            bias,
            stride,
            padding,
            dilation,
        )


@implements(aten.conv3d.default)
def _(func, types, args, kwargs):
    """The semantics of memory_format will match high precision counterparts
    i.e. if any of input or weight are in channels_last_3d format
    the output will be in channels_last_3d format, otherwise the output
    will be contiguous
    """
    (
        input_tensor,
        weight_tensor,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ) = fill_defaults(args, 7, [None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1])
    conv3d_output = _quantize_and_scaled_conv3d(
        input_tensor,
        weight_tensor,
        bias,
        stride,
        padding,
        dilation,
    )
    return conv3d_output


@implements(aten.conv2d.default)
def _(func, types, args, kwargs):
    """The semantics of memory_format will match high precision counterparts
    i.e. if any of input or weight are in channels_last_3d format
    the output will be in channels_last_3d format, otherwise the output
    will be contiguous
    """
    (
        input_tensor,
        weight_tensor,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ) = fill_defaults(args, 7, [None, [1, 1], [0, 0], [1, 1], 1])
    # (N, C, H, W) --> (N, C, 1, H, W)
    input_tensor = input_tensor.unsqueeze(2)
    weight_tensor = weight_tensor.unsqueeze(2)

    padding = [0, *padding]
    stride = [1, *stride]
    dilation = [1, *dilation]
    res = _quantize_and_scaled_conv3d(
        input_tensor,
        weight_tensor,
        bias,
        stride,
        padding,
        dilation,
    )
    assert res.shape[2] == 1
    res = res.squeeze(2)
    return res


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Supports slicing for 1d, 2d, and 3d tensors
    original tensor shape has dimension (N, K), or (E, N, K)
    qdata has dimension (N, K) or (E, N, K)
    scale (per row quantization) has dimension: (N,) or (E, N)

    since qdata has the same dimension as original tensor, we can directly slice that
    for scale, we'll do a slice when dim is 0, and don't need to do anything for dim 1

    Note that we need to call slice on the qdata and scale directly because slice
    is an operation that need to preserve aliasing
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1 or dim == 2, (
        f"Only dim==0,1,2 are supported, got: dim={dim}"
    )
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self.qdata.ndim == 2 or self.qdata.ndim == 3, (
        f"Expected packed weight to have dim==2,3 got: dim={self.qdata.ndim}"
    )

    # Always slice the qdata
    sliced_data = aten.slice.Tensor(self.qdata, dim, start, end, step)

    if self.scale.numel() == 1:
        # Per-tensor quantization - scale doesn't change
        sliced_scale = self.scale
    else:
        # Block-wise quantization - need to slice the scale appropriately
        sliced_scale = _slice_scale_for_dimension(
            self.scale, self.qdata.shape, dim, start, end, step
        )

    # adjust block_size since the shape has changed, block_size[i] should not be greater than shape[i]
    block_size = self.block_size.copy()
    for i in range(len(self.block_size)):
        block_size[i] = min(block_size[i], sliced_data.shape[i])

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Float8Tensor(
            sliced_data,
            sliced_scale,
            block_size,
            self.mm_config,
            self.act_quant_kwargs,
            self.kernel_preference,
            dtype=self.dtype,
        ),
    )


@implements(aten.cat.default)
def _(func, types, args, kwargs):
    """Concatenate multiple float8 quantized tensors
    (scale and qdata has the same rank)
    If the concatenation dimension is not the same as block_size, then we can just concatenate the
    qdata and scale directly
    If the concatention dimension is the same as block_size, theoretically we should either
      (1) check that scales from all tensors are equal and use the first scale
      (2) dequantize and requantize
    but for now we just use the first scale directly, which might have slight implication on accuaracy
    we can improve upon this a bit later
    """

    tensors, dim = fill_defaults(args, 2, [[], 0])
    tensor_0 = tensors[0]
    dim = dim % tensor_0.ndim

    for i in range(1, len(tensors)):
        assert tensor_0.qdata.ndim == tensors[i].qdata.ndim
        assert tensor_0.scale.ndim == tensors[i].scale.ndim
        assert tensor_0.block_size == tensors[i].block_size
        assert tensor_0.mm_config == tensors[i].mm_config
        assert tensor_0.act_quant_kwargs == tensors[i].act_quant_kwargs
        assert tensor_0.kernel_preference == tensors[i].kernel_preference

    qdatas = [t.qdata for t in tensors]
    scales = [t.scale for t in tensors]

    cat_qdata = aten.cat.default(qdatas, dim=dim)
    if tensor_0.block_size[dim] == 1:
        cat_scale = aten.cat.default(scales, dim=dim)
    else:
        for i in range(1, len(tensors)):
            assert torch.equal(tensor_0.scale, tensors[i].scale)
        cat_scale = scales[0]

    block_size = []
    for i in range(cat_qdata.ndim):
        block_size.append(cat_qdata.shape[i] // cat_scale.shape[i])

    new = tensor_0.__class__(
        cat_qdata,
        cat_scale,
        block_size,
        tensor_0.mm_config,
        tensor_0.act_quant_kwargs,
        tensor_0.kernel_preference,
        tensor_0.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args
    qdata = self.qdata.transpose(dim0, dim1)
    scale = self.scale.transpose(dim0, dim1)
    block_size = self.block_size.copy()

    block_size[dim0], block_size[dim1] = block_size[dim1], block_size[dim0]

    new = self.__class__(
        qdata,
        scale,
        block_size,
        self.mm_config,
        self.act_quant_kwargs,
        self.kernel_preference,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.view.default)
def _(func, types, args, kwargs):
    self, size = args
    original_shape = self.shape
    if len(original_shape) == 3 and len(size) == 2:
        assert original_shape[-1] == size[-1], (
            f"Only support reshaping when last dimension matches, requested: reshaping from {original_shape} to {size}"
        )
        # TODO: this seems wrong, we should merge the first two dimensions instead
        qdata = self.qdata.reshape(*size)
        scale = self.scale.reshape(*size)
        block_size = self.block_size.copy()
        block_size = [block_size[0] * block_size[1], block_size[2]]
    elif len(original_shape) == 2 and len(size) == 3:
        assert original_shape[-1] == size[-1], (
            f"Only support reshaping when last dimension matches, requested: reshaping from {original_shape} to {size}"
        )
        qdata = self.qdata.reshape(*size)
        block_size = self.block_size.copy()
        block_size = [1, block_size[0], block_size[1]]
        scale_shape = []
        for i in range(3):
            scale_shape.append(qdata.shape[i] // block_size[i])
        scale = self.scale.reshape(*scale_shape)
    elif len(original_shape) == len(size):
        assert all(x == y or y == -1 for x, y in zip(original_shape, size)), (
            f"Only support viewing with match dimensions or -1, got: {original_shape}, {size}"
        )
        qdata = self.qdata.reshape(*size)
        scale_shape = []
        for i in range(3):
            scale_shape.append(qdata.shape[i] // self.block_size[i])
        scale = self.scale.reshape(*scale_shape)
        block_size = self.block_size
    else:
        assert len(original_shape) == 2 and len(size) == 3, (
            f"Only support reshaping from 2D to 3D or from 3D to 2D, requested: reshaping from {original_shape} to {size}"
        )

    new = self.__class__(
        qdata,
        scale,
        block_size,
        self.mm_config,
        self.act_quant_kwargs,
        self.kernel_preference,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.squeeze.dim)
def _(func, types, args, kwargs):
    self, dim = args
    assert dim == 0, f"Only dim == 0 is supported, got: {dim}"
    qdata = self.qdata.squeeze(dim=dim)
    scale = self.scale.squeeze(dim=dim)
    block_size = []
    for i in range(len(qdata.shape)):
        block_size.append(qdata.shape[i] // scale.shape[i])

    new = self.__class__(
        qdata,
        scale,
        block_size,
        self.mm_config,
        self.act_quant_kwargs,
        self.kernel_preference,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.select.int)
def _(func, types, args, kwargs):
    old_float8_tensor, dim, index = args
    assert dim == 0, f"Float8Tensor aten.select.int with {dim=} is not yet supported"
    assert len(old_float8_tensor.qdata.shape) == len(old_float8_tensor.scale.shape), (
        "unsupported"
    )
    assert len(old_float8_tensor.qdata.shape) == len(old_float8_tensor.block_size), (
        "unsupported"
    )
    new_float8_tensor = old_float8_tensor.__class__(
        old_float8_tensor.qdata[index],
        old_float8_tensor.scale[index],
        old_float8_tensor.block_size[1:],
        old_float8_tensor.mm_config,
        old_float8_tensor.act_quant_kwargs,
        old_float8_tensor.kernel_preference,
        old_float8_tensor.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new_float8_tensor)


@implements(aten.unsqueeze.default)
def _(func, types, args, kwargs):
    self, dim = args
    qdata = self.qdata.unsqueeze(dim=dim)
    scale = self.scale.unsqueeze(dim=dim)
    block_size = []
    for i in range(len(qdata.shape)):
        block_size.append(qdata.shape[i] // scale.shape[i])

    new = self.__class__(
        qdata,
        scale,
        block_size,
        self.mm_config,
        self.act_quant_kwargs,
        self.kernel_preference,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.t.default)
def _(func, types, args, kwargs):
    assert len(args) == 1
    self = args[0]
    assert len(self.block_size) == 2
    new_tensor = self.__class__(
        self.qdata.t(),
        self.scale.t(),
        (self.block_size[1], self.block_size[0]),
        self.mm_config,
        self.act_quant_kwargs,
        self.kernel_preference,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new_tensor)


@implements_torch_function(torch.Tensor.t)
def _(func, types, args, kwargs):
    assert len(args) == 1
    self = args[0]
    assert len(self.block_size) == 2
    new_tensor = self.__class__(
        self.qdata.t(),
        self.scale.t(),
        (self.block_size[1], self.block_size[0]),
        self.mm_config,
        self.act_quant_kwargs,
        self.kernel_preference,
        self.dtype,
    )
    return new_tensor


@implements(aten.split.Tensor)
def _(func, types, args, kwargs):
    tensor, split_size_or_sections, dim = args
    assert isinstance(split_size_or_sections, int), "unimplemented"

    # 2D case
    #
    # orig
    #   qdata.shape [M, K]
    #   scale.shape [M, 1]
    #   block_size [1, K]
    #
    # split with size (K // 2) across dim -1:
    #   qdata.shape [M, K // 2], [M, K // 2]
    #   scale.shape [M, 1], [M, 1]
    #   block_size [1, K // 2], [1, K // 2]
    #
    # split with size (M // 2) across dim 0:
    #   qdata.shape [M // 2, K], [M // 2, K]
    #   scale.shape [M // 2, 1], [M // 2, 1]
    #   block_size [1, K], [1, K]

    # split the qdata
    new_qdatas = func(tensor.qdata, split_size_or_sections, dim)
    num_chunks = len(new_qdatas)

    # split the scale
    new_scales = []
    new_block_sizes = []
    if tensor.scale.shape[dim] == 1 and tensor.block_size[dim] == tensor.shape[dim]:
        # repeat the scale, split block_size
        for _ in range(num_chunks):
            new_scales.append(tensor.scale)
            new_block_size = tensor.block_size
            new_block_size[dim] = new_block_size[dim] // split_size_or_sections
            new_block_sizes.append(new_block_size)

    elif tensor.scale.shape[dim] == tensor.shape[dim] and tensor.block_size[dim] == 1:
        # repeat the block size, split scale
        new_scales = func(tensor.scale, split_size_or_sections, dim)
        for _ in range(num_chunks):
            new_block_sizes.append(tensor.block_size)

    else:
        raise AssertionError(
            f"`aten.split.Tensor` with {dim=} and {tensor.scale.shape=} is not yet implemented"
        )

    new_tensors_list = []
    for idx in range(num_chunks):
        new_tensor = tensor.__class__(
            new_qdatas[idx],
            new_scales[idx],
            new_block_sizes[idx],
            tensor.mm_config,
            tensor.act_quant_kwargs,
            tensor.kernel_preference,
            tensor.dtype,
        )
        new_tensor = return_and_correct_aliasing(func, args, kwargs, new_tensor)
        new_tensors_list.append(new_tensor)

    new_tensors_tuple = tuple(new_tensors_list)
    return new_tensors_tuple


Float8Tensor.__module__ = "torchao.quantization"

# Allow a model with Float8Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Float8Tensor, QuantizeTensorToFloat8Kwargs])
