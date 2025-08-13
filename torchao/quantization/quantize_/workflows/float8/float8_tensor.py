# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.utils import get_out_shape
from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _is_rowwise_scaled,
    _is_tensorwise_scaled,
    _slice_scale_for_dimension,
    addmm_float8_unwrapped_inference,
    preprocess_data,
    preprocess_scale,
)
from torchao.quantization.granularity import PerRow
from torchao.quantization.observer import get_block_size
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
from torchao.utils import (
    TorchAOBaseTensor,
    _is_fbgemm_genai_gpu_available,
    fill_defaults,
    is_sm_at_least_90,
)

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
        hp_value_lb (Optional[float]): the lower bound for high precision floating point value for calculating scale
        hp_value_ub (Optional[float]): the upper bound for high precision floating point value for calculating scale
        act_quant_kwargs (QuantizeTensorToFloat8Kwargs): the kwargs for Float8Tensor.to_float8
        kernel_preference (KernelPreference): the preference for quantize, mm etc. kernel to use,
        by default, this will be chosen for user based on hardware, library availabilities etc.
        dtype: Original Tensor dtype
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = [
        "block_size",
        "mm_config",
        "hp_value_lb",
        "hp_value_ub",
        "act_quant_kwargs",
        "kernel_preference",
        "dtype",
    ]

    def __new__(
        cls,
        qdata,
        scale,
        block_size,
        mm_config,
        hp_value_lb,
        hp_value_ub,
        act_quant_kwargs,
        kernel_preference,
        dtype,
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
        hp_value_lb: Optional[float] = None,
        hp_value_ub: Optional[float] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        dtype: Optional[torch.dtype] = None,
    ):
        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        self.mm_config = mm_config
        self.hp_value_lb = hp_value_lb
        self.hp_value_ub = hp_value_ub
        self.act_quant_kwargs = act_quant_kwargs
        self.kernel_preference = kernel_preference

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata=}, {self.scale=}, "
            f"{self.block_size=}, {self.mm_config=}, "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    def _quantization_type(self):
        return f"{self.act_quant_kwargs=}, {self.block_size=}, {self.mm_config=}, {self.scale.shape=}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        qdata, scale = self.qdata, self.scale
        return _dequantize_affine_float8(qdata, scale, output_dtype)

    @classmethod
    def to_float8(
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

        # for per row quantization and kernel_preference default setting, we'll use triton kernel for best performance
        if (
            kernel_preference == KernelPreference.AUTO
            and _is_fbgemm_genai_gpu_available()
            and (
                tuple(block_size)
                == (1,) * (hp_tensor.ndim - 1) + (hp_tensor.shape[-1],)
            )
        ):
            assert float8_dtype == torch.float8_e4m3fn, (
                f"Only torch.float8_e4m3fn is supported, got: {float8_dtype}"
            )
            if hp_value_ub is not None:
                maybe_hp_value_ub_tensor = torch.tensor(
                    hp_value_ub, dtype=torch.float, device=hp_tensor.device
                )
            else:
                maybe_hp_value_ub_tensor = None
            data, scale = torch.ops.triton.quantize_fp8_row(
                hp_tensor, scale_ub=maybe_hp_value_ub_tensor
            )
            scale_shape = []
            for i in range(hp_tensor.ndim):
                scale_shape.append(hp_tensor.shape[i] // block_size[i])
            scale = scale.reshape(*scale_shape)
        else:
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
            hp_value_lb=hp_value_lb,
            hp_value_ub=hp_value_ub,
            act_quant_kwargs=act_quant_kwargs,
            kernel_preference=kernel_preference,
            dtype=hp_dtype,
        )


implements = Float8Tensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert isinstance(weight_tensor, Float8Tensor), (
        f"Don't expect to reach here with an override other than weight currently, {type(input_tensor)} {type(weight_tensor)}"
    )

    act_quant_kwargs = weight_tensor.act_quant_kwargs
    # quantizing activation, if `act_quant_kwargs` is specified
    if act_quant_kwargs is not None:
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

    if isinstance(input_tensor, Float8Tensor):
        kernel_choice = None

        if weight_tensor.kernel_preference == KernelPreference.AUTO:
            kernel_choice = "torch"
            if _is_fbgemm_genai_gpu_available() and is_sm_at_least_90():
                kernel_choice = "fbgemm"
        elif weight_tensor.kernel_preference == KernelPreference.FBGEMM:
            kernel_choice = "fbgemm"
        else:
            assert weight_tensor.kernel_preference == KernelPreference.TORCH, (
                f"{weight_tensor.kernel_preference=} not handled"
            )
            kernel_choice = "torch"

        if kernel_choice == "fbgemm":
            assert _is_fbgemm_genai_gpu_available(), (
                "Expected fbgemm_gpu_genai package to be installed"
            )
            assert is_sm_at_least_90(), "Expected SM90+ for fbgemm_gpu_genai"

            out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)
            xq = input_tensor.qdata.reshape(-1, input_tensor.qdata.shape[-1])
            wq = weight_tensor.qdata
            x_scale = input_tensor.scale
            w_scale = weight_tensor.scale
            if _is_rowwise_scaled(weight_tensor):
                assert _is_rowwise_scaled(input_tensor), (
                    "Input tensor must be rowwise block size"
                )
                res = torch.ops.fbgemm.f8f8bf16_rowwise(
                    xq,
                    wq,
                    x_scale,
                    w_scale,
                ).reshape(out_shape)
            else:
                assert _is_tensorwise_scaled(weight_tensor)
                assert _is_tensorwise_scaled(input_tensor)
                res = torch.ops.fbgemm.f8f8bf16(
                    xq,
                    wq,
                    x_scale * w_scale,
                ).reshape(out_shape)
            if bias is not None:
                res = res + bias
            return res
        else:
            assert kernel_choice == "torch"
            scaled_mm_config = weight_tensor.mm_config
            assert scaled_mm_config is not None
            out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)

            # Extract tensor data and scales
            inpt_data = input_tensor.qdata.reshape(-1, input_tensor.qdata.shape[-1])
            w_data = weight_tensor.qdata
            input_scale = input_tensor.scale
            w_scale = weight_tensor.scale

            # Handle rowwise scaling
            if _is_rowwise_scaled(weight_tensor):
                assert _is_rowwise_scaled(input_tensor), (
                    "Input tensor must be rowwise block size"
                )
                w_scale = w_scale.transpose(-1, -2)

            input_scale = preprocess_scale(input_scale, input_tensor.shape)
            inpt_data, w_data = preprocess_data(inpt_data, w_data.T, scaled_mm_config)

            return addmm_float8_unwrapped_inference(
                inpt_data,
                input_scale,
                w_data,
                w_scale,
                output_dtype=input_tensor.dtype,
                bias=bias,
                use_fast_accum=scaled_mm_config.use_fast_accum,
            ).reshape(out_shape)
    else:
        assert not isinstance(input_tensor, TorchAOBaseTensor), (
            "Expecting input_tensor to be unquantized"
        )
        # when input is not `Float8Tensor`, we expect that it is not quantized
        # so this is float8 weight only quantization
        return torch.nn.functional.linear(
            input_tensor, weight_tensor.dequantize(), bias
        )


@implements(torch.bmm)
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
    assert _is_fbgemm_genai_gpu_available(), (
        "bmm is not supported when fbgemm_gpu_genai is not installed"
    )

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
        b_scale = weight_tensor.scale.squeeze(-1)
        assert b_data.is_contiguous(), "weight for bmm must be contiguous"

        assert (
            all(x == 1 for x in weight_tensor.block_size[:-1])
            and weight_tensor.block_size[-1] == weight_tensor.shape[-1]
        ), "bmm only works for per row weight quantization"
        assert (
            all(x == 1 for x in input_tensor.block_size[:-1])
            and input_tensor.block_size[-1] == input_tensor.shape[-1]
        ), "bmm only works for per row activation quantization"

        orig_out_features = b_data.shape[-2]

        res = torch.ops.fbgemm.f8f8bf16_rowwise_batched(
            a_data,
            b_data,
            a_scale,
            b_scale,
        )
        res = res.reshape(*orig_act_size[:-1], orig_out_features)
    else:
        raise NotImplementedError(
            "bmm only support float8 dynamic activation + float8 weight"
        )

    return res


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Only supports slicing for dim == 1 and dim == 2
    original tensor shape has dimension (N, K)
    qdata has dimension (N, K)
    scale (per row quantization) has dimension: (N,)

    since qdata has the same dimension as original tensor, we can directly slice that
    for scale, we'll do a slice when dim is 0, and don't need to do anything for dim 1

    Note that we need to call slice on the qdata and scale directly because slice
    is an operation that need to preserve aliasing
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self.qdata.ndim == 2, (
        f"Expected packed weight to have dim 2, got {self.qdata.dim}"
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
            self.hp_value_lb,
            self.hp_value_ub,
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
        assert tensor_0.hp_value_lb == tensors[i].hp_value_lb
        assert tensor_0.hp_value_ub == tensors[i].hp_value_ub
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
        tensor_0.hp_value_lb,
        tensor_0.hp_value_ub,
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
        self.hp_value_lb,
        self.hp_value_ub,
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
        self.hp_value_lb,
        self.hp_value_ub,
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
        self.hp_value_lb,
        self.hp_value_ub,
        self.act_quant_kwargs,
        self.kernel_preference,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


Float8Tensor.__module__ = "torchao.quantization"

# Allow a model with Float8Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Float8Tensor, QuantizeTensorToFloat8Kwargs])
