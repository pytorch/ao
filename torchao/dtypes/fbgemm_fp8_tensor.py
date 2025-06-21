# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.floatx.float8_layout import (
    preprocess_scale,
)
from torchao.dtypes.utils import get_out_shape
from torchao.float8.inference import (
    Float8MMConfig,
    addmm_float8_unwrapped_inference,
    preprocess_data,
)
from torchao.quantization.granularity import PerRow
from torchao.quantization.observer import get_block_size
from torchao.quantization.quant_primitives import (
    _choose_qparams_affine_float8,
    _quantize_affine_float8,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
    fill_defaults,
)

__all__ = [
    "to_fbgemm_fp8",
    "FbgemmFp8Tensor",
]

aten = torch.ops.aten


class FbgemmFp8Tensor(TorchAOBaseTensor):
    """
    Float8 Rowwise Quantized (weight) Tensor, with float8 rowwise dynamic quantization for activation.
    TODO: needs padding for cutlass kernels

    Tensor Attributes:
        float8_data: float8 raw data
        scale: the rowwise scale for float8 Tensor
        activation_scale_ub: upper bound for activation scale, used during dynamic quantization for activation

    Non-Tensor Attributes:
        rowwise_dim (int): the dimension for rowwise quantization, initially it's -1, but might change when we
        transpose the Tensor
        dtype: Original Tensor dtype
    """

    tensor_data_attrs = ["float8_data", "scale", "activation_scale_ub"]
    tensor_attributes = ["rowwise_dim", "mm_config", "kernel", "dtype"]
    _SUPPORTED_KERNELS = ["fbgemm", "aten"]

    def __new__(
        cls,
        float8_data,
        scale,
        activation_scale_ub,
        rowwise_dim,
        mm_config,
        kernel,
        dtype,
    ):
        shape = float8_data.shape
        kwargs = {}
        kwargs["device"] = float8_data.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        float8_data,
        scale,
        activation_scale_ub,
        rowwise_dim,
        mm_config,
        kernel,
        dtype,
    ):
        self.float8_data = float8_data
        self.scale = scale
        self.activation_scale_ub = activation_scale_ub
        self.rowwise_dim = rowwise_dim % self.float8_data.ndim
        self.mm_config = mm_config
        self.kernel = kernel

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
        return (
            f"{self.__class__.__name__}(weight={self.float8_data}, scale={self.scale}, "
            f"activation_scale_ub={self.activation_scale_ub}, rowwise_dim={self.rowwise_dim}, "
            f"mm_config={self.mm_config}, kernel={self.kernel}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, activation_scale_ub={self.activation_scale_ub}, rowwise_dim={self.rowwise_dim}, mm_config={self.mm_config}, kernel={self.kernel}, device={self.device}"

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.float8_data.to(device),
            self.scale.to(device),
            self.activation_scale_ub.to(device),
            self.rowwise_dim,
            self.mm_config,
            self.kernel,
            self.dtype,
        )

    def _transpose_and_reshape(self):
        """This is added for resharding support, since the resharding logic for the model we are
        working with only support 2D

        High level goal is to match the shape of the original unquantized Tensor and reshape
        it to 2D since resharding logic only supports 2D Tensor

        * transpose(1, 2) since we did a transpose initially to quantize the weight
        * reshape to 2D
        """
        assert len(self.shape) == 3, (
            f"Only expected to be used when the Tensor is 3D, got {len(self.shape)}"
        )
        dim0, dim1, dim2 = self.shape
        # because we first transpose the weight before quantization, we'll recover the original shape
        # by swapping dim1 and dim2
        original_shape = (dim0, dim2, dim1)
        # we must save this as 2D in the state dict, since loading code expects 2D weights
        new_shape = (-1, original_shape[-1])
        float8_data = self.float8_data
        float8_data = float8_data.transpose(1, 2).reshape(*new_shape).contiguous()
        scale = self.scale.transpose(1, 2).reshape(*new_shape).contiguous()
        if self.rowwise_dim in [0, 2]:
            rowwise_dim = 0
        else:
            rowwise_dim = 1

        return self.__class__(
            float8_data,
            scale,
            self.activation_scale_ub,
            rowwise_dim,
            self.mm_config,
            self.kernel,
            self.dtype,
        )

    def _unflatten(self, num_experts):
        """This is added for resharding support, since the resharding logic for the model we are
        working with only support 2D

        This is called after resharding logic, and it reverses the reshape to 2D in `_transpose_and_reshape`
        and gives a 3D tensor with `num_experts` as the first dimension
        """
        float8_data = self.float8_data
        scale = self.scale
        float8_data = float8_data.unflatten(0, (num_experts, -1)).squeeze(dim=0)
        scale = scale.unflatten(0, (num_experts, -1)).squeeze(dim=0)
        if self.rowwise_dim == 0:
            rowwise_dim = 1
        else:
            rowwise_dim = 2

        for d in range(len(float8_data.shape)):
            if float8_data.shape[d] != scale.shape[d] and scale.shape[d] == 1:
                rowwise_dim = d

        return self.__class__(
            float8_data,
            scale,
            self.activation_scale_ub,
            rowwise_dim,
            self.mm_config,
            self.kernel,
            self.dtype,
        )

    @classmethod
    def from_float(
        cls,
        w: torch.Tensor,
        input_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        activation_scale_ub: Optional[float] = None,
        mm_config: Optional[Float8MMConfig] = None,
        kernel: str = "fbgemm",
    ):
        assert kernel in cls._SUPPORTED_KERNELS

        assert input_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
        assert weight_dtype == input_dtype

        if activation_scale_ub is None:
            activation_scale_ub = 1200.0

        activation_scale_ub = torch.tensor(
            [activation_scale_ub],
            dtype=torch.float,
            device=w.device,
        )

        if kernel == "fbgemm":
            wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
            # add a last dimension for per row quantization to align the rank of
            # w_scale and wq
            w_scale = w_scale.unsqueeze(-1).contiguous()
            dtype = w.dtype
            del w
            return FbgemmFp8Tensor(
                wq,
                w_scale,
                activation_scale_ub=activation_scale_ub,
                rowwise_dim=wq.ndim - 1,
                mm_config=mm_config,
                kernel=kernel,
                dtype=dtype,
            )

        else:
            block_size = get_block_size(w.shape, PerRow())
            w_scale = _choose_qparams_affine_float8(
                w, float8_dtype=weight_dtype, block_size=block_size
            )
            wq = _quantize_affine_float8(w, w_scale, weight_dtype)
            dtype = w.dtype
            del w
            return FbgemmFp8Tensor(
                wq,
                w_scale,
                activation_scale_ub=activation_scale_ub,
                rowwise_dim=wq.ndim - 1,
                mm_config=mm_config,
                kernel=kernel,
                dtype=dtype,
            )


implements = FbgemmFp8Tensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    orig_act_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    if weight_tensor.kernel == "fbgemm":
        # not used
        num_tokens = torch.empty([input_tensor.size(0)], device=input_tensor.device)
        a_data, a_scale = torch.ops.fbgemm.quantize_fp8_per_row(
            input_tensor, num_tokens, weight_tensor.activation_scale_ub
        )

        b_data = weight_tensor.float8_data
        b_scale = weight_tensor.scale.squeeze(-1)

        res = torch.ops.fbgemm.f8f8bf16_rowwise(
            a_data,
            b_data,
            a_scale,
            b_scale,
            use_fast_accum=True,
        )
        res = res.reshape(*orig_act_size[:-1], orig_out_features)
        if bias is not None:
            res = res + bias
        return res
    else:
        scaled_mm_config = weight_tensor.mm_config
        assert scaled_mm_config is not None
        out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)

        block_size = get_block_size(input_tensor.shape, PerRow())
        weight_dtype = weight_tensor.float8_data.dtype
        # Note: we assume input dtype is the same as weight dtype
        input_scale = _choose_qparams_affine_float8(
            input_tensor, float8_dtype=weight_dtype, block_size=block_size
        )
        inpt_data = _quantize_affine_float8(input_tensor, input_scale, weight_dtype)

        # Extract tensor data and scales
        inpt_data = inpt_data.reshape(-1, inpt_data.shape[-1])

        w_data = weight_tensor.float8_data
        w_scale = weight_tensor.scale
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


@implements(torch.bmm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    orig_act_size = input_tensor.size()
    assert weight_tensor.kernel == "fbgemm", (
        f"Only fbgemm kernel support bmm right now, got {weight_tensor.kernel}"
    )

    # not used
    num_tokens = torch.empty([input_tensor.size(0)], device=input_tensor.device)
    a_data, a_scale = torch.ops.fbgemm.quantize_fp8_per_row(
        input_tensor, num_tokens, weight_tensor.activation_scale_ub
    )

    b_data = weight_tensor.float8_data
    b_scale = weight_tensor.scale.squeeze(-1)
    assert b_data.is_contiguous(), "weight for bmm must be contiguous"

    orig_out_features = b_data.shape[-2]

    res = torch.ops.fbgemm.f8f8bf16_rowwise_batched(
        a_data,
        b_data,
        a_scale,
        b_scale,
    )
    res = res.reshape(*orig_act_size[:-1], orig_out_features)
    return res


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


def _same_metadata(self: "FbgemmFp8Tensor", src: "FbgemmFp8Tensor") -> bool:
    return (
        isinstance(self, FbgemmFp8Tensor)
        and isinstance(src, FbgemmFp8Tensor)
        and self.shape == src.shape
        and self.float8_data.shape == src.float8_data.shape
        and self.scale.shape == src.scale.shape
        and self.activation_scale_ub.shape == src.activation_scale_ub.shape
        and self.rowwise_dim == src.rowwise_dim
        and self.dtype == src.dtype
    )


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if _same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return
    raise ValueError(
        f"Not supported args for copy_ due to metadata mismatch: {args[0], args[1]}"
    )


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Only supports slicing for dim == 1 and dim == 2
    original tensor shape has dimension (N, K)
    float8_data has dimension (N, K)
    scale (per row quantization) has dimension: (N,)

    since float8_data has the same dimension as original tensor, we can directly slice that
    for scale, we'll do a slice when dim is 0, and don't need to do anything for dim 1

    Note that we need to call slice on the float8_data and scale directly because slice
    is an operation that need to preserve aliasing, see `test_slice_and_copy_` in `test_fbgemm_fp8`
    for
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self.float8_data.ndim == 2, (
        f"Expected packed weight to have dim 2, got {self.float8_data.dim}"
    )

    # Always slice the float8_data
    sliced_data = aten.slice.Tensor(
        self.float8_data, dim, start, end, step
    ).contiguous()

    if dim != self.rowwise_dim:
        # scale has dimension (N,) where N is the dim 0 of `self`
        # so we do the same slice on scale for dimension 0
        sliced_scale = aten.slice.Tensor(self.scale, 0, start, end, step)
    else:
        # since scale is per row, slicing along the rowwise dimension does
        # not change the scale
        sliced_scale = self.scale

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        FbgemmFp8Tensor(
            sliced_data,
            sliced_scale,
            self.activation_scale_ub,
            self.rowwise_dim,
            self.mm_config,
            self.kernel,
            dtype=self.dtype,
        ),
    )


@implements(aten.cat.default)
def _(func, types, args, kwargs):
    """Concatenate multiple float8 quantized tensors
    (scale and float8_data has the same rank)
    If the concatenation dimension is not the same as rowwise_dim, then we can just concatenate the
    float8_data and scale directly
    If the concatention dimension is the same as rowwise_dim, theoretically we should either
      (1) check that scales from all tensors are equal and use the first scale
      (2) dequantize and requantize
    but for now we just use the first scale directly, which might have slight implication on accuaracy
    we can improve upon this a bit later
    """

    tensors, dim = fill_defaults(args, 2, [[], 0])
    tensor_0 = tensors[0]
    dim = dim % tensor_0.ndim

    # assert dim != tensor_0.rowwise_dim, f"Doesn't support concatenation over rowwise dimension: {dim=} {tensor_0.float8_data.shape=}, {tensor_0.rowwise_dim=} {tensor_0.scale.shape=}"

    for i in range(1, len(tensors)):
        assert tensor_0.float8_data.ndim == tensors[i].float8_data.ndim
        assert tensor_0.scale.ndim == tensors[i].scale.ndim
        assert tensor_0.activation_scale_ub == tensors[i].activation_scale_ub
        assert tensor_0.rowwise_dim == tensors[i].rowwise_dim

    float8_datas = [t.float8_data for t in tensors]
    scales = [t.scale for t in tensors]

    cat_float8_data = aten.cat.default(float8_datas, dim=dim)
    if dim != tensor_0.rowwise_dim:
        cat_scale = aten.cat.default(scales, dim=dim)
    else:
        # TODO: this is not exactly correct, we'll need to
        # figure out how to do this properly in the future
        cat_scale = scales[0]

    new = tensor_0.__class__(
        cat_float8_data,
        cat_scale,
        tensor_0.activation_scale_ub,
        tensor_0.rowwise_dim,
        tensor_0.mm_config,
        tensor_0.kernel,
        tensor_0.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args
    float8_data = self.float8_data.transpose(dim0, dim1).contiguous()
    scale = self.scale.transpose(dim0, dim1).contiguous()

    if self.rowwise_dim == dim0:
        rowwise_dim = dim1
    elif self.rowwise_dim == dim1:
        rowwise_dim = dim0

    new = self.__class__(
        float8_data,
        scale,
        self.activation_scale_ub,
        rowwise_dim,
        self.mm_config,
        self.kernel,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


to_fbgemm_fp8 = FbgemmFp8Tensor.from_float


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with FbgemmFp8Tensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([FbgemmFp8Tensor])
