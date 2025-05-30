# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout, get_out_shape
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
    to_sparse_semi_structured_cutlass_sm9x_f8,
)

aten = torch.ops.aten

def _pad_dense_input(dense_input: torch.Tensor) -> torch.Tensor:
    """
    Calculates padding for dense tensor and pads tensor if necessary.
    If padding is not required, this function returns the original tensor.
    """
    # only 2d matmul
    assert dense_input.dim() == 2

    # check shape
    m, n = dense_input.shape
    min_rows = 64
    min_cols = 64

    # calculate padding
    to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
    to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
    if to_pad_m or to_pad_n:
        return torch.nn.functional.pad(dense_input, (0, to_pad_n, 0, to_pad_m))
    else:
        return dense_input
        
def _pad_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Calculates padding for dense tensor and pads tensor if necessary.
    If padding is not required, this function returns the original tensor.
    """
    # only 2d matmul
    assert scale.dim() == 2

    # check shape
    m, n = scale.shape
    assert n == 1
    min_rows = 64
    # min_cols = 64

    # calculate padding
    to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
    # to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
    if to_pad_m:
        return torch.nn.functional.pad(scale, (0, 0, 0, to_pad_m))
    else:
        return scale

def _same_metadata(
    self: "CutlassSemiSparseTensorImpl", src: "CutlassSemiSparseTensorImpl"
) -> bool:
    return (
        isinstance(self, CutlassSemiSparseTensorImpl)
        and isinstance(src, CutlassSemiSparseTensorImpl)
        and self.shape == src.shape
        and self.sparse.shape == src.sparse.shape
        and self.meta.shape == src.meta.shape
        and self.scale.shape == src.scale.shape
        and type(self._layout) == type(src._layout)
    )


@dataclass(frozen=True)
class CutlassSemiSparseLayout(Layout):
    """Layout class for float8 2:4 sparsity layout for affine quantized tensor, for cutlass kernel."""

    # def pre_process(self, dense: torch.Tensor) -> torch.Tensor:
        # # prune to 2:4 if not already
        # from torchao.sparsity.utils import mask_creator

        # return dense * mask_creator(dense).bool()


@register_layout(CutlassSemiSparseLayout)
class CutlassSemiSparseTensorImpl(AQTTensorImpl):
    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = sparse.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else sparse.layout
        )
        kwargs["dtype"] = sparse.dtype
        kwargs["requires_grad"] = False
        # shape = (sparse.shape[0], 2 * sparse.shape[-1])
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        shape: torch.Size,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        self.sparse = sparse
        self.meta = meta
        self.scale = scale
        self._layout = _layout
        self._shape = shape

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        elif func is aten.copy_.default:
            self = args[0]
            src = args[1]
            if _same_metadata(self, src):
                self_tensors = self.__tensor_flatten__()[0]
                for tensor_name in self_tensors:
                    getattr(self, tensor_name).copy_(getattr(src, tensor_name))
                return
            raise ValueError(
                f"Not supported args for copy_ due to metadata mistach: {args[0], args[1]}"
            )

        raise NotImplementedError(
            f"CutlassSemiSparseTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        return ["sparse", "meta", "scale"], [self._layout, self._shape]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        sparse = tensor_data_dict["sparse"]
        meta = tensor_data_dict["meta"]
        scale = tensor_data_dict["scale"]
        (_layout, _shape) = tensor_attributes
        return cls(_shape, sparse, meta, scale, _layout)

    def get_plain(self):
        # No support in CUTLASS to convert back to dense from sparse
        # semi-structured format, so multiplying with identity matrix,
        # and using identity scale factors, for the conversion.
        # breakpoint()
        # raise NotImplementedError("get_plain not supported for CutlassSemiSparseTensorImpl")
        cols = self.shape[-1]
        input = torch.eye(cols, dtype=self.sparse.dtype, device=self.sparse.device)
        input_scale = torch.ones(
            (cols,), dtype=self.scale.dtype, device=self.sparse.device
        )
        sparse_scale = torch.ones_like(self.scale)
        out_dtype = torch.bfloat16
        dense = (
            rowwise_scaled_linear_sparse_cutlass_f8f8(
                input,
                input_scale,
                self.sparse,
                self.meta,
                sparse_scale,
                out_dtype=out_dtype,
            )
            .to(self.dtype)
            .t()
            .contiguous()
        )

        return dense, self.scale, None

    @classmethod
    def from_plain(
        cls,
        dense: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert zero_point is None or torch.all(zero_point == 0)
        # print(dense.shape)
        # dense_2d = dense.view(-1, dense.shape[-1])
        assert dense.ndim == 2
        assert dense.is_contiguous()

        dense_padded = _pad_dense_input(dense)
        scale_padded = _pad_scale(scale)

        # X_scale = torch.empty((dense.shape[0], 1), device=dense.device, dtype=torch.float32)
        Xq_sparse, X_meta = torch.ops.torchao.sparse24_sm90_sparsify(
            dense_padded,
            "cutlass",
            "identity",
            "largest",
            dtype=torch.float8_e4m3fn,
            scale=scale_padded,
        )

        res = cls(
            dense.shape,
            Xq_sparse,
            X_meta,
            scale_padded,
            _layout,
        )
        return res

    def get_layout(self) -> Layout:
        return self._layout

    def _apply_fn_to_data(self, fn):
        self.sparse = fn(self.sparse)
        self.meta = fn(self.meta)
        self.scale = fn(self.scale)
        return self


def _linear_fp8_act_fp8_weight_sparse_cutlass_check(input_tensor, weight_tensor, bias):
    from torchao.dtypes.floatx import Float8Layout

    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and isinstance(input_tensor._layout, Float8Layout)
        and input_tensor.dtype in (torch.float16, torch.bfloat16)
        and len(input_tensor.shape) >= 2
        and input_tensor.tensor_impl.scale.dtype == torch.float32
        and len(input_tensor.tensor_impl.scale.shape) == len(input_tensor.shape) - 1
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and isinstance(weight_tensor._layout, CutlassSemiSparseLayout)
        and weight_tensor.dtype == input_tensor.dtype
        and len(weight_tensor.shape) == 2
        and weight_tensor.tensor_impl.scale.dtype == torch.float32
        and len(weight_tensor.tensor_impl.scale.shape) == 1
        and (bias is None or bias.dtype == input_tensor.dtype)
        and (bias is None or len(bias.shape) == 1)
    )


def _linear_fp8_act_fp8_weight_sparse_cutlass_impl(input_tensor, weight_tensor, bias):
    from torchao.ops import rowwise_scaled_linear_sparse_cutlass_f8f8

    input = input_tensor.tensor_impl.float8_data
    input_scale = input_tensor.tensor_impl.scale
    weight = weight_tensor.tensor_impl.sparse
    weight_meta = weight_tensor.tensor_impl.meta
    weight_scale = weight_tensor.tensor_impl.scale
    out_dtype = input_tensor.dtype

    out = rowwise_scaled_linear_sparse_cutlass_f8f8(
        input, input_scale, weight, weight_meta, weight_scale, bias, out_dtype
    )

    return out

def _linear_fp8_act_sparse_fp8_weight_cutlass_check(input_tensor, weight_tensor, bias):
    from torchao.dtypes.floatx import Float8Layout

    res = (
        isinstance(input_tensor, AffineQuantizedTensor)
        and isinstance(input_tensor._layout, CutlassSemiSparseLayout)
        and input_tensor.dtype in (torch.float16, torch.bfloat16)
        and len(input_tensor.shape) >= 2
        and input_tensor.tensor_impl.scale.dtype == torch.float32
        and len(input_tensor.tensor_impl.scale.shape) == 2
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and isinstance(weight_tensor._layout, Float8Layout)
        and weight_tensor.dtype == input_tensor.dtype
        and len(weight_tensor.shape) == 2
        and weight_tensor.tensor_impl.scale.dtype == torch.float32
        and len(weight_tensor.tensor_impl.scale.shape) == 2
        and (bias is None or bias.dtype == input_tensor.dtype)
        and (bias is None or len(bias.shape) == 1)
    )
    return res

def _linear_fp8_act_sparse_fp8_weight_cutlass_impl(input_tensor, weight_tensor, bias):
    from torchao.ops import rowwise_scaled_linear_sparse_cutlass_f8f8

    input_sparse = input_tensor.tensor_impl.sparse
    input_meta = input_tensor.tensor_impl.meta
    input_scale = input_tensor.tensor_impl.scale
    weight = weight_tensor.tensor_impl.float8_data
    weight_scale = weight_tensor.tensor_impl.scale

    out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)
    rows, cols = (input_tensor.shape)

    out = torch.ops.torchao.sparse24_fp8_sm90_cutlass_gemm(
        input_sparse, input_meta, weight.t(), a_scale=input_scale, b_scale=weight_scale.t(),
    )[:rows, :].view(out_shape)
    
    return out
