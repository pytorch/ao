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
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
    to_sparse_semi_structured_cutlass_sm9x_f8,
)
from torchao.sparsity.utils import mask_creator

aten = torch.ops.aten


@dataclass(frozen=True)
class CutlassSemiSparseLayout(Layout):
    """Layout class for float8 2:4 sparsity layout for affine quantized tensor, for cutlass kernel."""

    def pre_process(self, dense: torch.Tensor) -> torch.Tensor:
        # prune to 2:4 if not already
        return dense * mask_creator(dense).bool()


@register_layout(CutlassSemiSparseLayout)
class CutlassSemiSparseTensorImpl(AQTTensorImpl):
    @staticmethod
    def __new__(
        cls,
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
        shape = (sparse.shape[0], 2 * sparse.shape[-1])
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        self.sparse = sparse
        self.meta = meta
        self.scale = scale
        self._layout = _layout

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"CutlassSemiSparseTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        return ["sparse", "meta", "scale"], [
            self._layout,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        sparse = tensor_data_dict["sparse"]
        meta = tensor_data_dict["meta"]
        scale = tensor_data_dict["scale"]
        _layout = tensor_attributes
        return cls(sparse, meta, scale, _layout)

    def get_plain(self):
        # No support in CUTLASS to convert back to dense from sparse
        # semi-structured format, so multiplying with identity matrix
        # for the conversion.
        cols = 2 * self.sparse.shape[1]
        input = torch.eye(cols, dtype=self.sparse.dtype, device=self.sparse.device)
        input_scale = torch.ones(
            (cols,), dtype=self.scale.dtype, device=self.sparse.device
        )
        dense = (
            rowwise_scaled_linear_sparse_cutlass_f8f8(
                input, input_scale, self.sparse, self.meta, self.scale
            )
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

        sparse, meta = to_sparse_semi_structured_cutlass_sm9x_f8(dense)
        return cls(
            sparse,
            meta,
            scale,
            _layout,
        )

    def get_layout(self) -> Layout:
        return self._layout

    def _apply_fn_to_data(self, fn):
        self.sparse = fn(self.sparse)
        self.meta = fn(self.meta)
        self.scale = fn(self.scale)
        return self


def _linear_float8_act_float8_weight_cutlass_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor._layout, CutlassSemiSparseLayout)
        and (input_tensor.dtype, weight_tensor.dtype)
        in [
            (torch.float8_e5m2, torch.float8_e4m3fn),
            (torch.float8_e4m3fn, torch.float8_e4m3fn),
        ]
        and (
            input_tensor.tensor_impl.scale.dtype,
            weight_tensor.tensor_impl.scale.dtype,
        )
        in [
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
        ]
        and (bias is None or bias.dtype in (torch.float16, torch.bfloat16))
        and len(input_tensor.shape) >= 2
        and len(weight_tensor.shape) == 2
        and len(input_tensor.tensor_impl.scale.shape) == len(input_tensor.shape) - 1
        and len(weight_tensor.tensor_impl.scale.shape) == 1
        and (bias is None or len(bias.shape) == 1)
    )


def _linear_float8_act_float8_weight_cutlass_impl(input_tensor, weight_tensor, bias):
    from torchao.ops import rowwise_scaled_linear_cutlass_s8s4

    input = input_tensor.tensor_impl.sparse
    input_scale = input_tensor.tensor_impl.scale
    weight = weight_tensor.tensor_impl.sparse
    weight_meta = weight_tensor.tensor_impl.meta
    weight_scale = weight_tensor.tensor_impl.scale

    out = rowwise_scaled_linear_sparse_cutlass_f8f8(
        input, input_scale, weight, weight_meta, weight_scale, bias
    )

    return out
