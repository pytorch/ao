# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from dataclasses import dataclass

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.uintx.tensor_core_tiled_layout import _aqt_is_tensor_core_tile_uint4
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
)

aten = torch.ops.aten


def _linear_fp_act_int4_weight_sparse_marlin_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_tensor_core_tile_uint4(weight_tensor)
        and input_tensor.dtype == torch.float16
        and len(weight_tensor.shape) == 2
        and weight_tensor.zero_point_domain == ZeroPointDomain.INT
        and isinstance(weight_tensor._layout, MarlinSparseLayout)
    )


def _linear_fp_act_int4_weight_sparse_marlin_impl(input_tensor, weight_tensor, bias):
    from torchao.ops import marlin_24_gemm
    from torchao.sparsity.marlin import marlin_24_workspace

    assert isinstance(weight_tensor, AffineQuantizedTensor)

    sparse_w_int4 = weight_tensor.tensor_impl.int_data
    scale = weight_tensor.tensor_impl.scale
    meta = weight_tensor.tensor_impl.meta
    original_shape = weight_tensor.tensor_impl.original_shape
    num_bits = weight_tensor.tensor_impl.num_bits

    # Folds batch dimension into the first dimension
    input_2d = input_tensor.view(-1, input_tensor.shape[-1])

    size_m = input_2d.shape[0]
    size_n = scale.shape[1]
    size_k = input_2d.shape[1]
    workspace_24 = marlin_24_workspace(original_shape[1])

    out = marlin_24_gemm(
        input_2d,
        sparse_w_int4,
        meta,
        scale,
        workspace_24,
        num_bits,
        size_m,
        size_n,
        size_k,
    )

    # Unfold the batch dimension
    out = out.reshape(input_tensor.shape[:-1] + (scale.shape[1],))

    if bias is not None:
        out += bias.to(out.dtype)
    return out


@dataclass(frozen=True)
class MarlinSparseLayout(Layout):
    """MarlinSparseLayout is a layout class for handling sparse tensor formats
    specifically designed for the Marlin sparse kernel. This layout is used
    to optimize the storage and computation of affine quantized tensors with
    2:4 sparsity patterns.

    The layout ensures that the tensor data is pre-processed and stored in a
    format that is compatible with the Marlin sparse kernel operations. It
    provides methods for preprocessing input tensors and managing the layout
    of quantized tensors.
    """

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor to be in the correct format for the Marlin sparse kernel.
            - 1º: the input tensor is transposed since the linear layer keeps the weights in a transposed format
            - 2º: tensor is injected with 2:4 sparsity
            - 3º: transposes it again because the quantization process will compute the scales for dim=-1

        Args:
            input (torch.Tensor): the input tensor to preprocess

        Returns:
            torch.Tensor: the preprocessed tensor
        """
        from torchao.sparsity.marlin import inject_24  # avoid circular import

        input_t = input.t()
        w_24, _ = inject_24(input_t, *input_t.shape)
        return w_24.t()


@register_layout(MarlinSparseLayout)
class MarlinSparseAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for sparse_marlin_24 layout for affine quantized tensor.

    Can be used with 4 bits and 8 bits quantization.

    Original marlin documentation and information:
    https://github.com/IST-DASLab/marlin/tree/master

    Sparse marlin documentation and information:
    https://github.com/IST-DASLab/Sparse-Marlin?tab=readme-ov-file

    fields:
        original_shape (torch.Size): the original shape of the tensor. used to unpack the tensor to the original shape
        group_size (int): the group size used to pack the tensor
        num_bits (int): the number of bits used to quantize the tensor
    """

    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        meta: torch.Tensor,
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        meta: torch.Tensor,
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        warnings.warn(
            "Models quantized with version 1 of Int4WeightOnlyConfig is deprecated and will no longer be supported in a future release, please upgrade torchao and quantize again, or download a newer torchao checkpoint, see https://github.com/pytorch/ao/issues/2948 for more details"
        )
        self.int_data = int_data
        self.scale_and_zero = None
        self.scale = scale
        self.zero = zero
        self.meta = meta
        self._layout = _layout
        self.original_shape = original_shape
        self.group_size = group_size
        self.num_bits = num_bits

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"MarlinSparseAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero", "meta"], [
            self._layout,
            self.original_shape,
            self.group_size,
            self.num_bits,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        zero = tensor_data_dict["zero"]
        meta = tensor_data_dict["meta"]
        _layout, original_shape, group_size, num_bits = tensor_attributes
        return cls(
            int_data,
            scale,
            zero,
            meta,
            _layout,
            original_shape,
            group_size,
            num_bits,
        )

    def get_plain(self):
        from torchao.sparsity.marlin import (
            unpack_from_marlin_24,
        )

        int_data_expanded, scales_expanded = unpack_from_marlin_24(
            self.int_data,
            self.scale,
            self.meta,
            self.original_shape,
            self.group_size,
            self.num_bits,
        )
        int_data_expanded_t = int_data_expanded.t()
        scales_expanded_t = scales_expanded.t()
        return int_data_expanded_t, scales_expanded_t, self.zero

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        _layout: Layout,
    ):
        from torchao.sparsity.marlin import (
            const,
            pack_to_marlin_24,
        )

        assert isinstance(_layout, MarlinSparseLayout)

        # Linear layers are (in_features, out_features) but the int_data that is reaching this point
        # is (out_features, in_features). We need to transpose it to match the expected shape in the marlin code.
        q_w_24 = int_data.t()
        # addressing the case when scale has dimension 1, happens when
        # weight_shape[-1] == group_size == 128
        if scale.ndim == 1:
            scale = scale.reshape(scale.shape[0], -1)

        scale_t = scale.t()

        if not torch.cuda.get_device_capability()[0] >= 8:
            raise ValueError(
                f"Can not use Sparse Marlin 2:4 int4*fp16 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for Marlin kernel."
            )

        if q_w_24.dtype != torch.int32:
            raise ValueError("Only `torch.int32` weights are supported.")

        in_features, out_features = q_w_24.shape
        if in_features % 128 != 0 or out_features != 256 == 0:
            raise ValueError(
                "`in_features` must be divisible by 64 and `out_features` by 256."
            )

        # NOTE: The current marlin 2:4 kernel supports both 4 and 8 bits quantization but fp8
        # will require a bit more work to get our current quantization flow to work with it.
        # Check the link for a reference: https://github.com/neuralmagic/nm-vllm/tree/main
        num_bits = 4 if torch.max(q_w_24) < 16 else -1
        if num_bits not in [4]:
            raise ValueError(f"Only {[4]} bits are supported, got {num_bits}.")

        group_size = in_features // scale_t.shape[0]
        if group_size == 0:
            group_size = in_features
        assert group_size <= in_features, (
            "Group size must be less than or equal to in_features."
        )

        if group_size not in const.SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Only {const.SUPPORTED_GROUP_SIZES} group sizes are supported, got {group_size}."
            )

        # Compress quantized weight to marlin 2:4 format
        marlin_24_q_w_comp, marlin_24_s, meta = pack_to_marlin_24(
            q_w_24, scale_t, num_bits, group_size
        )

        return cls(
            marlin_24_q_w_comp,
            marlin_24_s,
            zero,
            meta,
            _layout,
            q_w_24.shape,
            group_size,
            num_bits,
        )

    def get_layout(self) -> Layout:
        return self._layout

    def _apply_fn_to_data(self, fn):
        self.int_data = fn(self.int_data)
        self.scale = fn(self.scale)
        self.zero = fn(self.zero)
        self.meta = fn(self.meta)
        return self
