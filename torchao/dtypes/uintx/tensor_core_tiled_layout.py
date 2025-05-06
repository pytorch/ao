# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout, is_device
from torchao.quantization.quant_primitives import ZeroPointDomain, _get_reduction_params
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    fill_defaults,
    find_multiple,
)

aten = torch.ops.aten


def _aqt_is_tensor_core_tile_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    # TODO: use torch.uint4
    return (
        aqt.tensor_impl.dtype == torch.int32
        and aqt.quant_min == 0
        and aqt.quant_max == 15
    )


def _same_metadata(
    self: "TensorCoreTiledAQTTensorImpl", src: "TensorCoreTiledAQTTensorImpl"
) -> bool:
    return (
        isinstance(self, TensorCoreTiledAQTTensorImpl)
        and isinstance(src, TensorCoreTiledAQTTensorImpl)
        and self.shape == src.shape
        and self.packed_weight.shape == src.packed_weight.shape
        and self.scale_and_zero.shape == src.scale_and_zero.shape
        and self.transposed == src.transposed
        and type(self._layout) == type(src._layout)
    )


def _linear_bf16_act_uint4_weight_check(input_tensor, weight_tensor, bias):
    return (
        # input is native bfloat16 tensor
        not is_traceable_wrapper_subclass(input_tensor)
        and input_tensor.dtype == torch.bfloat16
        and
        # weight is uint4, group quantized tensor_core_tiled tensor impl affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_tensor_core_tile_uint4(weight_tensor)
        and weight_tensor.dtype == torch.bfloat16
        and len(weight_tensor.shape) == 2
        and weight_tensor.zero_point_domain == ZeroPointDomain.FLOAT
        and isinstance(weight_tensor._layout, TensorCoreTiledLayout)
    )


def _linear_bf16_act_uint4_weight_impl(input_tensor, weight_tensor, bias):
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    # TODO: check groupsize quantization
    # avoid circular dep, TODO: move this to a common util.py
    act_mat = input_tensor
    # weight is packed from padded (out_features, in_features) weight tensor
    # (same dimension requirement as F.linear weight)
    packed_weight = weight_tensor.tensor_impl.packed_weight
    scale_and_zero = weight_tensor.tensor_impl.scale_and_zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape and pad activation
    act_mat = act_mat.reshape(-1, act_mat.shape[-1]).to(torch.bfloat16)
    pad_size = find_multiple(act_mat.shape[-1], 1024)
    act_mat = torch.nn.functional.pad(act_mat, (0, pad_size - act_mat.shape[-1]))

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm(
        act_mat.contiguous(), packed_weight, groupsize, scale_and_zero
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


@dataclass(frozen=True)
class TensorCoreTiledLayout(Layout):
    """TensorCoreTiledLayout is a layout class for handling tensor core tiled layouts in affine quantized tensors. It provides methods for pre-processing and post-processing tensors to fit the required layout for efficient computation on tensor cores.

    Attributes:
        inner_k_tiles (int): An internal argument for the packing function of tensor core tiled layout that can affect the performance of the matmul kernel. Defaults to 8.
    """

    inner_k_tiles: int = 8

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        orig_out_features, orig_in_features = input.shape
        in_features = find_multiple(orig_in_features, 1024)
        out_features = find_multiple(orig_out_features, 8)
        input = torch.nn.functional.pad(
            input,
            (0, in_features - orig_in_features, 0, out_features - orig_out_features),
        )
        return input

    def pre_process_static(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = self.pre_process(input)
        orig_qparam_shape = scale.shape
        new_qparam_shape, reduction_dims = _get_reduction_params(
            block_size, input.size()
        )
        for dim in reduction_dims:
            new_qparam_shape.pop(dim)
        change_in_qparam_shape = [
            new_dim_size - orig_dim_size
            for new_dim_size, orig_dim_size in zip(new_qparam_shape, orig_qparam_shape)
        ]
        padding_changes = []
        for dim_change in change_in_qparam_shape:
            padding_changes = [0, dim_change] + padding_changes
        scale = torch.nn.functional.pad(scale, padding_changes)
        zero_point = torch.nn.functional.pad(zero_point, padding_changes)
        return input, scale, zero_point

    def post_process(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_out_features, orig_in_features = input.shape
        in_features = find_multiple(orig_in_features, 1024)
        out_features = find_multiple(orig_out_features, 8)
        input = torch.nn.functional.pad(
            input,
            (0, in_features - orig_in_features, 0, out_features - orig_out_features),
        )
        assert len(block_size) == 2, (
            f"TensorCoreTiledLayout only supports len(block_size) == 2, got: {block_size}"
        )
        scale_pad_dim_0 = (out_features - orig_out_features) // block_size[0]
        scale_pad_dim_1 = (in_features - orig_in_features) // block_size[1]
        scale = torch.nn.functional.pad(scale, (0, scale_pad_dim_1, 0, scale_pad_dim_0))
        zero_point = torch.nn.functional.pad(
            zero_point, (0, scale_pad_dim_1, 0, scale_pad_dim_0)
        )
        return input, scale, zero_point

    def extra_repr(self):
        return f"inner_k_tiles={self.inner_k_tiles}"


@register_layout(TensorCoreTiledLayout)
class TensorCoreTiledAQTTensorImpl(AQTTensorImpl):
    """TensorImpl for tensor_core_tiled layout for affine quantized tensor, this is for int4 only,
    used by tinygemm kernels `_weight_int4pack_mm`

    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 4-d tensor of
    dimension: [n / 8][k / (inner_k_tiles * 16)][32][inner_k_tiles / 2]
    (unpacked Tensor shape is n * k)
    where inner_k_tiles is an internal argument for packing function of tensor core tiled layout
    that can affect the performance of the matmul kernel (defaults to 8)

    Note: we also pack scale and zero point together here for tinygemm kernel

    Note: technically tensor core tiled layout should be the layout for the underlying packed weight
    (int Tensor) but since the scale and zero_point are also packed into the same tensor here which is not used
    in plain layout, we just created a layout for AQT right now, this could be improved if we split out
    int4 aqt into a separate tensor subclass

    fields:
      packed_weight (torch.Tensor): the 4-d packed tensor in a tensor_core_tiled layout
      scale_and_zero (torch.Tensor): the combined scale Tensor used to map between floating point tensor to quantized tensor and zero_point Tensor
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["layout"] = (
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else packed_weight.layout
        )
        kwargs["dtype"] = packed_weight.dtype
        kwargs["requires_grad"] = False
        shape = packed_weight.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scale_and_zero = scale_and_zero
        self.transposed = False
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["packed_weight", "scale_and_zero"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale_and_zero = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scale_and_zero"],
        )
        (
            transposed,
            _layout,
        ) = tensor_attributes
        return cls(packed_weight, scale_and_zero, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, TensorCoreTiledLayout)

        if TORCH_VERSION_AT_LEAST_2_5:
            int_data = (int_data[::, ::2] << 4 | int_data[::, 1::2]).to(torch.uint8)
            assert int_data.dtype == torch.uint8, (
                "torch.ops.aten._convert_weight_to_int4pack in torch 2.5 expects `uint8` dtype"
            )
        else:
            assert int_data.dtype == torch.int32, (
                "torch.ops.aten._convert_weight_to_int4pack in torch 2.4 expects `int32` dtype"
            )
        packed_weight = torch.ops.aten._convert_weight_to_int4pack(
            int_data, _layout.inner_k_tiles
        )
        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point, scale.dtype)
        return cls(packed_weight, scale_and_zero, False, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        # tensor core tiled layout supports both cpu and cuda but does not support the conversion
        # between these two devices, in the future we should not use the same layout for
        # cpu and cuda device: https://github.com/pytorch/ao/issues/1117
        if not is_device(torch.device(self.device).type, device):
            raise ValueError(
                f"TensorCoreTiledAQTTensorImpl does not support conversion from {self.device} to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale_and_zero.to(device),
            self.transposed,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        # self.packed_weight = fn(self.packed_weight)
        # self.scale_and_zero = fn(self.scale_and_zero)
        # return self
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale_and_zero),
            self.transposed,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten.copy_.default:
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

        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            transposed = TensorCoreTiledAQTTensorImpl(
                args[0].packed_weight,
                args[0].scale_and_zero,
                not args[0].transposed,
                args[0]._layout,
            )
            return return_and_correct_aliasing(func, args, kwargs, transposed)

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            cur_shape = self.shape
            assert len(cur_shape) == 4
            inner_k_tiles = cur_shape[-1] * 2
            original_shape = (cur_shape[0] * 8, cur_shape[1] * (inner_k_tiles * 16))

            n_by_8, k_by_inner_tiles, _, _ = self.packed_weight.shape
            sz_dim1, sz_dim0, _ = self.scale_and_zero.shape

            data_len = original_shape[dim]
            assert dim in [0, 1], (
                f"TensorCoreTiledAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
            )

            if dim == 0:
                pw_len = n_by_8
                sz_len = sz_dim0
            else:
                pw_len = k_by_inner_tiles
                sz_len = sz_dim1

            if pw_len == 0 or sz_len == 0:
                return return_and_correct_aliasing(
                    func,
                    args,
                    kwargs,
                    TensorCoreTiledAQTTensorImpl(
                        self.packed_weight,
                        self.scale_and_zero,
                        self.transposed,
                        self._layout,
                    ),
                )

            pw_ratio = data_len / pw_len
            start_pw = int(start / pw_ratio)
            end_pw = int(end / pw_ratio)

            sz_ratio = data_len / sz_len
            start_sz = int(start / sz_ratio)
            end_sz = int(end / sz_ratio)

            packed_weight = aten.slice(self.packed_weight, dim, start_pw, end_pw, step)
            scale_and_zero = aten.slice(
                self.scale_and_zero, 1 - dim, start_sz, end_sz, step
            )
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                TensorCoreTiledAQTTensorImpl(
                    packed_weight, scale_and_zero, self.transposed, self._layout
                ),
            )

        raise NotImplementedError(
            f"TensorCoreTiledAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @property
    def block_size(self):
        from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

        scale, zero = unpack_tinygemm_scales_and_zeros(self.scale_and_zero)
        cur_shape = self.shape
        assert len(cur_shape) == 4
        inner_k_tiles = cur_shape[-1] * 2
        original_shape = (cur_shape[0] * 8, cur_shape[1] * (inner_k_tiles * 16))
        groupsize = int(original_shape[1] / scale.shape[-2])
        return (1, groupsize)

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchao.quantization.quant_primitives import (
            ZeroPointDomain,
            quantize_affine,
        )
        from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

        scale, zero = unpack_tinygemm_scales_and_zeros(self.scale_and_zero)

        cur_shape = self.shape
        assert len(cur_shape) == 4
        inner_k_tiles = cur_shape[-1] * 2
        original_shape = (cur_shape[0] * 8, cur_shape[1] * (inner_k_tiles * 16))
        eye_shape = original_shape[1]
        groupsize = int(original_shape[1] / scale.shape[-2])
        block_size = (1, groupsize)
        device = self.device
        original_dtype = torch.bfloat16
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        zero_point_domain = ZeroPointDomain.FLOAT
        assert len(block_size) == 2 and block_size[0] == 1
        dequantized = torch.ops.aten._weight_int4pack_mm(
            torch.eye(eye_shape, device=device, dtype=original_dtype),
            self.packed_weight,
            groupsize,
            self.scale_and_zero,
        )
        dequantized = dequantized.t().contiguous()
        # TODO: move this to `unpack_tinygemm_scales_and_zeros`?
        scale = scale.reshape(scale.shape[:-1]).contiguous()
        zero = zero.reshape(zero.shape[:-1]).contiguous()
        int_data = quantize_affine(
            dequantized,
            block_size,
            scale,
            zero,
            target_dtype,
            quant_min,
            quant_max,
            zero_point_domain,
        )
        return int_data, scale, zero

    def get_layout(self) -> Layout:
        return self._layout
