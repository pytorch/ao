# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout, PlainLayout, is_device
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_6,
    fill_defaults,
)

from ..uintx.int4_cpu_layout import (
    _is_float,
)

aten = torch.ops.aten


@dataclass(frozen=True)
class Float8DynamicActFloat8WeightCPULayout(Layout):
    """Layout class for float8 da8w8 CPU layout for affine quantized tensor"""

    pass


@register_layout(Float8DynamicActFloat8WeightCPULayout)
class Float8DynActFloat8WeiCpuAQTTensorImpl(AQTTensorImpl):
    """TensorImpl for float8 da8w8 CPU layout for affine quantized tensor"""

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
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
        scales: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scales = scales
        self.transposed = transposed
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["packed_weight", "scales"], [
            self.transposed,
            self._layout,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scales = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scales"],
        )
        (transposed, _layout) = tensor_attributes
        return cls(packed_weight, scales, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, Float8DynamicActFloat8WeightCPULayout)
        assert data.dtype == torch.float8_e4m3fn, (
            "Float8 DA8W8 CPU: expects float8_e4m3fn weight"
        )
        if scale.dim() == 1:
            scale.unsqueeze_(-1)
        scale = scale.to(torch.float)

        N = data.size(0)
        K = data.size(-1)
        if N % 32 == 0 and K % 32 == 0:
            # Pack weight from [N, K] to [N / block_n, K / block_k, block_k, block_n].
            # Pack inner blocks [block_k, block_n] to VNNI layout if AMX is available.
            # Pack scales from [N, num_groups] to [N / block_n, num_groups, block_n].
            weight_packed, scales = torch.ops.torchao.float8_linear_prepack_cpu(
                data, scale
            )
        else:
            weight_packed = data
            scales = scale
            _layout = PlainLayout()
        return cls(weight_packed, scales, False, _layout)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scales),
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

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim in [0, 1]:
                assert step == 1, "Only step == 1 is supported in slicing right now"
                data, scale = self.get_plain()
                data_len = data.shape[dim]
                scale_len = scale.shape[dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                data = aten.slice.Tensor(data, dim, start, end, step)
                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                # this is to handle padding
                data, scale = self._layout.post_process(data, scale, self.block_size)
                sliced = self.from_plain(data, scale, self._layout)
                return return_and_correct_aliasing(func, args, kwargs, sliced)
            else:
                raise NotImplementedError(
                    f"{cls.__name__} dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @property
    def block_size(self):
        assert len(self.packed_weight.shape) == 2
        weight_shape = self.packed_weight.shape
        N = weight_shape[0]
        K = weight_shape[1]
        groups = self.scales.numel() // N
        group_size = K // groups
        return (1, group_size)

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(self._layout, PlainLayout):
            # If the layout is PlainLayout, return the packed weight and scales directly
            return (
                self.packed_weight,
                self.scales,
                torch.zeros_like(self.scales),
            )
        # Unpack weight by linear(eye(K), packed_weight).t()
        packed_w_shape = self.packed_weight.shape
        if len(packed_w_shape) == 4:
            K = packed_w_shape[1] * packed_w_shape[2]
        else:
            K = packed_w_shape[1]
        x = torch.eye(K).to(torch.float8_e4m3fn)
        x_scale = torch.ones(K).float()
        w_scale = torch.ones_like(self.scales).float()
        plain_weight = torch.ops.torchao.float8_linear_cpu.default(
            x,
            x_scale,
            self.packed_weight,
            w_scale,
            None,  # bias
            torch.float,  # out_dtype
        )
        plain_weight = plain_weight.t().contiguous()
        plain_weight = plain_weight.to(torch.float8_e4m3fn)

        if self.scales.dim() == 2:
            plain_scales = self.scales
        else:
            assert self.scales.dim() == 3
            packed_shape = self.scales.shape  # [Nc, G, block_n]
            plain_scales = (
                self.scales.permute([0, 2, 1]).contiguous().view([-1, packed_shape[1]])
            )

        return plain_weight, plain_scales, torch.zeros_like(plain_scales)


def _aqt_is_float8e4m3(aqt):
    """Check if an AffineQuantizedTensor is float8_e4m3fn quantized Tensor"""
    return aqt.tensor_impl.dtype == torch.float8_e4m3fn


def _float8_linear_cpu_check(input_tensor, weight_tensor, bias):
    return (
        TORCH_VERSION_AT_LEAST_2_6
        and is_device(input_tensor.device.type, "cpu")
        and is_device(weight_tensor.device.type, "cpu")
        and (bias is None or is_device(bias.device.type, "cpu"))
        and isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_float8e4m3(input_tensor)
        and _is_float(input_tensor.dtype)
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_float8e4m3(weight_tensor)
        and _is_float(weight_tensor.dtype)
        and isinstance(weight_tensor._layout, Float8DynamicActFloat8WeightCPULayout)
    )


def _float8_linear_cpu_impl(input_tensor, weight_tensor, bias):
    assert TORCH_VERSION_AT_LEAST_2_6, (
        f"Requires PyTorch version at least 2.6, but got: {torch.__version__}"
    )
    assert is_device(input_tensor.device.type, "cpu"), (
        f"For CPU device only but got: {input_tensor.device}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    act_mat = input_tensor
    act = act_mat.tensor_impl.int_data
    act_scales = act_mat.tensor_impl.scale

    packed_weight = weight_tensor.tensor_impl.packed_weight
    wei_scales = weight_tensor.tensor_impl.scales

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act = act.reshape(-1, act.shape[-1])

    y = torch.ops.torchao.float8_linear_cpu.default(
        act.contiguous(),
        act_scales,
        packed_weight,
        wei_scales,
        bias.float() if bias is not None else bias,  # requires bias to be float
        torch.float,  # out_dtype
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    return y.to(orig_dtype)
