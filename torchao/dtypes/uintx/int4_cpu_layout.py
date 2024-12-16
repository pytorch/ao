from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.utils import AQTTensorImpl, Layout, is_device
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    fill_defaults,
)

aten = torch.ops.aten


@dataclass(frozen=True)
class Int4CPULayout(Layout):
    """Only for PyTorch version at least 2.6"""

    pass


@register_layout(Int4CPULayout)
class Int4CPUAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for int4 CPU layout for affine quantized tensor, this is for int4 only,
    used by tinygemm kernels `_weight_int4pack_mm_for_cpu`
    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 2-d tensor of
    dimension: [n][k / 2] (uint8 dtype)
    (unpacked Tensor shape is n * k)
    Note: we also pack scale and zero point together here for tinygemm kernel
    Note: technically Int4 CPU layout should be the layout for the underlying packed weight
    (int Tensor) but since the scale and zero_point are also packed into the same tensor here which is not used
    in plain layout, we just created a layout for AQT right now, this could be improved if we split out
    int4 aqt into a separate tensor subclass
    fields:
      packed_weight (torch.Tensor): the 2-d packed tensor in a Int4 CPU layout
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
        assert isinstance(_layout, Int4CPULayout)

        if TORCH_VERSION_AT_LEAST_2_6:
            assert (
                int_data.dtype == torch.int32
            ), "torch.ops.aten._convert_weight_to_int4pack_for_cpu expects `int32` dtype"
            packed_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                int_data,
                1,  # TODO:remove
            )
        elif TORCH_VERSION_AT_LEAST_2_5:
            int_data = (int_data[::, ::2] << 4 | int_data[::, 1::2]).to(torch.uint8)
            assert (
                int_data.dtype == torch.uint8
            ), "torch.ops.aten._convert_weight_to_int4pack in torch 2.5 expects `uint8` dtype"
            packed_weight = torch.ops.aten._convert_weight_to_int4pack(
                int_data, _layout.inner_k_tiles
            )
        else:
            assert (
                int_data.dtype == torch.int32
            ), "torch.ops.aten._convert_weight_to_int4pack in torch 2.4 expects `int32` dtype"
            packed_weight = torch.ops.aten._convert_weight_to_int4pack(
                int_data, _layout.inner_k_tiles
            )

        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point)
        return cls(packed_weight, scale_and_zero, False, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device(torch.device(self.device).type, device):
            raise ValueError(
                f"Int4CPUAQTTensorImpl does not support conversion from {self.device} to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale_and_zero.to(device),
            self.transposed,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
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

        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            transposed = Int4CPUAQTTensorImpl(
                args[0].packed_weight,
                args[0].scale_and_zero,
                not args[0].transposed,
                args[0]._layout,
            )
            return return_and_correct_aliasing(func, args, kwargs, transposed)

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                int_data, scale, zero_point = self.get_plain()
                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                # this is to handle padding
                int_data = self._layout.post_process(int_data)
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return return_and_correct_aliasing(func, args, kwargs, sliced)
            elif dim == 1:
                int_data, scale, zero_point = self.get_plain()
                assert step == 1, "Only step == 1 is supported in slicing right now"
                data_len = int_data.shape[dim]
                scale_len = scale.shape[dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                # this is to handle padding
                int_data = self._layout.post_process(int_data)
                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                zero_point = aten.slice.Tensor(
                    zero_point, dim, start_scale, end_scale, step
                )
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return sliced
            else:
                raise NotImplementedError(
                    f"Int4CPUAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"Int4CPUAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchao.quantization.quant_primitives import (
            ZeroPointDomain,
            quantize_affine,
        )
        from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

        scale, zero = unpack_tinygemm_scales_and_zeros(self.scale_and_zero)

        cur_shape = self.shape
        assert len(cur_shape) == 2
        original_shape = (cur_shape[0], cur_shape[1] * 2)
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
        dequantized = torch.ops.aten._weight_int4pack_mm_for_cpu(
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
