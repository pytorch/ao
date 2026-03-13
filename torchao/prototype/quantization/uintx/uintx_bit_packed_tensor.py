# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor, fill_defaults

try:
    import gemlite
except Exception:
    gemlite = None

aten = torch.ops.aten


class UIntxBitPackedTensor(TorchAOBaseTensor):
    """Packed unsigned integer weight tensor using gemlite bit-packing.

    Supports 4-bit (asymmetric, grouped) and 8-bit (symmetric, per-channel) weight quantization
    with optional dynamic activation quantization.

    Tensor Attributes:
        packed_weight: gemlite-packed quantized weight data
        scale: quantization scale factors
        zero_point: quantization zero points (empty tensor for symmetric)

    Non-Tensor Attributes:
        gemlite_kwargs: dict with gemlite metadata (in_features, out_features, meta_args, etc.)
        bit_width: quantization bit width (4 or 8)
        group_size: quantization group size
        mode: "weight_only" or "dynamic"
        dtype: original weight dtype
    """

    tensor_data_names = ["packed_weight", "scale", "zero_point"]
    tensor_attribute_names = [
        "gemlite_kwargs",
        "bit_width",
        "group_size",
        "mode",
        "dtype",
    ]

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_kwargs: Dict,
        bit_width: int,
        group_size: int,
        mode: str,
        dtype: torch.dtype,
    ):
        shape = (gemlite_kwargs["out_features"], gemlite_kwargs["in_features"])
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=packed_weight.device,
            dtype=dtype,
            requires_grad=False,
        )

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_kwargs: Dict,
        bit_width: int,
        group_size: int,
        mode: str,
        dtype: torch.dtype,
    ):
        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self.gemlite_kwargs = gemlite_kwargs
        self.bit_width = bit_width
        self.group_size = group_size
        self.mode = mode
        # Note: don't set self.dtype, it's handled by __new__ via _make_wrapper_subclass

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        bit_width: int = 4,
        group_size: Optional[int] = 128,
        packing_bitwidth: Optional[int] = None,
        mode: str = "weight_only",
    ):
        """Create UIntxBitPackedTensor from a high-precision weight tensor.

        Args:
            hp_tensor: the float weight tensor (out_features, in_features)
            bit_width: 4 or 8
            group_size: quantization group size, None means per-channel
            packing_bitwidth: 8, 16, 32, or None (let gemlite choose)
            mode: "weight_only" or "dynamic"
        """
        if gemlite is None:
            raise ImportError("gemlite is required. Install with: pip install gemlite")

        assert bit_width in [4, 8], f"bit_width must be 4 or 8, got {bit_width}"
        assert hp_tensor.dtype in [torch.float16, torch.bfloat16], (
            f"dtype must be float16 or bfloat16, got {hp_tensor.dtype}"
        )
        assert group_size in [32, 64, 128, 256, 512, 1024, None]
        assert group_size is None or bit_width != 8, (
            "group_size must be None for bit_width=8"
        )
        assert packing_bitwidth in [8, 16, 32, None]
        assert mode in ["weight_only", "dynamic"]

        out_features, in_features = hp_tensor.shape
        effective_group_size = in_features if group_size is None else group_size
        device = hp_tensor.device

        # Step 1: Quantize the weight
        int_data, scale, zero_point = cls._quantize_weight(
            hp_tensor, bit_width, effective_group_size
        )

        # Step 2: Pack with gemlite
        if int_data.device.type != "cuda":
            int_data = int_data.cuda()

        if bit_width == 8 and effective_group_size == in_features:
            processor = (
                gemlite.helper.A8W8_int8_dynamic
                if mode == "dynamic"
                else gemlite.helper.A16W8
            )
            gemlite_linear = processor(device=int_data.device).from_weights(
                int_data, scales=scale, bias=None
            )
        else:
            if mode == "dynamic":
                # Try A8Wn_dynamic first (older gemlite), fall back to A8Wn_HQQ_INT_dynamic
                if hasattr(gemlite.helper, "A8Wn_dynamic"):
                    processor = gemlite.helper.A8Wn_dynamic(
                        device=int_data.device, packing_bitwidth=packing_bitwidth
                    )
                    gemlite_linear = processor.from_weights(
                        int_data,
                        scale,
                        zero_point,
                        bit_width,
                        effective_group_size,
                        bias=None,
                    )
                elif hasattr(gemlite.helper, "A8Wn_HQQ_INT_dynamic"):
                    # Newer gemlite API: W_nbits in constructor, not from_weights
                    processor = gemlite.helper.A8Wn_HQQ_INT_dynamic(
                        device=int_data.device,
                        packing_bitwidth=packing_bitwidth,
                        W_nbits=bit_width,
                    )
                    gemlite_linear = processor.from_weights(
                        int_data, scale, zero_point, bias=None
                    )
                else:
                    raise ImportError(
                        "gemlite does not have A8Wn_dynamic or A8Wn_HQQ_INT_dynamic helper"
                    )
            else:
                processor = gemlite.helper.A16Wn(
                    device=int_data.device, packing_bitwidth=packing_bitwidth
                )
                gemlite_linear = processor.from_weights(
                    int_data,
                    scale,
                    zero_point,
                    bit_width,
                    effective_group_size,
                    bias=None,
                )

        meta_args = gemlite_linear.get_meta_args()
        gemlite_kwargs = {
            "in_features": in_features,
            "out_features": out_features,
            "packing_bitwidth": packing_bitwidth,
            "data_contiguous": gemlite_linear.data_contiguous,
            "elements_per_sample": gemlite_linear.elements_per_sample,
            "W_group_mode": gemlite_linear.W_group_mode,
            "meta_args": meta_args,
        }

        packed_weight, scale, zero_point = gemlite_linear.get_tensor_args()
        packed_weight = packed_weight.to(device)
        if zero_point is None:
            zero_point = torch.tensor(
                [[]], device=packed_weight.device, dtype=torch.int32
            )

        return cls(
            packed_weight,
            scale,
            zero_point,
            gemlite_kwargs,
            bit_width,
            effective_group_size,
            mode,
            hp_tensor.dtype,
        )

    @classmethod
    def _quantize_weight(cls, hp_tensor, bit_width, group_size):
        """Quantize weight tensor to int data + scale + zero_point."""
        from torchao.quantization.quant_primitives import (
            MappingType,
            choose_qparams_affine,
            quantize_affine,
        )

        block_size = (1, group_size)

        if bit_width == 4:
            # Use HQQ for 4-bit (better quality)
            from torchao.quantization.quant_primitives import (
                _choose_qparams_and_quantize_affine_hqq,
            )

            int_data, scale, zero_point, _ = _choose_qparams_and_quantize_affine_hqq(
                hp_tensor,
                nbits=bit_width,
                group_size=group_size,
                raw_output=True,
            )
            return int_data, scale, zero_point

        else:
            # 8-bit: symmetric
            scale, zero_point = choose_qparams_affine(
                hp_tensor,
                mapping_type=MappingType.SYMMETRIC,
                block_size=block_size,
                target_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                eps=1e-5,
            )
            int_data = quantize_affine(
                hp_tensor,
                block_size=block_size,
                scale=scale,
                zero_point=zero_point,
                output_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
            )
            return int_data, scale, zero_point

    def dequantize(self, output_dtype=None):
        """Dequantize packed weight back to floating point."""
        device = self.packed_weight.device
        int_data = (
            gemlite.bitpack.unpack_over_rows(
                self.packed_weight.cuda(),
                W_nbits=self.bit_width,
                num_output_rows=self.gemlite_kwargs["in_features"],
                dtype=torch.uint8,
            )
            .to(device)
            .t()
        )

        if self.gemlite_kwargs["data_contiguous"]:
            int_data = int_data.contiguous()

        # Handle FMA mode: W_q * s + z -> (W_q - z) * s
        if self.gemlite_kwargs["W_group_mode"] == 4:
            scale_min_val = 1e-8
            scale = self.scale.clone().float()
            scale[torch.logical_and(scale >= 0, scale.abs() <= scale_min_val)] = (
                scale_min_val
            )
            scale[
                torch.logical_and(scale < 0, scale.abs() <= scale_min_val)
            ] = -scale_min_val
            zero_point = (-self.zero_point.float() / scale).clamp_(-100, 100)
            zero_point = zero_point.to(self.scale.dtype)
        else:
            zero_point = self.zero_point

        scale = self.scale.t().contiguous()
        zero_point = zero_point.t().contiguous()

        # Dequantize: (int_data - zero_point) * scale
        from torchao.quantization.quant_primitives import dequantize_affine

        block_size = (1, self.group_size)
        result = dequantize_affine(
            int_data,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point,
            input_dtype=int_data.dtype,
            output_dtype=output_dtype or self.dtype,
        )
        return result


# Register aten op implementations
implements = UIntxBitPackedTensor.implements
implements_torch_function = UIntxBitPackedTensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    assert isinstance(weight_tensor, UIntxBitPackedTensor)

    return gemlite.core.forward_functional(
        x=input_tensor,
        bias=bias,
        tensor_args=(
            weight_tensor.packed_weight,
            weight_tensor.scale,
            weight_tensor.zero_point,
        ),
        meta_args=weight_tensor.gemlite_kwargs["meta_args"],
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    # No-op: gemlite expects non-transposed weight, and F.linear decomposes to t + mm
    return return_and_correct_aliasing(
        func, args, kwargs or {}, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1, "Only step == 1 is supported"

    if dim not in [0, 1]:
        raise NotImplementedError(
            f"UIntxBitPackedTensor: slice on dim={dim} not supported"
        )

    # Data is stored transposed (K x N), so flip the dim
    data_dim = 1 - dim
    packed_weight = self.packed_weight
    scale = self.scale
    zero_point = self.zero_point

    gemlite_kwargs = self.gemlite_kwargs.copy()
    orig_shape = [
        gemlite_kwargs["in_features"],
        gemlite_kwargs["out_features"],
    ]
    elements_per_sample = gemlite_kwargs["elements_per_sample"]
    data_len = orig_shape[data_dim]
    scale_len = scale.shape[data_dim]
    ratio = data_len / scale_len
    start_scale = int(start / ratio)
    end_scale = int(end / ratio)

    # For packing only the K dimension
    div = elements_per_sample if data_dim == 0 else 1
    packed_weight = aten.slice.Tensor(
        packed_weight, data_dim, start // div, end // div, step
    )

    gemlite_kwargs["in_features"] = packed_weight.shape[0] * elements_per_sample
    gemlite_kwargs["out_features"] = packed_weight.shape[1]

    scale = aten.slice.Tensor(scale, data_dim, start_scale, end_scale, step)
    if zero_point is not None and zero_point.numel() > 0:
        zero_point = aten.slice.Tensor(
            zero_point, data_dim, start_scale, end_scale, step
        )

    sliced = UIntxBitPackedTensor(
        packed_weight,
        scale,
        zero_point,
        gemlite_kwargs,
        self.bit_width,
        self.group_size,
        self.mode,
        self.dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs or {}, sliced)
