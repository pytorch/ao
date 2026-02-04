# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch._inductor.ir import Pointwise, TensorBox
from torch._inductor.lowering import register_lowering, to_dtype
from torch._inductor.virtualized import ops


def _register_quantize_dequantize_fp8_lowering():
    @register_lowering(
        torch.ops.torchao.quantize_affine_float8_non_decomposed.default,
        type_promotion_kind=None,
    )
    def quantize_fp8(
        input: TensorBox,
        scale: TensorBox,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> TensorBox:
        # Expect scale to be a scalar tensor or a 1D tensor with size 1
        assert len(scale.get_size()) <= 1 and scale.get_numel() == 1, (
            "Only support per-tensor quantization for float8 now."
        )
        if input.get_dtype() != torch.float32:
            input = to_dtype(input, torch.float32)
        input_loader = input.make_loader()
        scale_loader = scale.make_loader()
        q_min = torch.finfo(float8_dtype).min
        q_max = torch.finfo(float8_dtype).max
        scale_idx = 0 if len(scale.get_size()) == 1 else []

        def inner_fn(idx):
            input = input_loader(idx)
            one = ops.constant(1.0, torch.float)
            inv_scale = ops.truediv(one, scale_loader(scale_idx))
            val = input * inv_scale
            qmin = ops.constant(q_min, torch.float32)
            qmax = ops.constant(q_max, torch.float32)
            clamped = ops.minimum(ops.maximum(val, qmin), qmax)
            return ops.to_dtype(clamped, float8_dtype)

        return Pointwise.create(
            device=input.get_device(),
            dtype=float8_dtype,
            inner_fn=inner_fn,
            ranges=input.get_size(),
        )

    @register_lowering(
        torch.ops.torchao.dequantize_affine_float8_non_decomposed.default,
        type_promotion_kind=None,
    )
    def dequantize_fp8(
        input: TensorBox,
        scale: TensorBox,
        output_dtype: torch.dtype = torch.float32,
    ) -> TensorBox:
        # Expect scale to be a scalar tensor or a 1D tensor with size 1
        assert len(scale.get_size()) <= 1 and scale.get_numel() == 1, (
            "Only support per-tensor dquantization for float8 now."
        )
        if input.get_dtype() != torch.float32:
            input = to_dtype(input, torch.float32)
        input_loader = input.make_loader()
        scale_loader = scale.make_loader()
        scale_idx = 0 if len(scale.get_size()) == 1 else []

        def inner_fn(idx):
            input = input_loader(idx)
            scale = scale_loader(scale_idx)
            val = input * scale
            return ops.to_dtype(val, output_dtype)

        return Pointwise.create(
            device=input.get_device(),
            dtype=output_dtype,
            inner_fn=inner_fn,
            ranges=input.get_size(),
        )
