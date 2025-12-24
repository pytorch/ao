# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch._inductor.ir import Pointwise, TensorBox
from torch._inductor.lowering import register_lowering, to_dtype
from torch._inductor.virtualized import ops


def _register_dequantize_fp8_lowering():
    @register_lowering(
        torch.ops.torchao.quantize_affine_float8_non_decomposed.default,
        type_promotion_kind=None,
    )
    def dequantize_fp8(
        input: TensorBox,
        scale: TensorBox,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> TensorBox:
        assert len(scale.get_size()) == 1 and scale.get_numel() == 1, (
            "Only support per-tensor quantization for float8 now."
        )
        if input.get_dtype() != torch.float32:
            input = to_dtype(input, torch.float32)
        input_loader = input.make_loader()
        scale_loader = scale.make_loader()
        q_min = torch.finfo(float8_dtype).min
        q_max = torch.finfo(float8_dtype).max

        def inner_fn(idx, scale):
            input = input_loader(idx)
            one = ops.constant(1.0, torch.float)
            inv_scale = ops.truediv(one, scale_loader(0))
            val = ops.round(input * inv_scale)
            qmin = ops.constant(q_min, torch.float32)
            qmax = ops.constant(q_max, torch.float32)
            clamped = ops.minimum(ops.maximum(val, qmin), qmax)
            return ops.to_dtype(clamped, float8_dtype)

        return Pointwise.create(
            device=input.get_device(),
            dtype=float8_dtype,
            inner_fn=functools.partial(inner_fn, scale=scale),
            ranges=input.get_size(),
        )
