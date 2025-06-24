# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.deep_gemm_float8_training.deep_gemm_utils import (
    scale_narrow_tiles,
    scale_square_tiles,
    scaled_mm_deep_gemm_128_1_128_1,
    scaled_mm_deep_gemm_128_1_128_128,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


@torch._dynamo.allow_in_graph
class deep_gemm_float8_fw_bw(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
    ):
        ctx.save_for_backward(input_hp, weight_hp)
        input_orig_shape = input_hp.shape
        input_hp = input_hp.reshape(-1, input_orig_shape[-1])
        assert input_hp.shape[-1] % 128 == 0, "unsupported"

        # cast input to float8
        input_fp8, input_scale = scale_narrow_tiles(input_hp, tile_size=128)

        # cast weight to float8 and save for bw
        weight_fp8, weight_scale = scale_square_tiles(weight_hp, tile_size=128)

        # float8 gemm
        output = scaled_mm_deep_gemm_128_1_128_128(
            input_fp8, weight_fp8, 1.0 / input_scale, 1.0 / weight_scale
        )
        output = output.reshape(*input_orig_shape[:-1], output.shape[-1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_hp, weight_hp = ctx.saved_tensors
        weight_hp_t = weight_hp.t().contiguous()

        input_orig_shape = input_hp.shape
        input_hp = input_hp.reshape(-1, input_orig_shape[-1])

        grad_output_orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, grad_output_orig_shape[-1])
        assert grad_output.shape[1] % 128 == 0, "unsupported"

        grad_output_fp8_dim0, grad_output_scale_dim0 = scale_narrow_tiles(
            grad_output, tile_size=128
        )
        # TODO reuse from forward instead of casting again
        weight_fp8, weight_scale = scale_square_tiles(weight_hp_t, tile_size=128)
        grad_input = scaled_mm_deep_gemm_128_1_128_128(
            grad_output_fp8_dim0,
            weight_fp8,
            1.0 / grad_output_scale_dim0,
            1.0 / weight_scale,
        )
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        if False:
            # passes unit tests, but broken in torchtitan with
            # https://gist.github.com/vkuzo/3a763e150dbb37e5b917833a460f7f92
            grad_output_fp8_dim1, grad_output_scale_dim1 = scale_narrow_tiles(
                grad_output.t().contiguous(), tile_size=128
            )
            input_hp_fp8_dim1, input_hp_scale_dim1 = scale_narrow_tiles(
                input_hp.t().contiguous(), tile_size=128
            )
            grad_weight = scaled_mm_deep_gemm_128_1_128_1(
                grad_output_fp8_dim1,
                input_hp_fp8_dim1,
                1.0 / grad_output_scale_dim1,
                1.0 / input_hp_scale_dim1,
            )
            grad_weight = grad_weight.to(grad_output.dtype)
        else:
            # workaround - leave this gemm in bf16
            grad_weight = grad_output.t() @ input_hp

        return grad_input, grad_weight


class DeepGemmFloat8Linear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = deep_gemm_float8_fw_bw.apply(
            input,
            self.weight,
        )
        # TODO add bias support
        return output

    @classmethod
    def from_float(
        cls,
        mod,
    ):
        assert mod.bias is None, "unsupported"
        assert mod.in_features % 128 == 0, "unsupported"
        assert mod.out_features % 128 == 0, "unsupported"
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class DeepGemmFloat8LinearConfig(AOBaseConfig):
    pass


@register_quantize_module_handler(DeepGemmFloat8LinearConfig)
def _deep_gemm_float8_inference_linear_transform(module, config):
    return DeepGemmFloat8Linear.from_float(module)
