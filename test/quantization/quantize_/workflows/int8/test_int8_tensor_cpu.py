# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from torch.testing._internal import common_utils

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
    quantize_,
)
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    choose_qparams_affine,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor
from torchao.quantization.utils import compute_error, get_block_size
from torchao.testing.model_architectures import ToyTwoLinearModel
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import (
    is_ROCM,
    should_reduce_range,
)


@common_utils.instantiate_parametrized_tests
class TestInt8TensorCPU(TorchAOIntegrationTestCase):
    # Note: The reduce_range parameter can be manually set by users via the config.
    # This UT only tests automatic reduce_range to avoid CI failures on CPUs without VNNI support.
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("config_mode", ["dynamic", "static"])
    @common_utils.parametrize(
        "granularity",
        [PerRow(), PerTensor(), [PerRow(), PerTensor()], [PerTensor(), PerRow()]],
    )
    @common_utils.parametrize(
        "act_mapping_type", [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]
    )
    def test_int8_tensor_cpu(
        self, act_mapping_type, granularity, config_mode, compile, dtype
    ):
        device = "cpu"
        if is_ROCM():
            self.skipTest("Don't test CPU for ROCM version of torch")

        torch.compiler.reset()

        M, N, K = 64, 256, 256
        input_tensor = torch.randn(M, K, dtype=dtype, device=device)
        model = ToyTwoLinearModel(K, N, K, dtype=dtype, device=device).eval()
        model_q = copy.deepcopy(model)
        reduce_range = should_reduce_range(input_tensor.device)

        if config_mode == "dynamic":
            config = Int8DynamicActivationInt8WeightConfig(
                version=2,
                granularity=granularity,
                act_mapping_type=act_mapping_type,
                reduce_range=reduce_range,
            )
        else:
            assert config_mode == "static", (
                f"Expected config_mode to be 'static', got {config_mode}"
            )
            act_granularity, _ = Int8Tensor._normalize_granularity(granularity)
            quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[torch.int8]
            if reduce_range:
                quant_min, quant_max = quant_min // 2, quant_max // 2
            block_size = get_block_size(input_tensor.shape, act_granularity)
            act_quant_scale, act_quant_zero_point = choose_qparams_affine(
                input=input_tensor,
                mapping_type=act_mapping_type,
                block_size=block_size,
                target_dtype=torch.int8,
                quant_min=quant_min,
                quant_max=quant_max,
                scale_dtype=dtype,
                zero_point_dtype=torch.int8,
                keepdim=True,
                eps=torch.finfo(torch.float32).eps,
            )
            config = Int8StaticActivationInt8WeightConfig(
                act_quant_scale=act_quant_scale,
                act_quant_zero_point=act_quant_zero_point,
                granularity=granularity,
                act_mapping_type=act_mapping_type,
                reduce_range=reduce_range,
            )

        quantize_(model_q, config)

        _, weight_granularity = Int8Tensor._normalize_granularity(config.granularity)
        if isinstance(weight_granularity, PerRow):
            self.assertEqual(model_q.linear2.weight.scale.shape, (K, 1))
        elif isinstance(weight_granularity, PerTensor):
            self.assertEqual(model_q.linear2.weight.scale.shape, (1, 1))

        self.assertEqual(model_q.linear2.weight.scale.ndim, 2)

        if compile:
            model_q = torch.compile(model_q, fullgraph=True)

        output_fp = model(input_tensor)
        output_quantized = model_q(input_tensor)

        assert compute_error(output_fp, output_quantized) > 20, (
            f"Quantization error is too high got a SQNR of {compute_error(output_fp, output_quantized)}"
        )


if __name__ == "__main__":
    common_utils.run_tests()
