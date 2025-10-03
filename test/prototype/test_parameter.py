# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import unittest

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.quantization.quant_api import (
    ModuleOrParamFqnToConfig,
    PerRow,
    quantize_,
)
from torchao.core.config import AOBaseConfig
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
from torchao.utils import is_fbcode, is_sm_at_least_90

if not is_fbcode():
    pass

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Checkpoints are produced in SM90+")
@unittest.skipIf(
    is_fbcode(),
    "Skipping the test in fbcode for now, not sure how to download from transformers",
)
class TestQuantizeFQNParam(TestCase):
    def test_quantize_param_fqn_exact(self):
        from transformers import AutoConfig
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        config = AutoConfig.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct"
        ).text_config
        model = Llama4TextMoe(config).to(torch.bfloat16).cuda()
        torch.randn(16, 128, config.hidden_size).cuda().bfloat16()

        quant_config = ModuleOrParamFqnToConfig(
            {
                "experts.gate_up_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                ),
            }
        )

        quantize_(
            model,
            quant_config,
        )

        assert isinstance(model.experts.gate_up_proj, Float8Tensor)

    def test_quantize_param_and_module_fqn(self):
        from transformers import AutoConfig
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
        from torchao.quantization import PerTensor

        config = AutoConfig.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct"
        ).text_config
        model = Llama4TextMoe(config).to(torch.bfloat16).cuda()
        torch.randn(16, 128, config.hidden_size).cuda().bfloat16()
        quant_config = ModuleOrParamFqnToConfig(
            {
                "experts.gate_up_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                ),
                "shared_expert.gate_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                ),
            }
        )

        quantize_(
            model,
            quant_config,
        )

    def test_quantize_param_and_module_fqn_regex(self):
        from transformers import AutoConfig
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
        from torchao.quantization import PerTensor

        config = AutoConfig.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct"
        ).text_config
        model = Llama4TextMoe(config).to(torch.bfloat16).cuda()
        torch.randn(16, 128, config.hidden_size).cuda().bfloat16()
        quant_config = ModuleOrParamFqnToConfig(
            {
                ".*gate_up_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                ),
                "shared_expert.gate_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                ),
            }
        )

        quantize_(
            model,
            quant_config,
        )

        assert isinstance(model.experts.gate_up_proj, Float8Tensor)
        assert isinstance(model.shared_expert.gate_proj.weight, Float8Tensor)

    def test_unsupported_param_config_raises_not_implemented_error(self):
        """Test that using an unsupported parameter config raises NotImplementedError."""
        from dataclasses import dataclass
        
        # Create a custom config that doesn't have a registered parameter handler
        @dataclass
        class UnsupportedParamConfig(AOBaseConfig):
            some_value: int = 42
        
        # Create a simple model
        model = nn.Linear(10, 5).cuda().bfloat16()
        
        # Create config with unsupported parameter handler
        quant_config = ModuleOrParamFqnToConfig(
            {
                "weight": UnsupportedParamConfig(),
            }
        )
        
        # This should raise NotImplementedError
        with self.assertRaises(NotImplementedError) as context:
            quantize_(model, quant_config)
        
        # Check that the error message contains the expected text
        self.assertIn("Parameter quantization for", str(context.exception))
        self.assertIn("not supported currently", str(context.exception))
        self.assertIn("UnsupportedParamConfig", str(context.exception))


if __name__ == "__main__":
    unittest.main()
