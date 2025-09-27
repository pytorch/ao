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
    ParamFqnToConfig,
    PerRow,
    quantize_,
)
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

        quant_config = ParamFqnToConfig(
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

    def test_quantize_param_fqn_regex(self):
        from transformers import AutoConfig
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        config = AutoConfig.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct"
        ).text_config
        model = Llama4TextMoe(config).to(torch.bfloat16).cuda()
        torch.randn(16, 128, config.hidden_size).cuda().bfloat16()
        # print(model.experts)
        for name, param in model.named_parameters():
            print(name)

        from torchao.quantization.quant_api import ParamFqnToConfig

        quant_config = ParamFqnToConfig(
            {
                ".*gate_up_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                ),
            }
        )

        quantize_(
            model,
            quant_config,
        )

        assert isinstance(model.experts.gate_up_proj, Float8Tensor)

    def test_quantize_param_root(self):
        param = nn.Parameter(torch.randn(1024, 1024).cuda().to(torch.bfloat16))
        new_param = quantize_(
            param, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        )
        assert isinstance(new_param, Float8Tensor)
