# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import unittest

import torch
from torch import nn
from torch.testing._internal import common_utils

from torchao.dtypes import MarlinSparseLayout, SemiSparseLayout
from torchao.quantization import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.sparsity import apply_fake_sparsity, semi_sparse_weight, sparsify_
from torchao.utils import is_sm_at_least_90
import torch.nn.functional as F

import re
import unittest
import warnings
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torchao.utils import is_fbcode, is_sm_at_least_90

if not is_fbcode():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

class TestMoE2d(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size, bias=None) for _ in range(self.num_experts)])

    def forward(self, inputs, expert_size):
        """
        Forward pass of the JetMoeParallelExperts module.

        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        # return True
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        
        assert len(input_list) == len(self.experts)
        for expert, expert_input in zip(self.experts, input_list):
            output_list.append(expert(expert_input))
        results = torch.cat(output_list, dim=0)
        return results


class TestMoE3d(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        This implementation is taken from:
        https://github.com/huggingface/transformers/blob/6cade29278c4aee3f174f8950f97a3873bdb212f/src/transformers/models/jetmoe/modeling_jetmoe.py#L141
        
        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        super().__init__()
        self.moe_weight = nn.Parameter(torch.randn(num_experts, output_size, input_size))
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs, expert_size):
        """
        Forward pass of the JetMoeParallelExperts module.

        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        # return True
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.moe_weight[i]))
        results = torch.cat(output_list, dim=0)
        return results

def print_model_fqn(model):
    print("=== Parameters ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    print("\n=== Modules ===")
    for name, module in model.named_modules():
        if name:  # Skip empty name for root module
            print(f"{name}: {type(module).__name__}")


class TestQuantizeParameterMoE(common_utils.TestCase):

    def test_2d_3d_moe_equivalent(self):
        test_input = torch.randn(1024, 1024).cuda()

        model_2d = TestMoE2d(2, 1024, 1024).cuda()
        model_3d = TestMoE3d(2, 1024, 1024).cuda()

        for i, expert in enumerate(model_2d.experts):
            model_3d.moe_weight.data[i] = expert.weight.detach().clone()

        output_2d = model_2d(test_input, 512)
        output_3d = model_3d(test_input, 512)
        
        torch.testing.assert_close(output_2d, output_3d, rtol=1e-3, atol=1e-3)


    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantize_parameter(self):
        test_input = torch.randn(1024, 1024).cuda().bfloat16()

        model_2d = TestMoE2d(2, 1024, 1024).cuda().bfloat16()
        model_3d = TestMoE3d(2, 1024, 1024).cuda().bfloat16()

        for i, expert in enumerate(model_2d.experts):
            model_3d.moe_weight.data[i] = expert.weight.detach().clone()

        # quantize all linears in 2d
        quantize_(
            model_2d,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        )
        
        quantize_(
            model_3d,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow(), param_name="moe_weight"),
            filter_fn=lambda mod, fqn: fqn is '' # top level module has no fqn
        )

        output_3d = model_3d(test_input, 512)
        output_2d = model_2d(test_input, 512)

        torch.testing.assert_close(output_2d, output_3d, rtol=1e-3, atol=1e-3)

@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Checkpoints are produced in SM90+")
@unittest.skipIf(
    is_fbcode(),
    "Skipping the test in fbcode for now, not sure how to download from transformers",
)
class TestTorchAOCheckpoint(TestCase):

    def test_comprehensive_checkpoint_loading(self):
        from transformers import AutoConfig, AutoModel
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        config = AutoConfig.from_pretrained("unsloth/Llama-4-Scout-17B-16E-Instruct")
        model = Llama4TextMoe(config.text_config).to(torch.bfloat16).cuda()
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
        print(model)
        print("DONE")
