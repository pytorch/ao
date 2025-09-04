# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# T tokens
# E experts
# D dim
# I intermediate dim
# A activated experts
# T'(e) tokens for expert e

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssMLP

from torchao.dtypes import Int4XPULayout
from torchao.prototype.moe_quant.quantizable_moe_modules import (
    MOEFeedForwardAOQuantizable,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.quant_primitives import ZeroPointDomain


def gpt_oss_moe_filter_fn(module, fqn):
    return isinstance(module, GptOssMLP)


def convert_fn(module):
    # get data
    hidden_dim = module.router.hidden_dim
    expert_dim = module.experts.expert_dim
    num_experts = module.router.num_experts
    top_k = module.router.top_k
    act_fn = torch.nn.functional.sigmoid
    shared_expert = None
    return_scores = True
    new_mod = MOEFeedForwardAOQuantizable(
        hidden_dim,
        expert_dim,
        num_experts,
        top_k,
        act_fn,
        shared_expert,
        return_scores,
        empty_init=True,
        with_bias=True,
        gpt_oss_mlp=True,
        limit=module.experts.limit,  # GPT-OSS uses a limit for the activation
        alpha=module.experts.alpha,  # GPT-OSS uses a different activation function
    )

    router = module.router
    up_proj = module.experts.gate_up_proj
    w1 = up_proj[..., ::2].permute(0, 2, 1).contiguous()  # To Do
    w3 = up_proj[..., 1::2].permute(0, 2, 1).contiguous()  # To Do
    w2 = module.experts.down_proj.permute(0, 2, 1)

    bias1 = module.experts.gate_up_proj_bias[..., ::2].contiguous()
    bias3 = module.experts.gate_up_proj_bias[..., 1::2].contiguous()
    bias2 = module.experts.down_proj_bias

    new_mod.router = router
    new_mod.experts.w1 = nn.Parameter(w1, requires_grad=False)
    new_mod.experts.bias1 = nn.Parameter(bias1, requires_grad=False)

    new_mod.experts.w2 = nn.Parameter(w2, requires_grad=False)
    new_mod.experts.bias2 = nn.Parameter(bias2, requires_grad=False)

    new_mod.experts.w3 = nn.Parameter(w3, requires_grad=False)
    new_mod.experts.bias3 = nn.Parameter(bias3, requires_grad=False)

    return new_mod


model_id = "unsloth/gpt-oss-20b-BF16"
model = GptOssForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

_replace_with_custom_fn_if_matches_filter(
    model,
    convert_fn,
    gpt_oss_moe_filter_fn,
)

from torchao.prototype.moe_quant.utils import (
    MoEQuantConfig,
    cond_ffn_filter,
)
from torchao.quantization import Int4WeightOnlyConfig, quantize_

quantize_(
    model,
    MoEQuantConfig(
        Int4WeightOnlyConfig(
            group_size=64, layout=Int4XPULayout(), zero_point_domain=ZeroPointDomain.INT
        )
    ),
    cond_ffn_filter,
    device="xpu",
)
quantize_(
    model,
    Int4WeightOnlyConfig(
        group_size=64, layout=Int4XPULayout(), zero_point_domain=ZeroPointDomain.INT
    ),
)

model.xpu()

prompt = "He is here, the one who will tear apart the very stars"
inputs = tokenizer(prompt, return_tensors="pt")
model.generate(inputs.input_ids.xpu(), max_length=30)
model.generate(inputs.input_ids.xpu(), max_length=30)
generate_ids = model.generate(inputs.input_ids.xpu(), max_length=50, do_sample=False)
out = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(out)
