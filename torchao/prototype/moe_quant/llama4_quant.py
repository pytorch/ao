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
from transformers import AutoTokenizer, Llama4ForCausalLM
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from torchao.prototype.moe_quant.quantizable_moe_modules import (
    MOEFeedForwardAOQuantizable,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter


def llama4_moe_filter_fn(module, fqn):
    return isinstance(module, Llama4TextMoe)


def convert_fn(module):
    # get data
    hidden_dim = module.hidden_dim
    expert_dim = module.experts.expert_dim
    num_experts = module.num_experts
    top_k = module.top_k
    act_fn = module.experts.act_fn
    shared_expert = module.shared_expert
    return_scores = True
    new_mod = MOEFeedForwardAOQuantizable(
        hidden_dim,
        expert_dim,
        num_experts,
        top_k,
        act_fn,
        shared_expert,
        return_scores,
    )

    router = module.router
    up_proj = module.experts.gate_up_proj
    w1, w3 = up_proj.permute(0, 2, 1).chunk(2, dim=1)
    w2 = module.experts.down_proj.permute(0, 2, 1)

    new_mod.router = router
    new_mod.experts.w1 = nn.Parameter(w1, requires_grad=False)
    new_mod.experts.w2 = nn.Parameter(w2, requires_grad=False)
    new_mod.experts.w3 = nn.Parameter(w3, requires_grad=False)
    return new_mod


model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = Llama4ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

_replace_with_custom_fn_if_matches_filter(
    model,
    convert_fn,
    llama4_moe_filter_fn,
)

model = model

from torchao.prototype.moe_quant.utils import (
    MoEQuantConfig,
    cond_ffn_filter,
)
from torchao.quantization import Int4WeightOnlyConfig, quantize_

quantize_(
    model,
    MoEQuantConfig(Int4WeightOnlyConfig(version=1)),
    cond_ffn_filter,
    device="cuda",
)

model.cuda()

model = torch.compile(model, mode="reduce-overhead")

prompt = "He is here, the one who will tear apart the very stars"
inputs = tokenizer(prompt, return_tensors="pt")
model.generate(inputs.input_ids.cuda(), max_length=30)
model.generate(inputs.input_ids.cuda(), max_length=30)
generate_ids = model.generate(inputs.input_ids.cuda(), max_length=50)
out = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(out)
