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

from time import time

import torch
from transformers import AutoTokenizer, Llama4ForCausalLM
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from torchao.prototype.moe_quant.utils import (
    MoEMapping,
    MoEQuantConfig,
)
from torchao.quantization.quant_api import Int4WeightOnlyConfig, quantize_


def llama4_moe_filter_fn(module, fqn):
    return isinstance(module, Llama4TextMoe)


max_tok = 200
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = Llama4ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

moe_mapping = MoEMapping(
    target_module_type=Llama4TextMoe,
    router_fqn="router",
    top_k_fqn="top_k",
    up_proj_fqn="experts.gate_up_proj",
    up_proj_part2_fqn=None,
    down_proj_fqn="experts.down_proj",
    order_of_weight_indices=(0, 2, 1),
    act_fn_fqn="experts.act_fn",
    shared_expert_fqn="shared_expert",
    return_scores=True,
    decompose_grouped_mm=True,
)
base_config = Int4WeightOnlyConfig()

config = MoEQuantConfig(base_config, moe_mapping)
quantize_(model, config, llama4_moe_filter_fn, device="cuda")
model = torch.compile(model, mode="reduce-overhead")

prompt = "He is here, the one who will tear apart the very stars"
inputs = tokenizer(prompt, return_tensors="pt")
inputs.input_ids = inputs.input_ids.cuda()
model.generate(inputs.input_ids, max_length=30)
model.generate(inputs.input_ids, max_length=30)
generate_ids = model.generate(inputs.input_ids, max_length=max_tok)
out = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(out)

do_bench = True
if do_bench:
    start = time()
    for i in range(10):
        model.generate(inputs.input_ids, max_length=max_tok)
    elapsed = (time() - start) / 10
    print(
        f"took {elapsed:.2f} seconds, {(max_tok - inputs.input_ids.numel()) / elapsed:.2f} tok/s"
    )
