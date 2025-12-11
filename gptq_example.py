# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.prototype.numerics import GPTQConfig, ObserverConfig
from torchao.quantization import quantize_

model_id = "unsloth/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# # get baseline
# print("get baseline")
# lm = HFLM(pretrained=model, batch_size=16)
# results = lm_eval.simple_evaluate(
#     model=lm,
#     tasks=["hellaswag"],
# )
# print(results["results"])

# calibrate on data
print("calibrating")
quantize_(model, ObserverConfig())
lm = HFLM(pretrained=model, batch_size=16)
results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["hellaswag"],
    limit=1000,
)
print(results["results"])

print("converting to int4 and eval")
# convert to int4 and eval
quantize_(model, GPTQConfig())

model.save_pretrained("llama3-int4-gptq", safe_serialization=False)
tokenizer.save_pretrained("llama3-int4-gptq")

print("model saved!")

# lm = HFLM(pretrained=model, batch_size=16)
# results = lm_eval.simple_evaluate(
#     model=lm,
#     tasks=["hellaswag"],
# )

# print(results["results"])
