
from transformers import AutoModelForCausalLM, TorchAoConfig, AutoTokenizer
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
import unittest
import copy

import torch
import torch.nn as nn
from torchao.prototype.numerics import ObserverConfig, ObserverTensor
from torchao.quantization import quantize_, FqnToConfig, Int4WeightOnlyConfig

from lm_eval.evaluator import simple_evaluate
import lm_eval
from lm_eval.utils import setup_logging
# setup_logging("DEBUG")

config = FqnToConfig(
    {
        r"re:model.layers.0.feed_forward.experts.gate_up_proj": ObserverConfig(),
        r"re:model.layers.0.feed_forward.experts.down_proj": ObserverConfig(),
    }
)
quant_config = TorchAoConfig(quant_type=config)
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    dtype=torch.bfloat16,
    # quantization_config=quant_config,
)

lm_eval_model = lm_eval.models.huggingface.HFLM(pretrained=model)

results = simple_evaluate(
    model=lm_eval_model,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size=4,
    device="cuda",
    limit=50,
    use_cache=None,  # Path to cache file (None means no caching)
)

print(results["results"]["gsm8k"])
print("\n DONE CALIBRATING \n ")
breakpoint()

convert_config = FqnToConfig(
    {
        r"re:model.layers.0.feed_forward.experts.gate_up_proj": ObserverConfig(step="convert"),
        r"re:model.layers.0.feed_forward.experts.down_proj": ObserverConfig(step="convert"),
    }
)
quantize_(model, convert_config, filter_fn=None)
lm_eval_model = lm_eval.models.huggingface.HFLM(pretrained=model)

results = simple_evaluate(
    model=lm_eval_model,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size=4,
    device="cuda",
    limit=50,
    use_cache=None,  # Path to cache file (None means no caching)
)

print("\n AFTER CALIBRATING \n ")
print(results["results"]["gsm8k"])
