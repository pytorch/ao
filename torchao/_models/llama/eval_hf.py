# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow


def quantize_model_and_save(model_id, quant_config, output_dir="results"):
    """Quantize the model and save it to the output directory."""
    quantization_config = TorchAoConfig(quant_type=quant_config)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantized_model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir, safe_serialization=False)
    return quantized_model, tokenizer


# Run lm_eval
# lm_eval --model hf --model_args pretrained=llama-fp8 --tasks hellaswag --device cuda:0 --batch_size 8

import subprocess


def run_lm_eval(model_dir, tasks="hellaswag", device="cuda:0", batch_size=8):
    """Run the lm_eval command using subprocess."""
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_dir}",
        "--tasks",
        f"{tasks}",
        "--device",
        f"{device}",
        "--batch_size",
        f"{batch_size}",
    ]
    subprocess.run(command, check=True)


# def push_to_hub(user_id, model_name, quant_recipe='float8dq'):
#     """Push to hub"""
#     save_to = f"{user_id}/{model_name}-{quant_recipe}"
#     quantized_model.push_to_hub(save_to, safe_serialization=False)
#     tokenizer.push_to_hub(save_to)

# def prompt_testing(quantized_model, tokenizer):
#     # Manual Testing
#     prompt = "Hey, are you conscious? Can you talk to me?"
#     messages = [
#         {
#             "role": "system",
#             "content": "",
#         },
#         {"role": "user", "content": prompt},
#     ]
#     templated_prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     print("Prompt:", prompt)
#     print("Templated prompt:", templated_prompt)
#     inputs = tokenizer(
#         templated_prompt,
#         return_tensors="pt",
#     ).to("cuda")
#     generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
#     output_text = tokenizer.batch_decode(
#         generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     print("Response:", output_text[0][len(prompt):])


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.1-8B"
    model_name = model_id.split("/")[-1]
    model_output_dir = f"quantized_model/{model_name}"
    quant_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    quantized_model, tokenizer = quantize_model_and_save(
        model_id, quant_config=quant_config, output_dir=model_output_dir
    )
    run_lm_eval(model_output_dir)
    # prompt_testing(quantized_model, tokenizer)
