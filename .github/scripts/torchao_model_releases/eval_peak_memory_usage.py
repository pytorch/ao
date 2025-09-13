# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def eval_peak_memory_usage(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    torch.cuda.reset_peak_memory_stats()

    prompt = "Hey, are you conscious? Can you talk to me?"
    messages = [
        {
            "role": "system",
            "content": "",
        },
        {"role": "user", "content": prompt},
    ]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Prompt:", prompt)
    print("Templated prompt:", templated_prompt)
    inputs = tokenizer(
        templated_prompt,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Response:", output_text[0][len(prompt) :])

    mem = torch.cuda.max_memory_reserved() / 1e9
    print(f"Peak Memory Usage: {mem:.02f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with the specified parameters."
    )
    parser.add_argument(
        "--model_id", type=str, help="Huggingface hub model ID of the model."
    )
    args = parser.parse_args()
    eval_peak_memory_usage(args.model_id)
