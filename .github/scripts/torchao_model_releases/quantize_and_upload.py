# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from huggingface_hub import ModelCard, get_token, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    PerAxis,
    PerGroup,
    PerRow,
)


def _get_username():
    token = get_token()
    username = whoami(token=token)["name"]
    return username


def _untie_weights_and_save_locally(model_id):
    untied_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from transformers.modeling_utils import find_tied_parameters

    if getattr(
        untied_model.config.get_text_config(decoder=True), "tie_word_embeddings"
    ):
        setattr(
            untied_model.config.get_text_config(decoder=True),
            "tie_word_embeddings",
            False,
        )

    untied_model._tied_weights_keys = []
    untied_model.lm_head.weight = torch.nn.Parameter(
        untied_model.lm_head.weight.clone()
    )

    print("tied weights:", find_tied_parameters(untied_model))

    MODEL_NAME = model_id.split("/")[-1]
    # save locally
    save_to_local_path = f"{MODEL_NAME}-untied-weights"
    untied_model.save_pretrained(save_to_local_path)
    tokenizer.save_pretrained(save_to_local_path)
    return save_to_local_path


MODEL_CARD = """---
base_model: {base_model}
tags:
- transformers
- torchao
- {model_type}
license: apache-2.0
language:
- en
---

# {quant} {base_model} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Quantized from model :** {base_model}

"""


def quantize_and_upload(model_id, quant):
    _8da4w_linear_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
        weight_scale_dtype=torch.bfloat16,
    )
    _8da4w_embedding_config = IntxWeightOnlyConfig(
        weight_dtype=torch.int8,
        granularity=PerAxis(0),
    )
    quant_to_config = {
        "float8": Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        "int4": Int4WeightOnlyConfig(group_size=128),
        "8da4w": ModuleFqnToConfig(
            {
                "_default": _8da4w_linear_config,
                "model.embed_tokens": _8da4w_embedding_config,
            }
        ),
    }
    assert quant in quant_to_config, f"Unsupported quant option: {quant}"
    quant_config = quant_to_config[quant]

    model_to_quantize = model_id
    if quant == "8da4w":
        model_to_quantize = _untie_weights_and_save_locally()

    quantization_config = TorchAoConfig(quant_type=quant_config)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_to_quantize,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    username = _get_username()

    # model card
    content = MODEL_CARD.format(
        username=username,
        base_model=quantized_model.config._name_or_path,
        model_type=quantized_model.config.model_type,
        quant=quant,
    )
    card = ModelCard(content)

    # Push to hub
    MODEL_NAME = model_id.split("/")[-1]
    save_to = f"{username}/{MODEL_NAME}-{quant}"
    quantized_model.push_to_hub(save_to, safe_serialization=False)
    tokenizer.push_to_hub(save_to)
    card.push_to_hub(save_to)

    # Manual Testing
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
    generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Response:", output_text[0][len(prompt) :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with the specified parameters."
    )
    parser.add_argument(
        "--model_id", type=str, help="Huggingface hub model ID of the model."
    )
    parser.add_argument(
        "--quant",
        type=str,
        help="Quantization method. Options are float8, int4, 8da4w, int4-awq",
    )
    args = parser.parse_args()
    quantize_and_upload(args.model_id, args.quant)
