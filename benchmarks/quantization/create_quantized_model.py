# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from utils import string_to_config


def quantize_model_and_save(
    model_id,
    quant_config,
    output_dir,
    safe_serialization=False,
):
    """Quantize the model and save it to the output directory."""
    print("Quantizing model with config: ", quant_config)
    quantization_config = (
        TorchAoConfig(quant_type=quant_config) if quant_config else None
    )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantized_model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir, safe_serialization=safe_serialization)
    return quantized_model, tokenizer


def get_size_of_dir(model_output_dir):
    # get dir size from shell, to skip complexity of dealing with tensor
    # subclasses
    result = subprocess.run(
        ["du", "-sb", model_output_dir], capture_output=True, text=True
    )
    size = int(result.stdout.split()[0])
    return size


def run(
    model_id: str,
    quant_recipe_name: str | None,
    model_output_dir,
):
    print(f"\nRunning {model_id=} with {quant_recipe_name=}\n")
    quant_config = string_to_config(quant_recipe_name)
    quantized_model, tokenizer = quantize_model_and_save(
        model_id, quant_config=quant_config, output_dir=model_output_dir
    )
    print(quantized_model)
    print(f"saved {model_id=}, {quant_recipe_name=} to {model_output_dir=}")
    model_size = get_size_of_dir(model_output_dir) / 1e9
    print(f"checkpoint size: {model_size} GB")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load a model from HuggingFace, quantize it and save it to disk."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="The model ID to use.",
    )
    parser.add_argument(
        "--quant_recipe_name",
        type=str,
        help="The quantization recipe to use, 'None' means no quantization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmarks/data/quantized_model/test",
        help="Output directory for quantized model.",
    )
    args = parser.parse_args()

    # Use parsed arguments
    run(
        model_id=args.model_id,
        quant_recipe_name=args.quant_recipe_name,
        model_output_dir=args.output_dir,
    )
