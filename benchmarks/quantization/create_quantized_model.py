# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers.quantizers.auto import get_hf_quantizer
from utils import string_to_config

from torchao.prototype.awq import AWQConfig
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization.quant_api import _is_linear, quantize_


def _apply_calibration_based_quantization(
    model,
    config_class,
    base_config,
    calibration_tasks,
    calibration_limit,
    tokenizer,
    max_seq_length,
    safe_serialization,
    filter_fn=None,
):
    """Apply prepare->calibrate->convert workflow for AWQ/SmoothQuant."""
    # Prepare
    quantize_(model, config_class(base_config, step="prepare"), filter_fn=filter_fn)

    # Calibrate
    print(
        f"Calibrating {config_class.__name__} with tasks: {calibration_tasks}, limit: {calibration_limit}"
    )
    evaluator.simple_evaluate(
        HFLM(pretrained=model, tokenizer=tokenizer),
        tasks=calibration_tasks,
        limit=calibration_limit,
        batch_size=1,
    )

    # Convert (quantize)
    quantize_(model, config_class(base_config, step="convert"), filter_fn=filter_fn)

    # Prepare for loading
    load_config = config_class(base_config, step="prepare_for_loading")
    # TODO: safe serialization support for SmoothQuant
    if safe_serialization and config_class == AWQConfig:
        model.config.quantization_config = TorchAoConfig(load_config).to_dict()
        hf_quantizer, _, _, _ = get_hf_quantizer(
            config=model.config,
            quantization_config=None,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            weights_only=True,
            user_agent={
                "file_type": "model",
                "framework": "pytorch",
                "from_auto_class": False,
            },
        )
        model.hf_quantizer = hf_quantizer
    else:
        model.config.quantization_config = TorchAoConfig(load_config)


def quantize_model_and_save(
    model_id,
    quant_config,
    output_dir,
    calibration_tasks=None,
    calibration_limit=10,
    max_seq_length=2048,
    safe_serialization=False,
):
    """Quantize the model and save it to the output directory."""
    print("Quantizing model with config: ", quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quant_config == "awq_int4_weight_only":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda:0", dtype=torch.bfloat16
        )
        filter_fn = (
            (lambda module, fqn: fqn != "lm_head" and _is_linear(module, fqn))
            if safe_serialization
            else None
        )
        _apply_calibration_based_quantization(
            model,
            AWQConfig,
            quant_config,
            calibration_tasks,
            calibration_limit,
            tokenizer,
            max_seq_length,
            safe_serialization,
            filter_fn,
        )
        quantized_model = model

    elif quant_config == "smoothquant_int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=torch.bfloat16
        )
        _apply_calibration_based_quantization(
            model,
            SmoothQuantConfig,
            quant_config,
            calibration_tasks,
            calibration_limit,
            tokenizer,
            max_seq_length,
            safe_serialization,
        )
        quantized_model = model
    else:
        # Calibration-free
        quantization_config = (
            TorchAoConfig(quant_type=quant_config) if quant_config else None
        )
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

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
    calibration_tasks=None,
    calibration_limit=10,
    max_seq_length=2048,
):
    print(f"\nRunning {model_id=} with {quant_recipe_name=}\n")
    quant_config = string_to_config(quant_recipe_name)
    quantized_model, tokenizer = quantize_model_and_save(
        model_id,
        quant_config=quant_config,
        output_dir=model_output_dir,
        calibration_tasks=calibration_tasks,
        calibration_limit=calibration_limit,
        max_seq_length=max_seq_length,
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
    parser.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="Tasks to use for calibration (for AWQ/SmoothQuant).",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=10,
        help="Number of samples to use for calibration (for AWQ/SmoothQuant).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration.",
    )
    args = parser.parse_args()

    # Use parsed arguments
    run(
        model_id=args.model_id,
        quant_recipe_name=args.quant_recipe_name,
        model_output_dir=args.output_dir,
        calibration_tasks=args.calibration_tasks,
        calibration_limit=args.calibration_limit,
        max_seq_length=args.max_seq_length,
    )
