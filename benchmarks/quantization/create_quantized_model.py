# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers.quantizers.auto import get_hf_quantizer
from utils import string_to_config

from torchao._models._eval import TransformerEvalWrapper
from torchao.prototype.awq import AWQConfig
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization.quant_api import _is_linear, quantize_


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

    if "AWQ" in quant_config[0]:
        # AWQ workflow: prepare -> eval -> convert -> prepare_for_loading
        assert quant_config == "AWQ-INT4", "Only support AWQ-INT4 for now"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )

        def filter_fn_skip_lmhead(module, fqn):
            if fqn == "lm_head":
                return False
            return _is_linear(module, fqn)

        base_config = quant_config
        awq_config = AWQConfig(base_config, step="prepare")
        if safe_serialization:
            quantize_(model, awq_config, filter_fn=filter_fn_skip_lmhead)
        else:
            quantize_(model, awq_config)

        print(
            f"Calibrating AWQ with tasks: {calibration_tasks}, limit: {calibration_limit}"
        )
        TransformerEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        ).run_eval(
            tasks=calibration_tasks,
            limit=calibration_limit,
        )

        awq_config = AWQConfig(base_config, step="convert")
        if safe_serialization:
            quantize_(model, awq_config, filter_fn=filter_fn_skip_lmhead)
        else:
            quantize_(model, awq_config)

        quantized_model = model
        load_config = AWQConfig(base_config, step="prepare_for_loading")
        if safe_serialization:
            quantization_config = TorchAoConfig(load_config).to_dict()
            quantized_model.config.quantization_config = quantization_config

            hf_quantizer, _, _, _ = get_hf_quantizer(
                config=quantized_model.config,
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
            quantized_model.hf_quantizer = hf_quantizer
        else:
            quantized_model.config.quantization_config = TorchAoConfig(load_config)
    elif "SmoothQuant" in quant_config[0]:
        # SmoothQuant workflow: prepare -> eval -> convert -> prepare_for_loading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        base_config = quant_config
        smoothquant_config = SmoothQuantConfig(base_config, step="prepare")
        quantize_(model, smoothquant_config)

        print(
            f"Calibrating SmoothQuant with tasks: {calibration_tasks}, limit: {calibration_limit}"
        )
        TransformerEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        ).run_eval(
            tasks=calibration_tasks,
            limit=calibration_limit,
        )

        smoothquant_config = SmoothQuantConfig(base_config, step="convert")
        quantize_(model, smoothquant_config)

        quantized_model = model

        load_config = SmoothQuantConfig(base_config, step="prepare_for_loading")
        quantized_model.config.quantization_config = TorchAoConfig(load_config)
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
