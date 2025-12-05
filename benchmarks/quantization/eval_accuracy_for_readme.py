# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    PerRow,
)


def string_to_config(s):
    if s is None:
        return None
    elif s == "float8_rowwise":
        return Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    elif s == "int4_weight_float8_rowwise_activation":
        return Float8DynamicActivationInt4WeightConfig()
    elif s == "int4_weight_only_hqq":
        return Int4WeightOnlyConfig(
            group_size=32,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq",
        )
    elif s == "int8_weight_only":
        return Int8WeightOnlyConfig()
    elif s == "int8":
        return Int8DynamicActivationInt8WeightConfig()
    else:
        raise AssertionError(f"unsupported {s}")


def quantize_model_and_save(model_id, quant_config, output_dir="results"):
    """Quantize the model and save it to the output directory."""
    print("Quantizing model with config: ", quant_config)
    if quant_config is None:
        quantization_config = None
    else:
        quantization_config = TorchAoConfig(quant_type=quant_config)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantized_model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir, safe_serialization=False)
    return quantized_model, tokenizer


def run_lm_eval(model_dir, tasks_list=["hellaswag"], device="cuda:0", batch_size=8):
    """Run the lm_eval command using subprocess."""
    tasks_str = ",".join(tasks_list)
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_dir}",
        "--tasks",
        f"{tasks_str}",
        "--device",
        f"{device}",
        "--batch_size",
        f"{batch_size}",
        "--output_path",
        f"{model_dir}/lm_eval_outputs/",
    ]
    subprocess.run(command, check=True)


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
    tasks,
    device,
    batch_size,
    model_output_dir,
):
    print(f"\nRunning {model_id=} with {quant_recipe_name=}\n")
    model_name = model_id.split("/")[-1]
    model_output_dir = (
        f"benchmarks/data/quantized_model/{model_name}-{quant_recipe_name}"
    )
    quant_config = string_to_config(quant_recipe_name)
    quantized_model, tokenizer = quantize_model_and_save(
        model_id, quant_config=quant_config, output_dir=model_output_dir
    )
    print(quantized_model)

    model_size = get_size_of_dir(model_output_dir) / 1e9
    print(f"checkpoint size: {model_size} GB")

    run_lm_eval(
        model_output_dir, tasks_list=tasks, device=device, batch_size=batch_size
    )
    print("done\n")


if __name__ == "__main__":
    try:
        import lm_eval  # noqa: F401
    except:
        print(
            "lm_eval is required to run this script. Please install it using pip install lm-eval."
        )
        exit(0)

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Quantize a model and evaluate its throughput."
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
        default=None,
        help="The quantization recipe to use.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="List of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on."
    )
    parser.add_argument(
        "--batch_size", type=str, default="auto", help="Batch size for lm_eval."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="quantized_models",
        help="Output directory for quantized model.",
    )
    args = parser.parse_args()

    # Use parsed arguments
    run(
        model_id=args.model_id,
        quant_recipe_name=args.quant_recipe_name,
        tasks=args.tasks,
        device=args.device,
        batch_size=args.batch_size,
        model_output_dir=args.output_dir,
    )
