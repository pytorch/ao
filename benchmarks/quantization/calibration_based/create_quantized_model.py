# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.prototype.awq import AWQConfig
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization.quant_api import quantize_

from ..utils import get_size_of_dir
from .utils import string_to_calibration_config


def _apply_calibration(
    model, config_class, base_config, tasks, limit, tokenizer, filter_fn=None
):
    """Apply prepare->calibrate->convert workflow for AWQ/SmoothQuant."""
    # Prepare
    quantize_(model, config_class(base_config, step="prepare"), filter_fn=filter_fn)
    print(f"Calibrating with tasks={tasks}, limit={limit}")

    # Calibrate
    evaluator.simple_evaluate(
        HFLM(pretrained=model, tokenizer=tokenizer),
        tasks=tasks,
        limit=limit,
        batch_size=1,
    )
    quantize_(model, config_class(base_config, step="convert"), filter_fn=filter_fn)
    load_config = config_class(base_config, step="prepare_for_loading")
    model.config.quantization_config = TorchAoConfig(load_config)


def quantize_model_and_save(
    model: str,
    recipe: str,
    base_config_cls: dict,
    output_dir: str,
    tasks: list[str],
    limit: int,
):
    """Quantize model with calibration and save."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model, device_map="cuda:0", dtype=torch.bfloat16
    )

    if base_config_cls is None:
        pass
    elif recipe == "awq_int4_weight_only":
        _apply_calibration(model, AWQConfig, base_config_cls, tasks, limit, tokenizer)
    elif recipe == "smoothquant_int8":
        _apply_calibration(
            model, SmoothQuantConfig, base_config_cls, tasks, limit, tokenizer
        )
    else:
        raise AssertionError(f"unsupported recipe: {recipe}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize model with calibration (AWQ/SmoothQuant)"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--recipe",
        required=True,
        help="awq_int4_weight_only, smoothquant_int8, or None (no quantization)",
    )
    parser.add_argument("--output_dir", default="benchmarks/data/quantized_model/test")
    parser.add_argument("--calibration_tasks", nargs="+", default=["wikitext"])
    parser.add_argument("--calibration_limit", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{args.model} with {args.recipe}\n")
    base_config = string_to_calibration_config(args.recipe)
    model, _ = quantize_model_and_save(
        args.model,
        args.recipe,
        base_config,
        args.output_dir,
        args.calibration_tasks,
        args.calibration_limit,
    )
    print(f"Saved to {args.output_dir}")
    print(f"Size: {get_size_of_dir(args.output_dir) / 1e9:.2f} GB")
