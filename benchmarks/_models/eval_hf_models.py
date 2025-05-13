# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from benchmarks.microbenchmarks.utils import string_to_config
from torchao.quantization import *  # noqa: F401, F403
from torchao.quantization.utils import _lm_eval_available


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
        torch_dtype=torch.bfloat16,
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
    ]
    subprocess.run(command, check=True)


def get_model_size_in_bytes(model, ignore_embeddings=False):
    """
    Returns the model size in bytes. The option to ignore embeddings
    is useful for models with disproportionately large embeddings compared
    to other model parameters that get quantized/sparsified.
    """

    def flat_size(tensor):
        if hasattr(tensor, "__tensor_flatten__"):
            size = 0
            # 0th element is a list of attributes that
            # hold tensors
            for attr_name in tensor.__tensor_flatten__()[0]:
                sub_tensor = getattr(tensor, attr_name)
                size += flat_size(sub_tensor)
            return size
        else:
            return tensor.numel() * tensor.element_size()

    model_size = 0
    for _, child in model.named_children():
        if not (isinstance(child, torch.nn.Embedding) and ignore_embeddings):
            for p in itertools.chain(
                child.parameters(recurse=False), child.buffers(recurse=False)
            ):
                model_size += flat_size(p)
            model_size += get_model_size_in_bytes(child, ignore_embeddings)
    return model_size


def run(
    model_id,
    quantization,
    tasks,
    device,
    batch_size,
    model_output_dir,
):
    print(f"Running model {model_id} with quantization {quantization}")
    model_name = model_id.split("/")[-1]
    model_output_dir = f"quantized_model/{model_name}-{quantization}"
    quant_config = string_to_config(quantization, None)
    quantized_model, tokenizer = quantize_model_and_save(
        model_id, quant_config=quant_config, output_dir=model_output_dir
    )
    print("Compiling model ....")
    quantized_model = torch.compile(
        quantized_model,
        mode="reduce-overhead",
        fullgraph=True,
    )
    run_lm_eval(
        model_output_dir, tasks_list=tasks, device=device, batch_size=batch_size
    )
    model_size = get_model_size_in_bytes(quantized_model, ignore_embeddings=True) / 1e9
    print(f"Model size: {model_size:.2f} GB")


if __name__ == "__main__":
    if not _lm_eval_available:
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
        "--quantization",
        type=str,
        default=None,
        help="The quantization method to use.",
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
        "--batch_size", type=int, default=1, help="Batch size for lm_eval."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are we having for dinner?",
        help="Prompt for model throughput evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Max new tokens to generate for throughput evaluation.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for throughput evaluation.",
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
        quantization=args.quantization,
        tasks=args.tasks,
        device=args.device,
        batch_size=args.batch_size,
        model_output_dir=args.output_dir,
    )
