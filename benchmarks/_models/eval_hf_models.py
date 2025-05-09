# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import subprocess
import time

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


def model_throughput(
    model,
    tokenizer,
    prompt="What are we having for dinner?",
    max_new_tokens=10,
    num_runs=5,
):
    """
    Calculate model throughput in tokens per second.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        prompt: The input prompt
        max_new_tokens: Number of tokens to generate
        num_runs: Number of runs to average over for more accurate measurement
        print_all_responses: Whether to print responses from all runs or just the last one

    Returns:
        float: Throughput in tokens per second
    """
    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    # Warmup run
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Measure generation time over multiple runs
    total_tokens = 0
    total_time = 0
    generated_ids = None

    for _ in range(num_runs):
        # Start timing
        torch.cuda.synchronize()
        start_time = time.time()

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # End timing
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate tokens generated (excluding prompt tokens)
        prompt_length = inputs.input_ids.shape[1]
        total_length = generated_ids.shape[1]
        new_tokens = total_length - prompt_length

        total_tokens += new_tokens
        total_time += end_time - start_time

    # Calculate throughput
    throughput = total_tokens / total_time

    # Get the output text for the last run
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"Response: {output_text[0][len(prompt) :]}")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(
        f"Average generation time: {(total_time / num_runs) * 1000:.2f} ms for {max_new_tokens} tokens"
    )

    return throughput


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
    prompt,
    max_new_tokens,
    num_runs,
    model_output_dir,
):
    print(f"Running model {model_id} with quantization {quantization}")
    model_name = model_id.split("/")[-1]
    model_output_dir = f"quantized_model/{model_name}-{quantization}"
    quant_config = string_to_config(quantization, None)
    quantized_model, tokenizer = quantize_model_and_save(
        model_id, quant_config=quant_config, output_dir=model_output_dir
    )
    run_lm_eval(
        model_output_dir, tasks_list=tasks, device=device, batch_size=batch_size
    )
    model_throughput(
        quantized_model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_runs=num_runs,
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
        "--batch_size", type=int, default=8, help="Batch size for lm_eval."
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
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        model_output_dir=args.output_dir,
    )
