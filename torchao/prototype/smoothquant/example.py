# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.prototype.awq.example import get_calib_dataset
from torchao.prototype.smoothquant import (
    SmoothQuantConfig,
)
from torchao.prototype.smoothquant.core import SmoothQuantStep
from torchao.quantization import quantize_
from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig


# TODO: Build benchmark within vLLM ecosystem with more quantization APIs
# See https://github.com/pytorch/ao/issues/2815 for more details
def benchmark(model, tokenizer, max_seq_length=512, tasks=["PPL"], device="cuda"):
    """Benchmark model with perplexity calculation on WikiText-2"""
    # Load WikiText-2 test set
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Prepare text data and truncate if necessary
    text = "\n\n".join(dataset["text"])
    # Get model's maximum sequence length
    model_max_length = getattr(tokenizer, "model_max_length", max_seq_length)
    if model_max_length > 1000000:  # Default large value, use our max_seq_length
        model_max_length = max_seq_length

    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=model_max_length
    )

    # Calculate perplexity
    model.eval()
    nlls = []

    with torch.no_grad():
        seq_len = encodings.input_ids.size(1)
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, max_seq_length):
            end_loc = min(begin_loc + max_seq_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            # Measure inference time
            start_time = time.time()
            outputs = model(input_ids, labels=target_ids)
            inference_time = time.time() - start_time

            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return {
        "perplexity": ppl.item(),
        "tokens_per_sec": input_ids.size(1) / inference_time,
    }


def quantize_and_eval(
    repo_id: str,
    alpha: float,
    tasks: list[str],
    max_seq_length: int,
    calibration_limit: int,
    device: str,
    precision: torch.dtype,
    compile: bool,
    model_save_path: str,
    model_save_hf_hub_path: str,
):
    print(f"Loading model on {device}...")
    torch.manual_seed(34)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = (
        AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision)
        .eval()
        .to(device)
    )
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # Step 1: Prepare - insert observers
    print("running SmoothQuant prepare and calibrate")
    t0 = time.time()
    quant_config = SmoothQuantConfig(
        base_config=Int8DynamicActivationInt8WeightConfig(),
        step=SmoothQuantStep.PREPARE,
        alpha=alpha,
    )
    quantize_(model, quant_config)

    # Step 2: Calibration
    calibration_data = get_calib_dataset(
        tokenizer=tokenizer, n_samples=calibration_limit, block_size=max_seq_length
    )
    for batch in calibration_data:
        model(batch.to(device))
        batch.to("cpu")

    print(f"time for prepare and calibration: {time.time() - t0:.02f} seconds")

    # Step 3: Convert to quantized model
    print("running SmoothQuant convert")
    t0 = time.time()
    quant_config.step = SmoothQuantStep.CONVERT
    quantize_(model, quant_config)
    print(f"time for convert: {time.time() - t0:.02f} seconds")

    # Set up config for loading
    quant_config.step = SmoothQuantStep.PREPARE_FOR_LOADING
    model.config.quantization_config = TorchAoConfig(quant_config)

    if model_save_path is not None:
        print(f"Saving model to {model_save_path}")
        torch.save(model, model_save_path)

    if model_save_hf_hub_path is not None:
        print("pushing model to hub:", model_save_hf_hub_path)
        model.push_to_hub(model_save_hf_hub_path, safe_serialization=False)
        tokenizer.push_to_hub(model_save_hf_hub_path)

    if compile:
        model = torch.compile(model)

    print("Benchmarking SmoothQuant model...")
    return benchmark(model, tokenizer, max_seq_length, tasks=tasks, device=device)


def compare_models(
    repo_id: str,
    alpha: float,
    tasks: list[str],
    max_seq_length: int,
    calibration_limit: int,
    device: str,
    precision: torch.dtype,
    compile: bool,
    model_save_path: str,
    model_save_hf_hub_path: str,
):
    """Compare perplexity and speed for behchmarking SmoothQuant"""

    # Case 1: Base model without quantization
    print("Benchmarking base model...")
    torch.manual_seed(34)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = (
        AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision)
        .eval()
        .to(device)
    )
    if compile:
        model = torch.compile(model)
    base_results = benchmark(
        model, tokenizer, max_seq_length, tasks=tasks, device=device
    )

    # Case 2: W4A8-dynamic without SmoothQuant
    print("Benchmarking W4A8-dynamic without SmoothQuant...")
    torch.manual_seed(34)
    w4a8_model = (
        AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision)
        .eval()
        .to(device)
    )
    quantize_(w4a8_model, Int8DynamicActivationInt8WeightConfig())
    if compile:
        w4a8_model = torch.compile(w4a8_model)
    w4a8_results = benchmark(
        w4a8_model, tokenizer, max_seq_length, tasks=tasks, device=device
    )

    # Case 3: SmoothQuant + W4A8-dynamic
    print("Benchmarking SmoothQuant with W4A8-dynamic...")
    smoothquant_results = quantize_and_eval(
        repo_id,
        alpha,
        tasks,
        max_seq_length,
        calibration_limit,
        device,
        precision,
        compile,
        model_save_path,
        model_save_hf_hub_path,
    )

    # Calculate changes and display results
    w4a8_ppl_change = (
        (w4a8_results["perplexity"] - base_results["perplexity"])
        / base_results["perplexity"]
        * 100
    )
    w4a8_speed_change = (
        (w4a8_results["tokens_per_sec"] - base_results["tokens_per_sec"])
        / base_results["tokens_per_sec"]
        * 100
    )

    smoothquant_ppl_change = (
        (smoothquant_results["perplexity"] - base_results["perplexity"])
        / base_results["perplexity"]
        * 100
    )
    smoothquant_speed_change = (
        (smoothquant_results["tokens_per_sec"] - base_results["tokens_per_sec"])
        / base_results["tokens_per_sec"]
        * 100
    )

    # Print results
    print(
        f"\nBase: PPL={base_results['perplexity']:.2f}, Speed={base_results['tokens_per_sec']:.2f} tokens/sec"
    )
    print(
        f"W4A8-Dynamic: PPL={w4a8_results['perplexity']:.2f}, Speed={w4a8_results['tokens_per_sec']:.2f} tokens/sec"
    )
    print(
        f"SmoothQuant+W4A8: PPL={smoothquant_results['perplexity']:.2f}, Speed={smoothquant_results['tokens_per_sec']:.2f} tokens/sec"
    )
    print(f"W4A8 Changes: PPL {w4a8_ppl_change:+.2f}%, Speed {w4a8_speed_change:+.2f}%")
    print(
        f"SmoothQuant Changes: PPL {smoothquant_ppl_change:+.2f}%, Speed {smoothquant_speed_change:+.2f}%"
    )

    return {
        "base_model": base_results,
        "w4a8_model": w4a8_results,
        "smoothquant_model": smoothquant_results,
        "w4a8_ppl_change_percent": w4a8_ppl_change,
        "w4a8_speed_improvement_percent": w4a8_speed_change,
        "smoothquant_ppl_change_percent": smoothquant_ppl_change,
        "smoothquant_speed_improvement_percent": smoothquant_speed_change,
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a model with SmoothQuant quantization."
    )

    parser.add_argument(
        "--repo", type=str, required=True, help="Repository ID of the model."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="The alpha hyperparameter for SmoothQuant. Default is 0.5.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        help="Task to benchmark model on.",
        default=["PPL"],
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=10,
        help="Number of samples to use for calibration. Default is 10.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the evaluation on. Default is 'cuda'.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        help="Precision type. Default is 'bfloat16'.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length. Default is 512",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Flag to indicate if compilation is required.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to store the quantized model.",
    )
    parser.add_argument(
        "--model_save_hf_hub_path",
        type=str,
        default=None,
        help="Huggingface hub path to store the quantized model and tokenizer.",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Convert precision argument to torch dtype
    precision_dtype = getattr(torch, args.precision, torch.bfloat16)
    result = compare_models(
        args.repo,
        args.alpha,
        args.tasks,
        args.max_seq_length,
        args.calibration_limit,
        args.device,
        precision_dtype,
        args.compile,
        args.model_save_path,
        args.model_save_hf_hub_path,
    )
