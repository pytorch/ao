# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import time
from typing import Any, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.prototype.gptq import GPTQConfig
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
    NVFP4WeightOnlyConfig,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
from torchao.quantization.granularity import PerRow
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

"""
GPTQ sequential quantization example for huggignface models.

Suppose we have a two layer model that we want to quantize. We can either use the unquantized or quantized output of the first layer as  our observed input to the the second layer.

To do this for huggingface models, we interate through the layers one at a time and quantize each block respectively with GPTQ.

Depending on your exact task, you may see a difference in accuraccy between the two approaches. Users need to implement sequential quantization for their specific model type.
"""


# run with no grad otherwise keeping all the tensors around for the backwards will cause oom
@torch.no_grad()
def sequential_quantize(
    model,
    calibration_data: List[torch.Tensor],
    config: Any,
) -> None:
    # Get device from embed_tokens layer (supports device_map="auto")
    embed_device = next(model.model.embed_tokens.parameters()).device

    # Prepare embeddings
    inputs = []
    position_ids = []
    position_embeddings = []

    # Generate embeddings for each sequence
    for seq in calibration_data:
        seq_length = seq.shape[1]
        embedded = model.model.embed_tokens(seq.to(embed_device))
        inputs.append(embedded)
        pid = torch.arange(
            0, seq_length, dtype=torch.long, device=embed_device
        ).unsqueeze(0)
        position_ids.append(pid)
        position_embeddings.append(model.model.rotary_emb(embedded, pid))

    # Process each transformer block sequentially
    num_blocks = len(model.model.layers)
    for block_idx in tqdm(range(num_blocks), desc="Quantizing blocks"):
        block = model.model.layers[block_idx]
        print(f"Working on block {block_idx} ...")

        for i in range(len(inputs)):
            block(
                inputs[i].to(next(block.parameters()).device),
                position_ids=position_ids[i],
                position_embeddings=position_embeddings[i],
            )

        quantize_(block, config)

        # Synchronize across devices after quantizing each block
        for i in tqdm(range(len(inputs)), desc="propogating activations"):
            inputs[i] = block(
                inputs[i].to(next(block.parameters()).device),
                position_ids=position_ids[i],
                position_embeddings=position_embeddings[i],
            )


def prepare_dataset(
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    dataset_id: str = "ultrachat200k",
    dataset_split: str = "train_sft",
    seed: int = 42,
) -> List[torch.Tensor]:
    # Map dataset names to HuggingFace IDs
    dataset_map = {
        "hellaswag": "Rowan/hellaswag",
        "ultrachat200k": "HuggingFaceH4/ultrachat_200k",
    }

    hf_dataset_id = dataset_map.get(dataset_id, dataset_id)

    # Load dataset and preprocess
    train_dataset_raw = load_dataset(hf_dataset_id, split=dataset_split, streaming=True)
    train_dataset_raw = train_dataset_raw.shuffle(seed=seed, buffer_size=1_000)

    def preprocess_hellaswag(example):
        # HellaSwag format: context + correct ending
        context = example["ctx"]
        endings = example["endings"]
        correct_ending = endings[int(example["label"])]
        text = context + " " + correct_ending
        return {"text": text}

    def preprocess_ultrachat(example):
        # UltraChat format: conversation messages
        messages = example.get("messages", [])
        # Concatenate all messages into a single text
        text = " ".join([msg.get("content", "") for msg in messages])
        return {"text": text}

    # Choose preprocessing based on dataset
    if dataset_id == "hellaswag":
        train_dataset_raw = train_dataset_raw.map(preprocess_hellaswag)
    elif dataset_id == "ultrachat200k":
        train_dataset_raw = train_dataset_raw.map(preprocess_ultrachat)

    train_dataset = []
    for i, sample in enumerate(train_dataset_raw):
        if i == num_calibration_samples:
            break
        tokenized_sample = tokenizer(
            sample["text"],
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        train_dataset.append(tokenized_sample["input_ids"])
    return train_dataset


def dequantize_mx_tensors(model):
    """Dequantize any MXTensor parameters before saving."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    # First pass: dequantize all module parameters (weight, bias, etc.)
    for name, module in model.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, MXTensor):
                print(f"Dequantizing MXTensor in {name}.{param_name}")
                dequantized = param.dequantize(output_dtype=param._orig_dtype)
                setattr(module, param_name, torch.nn.Parameter(dequantized, requires_grad=False))

    # Second pass: check all parameters in state_dict to catch any remaining MXTensors
    state_dict = model.state_dict()
    for param_name, param in list(state_dict.items()):
        if isinstance(param, MXTensor):
            print(f"Dequantizing remaining MXTensor: {param_name}")
            dequantized = param.dequantize(output_dtype=param._orig_dtype)
            # Update the parameter in the model
            keys = param_name.split('.')
            module = model
            for key in keys[:-1]:
                if key.isdigit():
                    module = module[int(key)]
                else:
                    module = getattr(module, key)
            setattr(module, keys[-1], torch.nn.Parameter(dequantized, requires_grad=False))


def dequantize_nvfp4_tensors(model):
    """Dequantize any NVFP4Tensor parameters before saving."""
    # First pass: dequantize all module parameters (weight, bias, etc.)
    for name, module in model.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, NVFP4Tensor):
                print(f"Dequantizing NVFP4Tensor in {name}.{param_name}")
                dequantized = param.dequantize(output_dtype=param.orig_dtype)
                setattr(module, param_name, torch.nn.Parameter(dequantized, requires_grad=False))

    # Second pass: check all parameters in state_dict to catch any remaining NVFP4Tensors
    state_dict = model.state_dict()
    for param_name, param in list(state_dict.items()):
        if isinstance(param, NVFP4Tensor):
            print(f"Dequantizing remaining NVFP4Tensor: {param_name}")
            dequantized = param.dequantize(output_dtype=param.orig_dtype)
            keys = param_name.split('.')
            module = model
            for key in keys[:-1]:
                if key.isdigit():
                    module = module[int(key)]
                else:
                    module = getattr(module, key)
            setattr(module, keys[-1], torch.nn.Parameter(dequantized, requires_grad=False))


def run_lm_eval(model, tokenizer, tasks: str, num_fewshot: int, batch_size: str):
    """Run lm_eval on an in-memory model."""
    import lm_eval

    print(f"\n{'=' * 60}")
    print(f"Evaluating in-memory model")
    print(f"{'=' * 60}\n")

    task_list = [t.strip() for t in tasks.split(",")]
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "tokenizer": tokenizer, "batch_size": batch_size},
        tasks=task_list,
        num_fewshot=num_fewshot,
    )

    if results and "results" in results:
        for task_name, task_results in results["results"].items():
            for metric, value in task_results.items():
                if metric not in ("alias",) and not metric.endswith("_stderr"):
                    stderr_key = f"{metric}_stderr"
                    stderr = task_results.get(stderr_key, "")
                    stderr_str = f" +/- {stderr:.4f}" if isinstance(stderr, float) else ""
                    if isinstance(value, float):
                        print(f"  {task_name:20s} {metric:15s} {value:.4f}{stderr_str}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPTQ quantization example for language models"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID to quantize",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=128,
        help="Number of calibration samples to use",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: use model's max_length)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="ultrachat200k",
        choices=["hellaswag", "ultrachat200k"],
        help="Dataset for calibration (hellaswag or ultrachat200k)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int4-gptq-sequential",
        choices=[
            "none",
            "int4-rtn",
            "int4-gptq-sequential",
            "int4-gptq-nonsequential",
            "int8-rtn",
            "int8-gptq-sequential",
            "int8-gptq-nonsequential",
            "mxfp8-rtn",
            "mxfp8-gptq-sequential",
            "mxfp8-gptq-nonsequential",
            "mxfp4-rtn",
            "mxfp4-gptq-sequential",
            "mxfp4-gptq-nonsequential",
            "nvfp4-rtn",
            "nvfp4-gptq-sequential",
            "nvfp4-gptq-nonsequential",
            "nvfp4-dynamic-rtn",
            "nvfp4-dynamic-gptq-sequential",
            "nvfp4-dynamic-gptq-nonsequential",
        ],
        help="Quantization method to use",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.5,
        help="Percentage damping for GPTQ",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization",
    )
    parser.add_argument(
        "--gptq-block-size",
        type=int,
        default=1024,
        help="Block size for GPTQ quantization",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after quantization",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        default="arc_challenge,arc_easy,hellaswag,piqa,winogrande",
        help="Comma-separated list of lm_eval tasks",
    )
    parser.add_argument(
        "--eval-num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for evaluation",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=str,
        default="auto",
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get("bfloat16", torch.bfloat16)

    print(f"Loading model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="cuda:0",
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Model config: {model.config}")

    # Determine max sequence length
    max_seq_length = args.max_sequence_length
    if max_seq_length is None:
        max_seq_length = getattr(model.config, "max_length", 2048)
        print(f"Using model's max_length: {max_seq_length}")

    # Generate output directory name from args
    model_name = args.model_id.split("/")[-1]  # Get last part of model ID
    output_dir = f"{model_name}_{args.quantization}"

    if args.quantization != "none":
        output_dir += f"_gs{args.group_size}"

    if args.quantization in [
        "int4-gptq-sequential",
        "int4-gptq-nonsequential",
        "int8-gptq-sequential",
        "int8-gptq-nonsequential",
        "mxfp8-gptq-sequential",
        "mxfp8-gptq-nonsequential",
        "mxfp4-gptq-sequential",
        "mxfp4-gptq-nonsequential",
        "nvfp4-gptq-sequential",
        "nvfp4-gptq-nonsequential",
        "nvfp4-dynamic-gptq-sequential",
        "nvfp4-dynamic-gptq-nonsequential",
    ]:
        output_dir += f"_{args.dataset_id}_n{args.num_calibration_samples}"
        output_dir += f"_damp{args.percdamp}_bs{args.gptq_block_size}"

    print(f"Output directory: {output_dir}")

    # Handle different quantization methods
    quantization_start_time = time.time()

    if args.quantization == "int4-rtn":
        print("Applying Int4 RTN (Round-To-Nearest) quantization...")
        config = Int4WeightOnlyConfig(group_size=args.group_size)
        quantize_(model, config, filter_fn=None)

    elif args.quantization == "int8-rtn":
        print("Applying Int8 RTN (Round-To-Nearest) quantization...")
        config = Int8WeightOnlyConfig(version=2, granularity=PerRow())
        quantize_(model, config, filter_fn=None)

    elif args.quantization == "mxfp8-rtn":
        print("Applying MXFP8 RTN (Round-To-Nearest) quantization...")
        config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.AUTO,
        )
        quantize_(model, config, filter_fn=None)

    elif args.quantization == "mxfp4-rtn":
        print("Applying MXFP4 RTN (Round-To-Nearest) quantization...")
        config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float4_e2m1fn_x2,
            weight_dtype=torch.float4_e2m1fn_x2,
            kernel_preference=KernelPreference.AUTO,
        )
        quantize_(model, config, filter_fn=None)

    elif args.quantization == "nvfp4-rtn":
        print("Applying NVFP4 weight-only RTN quantization...")
        config = NVFP4WeightOnlyConfig()
        quantize_(model, config, filter_fn=None)

    elif args.quantization == "nvfp4-dynamic-rtn":
        print("Applying NVFP4 dynamic activation RTN quantization...")
        config = NVFP4DynamicActivationNVFP4WeightConfig()
        quantize_(model, config, filter_fn=None)

    elif args.quantization in [
        "int4-gptq-sequential",
        "int4-gptq-nonsequential",
        "int8-gptq-sequential",
        "int8-gptq-nonsequential",
        "mxfp8-gptq-sequential",
        "mxfp8-gptq-nonsequential",
        "mxfp4-gptq-sequential",
        "mxfp4-gptq-nonsequential",
        "nvfp4-dynamic-gptq-sequential",
        "nvfp4-dynamic-gptq-nonsequential",
        "nvfp4-gptq-sequential",
        "nvfp4-gptq-nonsequential",
    ]:
        # Determine base config based on quantization type
        if "int4" in args.quantization:
            base_config = Int4WeightOnlyConfig(group_size=args.group_size)
            quant_type = "Int4"
        elif "int8" in args.quantization:
            base_config = Int8WeightOnlyConfig(granularity=PerRow(), version=2)
            quant_type = "Int8"
        elif "nvfp4-dynamic" in args.quantization:
            base_config = NVFP4DynamicActivationNVFP4WeightConfig()
            quant_type = "NVFP4-Dynamic"
        elif "nvfp4" in args.quantization:
            base_config = NVFP4WeightOnlyConfig()
            quant_type = "NVFP4"
        elif "mxfp4" in args.quantization:
            base_config = MXDynamicActivationMXWeightConfig(
                activation_dtype=torch.float4_e2m1fn_x2,
                weight_dtype=torch.float4_e2m1fn_x2,
                kernel_preference=KernelPreference.AUTO,  # Use non-emulated mode
            )
            quant_type = "MXFP4"
        else:  # mxfp8
            base_config = MXDynamicActivationMXWeightConfig(
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                kernel_preference=KernelPreference.AUTO,  # Use non-emulated mode
            )
            quant_type = "MXFP8"

        # First application: wrap weights with GPTQObserverTensor (observe step)
        print(
            f"Wrapping weights with GPTQObserverTensor for {quant_type} calibration..."
        )
        observe_config = GPTQConfig(
            step="observe",
            base_config=base_config,
            percdamp=args.percdamp,
            gptq_quantize_block_size=args.gptq_block_size,
        )
        quantize_(model, observe_config, filter_fn=None)

        # Prepare calibration dataset
        print(
            f"Preparing {args.num_calibration_samples} calibration samples from {args.dataset_id}..."
        )
        dataset = prepare_dataset(
            tokenizer,
            max_seq_length,
            args.num_calibration_samples,
            dataset_id=args.dataset_id,
            dataset_split="train_sft",
            seed=42,
        )

        # Second application: apply GPTQ quantization (convert step)
        convert_config = GPTQConfig(
            step="convert",
            base_config=base_config,
            percdamp=args.percdamp,
            gptq_quantize_block_size=args.gptq_block_size,
        )

        if "nonsequential" in args.quantization:
            print(f"Applying {quant_type} GPTQ quantization (non-sequential)...")
            # Get device for input (from embedding layer, supports device_map="auto")
            input_device = next(model.model.embed_tokens.parameters()).device

            # Run calibration
            for seq in tqdm(dataset, desc="Calibrating"):
                model(seq.to(input_device))
            # Apply quantization
            quantize_(model, convert_config, filter_fn=None)
        else:  # sequential
            print(f"Applying {quant_type} GPTQ quantization (sequential)...")
            sequential_quantize(model, dataset, convert_config)

    quantization_end_time = time.time()
    quantization_time = quantization_end_time - quantization_start_time

    if args.quantization != "none":
        print(f"\n{'=' * 60}")
        print(
            f"Quantization completed in {quantization_time:.2f} seconds ({quantization_time / 60:.2f} minutes)"
        )
        print(f"{'=' * 60}\n")

    # Optionally run evaluation on in-memory model (before saving)
    if args.evaluate:
        print("Running evaluation...")
        run_lm_eval(model, tokenizer, args.eval_tasks, args.eval_num_fewshot, args.eval_batch_size)

    # Before saving, dequantize any MX/NVFP4 tensors
    if args.quantization in ["mxfp8-rtn", "mxfp4-rtn", "mxfp8-gptq-sequential", "mxfp8-gptq-nonsequential", "mxfp4-gptq-sequential", "mxfp4-gptq-nonsequential"]:
        print("Dequantizing MX tensors before saving...")
        dequantize_mx_tensors(model)
    if args.quantization in ["nvfp4-rtn", "nvfp4-dynamic-rtn", "nvfp4-gptq-sequential", "nvfp4-gptq-nonsequential", "nvfp4-dynamic-gptq-sequential", "nvfp4-dynamic-gptq-nonsequential"]:
        print("Dequantizing NVFP4 tensors before saving...")
        dequantize_nvfp4_tensors(model)

    # Save model to generated output directory
    print(f"Saving model to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)

    print("DONE!")


if __name__ == "__main__":
    main()
