# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import gc
import subprocess
import time
from typing import Any, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.prototype.gptq import GPTQConfig
from torchao.quantization import quantize_


def sequential_quantize(
    model,
    calibration_data: List[torch.Tensor],
    config: Any,
) -> None:
    # run with no grad otherwise keeping all the tensors around for the backwards will cause oom
    with torch.no_grad():
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
    dataset_id: str = "hellaswag",
    dataset_split: str = "train",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPTQ quantization example for language models"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID to quantize",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=5000,
        help="Number of calibration samples to use",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=8192,
        help="Maximum sequence length (default: use model's max_length)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="hellaswag",
        choices=["hellaswag", "ultrachat200k"],
        help="Dataset for calibration (hellaswag or ultrachat200k)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int4-gptq-sequential",
        choices=["none", "int4-rtn", "int4-gptq-sequential", "int4-gptq-nonsequential"],
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

    if args.quantization in ["int4-gptq-sequential", "int4-gptq-nonsequential"]:
        output_dir += f"_{args.dataset_id}_n{args.num_calibration_samples}"
        output_dir += f"_damp{args.percdamp}_bs{args.gptq_block_size}"

    print(f"Output directory: {output_dir}")

    # Handle different quantization methods
    quantization_start_time = time.time()

    if args.quantization == "int4-rtn":
        print("Applying Int4 RTN (Round-To-Nearest) quantization...")
        from torchao.quantization import Int4WeightOnlyConfig

        config = Int4WeightOnlyConfig(group_size=args.group_size)
        quantize_(model, config, filter_fn=None)

    elif args.quantization in ["int4-gptq-sequential", "int4-gptq-nonsequential"]:
        # First application: wrap weights with ObserverTensor (observe step)
        print("Wrapping weights with ObserverTensor for calibration...")
        observe_config = GPTQConfig(
            step="observe",
            group_size=args.group_size,
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
            dataset_split="train",
            seed=42,
        )

        # Second application: apply GPTQ quantization (convert step)
        convert_config = GPTQConfig(
            step="convert",
            group_size=args.group_size,
            percdamp=args.percdamp,
            gptq_quantize_block_size=args.gptq_block_size,
        )

        if args.quantization == "int4-gptq-sequential":
            print("Applying GPTQ quantization (sequential)...")
            sequential_quantize(model, dataset, convert_config)
        else:  # int4-gptq-nonsequential
            print("Applying GPTQ quantization (non-sequential)...")
            # Get device for input (from embedding layer, supports device_map="auto")
            input_device = next(model.model.embed_tokens.parameters()).device

            # Run calibration
            for seq in tqdm(dataset, desc="Calibrating"):
                model(seq.to(input_device))
            # Apply quantization
            quantize_(model, convert_config, filter_fn=None)

    quantization_end_time = time.time()
    quantization_time = quantization_end_time - quantization_start_time

    if args.quantization != "none":
        print(f"\n{'=' * 60}")
        print(
            f"Quantization completed in {quantization_time:.2f} seconds ({quantization_time / 60:.2f} minutes)"
        )
        print(f"{'=' * 60}\n")

    # Save model to generated output directory
    print(f"Saving model to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)

    print("DONE!")

    # Clear GPU memory before running lm_eval
    print("\nClearing GPU memory...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleared.")

    # Run lm_eval on the saved model
    print(f"\n{'=' * 60}")
    print("Running lm_eval on the quantized model...")
    print(f"{'=' * 60}\n")

    lm_eval_cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={output_dir}",
        "--tasks",
        "hellaswag",
        "--batch_size",
        "auto",
    ]

    print(f"Running command: {' '.join(lm_eval_cmd)}")
    try:
        subprocess.run(lm_eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"lm_eval failed with error: {e}")
    except FileNotFoundError:
        print("lm_eval not found. Please install it with: pip install lm-eval")


if __name__ == "__main__":
    main()
