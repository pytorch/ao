# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import gc
import subprocess
import time
from contextlib import nullcontext
from typing import Any, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer

from torchao.prototype.gptq import GPTQConfig
from torchao.prototype.gptq.observer import GPTQObserverTensor
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.quantization import (
    FqnToConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerRow

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
        "c4": ("allenai/c4", "en"),
    }

    dataset_entry = dataset_map.get(dataset_id, dataset_id)
    if isinstance(dataset_entry, tuple):
        hf_dataset_id, hf_config_name = dataset_entry
    else:
        hf_dataset_id, hf_config_name = dataset_entry, None

    # Load dataset and preprocess
    train_dataset_raw = load_dataset(
        hf_dataset_id, hf_config_name, split=dataset_split, streaming=True
    )
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
        choices=["hellaswag", "ultrachat200k", "c4"],
        help="Dataset for calibration (hellaswag or ultrachat200k)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train_sft",
        help="Dataset split to use for calibration",
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
            "nvfp4-rtn",
            "nvfp4-gptq-sequential",
            "nvfp4-gptq-nonsequential",
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
        "--lm-eval-tasks",
        type=str,
        default="leaderboard_bbh",
        help="Comma-separated tasks for lm_eval",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=3,
        help="Number of few-shot examples for lm_eval (0 to disable)",
    )
    parser.add_argument(
        "--lm-eval-batch-size",
        type=str,
        default="auto",
        help="Batch size for lm_eval (default: auto)",
    )
    parser.add_argument(
        "--lm-eval-limit",
        type=int,
        default=None,
        help="Limit number of examples per task for lm_eval (default: no limit)",
    )
    parser.add_argument(
        "--output-dir-prefix",
        type=str,
        required=True,
        help="Prefix for the output directory (e.g. /home/user/tmp/20260420)",
    )
    parser.add_argument(
        "--skip-lm-eval",
        action="store_true",
        default=False,
        help="Skip running lm_eval after quantization, useful for quickly iterating on lm_eval arguments",
    )
    parser.add_argument(
        "--o-proj-only",
        action="store_true",
        default=False,
        help="Only quantize `o_proj` layers, useful for faster GPTQ runs for debugging",
    )
    return parser.parse_args()


OLMOE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"


def _verify_olmoe_experts_quantized(model):
    """Assert every OlmoeExperts module has NVFP4Tensor for both expert weights."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    found = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, OlmoeExperts):
            continue
        for pname in ("gate_up_proj", "down_proj"):
            param = getattr(mod, pname)
            assert isinstance(param, NVFP4Tensor), (
                f"{name}.{pname} is {type(param).__name__}, expected NVFP4Tensor"
            )
        found += 1
    assert found > 0, "no OlmoeExperts modules found to verify"
    print(f"Verified NVFP4 quantization on {found} OlmoeExperts modules")


def main():
    args = parse_args()

    is_olmoe = args.model_id == OLMOE_MODEL_ID
    if is_olmoe and args.quantization not in (
        "none",
        "nvfp4-rtn",
        "nvfp4-gptq-nonsequential",
    ):
        raise ValueError(
            f"model {args.model_id} only supports 'none', 'nvfp4-rtn', or "
            f"'nvfp4-gptq-nonsequential', got '{args.quantization}'"
        )

    # lm_eval batch_size="auto" with nvfp4 gptq causes the error in
    # MSLK nvfp4 triton kernel, likely an unsupported shape:
    # https://gist.github.com/vkuzo/b71ca46365dee017d1602e9638d91603
    # TODO(future): debug and fix this. For now, the workaround is
    # for the user to manually specify lm_eval batch_size.
    if "nvfp4" in args.quantization:
        assert args.lm_eval_batch_size != "auto", "unsupported"

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get("bfloat16", torch.bfloat16)

    print(f"Loading model {args.model_id}...")
    from_pretrained_kwargs = dict(device_map="cuda:0", dtype=dtype)
    if is_olmoe:
        from_pretrained_kwargs["experts_implementation"] = "grouped_mm"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, **from_pretrained_kwargs
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
    output_dir = f"{args.output_dir_prefix}_{model_name}_{args.quantization}"

    if args.quantization != "none":
        output_dir += f"_gs{args.group_size}"

    if args.quantization in [
        "int4-gptq-sequential",
        "int4-gptq-nonsequential",
        "int8-gptq-sequential",
        "int8-gptq-nonsequential",
        "nvfp4-gptq-sequential",
        "nvfp4-gptq-nonsequential",
    ]:
        output_dir += f"_{args.dataset_id}_n{args.num_calibration_samples}"
        output_dir += f"_damp{args.percdamp}_bs{args.gptq_block_size}"

    print(f"Output directory: {output_dir}")

    # Handle different quantization methods
    quantization_start_time = time.time()

    def skip_lm_head(module, fqn):
        return isinstance(module, torch.nn.Linear) and "lm_head" not in fqn

    def skip_lm_head_o_proj(module, fqn):
        return (
            isinstance(module, torch.nn.Linear)
            and "lm_head" not in fqn
            and "o_proj" in fqn
        )

    filter_fn_to_use = skip_lm_head
    if args.o_proj_only:
        filter_fn_to_use = skip_lm_head_o_proj

    if args.quantization == "int4-rtn":
        print("Applying Int4 RTN (Round-To-Nearest) quantization...")
        config = Int4WeightOnlyConfig(group_size=args.group_size)
        quantize_(model, config, filter_fn=filter_fn_to_use)

    elif args.quantization == "int8-rtn":
        print("Applying Int8 RTN (Round-To-Nearest) quantization...")
        config = Int8WeightOnlyConfig(version=2, granularity=PerRow())
        quantize_(model, config, filter_fn=filter_fn_to_use)

    elif args.quantization == "nvfp4-rtn":
        print("Applying NVFP4 RTN (Round-To-Nearest) quantization...")

        config = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        )
        if is_olmoe:
            quantize_(
                model,
                FqnToConfig(
                    {
                        r"re:.*\.experts\.gate_up_proj": config,
                        r"re:.*\.experts\.down_proj": config,
                    }
                ),
                filter_fn=None,
            )
            _verify_olmoe_experts_quantized(model)
        else:
            quantize_(model, config, filter_fn=filter_fn_to_use)
        print(model)

    elif args.quantization in [
        "int4-gptq-sequential",
        "int4-gptq-nonsequential",
        "int8-gptq-sequential",
        "int8-gptq-nonsequential",
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
        else:  # nvfp4
            base_config = NVFP4DynamicActivationNVFP4WeightConfig(
                use_dynamic_per_tensor_scale=True,
                use_triton_kernel=True,
            )
            quant_type = "NVFP4"

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
        if is_olmoe:
            quantize_(
                model,
                FqnToConfig(
                    {
                        r"re:.*\.experts\.gate_up_proj": observe_config,
                        r"re:.*\.experts\.down_proj": observe_config,
                    }
                ),
                filter_fn=None,
            )
        else:
            quantize_(model, observe_config, filter_fn=filter_fn_to_use)
        print(model)

        # Prepare calibration dataset
        print(
            f"Preparing {args.num_calibration_samples} calibration samples from {args.dataset_id}..."
        )
        dataset = prepare_dataset(
            tokenizer,
            max_seq_length,
            args.num_calibration_samples,
            dataset_id=args.dataset_id,
            dataset_split=args.dataset_split,
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
            # Print total # of GPTQ modules
            num_gptq_weights = 0
            for name, param in model.named_parameters():
                if isinstance(param, GPTQObserverTensor):
                    num_gptq_weights += 1
            print(f"Total GPTQ weights to convert: {num_gptq_weights}")
            # Apply quantization
            if is_olmoe:
                quantize_(
                    model,
                    FqnToConfig(
                        {
                            r"re:.*\.experts\.gate_up_proj": convert_config,
                            r"re:.*\.experts\.down_proj": convert_config,
                        }
                    ),
                    filter_fn=None,
                )
                _verify_olmoe_experts_quantized(model)
            else:
                quantize_(model, convert_config, filter_fn=filter_fn_to_use)
        else:  # sequential
            print(f"Applying {quant_type} GPTQ quantization (sequential)...")
            assert filter_fn_to_use == skip_lm_head, "unsupported"
            sequential_quantize(model, dataset, convert_config)

    if is_olmoe:
        # generate() switches to batched_mm for decoding, which doesn't support
        # NVFP4Tensor (needs aten.index.Tensor). Override to keep grouped_mm.
        # TODO(future): remove when NVFP4 MoE supports bmm-style decode
        model._optimize_model_for_decode = nullcontext

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
    print(model)

    if "nvfp4" in args.quantization:
        import inspect

        source = inspect.getsource(TorchAoHfQuantizer.get_weight_conversions)
        if "per_tensor_scale" not in source:
            raise RuntimeError(
                "Your version of `transformers` does not support NVFP4 serialization. "
                "Please install a version that includes "
                "https://github.com/huggingface/transformers/pull/45573"
            )
        if is_olmoe and "gate_up_proj" not in source:
            raise RuntimeError(
                "Your version of `transformers` does not support NVFP4 MoE serialization. "
                "Please install a version that includes "
                "https://github.com/huggingface/transformers/pull/45609"
            )

    if args.quantization != "none":
        # Attach hf_quantizer so save_pretrained uses the flatten path for tensor
        # subclasses (e.g. NVFP4Tensor) that don't have a valid storage pointer.
        ao_config = base_config if "gptq" in args.quantization else config
        torchao_config = TorchAoConfig(quant_type=ao_config)
        model.config.quantization_config = torchao_config
        model.hf_quantizer = TorchAoHfQuantizer(torchao_config)

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
        args.lm_eval_tasks,
        "--batch_size",
        args.lm_eval_batch_size,
    ]

    if args.num_fewshot > 0:
        lm_eval_cmd += ["--num_fewshot", str(args.num_fewshot)]
    if args.lm_eval_limit is not None:
        lm_eval_cmd += ["--limit", str(args.lm_eval_limit)]

    print(f"Running command: {' '.join(lm_eval_cmd)}")
    if args.skip_lm_eval:
        print("Terminating early due to skip_lm_eval=True")
        return
    try:
        subprocess.run(lm_eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"lm_eval failed with error: {e}")
    except FileNotFoundError:
        print("lm_eval not found. Please install it with: pip install lm-eval")


if __name__ == "__main__":
    main()
