"""
SFT with Qwen3-30B-A3B on configurable tasks.

Fine-tunes on training examples and saves a bf16 checkpoint.
Use ``temp_eval.py`` to evaluate (bf16 or NVFP4).

Usage::

    # SFT on GSM8K (default)
    python torchao/prototype/qat/temp_finetune.py
    python torchao/prototype/qat/temp_finetune.py --qat

    # SFT on ARC-Challenge
    python torchao/prototype/qat/temp_finetune.py --task arc_challenge
    python torchao/prototype/qat/temp_finetune.py --task arc_challenge --qat
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-30B-A3B"


# ---------------------------------------------------------------------------
# Dataset formatters (each returns a list of {"messages": [...]} dicts)
# ---------------------------------------------------------------------------

def format_gsm8k(example: dict) -> dict:
    """Convert a GSM8K example into the ``messages`` format expected by SFTTrainer."""
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def format_arc_challenge(example: dict) -> dict:
    """Convert an ARC-Challenge example into the ``messages`` format."""
    choices = example["choices"]
    choices_str = "\n".join(
        f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])
    )
    question = f"{example['question']}\n\n{choices_str}"
    answer = example["answerKey"]
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "gsm8k": {
        "dataset": ("openai/gsm8k", "main"),
        "formatter": format_gsm8k,
        "default_output_dir": "./qwen3-30b-a3b-gsm8k-sft",
    },
    "arc_challenge": {
        "dataset": ("allenai/ai2_arc", "ARC-Challenge"),
        "formatter": format_arc_challenge,
        "default_output_dir": "./qwen3-30b-a3b-arc-challenge-sft",
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT on configurable tasks")
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        choices=list(TASKS.keys()),
        help=f"Training task (default: gsm8k). Choices: {list(TASKS.keys())}.",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Apply NVFP4 QAT to MoE expert layers during training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100). Set to 0 to skip training and only save bf16 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the checkpoint.",
    )
    args = parser.parse_args()

    task_cfg = TASKS[args.task]
    output_dir = args.output_dir or (task_cfg["default_output_dir"] + ("-qat" if args.qat else ""))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        experts_implementation="grouped_mm",
    )

    if args.qat:
        from torchao.prototype.qat.nvfp4_moe import apply_nvfp4_moe_qat
        model = apply_nvfp4_moe_qat(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_steps > 0:
        ds_name, ds_config = task_cfg["dataset"]
        ds = load_dataset(ds_name, ds_config)
        train_dataset = ds["train"].map(task_cfg["formatter"])

        training_args = SFTConfig(
            output_dir=output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            max_length=1024,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        if args.qat:
            from torchao.prototype.qat.nvfp4_moe import remove_nvfp4_moe_qat
            remove_nvfp4_moe_qat(trainer.model)

        # Save bf16 checkpoint
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nbf16 checkpoint saved to {output_dir}")
    else:
        print("\nSkipping training (--max-steps 0)")

        if args.qat:
            from torchao.prototype.qat.nvfp4_moe import remove_nvfp4_moe_qat
            remove_nvfp4_moe_qat(model)

        # Still save the bf16 checkpoint (e.g. from base model)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nbf16 checkpoint saved to {output_dir}")
