"""
SFT with Qwen3-30B-A3B on configurable tasks.

Fine-tunes on training examples and saves a bf16 checkpoint.
Use ``temp_eval.py`` to evaluate (bf16 or NVFP4).

Usage::

    # SFT on GSM8K (default)
    python torchao/prototype/qat/temp_finetune.py
    python torchao/prototype/qat/temp_finetune.py --qat
    python torchao/prototype/qat/temp_finetune.py --qat --qat-impl reference_module_swap

    # SFT on ARC-Challenge
    python torchao/prototype/qat/temp_finetune.py --task arc_challenge
    python torchao/prototype/qat/temp_finetune.py --task arc_challenge --qat

    # Tailpatch QAT: resume from SFT checkpoint, apply QAT for 50 steps
    python torchao/prototype/qat/temp_finetune.py --resume-from ./sft-checkpoint --qat-impl simple --max-steps 50
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
    """ARC-Challenge in completion format matching lm_eval's eval prompt:
    ``Question: {question}\\nAnswer: {answer_text}``"""
    choices = example["choices"]
    answer_idx = choices["label"].index(example["answerKey"])
    answer_text = choices["text"][answer_idx]
    return {
        "text": f"Question: {example['question']}\nAnswer: {answer_text}"
    }


# def format_arc_challenge_chat(example: dict) -> dict:
#     """Convert an ARC-Challenge example into the ``messages`` format.
#     Answer is just the letter (A, B, C, D)."""
#     choices = example["choices"]
#     choices_str = "\n".join(
#         f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])
#     )
#     question = f"{example['question']}\n\n{choices_str}"
#     answer = example["answerKey"]
#     return {
#         "messages": [
#             {"role": "user", "content": question},
#             {"role": "assistant", "content": answer},
#         ]
#     }


def format_open_platypus(example: dict) -> dict:
    """Open-Platypus: instruction + optional input -> output."""
    content = example["instruction"]
    if example["input"]:
        content += f"\n\n{example['input']}"
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def format_alpaca(example: dict) -> dict:
    """Alpaca: instruction + optional input -> output."""
    content = example["instruction"]
    if example["input"]:
        content += f"\n\n{example['input']}"
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def format_slimorca(example: dict) -> dict:
    """SlimOrca: conversations list with system/human/gpt roles.
    Drops system messages to avoid SFTTrainer tokenization issues."""
    role_map = {"human": "user", "gpt": "assistant"}
    messages = []
    for msg in example["conversations"]:
        if msg["from"] == "system":
            continue
        messages.append({
            "role": role_map[msg["from"]],
            "content": msg["value"],
        })
    return {"messages": messages}


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
    "open_platypus": {
        "dataset": ("garage-bAInd/Open-Platypus", None),
        "formatter": format_open_platypus,
        "default_output_dir": "./qwen3-30b-a3b-platypus-sft",
    },
    "alpaca": {
        "dataset": ("tatsu-lab/alpaca", None),
        "formatter": format_alpaca,
        "default_output_dir": "./qwen3-30b-a3b-alpaca-sft",
    },
    "slimorca": {
        "dataset": ("Open-Orca/SlimOrca", None),
        "formatter": format_slimorca,
        "default_output_dir": "./qwen3-30b-a3b-slimorca-sft",
    },
}


def _get_qat_fns(qat_impl):
    """Return (apply_fn, remove_fn) for the given QAT implementation."""
    if qat_impl == "reference_subclass":
        from torchao.prototype.qat.nvfp4_moe import (
            apply_nvfp4_moe_qat,
            remove_nvfp4_moe_qat,
        )

        return apply_nvfp4_moe_qat, remove_nvfp4_moe_qat
    elif qat_impl == "simple_subclass":
        from torchao.prototype.qat.nvfp4_moe_simple import (
            apply_simple_fp4_moe_qat,
            remove_simple_fp4_moe_qat,
        )

        return apply_simple_fp4_moe_qat, remove_simple_fp4_moe_qat
    else:
        from torchao.prototype.qat.nvfp4_moe_module_swap import (
            apply_nvfp4_moe_qat,
            remove_nvfp4_moe_qat,
        )

        return apply_nvfp4_moe_qat, remove_nvfp4_moe_qat


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
        "--qat-impl",
        type=str,
        default=None,
        choices=["reference_subclass", "reference_module_swap", "simple_subclass"],
        help="QAT implementation (default: reference_subclass). Implies --qat. "
        "reference_subclass intercepts torch._grouped_mm via a tensor subclass "
        "matching the flashinfer kernel's two-level NvFP4 scaling; "
        "simple intercepts torch._grouped_mm with a simpler per-tensor FP4 "
        "fake quantize (no two-level scaling or per-expert quantization); "
        "reference_module_swap replaces the SparseMoeBlock with a custom module "
        "(currently only supports Qwen3 MoE).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Load model from this checkpoint instead of the base pretrained model. "
        "Useful for tailpatch QAT: first SFT without QAT, then resume with QAT.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100). Set to 0 to skip training and only save bf16 checkpoint.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device train batch size (default: 16).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Number of warmup steps (default: 20).",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="LR scheduler type (default: cosine).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the checkpoint.",
    )
    args = parser.parse_args()

    # --qat-impl implies --qat; default to reference_subclass when --qat is used
    if args.qat_impl is not None:
        args.qat = True

    if args.qat and args.qat_impl is None:
        args.qat_impl = "reference_subclass"

    task_cfg = TASKS[args.task]
    output_dir = args.output_dir or (
        task_cfg["default_output_dir"] + ("-qat" if args.qat else "")
    )

    model_path = args.resume_from or MODEL_NAME
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        experts_implementation="grouped_mm",
    )

    apply_qat, remove_qat = None, None
    if args.qat:
        apply_qat, remove_qat = _get_qat_fns(args.qat_impl)
        model = apply_qat(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_steps > 0:
        ds_name, ds_config = task_cfg["dataset"]
        ds = load_dataset(ds_name, ds_config)
        train_dataset = ds["train"].map(task_cfg["formatter"])

        training_args = SFTConfig(
            output_dir=output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            max_grad_norm=1.0,
            lr_scheduler_type=args.lr_scheduler,
            warmup_steps=args.warmup_steps,
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
        model = trainer.model
    else:
        print("\nSkipping training (--max-steps 0)")

    if args.qat:
        remove_qat(model)

    # Save bf16 checkpoint
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nbf16 checkpoint saved to {output_dir}")
