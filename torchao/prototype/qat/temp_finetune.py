"""
SFT on GSM8K (grade-school math) with Qwen3-30B-A3B.

Fine-tunes on chain-of-thought GSM8K training examples and saves a bf16
checkpoint. Use ``temp_eval.py`` to evaluate (bf16 or NVFP4).

Usage::

    # Standard SFT
    python torchao/prototype/qat/temp_finetune.py
    #   bf16: ./qwen3-30b-a3b-gsm8k-sft

    # SFT with NVFP4 QAT
    python torchao/prototype/qat/temp_finetune.py --qat
    #   bf16: ./qwen3-30b-a3b-gsm8k-sft-qat
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
BASE_OUTPUT_DIR = "./qwen3-30b-a3b-gsm8k-sft"


def format_gsm8k(example: dict) -> dict:
    """Convert a GSM8K example into the ``messages`` format expected by SFTTrainer."""
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT on GSM8K")
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
        help="Output directory for the checkpoint (default: BASE_OUTPUT_DIR[-qat]).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (BASE_OUTPUT_DIR + ("-qat" if args.qat else ""))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.qat:
        from torchao.prototype.qat.nvfp4_moe import apply_nvfp4_moe_qat
        model = apply_nvfp4_moe_qat(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_steps > 0:
        # Load GSM8K
        gsm8k = load_dataset("openai/gsm8k", "main")
        train_dataset = gsm8k["train"].map(format_gsm8k)

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
