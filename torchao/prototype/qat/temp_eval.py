"""
Evaluate a model on GSM8K (zero-shot CoT) using lm_eval.

For NVFP4 eval, this script quantizes a bf16 checkpoint to NVFP4 using
ModelOpt, saves the NVFP4 checkpoint, then evaluates it via vllm.

Usage::

    # Quantize bf16 checkpoint to NVFP4 and evaluate via vllm
    python torchao/prototype/qat/temp_eval.py --checkpoint ./qwen3-30b-a3b-gsm8k-sft

    # Evaluate bf16 checkpoint via HF backend
    python torchao/prototype/qat/temp_eval.py --checkpoint ./qwen3-30b-a3b-gsm8k-sft --bf16

    # Evaluate base model in bf16
    python torchao/prototype/qat/temp_eval.py --checkpoint Qwen/Qwen3-30B-A3B --bf16

    # Evaluate only the first 100 examples (default: all)
    python torchao/prototype/qat/temp_eval.py --limit 100
"""

import argparse

import torch
from lm_eval import simple_evaluate


def quantize_to_nvfp4(
    bf16_checkpoint: str,
    nvfp4_dir: str,
    num_calib_samples: int = 128,
) -> str:
    """Load a bf16 checkpoint, quantize to NVFP4 via ModelOpt, and save.

    Returns the path to the NVFP4 checkpoint directory.
    """
    import modelopt.torch.quantization as mtq
    from datasets import load_dataset
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading bf16 checkpoint from {bf16_checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        bf16_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(bf16_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build calibration data from GSM8K train split.
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    calib_texts = [ex["question"] for ex in gsm8k.select(range(num_calib_samples))]
    calib_inputs = tokenizer(
        calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
    )

    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for i in range(0, len(calib_texts), 8):
                batch = {k: v[i:i+8].to(model.device) for k, v in calib_inputs.items()}
                model(**batch)

    print("Quantizing model to NVFP4...")
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)

    print(f"Exporting NVFP4 checkpoint to {nvfp4_dir}...")
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=nvfp4_dir)
    tokenizer.save_pretrained(nvfp4_dir)

    print(f"NVFP4 checkpoint saved to {nvfp4_dir}")

    # Free GPU memory so vllm can use it for inference.
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return nvfp4_dir


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_gsm8k_eval_nvfp4(
    model_path: str,
    limit: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Evaluate using vllm with the NVFP4 trtllm kernel (modelopt_fp4)."""
    return simple_evaluate(
        model="vllm",
        model_args={
            "pretrained": model_path,
            "quantization": "modelopt_fp4",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.65,
            "max_model_len": 2048,
        },
        tasks=["gsm8k_cot_zeroshot"],
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )


def run_gsm8k_eval_bf16(
    model_path: str,
    limit: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Evaluate using HuggingFace backend (bf16 inference)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    return simple_evaluate(
        model=lm,
        tasks=["gsm8k_cot_zeroshot"],
        num_fewshot=0,
        limit=limit,
        log_samples=False,
    )


def print_results(results: dict, label: str) -> float:
    """Print GSM8K accuracy from lm_eval results and return flexible-extract accuracy."""
    task = results["results"]["gsm8k_cot_zeroshot"]
    acc_strict = task.get("exact_match,strict-match", None)
    acc_flex = task.get("exact_match,flexible-extract", None)
    print(f"\n=== {label} ===")
    if acc_strict is not None:
        print(f"  strict-match:     {acc_strict:.2%}")
    if acc_flex is not None:
        print(f"  flexible-extract: {acc_flex:.2%}")
    return acc_flex if acc_flex is not None else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./qwen3-30b-a3b-gsm8k-sft",
        help="Path to bf16 checkpoint (default: ./qwen3-30b-a3b-gsm8k-sft).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of test examples to evaluate (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for eval (default: 4).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Evaluate in bf16 using HF backend instead of NVFP4 via vllm.",
    )
    args = parser.parse_args()

    model_path = args.checkpoint

    if args.bf16:
        label = f"checkpoint: {model_path} (bf16)"
        n = f"{args.limit} examples" if args.limit else "all examples"
        print(f"Evaluating {model_path} on GSM8K ({n}, bf16)")
        results = run_gsm8k_eval_bf16(
            model_path, limit=args.limit, batch_size=args.batch_size,
        )
    else:
        # Quantize bf16 checkpoint to NVFP4, then evaluate via vllm
        nvfp4_dir = model_path.rstrip("/") + "-nvfp4"
        nvfp4_dir = quantize_to_nvfp4(model_path, nvfp4_dir)
        label = f"checkpoint: {nvfp4_dir} (nvfp4 kernel)"
        n = f"{args.limit} examples" if args.limit else "all examples"
        print(f"Evaluating {nvfp4_dir} on GSM8K ({n}, nvfp4)")
        results = run_gsm8k_eval_nvfp4(
            nvfp4_dir, limit=args.limit, batch_size=args.batch_size,
        )

    print_results(results, label)
