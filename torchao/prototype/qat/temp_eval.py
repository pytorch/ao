"""
Evaluate a model on configurable tasks using lm_eval.

For NVFP4 eval, this script quantizes a bf16 checkpoint to NVFP4 using
ModelOpt, saves the NVFP4 checkpoint, then evaluates it via vllm.

Usage::

    # Quantize bf16 checkpoint to NVFP4 and evaluate via vllm
    python torchao/prototype/qat/temp_eval.py --checkpoint ./qwen3-30b-a3b-gsm8k-sft

    # Evaluate bf16 checkpoint via HF backend
    python torchao/prototype/qat/temp_eval.py --checkpoint ./qwen3-30b-a3b-gsm8k-sft --bf16

    # Evaluate on ARC-Challenge instead of GSM8K
    python torchao/prototype/qat/temp_eval.py --task arc_challenge --checkpoint ./qwen3-30b-a3b-arc-challenge-sft

    # Evaluate only the first 100 examples (default: all)
    python torchao/prototype/qat/temp_eval.py --limit 100
"""

import argparse

import torch
from lm_eval import simple_evaluate

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "gsm8k": {
        "eval_task": "gsm8k_cot_zeroshot",
        "calib_dataset": ("openai/gsm8k", "main", "train"),
        "calib_text_key": "question",
        "default_checkpoint": "./qwen3-30b-a3b-gsm8k-sft",
    },
    "arc_challenge": {
        "eval_task": "arc_challenge",
        "calib_dataset": ("allenai/ai2_arc", "ARC-Challenge", "train"),
        "calib_text_key": "question",
        "default_checkpoint": "./qwen3-30b-a3b-arc-challenge-sft",
    },
}


def quantize_to_nvfp4(
    bf16_checkpoint: str,
    nvfp4_dir: str,
    task_name: str = "gsm8k",
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

    # Build calibration data from training split of the task.
    task_cfg = TASKS[task_name]
    ds_name, ds_config, ds_split = task_cfg["calib_dataset"]
    ds = load_dataset(ds_name, ds_config, split=ds_split)
    calib_texts = [
        ex[task_cfg["calib_text_key"]] for ex in ds.select(range(num_calib_samples))
    ]
    calib_inputs = tokenizer(
        calib_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for i in range(0, len(calib_texts), 8):
                batch = {
                    k: v[i : i + 8].to(model.device) for k, v in calib_inputs.items()
                }
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


def run_eval_nvfp4(
    model_path: str,
    eval_task: str,
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
        tasks=[eval_task],
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )


def run_eval_bf16(
    model_path: str,
    eval_task: str,
    limit: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Evaluate using HuggingFace backend (bf16 inference)."""
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        tasks=[eval_task],
        num_fewshot=0,
        limit=limit,
        log_samples=False,
    )


def print_results(results: dict, eval_task: str, label: str) -> None:
    """Print accuracy metrics from lm_eval results."""
    task_results = results["results"][eval_task]
    print(f"\n=== {label} ===")
    for metric, value in sorted(task_results.items()):
        if isinstance(value, (int, float)) and not metric.endswith("_stderr"):
            print(f"  {metric}: {value:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on configurable tasks")
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        choices=list(TASKS.keys()),
        help=f"Eval task (default: gsm8k). Choices: {list(TASKS.keys())}.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to bf16 checkpoint.",
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

    task_cfg = TASKS[args.task]
    eval_task = task_cfg["eval_task"]
    model_path = args.checkpoint or task_cfg["default_checkpoint"]

    n = f"{args.limit} examples" if args.limit else "all examples"

    if args.bf16:
        label = f"checkpoint: {model_path} (bf16)"
        print(f"Evaluating {model_path} on {eval_task} ({n}, bf16)")
        results = run_eval_bf16(
            model_path,
            eval_task,
            limit=args.limit,
            batch_size=args.batch_size,
        )
    else:
        # Quantize bf16 checkpoint to NVFP4, then evaluate via vllm
        nvfp4_dir = model_path.rstrip("/") + "-nvfp4"
        nvfp4_dir = quantize_to_nvfp4(model_path, nvfp4_dir, task_name=args.task)
        label = f"checkpoint: {nvfp4_dir} (nvfp4 kernel)"
        print(f"Evaluating {nvfp4_dir} on {eval_task} ({n}, nvfp4)")
        results = run_eval_nvfp4(
            nvfp4_dir,
            eval_task,
            limit=args.limit,
            batch_size=args.batch_size,
        )

    print_results(results, eval_task, label)
