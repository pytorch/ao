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


def _unfuse_experts_model(model: torch.nn.Module) -> None:
    """Convert fused Qwen3MoeExperts to per-expert nn.Linear modules in-place.

    Transformers 5.x ``Qwen3MoeExperts`` stores expert weights as fused 3D
    ``nn.Parameter`` tensors (``gate_up_proj`` [E, 2*I, H] and ``down_proj``
    [E, H, I]).  ModelOpt 0.42.0 cannot quantize or export this format because
    it expects per-expert ``nn.Linear`` modules.

    This function replaces each ``Qwen3MoeExperts`` with an ``nn.ModuleList``
    of expert modules (each containing ``gate_proj``, ``up_proj``, ``down_proj``
    as ``nn.Linear``), wrapped in a callable module so the ``SparseMoeBlock``
    forward continues to work for calibration.

    The conversion is lossless and in-place (modifies the model, returns None).
    """
    import torch.nn as nn

    class _UnfusedExperts(nn.Module):
        """Drop-in replacement for Qwen3MoeExperts using per-expert nn.Linear.

        Implements the same forward (per-expert loop) and is iterable so
        ModelOpt's export code can enumerate expert linear layers.
        """

        def __init__(self, fused: nn.Module):
            super().__init__()
            self.num_experts = fused.num_experts
            self.act_fn = fused.act_fn
            E = fused.num_experts
            I = fused.intermediate_dim
            H = fused.hidden_dim

            for i in range(E):
                gate_w = fused.gate_up_proj.data[i, :I, :].contiguous()
                up_w = fused.gate_up_proj.data[i, I:, :].contiguous()
                down_w = fused.down_proj.data[i].contiguous()

                expert = nn.Module()
                expert.gate_proj = nn.Linear(
                    H, I, bias=False, device=gate_w.device, dtype=gate_w.dtype
                )
                expert.gate_proj.weight = nn.Parameter(gate_w)
                expert.up_proj = nn.Linear(
                    H, I, bias=False, device=up_w.device, dtype=up_w.dtype
                )
                expert.up_proj.weight = nn.Parameter(up_w)
                expert.down_proj = nn.Linear(
                    I, H, bias=False, device=down_w.device, dtype=down_w.dtype
                )
                expert.down_proj.weight = nn.Parameter(down_w)
                # Register as numbered child so state_dict keys are
                # "experts.{i}.gate_proj.weight" (no extra prefix).
                self.add_module(str(i), expert)

        def __iter__(self):
            for i in range(self.num_experts):
                yield self._modules[str(i)]

        def __len__(self):
            return self.num_experts

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0
                ).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                expert = self._modules[str(expert_idx.item())]
                gate = expert.gate_proj(current_state)
                up = expert.up_proj(current_state)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = expert.down_proj(current_hidden_states)
                current_hidden_states = (
                    current_hidden_states
                    * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )

            return final_hidden_states

    count = 0
    for module in model.modules():
        if (
            hasattr(module, "experts")
            and hasattr(module.experts, "gate_up_proj")
            and hasattr(module.experts, "num_experts")
        ):
            module.experts = _UnfusedExperts(module.experts)
            count += 1

            # ModelOpt 0.42.0's _QuantSparseMoe.forward expects top_k and
            # num_experts on the SparseMoeBlock itself, but transformers 5.x
            # stores them on the router (module.gate).  Copy them over so
            # the calibration forward doesn't crash.
            if hasattr(module, "gate") and not hasattr(module, "top_k"):
                module.top_k = module.gate.top_k
            if hasattr(module, "gate") and not hasattr(module, "num_experts"):
                module.num_experts = module.gate.num_experts

    print(f"Converted {count} fused Qwen3MoeExperts to per-expert nn.Linear")


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

    # ModelOpt 0.42.0 cannot quantize or export transformers 5.x
    # Qwen3MoeExperts (fused 3D parameter tensors).  Convert to per-expert
    # nn.Linear so ModelOpt can handle them as standard quantized linears.
    _unfuse_experts_model(model)

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

    # Fix config.json for vllm compatibility.  ModelOpt's export with the
    # unfused experts layout omits gate (router) layers from the quantization
    # ignore list and uses transformers 5.x config keys that vllm may not
    # recognise.
    import json
    from pathlib import Path

    config_path = Path(nvfp4_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_layers = config.get("num_hidden_layers", 0)
    qcfg = config.get("quantization_config", {})
    ignore = qcfg.get("ignore", [])
    for layer_idx in range(num_layers):
        gate_name = f"model.layers.{layer_idx}.mlp.gate"
        if gate_name not in ignore:
            ignore.append(gate_name)
    qcfg["ignore"] = ignore

    # vllm expects "num_experts"; transformers 5.x writes "num_local_experts".
    if "num_experts" not in config and "num_local_experts" in config:
        config["num_experts"] = config["num_local_experts"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

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
