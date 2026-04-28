"""Minimal OLMoE inference example with optional NVFP4 quantization for expert weights."""

import gc
import inspect
import subprocess
import tempfile
from contextlib import nullcontext

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer

from torchao.prototype.gptq.gptq_example import prepare_dataset
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.quantization import FqnToConfig, quantize_


def install_expert_counters(model):
    """Install forward pre-hooks on OlmoeExperts to count per-expert token routing.

    Returns a dict mapping module FQN to a (num_experts,) int64 tensor of counts,
    and a list of hook handles for removal.
    """
    expert_counts = {}
    handles = []

    for name, mod in model.named_modules():
        if not isinstance(mod, OlmoeExperts):
            continue
        counts = torch.zeros(mod.num_experts, dtype=torch.int64)
        expert_counts[name] = counts

        def make_hook(c, n):
            def hook(module, args, kwargs):
                top_k_index = args[1]
                c.add_(top_k_index.flatten().bincount(minlength=n).cpu())

            return hook

        handles.append(
            mod.register_forward_pre_hook(
                make_hook(counts, mod.num_experts), with_kwargs=True
            )
        )

    return expert_counts, handles


def print_expert_counts(expert_counts):
    print("\n=== Per-expert token counts ===")
    for name, counts in expert_counts.items():
        total = counts.sum().item()
        print(f"{name}: total={total}, per_expert={counts.tolist()}")

    print("\n=== Global expert utilization summary ===")
    all_counts = torch.cat([c for c in expert_counts.values()])
    n = len(all_counts)
    for threshold in range(129):
        if threshold % 10 != 0:
            continue
        num = int((all_counts <= threshold).sum().item())
        print(f"experts with <= {threshold} tokens: {num}/{n} ({num / n * 100:.1f}%)")


def main(
    recipe: str = "bf16",
    run_lm_eval: bool = False,
    calibrate_on_c4: bool = False,
    num_calibration_samples: int = 128,
    max_sequence_length: int = 2048,
):
    print(f"{recipe=}")
    model_id = "allenai/OLMoE-1B-7B-0924"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        experts_implementation="grouped_mm",
    )
    print(model)

    if recipe == "nvfp4":
        config = NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)
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

        # Verify quantization worked
        for name, mod in model.named_modules():
            if isinstance(mod, OlmoeExperts):
                for pname in ("gate_up_proj", "down_proj"):
                    param = getattr(mod, pname)
                    assert isinstance(param, NVFP4Tensor), (
                        f"{name}.{pname} is {type(param).__name__}, expected NVFP4Tensor"
                    )

    elif recipe == "bf16":
        pass
    else:
        raise ValueError(f"Unknown recipe: {recipe}")

    # generate() switches to batched_mm for decoding, which doesn't support
    # NVFP4Tensor (needs aten.index.Tensor). Override to keep grouped_mm.
    # TODO(future PR): implement bmm for nvfp4 and remove this workaround
    model._optimize_model_for_decode = nullcontext

    if calibrate_on_c4:
        assert recipe == "bf16", (
            "calibrate_on_c4 is only supported with recipe=bf16 for now"
        )

        expert_counts, hooks = install_expert_counters(model)

        dataset = prepare_dataset(
            tokenizer,
            max_sequence_length,
            num_calibration_samples=num_calibration_samples,
            dataset_id="c4",
            dataset_split="train",
        )
        print(f"Running calibration on {len(dataset)} C4 samples...")
        with torch.no_grad():
            for seq in dataset:
                model(seq.to("cuda"))
        print("Calibration complete.")

        print_expert_counts(expert_counts)
        for h in hooks:
            h.remove()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    print(tokenizer.decode(output[0], skip_special_tokens=True))

    if run_lm_eval:
        if recipe == "nvfp4":
            source = inspect.getsource(TorchAoHfQuantizer.get_weight_conversions)
            if "gate_up_proj" not in source:
                raise RuntimeError(
                    "Your version of `transformers` does not support NVFP4 MoE serialization. "
                    "Please install a version that includes "
                    "https://github.com/huggingface/transformers/pull/45609"
                )

        with tempfile.TemporaryDirectory() as output_dir:
            print(f"\nSaving model to {output_dir}...")

            if recipe != "bf16":
                from transformers import TorchAoConfig

                torchao_config = TorchAoConfig(quant_type=config)
                model.config.quantization_config = torchao_config
                model.hf_quantizer = TorchAoHfQuantizer(torchao_config)

            model.save_pretrained(output_dir, safe_serialization=False)
            tokenizer.save_pretrained(output_dir)

            del model
            gc.collect()
            torch.cuda.empty_cache()

            lm_eval_cmd = [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={output_dir}",
                "--tasks",
                "wikitext",
                "--num_fewshot",
                "0",
                "--batch_size",
                "16",
            ]
            print(f"Running: {' '.join(lm_eval_cmd)}")
            subprocess.run(lm_eval_cmd, check=True)


if __name__ == "__main__":
    fire.Fire(main)
