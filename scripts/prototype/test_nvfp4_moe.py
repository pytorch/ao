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

from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.quantization import FqnToConfig, quantize_


def main(recipe: str = "bf16", run_lm_eval: bool = False):
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

        # generate() switches to batched_mm for decoding, which doesn't support
        # NVFP4Tensor (needs aten.index.Tensor). Override to keep grouped_mm.
        # TODO(future PR): implement bmm for nvfp4 and remove this workaround
        model._optimize_model_for_decode = nullcontext
    elif recipe == "bf16":
        pass
    else:
        raise ValueError(f"Unknown recipe: {recipe}")

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
