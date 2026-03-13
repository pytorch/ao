# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Create a quantized `meta-llama/Meta-Llama-3.1-8B-Instruct` model and save
it to disk for local benchmarking with `vllm`.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
)


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a model with TorchAO")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the quantized model",
    )
    return parser.parse_args()


def main(args):
    """
    Args:
        args: Parsed command line arguments containing:
            output_dir: Directory to save the quantized model
            max_new_tokens: Max tokens to generate for testing
            convert_llama_4_expert_weights_to_mnk: if True, converts LLaMa 4 Scout expert weights from MKN to MNK memory layout
            no_save_model_to_disk: if True, skips saving quantized model to local disk
            no_load_model_from_disk: if True, skips reloading model from disk to test it again
    """

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_map = "auto"
    max_new_tokens = 20

    # Test prompts
    prompts = [
        "Why is Pytorch 2.0 the best machine learning compiler?",
    ]

    # Set seed before creating the model
    set_seed(42)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get quantization config
    # quantization_config = TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())
    quantization_config = TorchAoConfig(
        MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
        )
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and quantize model
    print("Loading and quantizing model...")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map=device_map,
        quantization_config=quantization_config,
    )
    print(quantized_model)

    if False:
        # Test generation
        print("\nTesting quantized model generation...")
        input_ids = tokenizer(prompts, return_tensors="pt", padding=False).to(
            quantized_model.device
        )
        outputs = quantized_model.generate(**input_ids, max_new_tokens=max_new_tokens)

        for i, (prompt, output) in enumerate(zip(prompts, outputs, strict=False)):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Save quantized model
    print(f"\nSaving quantized model to: {output_dir}")
    quantized_model.save_pretrained(
        output_dir,
        safe_serialization=False,
    )
    tokenizer.save_pretrained(output_dir)

    if False:
        # Load saved model to verify
        # TODO: do we really need `weights_only=False` here?
        loaded_model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            device_map=device_map,
            torch_dtype="auto",
            weights_only=False,
        )

        # Test loaded model with first prompt
        test_prompt = prompts[0]
        input_ids = tokenizer(test_prompt, return_tensors="pt").to(loaded_model.device)
        output = loaded_model.generate(**input_ids, max_new_tokens=args.max_new_tokens)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(
            f"Verification - Prompt: {test_prompt!r}, Generated text: {generated_text!r}"
        )

    print("\nQuantization process completed successfully.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
