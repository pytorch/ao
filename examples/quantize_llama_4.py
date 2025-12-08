# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
A script demonstrating quantization of the routed experts of
the `meta-llama/Llama-4-Scout-17B-16E-Instruct` model from HuggingFace
to w8a8 with float8 rowwise weights and activations.
"""

import argparse
import random
from pathlib import Path

import fbgemm_gpu
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    FqnToConfig,
    PerRow,
)
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    Float8Tensor,
)


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_quantization_config():
    expert_3d_weight_single_config = Float8DynamicActivationFloat8WeightConfig(
        # the weights of this model are stored in (B, K, N) layout, and we
        # need to quantize rowwise across the K axis, which is `PerRow(1)`.
        granularity=[PerRow(), PerRow(1)],
        # guard against activations with groups of all-zeroes
        activation_value_lb=1.0e-12,
    )
    fqn_to_config = FqnToConfig(
        {
            # only quantize the routed experts, the rest of the model is left
            # in high precision
            r"re:.*\.feed_forward\.experts\.gate_up_proj": expert_3d_weight_single_config,
            r"re:.*\.feed_forward\.experts\.down_proj": expert_3d_weight_single_config,
        }
    )
    return TorchAoConfig(quant_type=fqn_to_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a model with TorchAO")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the quantized model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max tokens to generate for testing (default: 64)",
    )
    parser.add_argument(
        "--convert_llama_4_expert_weights_to_mnk",
        action="store_true",
        help="If set, converts LLaMa 4 Scout expert weights from MKN to MNK memory layout",
    )
    parser.add_argument(
        "--no_save_model_to_disk",
        action="store_true",
        help="If set, skips saving quantized model to local disk",
    )
    parser.add_argument(
        "--no_load_model_from_disk",
        action="store_true",
        help="If set, skips reloading model from disk to test it again",
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

    # ensure relevant dependency versions are satisfied
    t_v = str(transformers.__version__)
    assert t_v >= "4.58", (
        f"transformers version {t_v} too old, please upgrade to a transformers version with https://github.com/huggingface/transformers/pull/41894"
    )
    f_v = str(fbgemm_gpu.__version__)
    if f_v.startswith("202"):
        # nightly version, such as '2025.11.22+cu128'
        assert f_v >= "2025.11.22", (
            f"fbgemm_gpu nightly version  {f_v} too old, please upgrade to a nightly from 2025-11-22 or later"
        )
    else:
        # stable version, such as '1.4.1'
        assert f_v >= "1.5", (
            f"fbgemm_gpu stable version  {f_v} too old, please upgrade to 1.5 or later"
        )

    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    device_map = "auto"

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
    quantization_config = get_quantization_config()

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

    # Test generation
    print("\nTesting quantized model generation...")
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(
        quantized_model.device
    )
    outputs = quantized_model.generate(**input_ids, max_new_tokens=args.max_new_tokens)

    for i, (prompt, output) in enumerate(zip(prompts, outputs, strict=False)):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    save_model_to_disk = not args.no_save_model_to_disk
    load_model_from_disk = not args.no_load_model_from_disk

    if save_model_to_disk:
        # Save quantized model
        print(f"\nSaving quantized model to: {output_dir}")

        if args.convert_llama_4_expert_weights_to_mnk:
            print("\nConverting LLaMa 4 expert weights from MKN to MNK layout")

            # source: https://github.com/huggingface/transformers/blob/6f6095e0cf509f7384d3ce0c1804013ef6cafd5f/src/transformers/modeling_utils.py#L3466
            def save_function(shard, filename):
                # `save_pretrained` default logic calls tensor.contiguous() before
                # saving, so if we do mkn -> mnk before saving it will be
                # converted back to mkn.
                # We undo this in the custom save_function, which runs after
                # the contiguous call in `save_pretrained`.:)
                for k, v in shard.items():
                    # hacky check for LLaMa 4 experts
                    if isinstance(v, Float8Tensor) and len(v.shape) == 3:
                        v.qdata = (
                            v.qdata.transpose(-2, -1).contiguous().transpose(-2, -1)
                        )
                torch.save(shard, filename)

        else:
            save_function = torch.save

        quantized_model.save_pretrained(
            output_dir,
            safe_serialization=False,
            save_function=save_function,
        )
        tokenizer.save_pretrained(output_dir)

    if load_model_from_disk:
        assert save_model_to_disk, "unimplemented"
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
