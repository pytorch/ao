# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import json
from pathlib import Path

from transformers import AutoModel


def create_weight_map(checkpoint_dir: Path):
    """
    This function, create_weight_map, generates a mapping of a model's weights to a file (pytorch_model.bin)
    and saves this mapping, along with the model's total size, to a JSON file (pytorch_model.bin.index.json).
    The model is loaded from a pre-trained model specified by model_name.
    This weight map is used by the HF conversion script (convert_hf_checkpoint.py).
    """
    # Load the model
    model_name = checkpoint_dir.parent.name + "/" + checkpoint_dir.name
    print(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Get the state dict
    state_dict = model.state_dict()
    # Create the weight map
    weight_map = {}
    for key, tensor in state_dict.items():
        # In this example, we're assuming all weights are in a single file
        # You may need to adjust this if your model uses sharded weights
        weight_map[key] = "pytorch_model.bin"
    # Create the index dictionary
    index_dict = {
        "metadata": {
            "total_size": sum(
                param.numel() * param.element_size() for param in model.parameters()
            )
        },
        "weight_map": weight_map,
    }
    # Save the index dictionary to a JSON file
    with open(f"{checkpoint_dir}/pytorch_model.bin.index.json", "w") as f:
        json.dump(index_dict, f, indent=2)
    print("Created pytorch_model.bin.index.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create weight map for hf model")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/Xenova/llama2.c-stories15M"),
    )

    args = parser.parse_args()
    create_weight_map(args.checkpoint_dir)
