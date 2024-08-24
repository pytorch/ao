"""
This file produces a file named pytorch_model.bin.index.json based on the downloaded model weights. 
It was primarily used to create run evals on llama2.c-stories15M model.
"""
import json
import torch
from transformers import AutoModel

def create_weight_map(model_name):
    # Load the model
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
        "metadata": {"total_size": sum(param.numel() * param.element_size() for param in model.parameters())},
        "weight_map": weight_map
    }

    # Save the index dictionary to a JSON file
    with open("pytorch_model.bin.index.json", "w") as f:
        json.dump(index_dict, f, indent=2)

    print("Created pytorch_model.bin.index.json")

# Usage
model_name = "checkpoints/Xenova/llama2.c-stories15M"
create_weight_map(model_name)