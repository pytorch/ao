# ==------------------------------------------------------------------------------------------==
# Utils for the auto-round (put here temporarily)
# ==------------------------------------------------------------------------------------------==
from typing import Optional, Tuple
import random
import torch
import numpy as np
from collections import UserDict


def freeze_random(seed=0):
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)


def get_tokenizer_function(tokenizer, seqlen):
    def default_tokenizer_function(examples):
        example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
        return example

    return default_tokenizer_function


def get_dataloader(tokenizer, seqlen=1024, dataset_name="NeelNanda/pile-10k", split="train", seed=42, batch_size=4):
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        for text in batch:
            input_ids = text["input_ids"]
            if input_ids.shape[0] < seqlen:
                continue
            input_ids = input_ids[:seqlen]
            input_ids_list = input_ids.tolist()
            if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                continue
            input_ids_new.append(input_ids)
        # TODO: need to handle the case where all input_ids are empty
        if len(input_ids_new) == 0:
            return None
        tmp = torch.vstack(input_ids_new)
        res = {"input_ids": tmp}
        return res

    calib_dataset = load_dataset(dataset_name, split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)
    calib_dataset.set_format(type="torch", columns=["input_ids"])
    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader


def move_input_to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = move_input_to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res = []
        for inp in input:
            input_res.append(move_input_to_device(inp, device))
        input = input_res
    else:
        raise ValueError(f"Unsupported type: {type(input)}")
    return input


@torch.no_grad()
def gen_text(model, tokenizer, msg=""):
    device = next(model.parameters()).device
    inputs = tokenizer("What are we having for dinner?", return_tensors="pt")
    new_tokens = model.generate(**inputs.to(device), max_length=20)
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    print(f"Generated text ({msg}): {text}")


def get_float_model_info(model_name_or_path):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    if "Llama" in model_name_or_path:
        decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    elif "opt" in model_name_or_path:
        decoder_cls = transformers.models.opt.modeling_opt.OPTDecoderLayer
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    return model, tokenizer, decoder_cls


def get_example_inputs(tokenizer):
    iters = 4
    prompt = "What are we having for dinner?"
    example_inputs = tokenizer(prompt, return_tensors="pt")
    for i in range(iters):
        yield example_inputs


def check_package(package_name: str):
    try:
        __import__(package_name)
        return True
    except ImportError:
        print(f"Package {package_name} not found.")
        return False
