# ==------------------------------------------------------------------------------------------==
# Utils
# ==------------------------------------------------------------------------------------------==
from typing import Optional, Callable, Any, List, Tuple, Dict

import random
import os
import torch
import numpy as np


def freeze_random():
    seed = 0

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)


def assert_same(
    a: Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
    b: Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
):
    assert len(a) == len(b), f"len: {len(a)} != {len(b)}"
    for i, _ in enumerate(a):
        assert type(a[i]) == type(b[i]), f"type: {type(a[i])} != {type(b[i])}"
        if isinstance(a[i], torch.Tensor):
            torch.testing.assert_allclose(a[i], b[i])
        elif isinstance(a[i], tuple):
            assert_same(a[i], b[i])
        elif isinstance(a[i], dict):
            for k in a[i].keys():
                assert k in b[i], f"key: {k} not in {b[i]}"
                assert_same(a[i][k], b[i].get(k))
        elif a[i] is None:
            assert b[i] is None
        else:
            raise ValueError(f"Unsupported type: {type(a[i])}")
    print("Same!")


def inspect_module_inputs(inputs, indent=""):
    if isinstance(inputs, torch.Tensor):
        print(f"{indent}Tensor: {inputs.shape}")
    elif isinstance(inputs, tuple) or isinstance(inputs, list):
        for i in inputs:
            inspect_module_inputs(i, indent + "  ")
    elif isinstance(inputs, dict):
        for k, v in inputs.items():
            print(f"{indent}{k}:")
            inspect_module_inputs(v, indent + "  ")
    elif inputs is None:
        print(f"{indent}None")
    else:
        print(f"{indent}{type(inputs)}")


def get_tokenizer_function(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length of
    seqlen to the "text" field of examples.
    """

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
