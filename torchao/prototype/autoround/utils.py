# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# ==------------------------------------------------------------------------------------------==
# Utils for the auto-round
# ==------------------------------------------------------------------------------------------==
import collections
import logging
import random

import numpy as np
import torch


def _is_package_available(pkg_name, metadata_name=None):
    # Copied from Accelerate https://github.com/huggingface/accelerate
    import importlib

    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            # Some libraries have different names in the metadata
            _ = importlib.metadata.metadata(
                pkg_name if metadata_name is None else metadata_name
            )
            return True
        except importlib.metadata.PackageNotFoundError:
            return False


def is_auto_round_available() -> bool:
    return _is_package_available("auto_round")


def import_dataloader():
    if is_auto_round_available():
        import auto_round

        get_dataloader = auto_round.calib_dataset.get_dataloader
        return get_dataloader
    else:
        raise ImportError(
            (
                "This example requires the `auto-round` library."
                "Please install it with `pip install git+https://github.com/intel/auto-round.git@patch-for-ao-2`"
            )
        )


def singleton(cls):
    """Singleton decorator."""
    instances = {}

    def _singleton(*args, **kw):
        """Create a singleton object."""
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


def freeze_random(seed=0):
    random.seed(seed)

    torch.manual_seed(seed)

    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_tensor_of_type(mod, cls):
    res = 0
    for name, param in mod.named_parameters():
        if isinstance(param, cls):
            res += 1
    return res


def see_memory_usage(message: str = "", force=True):
    # Modified from DeepSpeed https://github.com/microsoft/DeepSpeed
    import gc
    import logging

    import torch.distributed as dist

    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    gc.collect()

    # Print message except when distributed but not rank 0
    logging.warning(message)
    bytes_to_gb = 1024 * 1024 * 1024
    logging.warning(
        f"AllocatedMem {round(torch.cuda.memory_allocated() / (bytes_to_gb), 2)} GB \
        MaxAllocatedMem {round(torch.cuda.max_memory_allocated() / (bytes_to_gb), 2)} GB \
        ReservedMem {round(torch.cuda.memory_reserved() / (bytes_to_gb), 2)} GB \
        MaxReservedMem {round(torch.cuda.max_memory_reserved() / (bytes_to_gb))} GB "
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def gen_text(model, tokenizer, msg="", device=None, prompt="What's AI?", max_length=20):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    model = model.to(device)
    new_tokens = model.generate(**inputs.to(device), max_length=max_length)
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    print(f"Generated text ({msg}): {text}")


def gen_example_inputs(tokenizer, device, max_length=20):
    inputs = tokenizer(
        "What's AI?", return_tensors="pt", padding="max_length", max_length=max_length
    )
    input_ids = inputs["input_ids"].to(device)
    return (input_ids,)


def _auto_detect_decoder_cls(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            first_module = module[0]
            return type(first_module)


def get_float_model_info(model_name_or_path, torch_dtype=torch.float32):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    decoder_cls = _auto_detect_decoder_cls(model)
    logging.warning(f"Detected decoder class: {decoder_cls}")
    if decoder_cls is None:
        raise ValueError(
            "Cannot detect the decoder class from the model, please provide it manually."
        )
    return model, tokenizer, decoder_cls


execution_records = collections.defaultdict(list)


def dump_elapsed_time(customized_msg="", record=False):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """
    import logging
    import time

    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            dur = round((end - start) * 1000, 2)
            if record:
                execution_records[func.__qualname__].append(dur)
            logging.warning(
                "%s elapsed time: %s ms"
                % (
                    customized_msg if customized_msg else func.__qualname__,
                    dur,
                )
            )
            if record:
                avg_time = sum(execution_records[func.__qualname__]) / len(
                    execution_records[func.__qualname__]
                )
                std_time = np.std(execution_records[func.__qualname__])
                logging.warning(
                    f"For {func.__qualname__}, the average elapsed time: {avg_time: .2f} ms, the std: {std_time: .2f} ms"
                )
            return res

        return fi

    return f
