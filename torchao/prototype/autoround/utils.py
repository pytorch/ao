# ==------------------------------------------------------------------------------------------==
# Utils for the auto-round (put here temporarily)
# ==------------------------------------------------------------------------------------------==
import random

import auto_round
import logging
import numpy as np
import torch

get_dataloader = auto_round.calib_dataset.get_dataloader


def freeze_random(seed=0):
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)


def count_tensor_of_type(mod, cls):
    res = 0
    for name, param in mod.named_parameters():
        if isinstance(param, cls):
            res += 1
    return res


def see_memory_usage(message, force=True):
    # Modified from DeepSpeed
    import gc
    import logging

    import torch.distributed as dist

    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    gc.collect()

    # Print message except when distributed but not rank 0
    logging.info(message)
    logging.info(
        f"AllocatedMem {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        MaxAllocatedMem {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        ReservedMem {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        MaxReservedMem {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB "
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()


def move_data_to_device(input, device=torch.device("cpu")):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, dict):
        for inp in input.keys():
            input[inp] = move_data_to_device(input[inp], device)
    elif isinstance(input, (list, tuple)):
        input = [move_data_to_device(inp, device) for inp in input]
    return input


@torch.no_grad()
def gen_text(
    model, tokenizer, msg="", device="cuda", prompt="What's AI?", max_length=20
):
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


def auto_detect_decoder_cls(model):
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
    if "Llama" in model_name_or_path:
        decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    elif "opt" in model_name_or_path:
        decoder_cls = transformers.models.opt.modeling_opt.OPTDecoderLayer
    else:
        decoder_cls = auto_detect_decoder_cls(model)
        logging.warning(f"auto detect decoder_cls: {decoder_cls}")
        if decoder_cls is None:
            raise ValueError(f"Unsupported model: {model_name_or_path}")
    return model, tokenizer, decoder_cls


def dump_elapsed_time(customized_msg=""):
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
            logging.warning(
                "%s elapsed time: %s ms"
                % (
                    customized_msg if customized_msg else func.__qualname__,
                    round((end - start) * 1000, 2),
                )
            )
            return res

        return fi

    return f
