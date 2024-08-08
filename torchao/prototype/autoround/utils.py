# ==------------------------------------------------------------------------------------------==
# Utils for the auto-round (put here temporarily)
# ==------------------------------------------------------------------------------------------==
import random

import auto_round

import numpy as np
import torch

get_dataloader = auto_round.calib_dataset.get_dataloader



def freeze_random(seed=0):
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)


def has_tensor_of_type(mod, cls):
    for name, param in mod.named_parameters():
        if isinstance(param, cls):
            return True
    return False


def move_data_to_device(input, device=torch.device("cpu")):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict):
        for inp in input.keys():
            input[inp] = move_data_to_device(input[inp], device)
    elif isinstance(input, (list, tuple)):
        input = [move_data_to_device(inp, device) for inp in input]
    return input


@torch.no_grad()
def gen_text(
    model, tokenizer, msg="", device="cuda", prompt="What's AI?", max_length=20
):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    new_tokens = model.generate(**inputs.to(device), max_length=max_length)
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    print(f"Generated text ({msg}): {text}")


def get_float_model_info(model_name_or_path):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    if "Llama" in model_name_or_path:
        decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    elif "opt" in model_name_or_path:
        decoder_cls = transformers.models.opt.modeling_opt.OPTDecoderLayer
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    return model, tokenizer, decoder_cls
