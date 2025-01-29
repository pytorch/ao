import csv
import json

import torch
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from naive_intNwo import intN_weight_only
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.quantization import quantize_


def write_history_to_csv(history, output_file, keyword):
    # keyword example: ['cal_PPL', 'cal_throughput', 'config']

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(keyword)

        for eval_results, config in history:
            obj1 = eval_results[keyword[0]][0]
            obj2 = eval_results[keyword[1]][0]

            writer.writerow([obj1, obj2, config])


# quantize a model based on a given quantization configuration
def quantize_by_fqn_to_config(model, device, fqn_to_config):
    it = iter(fqn_to_config.items())
    while True:
        try:
            k1, v1 = next(it)
            k2, v2 = next(it)
            fqn = k1[8:]
            bit_width, groupsize = v1, v2

            def filter_fn_sen(child: torch.nn.Module, cur_fqn: str) -> bool:
                return isinstance(child, torch.nn.Linear) and (fqn in cur_fqn)

            quantize_(
                model.to(device=device),
                intN_weight_only(n=bit_width, group_size=groupsize),
                filter_fn_sen,
            )
        except StopIteration:
            break


# calculate perplexity on wikitext-document, need to support more tasks
def cal_wikitext_ppl(model, tokenizer, limit=62):
    with torch.no_grad():
        result = evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1),
            get_task_dict("wikitext"),
            limit=limit,
        )

    return result["results"]["wikitext"]["word_perplexity,none"]


# TODO: make it generalize to more models
def cal_model_size(model, fqn_to_config):
    _sum = 0
    fqn_cofg_dict = dict()

    it = iter(fqn_to_config.items())
    while True:
        try:
            k1, v1 = next(it)
            k2, v2 = next(it)
            bit_width, groupsize = v1, v2
            bit_zeropoint = 32
            bit_scale = 8
            fqn = k1[8:]
            fqn_cofg_dict[fqn] = (bit_width, groupsize, bit_zeropoint, bit_scale)
        except StopIteration:
            break

    for name, parameter in model.named_parameters():
        flag = 0
        for fqn in fqn_cofg_dict:
            if fqn in name:
                flag = 1
                if "self_attn" in name or "mlp" in name:
                    _sum += parameter.numel() * fqn_cofg_dict[fqn][
                        0
                    ] + parameter.numel() // fqn_cofg_dict[fqn][1] * (
                        fqn_cofg_dict[fqn][2] + fqn_cofg_dict[fqn][3]
                    )
        if flag == 0:
            _sum += parameter.numel() * 16

    _sum_in_byte = _sum / 8.0
    _sum_in_GB = _sum_in_byte / (1024**3) / 1.0
    return _sum_in_GB


def load_model(repo_id, device):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16
    ).to(device=device)
    return model, tokenizer


def load_parameters_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)

    bitwidth_config = next(
        param for param in config["parameters"] if param["name"] == "bitwidth"
    )
    groupsize_config = next(
        param for param in config["parameters"] if param["name"] == "groupsize"
    )

    parameters_list = []

    # Ensure that we are interleaving bitwidth and groupsize for each layer
    for bw_layer, gs_layer in zip(
        bitwidth_config["layers"], groupsize_config["layers"]
    ):
        start, end = bw_layer["range"]
        for i in range(start, end):
            # Add bitwidth parameter
            bitwidth_param = {
                "name": bitwidth_config["name_format"].format(i=i),
                "type": bw_layer["type"],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": True,
            }
            if bw_layer["type"] == "fixed":
                bitwidth_param["value"] = bw_layer["value"]
            elif bw_layer["type"] == "choice":
                bitwidth_param["values"] = bw_layer["values"]
            parameters_list.append(bitwidth_param)

            # Add groupsize parameter
            groupsize_param = {
                "name": groupsize_config["name_format"].format(i=i),
                "type": gs_layer["type"],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": True,
            }
            if gs_layer["type"] == "fixed":
                groupsize_param["value"] = gs_layer["value"]
            elif gs_layer["type"] == "choice":
                groupsize_param["values"] = gs_layer["values"]
            parameters_list.append(groupsize_param)

    return parameters_list


def load_initial_samples(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config["initial_samples"]
