import torch
import torch.nn as nn

from torchao.quantization import quantize_

import copy
import random

from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

from naive_intNwo import intN_weight_only
from transformers import AutoModelForCausalLM, AutoTokenizer

from ax.service.ax_client import AxClient, ObjectiveProperties


# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "/home/hanxianhuang/ao/torchao/quantization/prototype/mixed_precision/checkpoints/meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to(
    device=device
)


# quantize a model based on given configuration
def quantize_by_fqn_to_config(model, fqn_to_config):
    for fqn, config in fqn_to_config.items():
        bit_width, groupsize = config.split("_")
        bit_width = int(bit_width)
        groupsize = int(groupsize)

        def filter_fn_sen(child: torch.nn.Module, cur_fqn: str) -> bool:
            return isinstance(child, torch.nn.Linear) and (fqn in cur_fqn)

        quantize_(
            model.to(device=device),
            intN_weight_only(n=bit_width, group_size=groupsize),
            filter_fn_sen,
        )

# calculate perplexity, need to support more tasks
def cal_ppl(model):
    with torch.no_grad():

        result = evaluate(
            HFLM(pretrained=model.cuda(), tokenizer=tokenizer, batch_size=1),
            get_task_dict("wikitext"),
        )

    return result["results"]["wikitext"]["word_perplexity,none"]


# calculate model size
def cal_model_size(model, fqn_to_config):
    _sum = 0
    fqn_cofg_dict = dict()

    for fqn, config in fqn_to_config:
        bit_width, groupsize = config.split("_")
        bit_width = int(bit_width)
        groupsize = int(groupsize)
        bit_zeropoint = bit_width
        bit_scale = bit_width
        fqn_cofg_dict[fqn] = (bit_width, groupsize, bit_zeropoint, bit_scale)

    for name, parameter in model.named_parameters():
        flag = 0
        for fqn in fqn_cofg_dict:
            if fqn in name:
                flag = 1
                if "self_attn" in name or "mlp" in name:
                    # print(name, fqn_cofg_dict[fqn][0])
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


# return evaluation results
def eval(model, fqn_to_config):
    return {
        "cal_PPL": (cal_ppl(model), 0.0),
        "model_size": (cal_model_size(model, fqn_to_config), 0.0),
    }


# define the search space
parameter_choices_list = []
for i in [2, 3, 4]:
    for j in [32, 64]:
        parameter_choices_list.append(str(i) + "_" + str(j))

for i in [5, 6, 8]:
    for j in [32, 64, 128, 256]:
        parameter_choices_list.append(str(i) + "_" + str(j))
        
#initialize ax_client
ax_client = AxClient()

# initialize the search space
parameters_list = []
for i in range(3, 30):  # skip the first3 and last2
    parameters_list.append(
        {
            "name": f".{i}.",
            "type": "choice",
            "value_type": "str",
            "values": parameter_choices_list,
            "is_ordered": False,
            "sort_values": False,
        }
    )



# add initial search points based on the sensitivity score, need to automate this part
initial_points_set = []

for _ in range(50):
    initial_points = []

    # auto sample the bit choices with random choice probability positive correlated to FIT score
    for i in range(3, 18):
        if i in [5,6,7,10,11,12,16]:
            initial_points.append(
                ("." + str(i) + ".", random.choices(['5_32','5_64','4_32','4_64'], [0,0,50,50])[0])
            )
        else:
            initial_points.append(
                ("." + str(i) + ".", random.choices(['5_32','5_64','4_32','4_64'], [5,5,45,45])[0])
            )

    for i in range(18, 30):
        if i in [22,23,24]:
            initial_points.append(
                ("." + str(i) + ".", random.choices(['5_32','5_64','5_128','4_32','4_64','3_32','3_64','2_32'], [0,0,0,20,20,30,20,10])[0])
            )
        else:
            initial_points.append(
                ("." + str(i) + ".", random.choices(['5_32','5_64','5_128','4_32','4_64','3_32','3_64','2_32'], [5,5,5,30,30,10,10,5])[0])
            )

    initial_points_set.append(initial_points)


# create experiment
ax_client.create_experiment(
    parameters=parameters_list,
    name="test_quantize_BO",
    objectives={"cal_PPL": ObjectiveProperties(minimize=True)},
    choose_generation_strategy_kwargs={
        "num_initialization_trials": 20,
    },
    outcome_constraints=["model_size <= 4.0"],
)



# add 50 initial points into the BO trials
history=[]

trial_id = 0
for i in range(len(initial_points_set)):

    ax_client.attach_trial(parameters=dict(initial_points_set[i]))

    initial_points_set[i].append((".0.", "5_32"))
    initial_points_set[i].append((".1.", "5_32"))
    initial_points_set[i].append((".2.", "5_32"))
    initial_points_set[i].append((".30.", "5_32"))
    initial_points_set[i].append((".31.", "5_32"))

    m = copy.deepcopy(model).to(device=device)
    quantize_by_fqn_to_config(m, dict(initial_points_set[i]))

    eval_results = eval(m, initial_points_set[i])
    
    print("------------")
    print(trial_id, initial_points_set[i], eval_results)

    history.append((eval_results, initial_points_set[i]))
    ax_client.complete_trial(
        trial_index=trial_id,
        raw_data=eval_results,
    )
    trial_id += 1
    del m
    torch.cuda.empty_cache()


# other search trials
for k_ in range(150):
    parameters, trial_idx = ax_client.get_next_trial()

    parameter_tuple = []
    for k, v in parameters.items():
        parameter_tuple.append((k, v))
    parameter_tuple.append((".0.", "5_32"))
    parameter_tuple.append((".1.", "5_32"))
    parameter_tuple.append((".2.", "5_32"))
    parameter_tuple.append((".30.", "5_32"))
    parameter_tuple.append((".31.", "5_32"))

    m = copy.deepcopy(model).to(device=device)

    quantize_by_fqn_to_config(m, dict(parameter_tuple))

    eval_results = eval(m, parameter_tuple)
    
    print("------------")
    print(trial_idx, parameter_tuple, eval_results)
    history.append((eval_results, parameter_tuple))
    
    ax_client.complete_trial(
        trial_index=trial_idx,
        raw_data=eval_results,
    )

    del m
    torch.cuda.empty_cache()


print("------Finish BO------")
for h in history:
    print(h)

print("------Best config------")
best_parameters, values = ax_client.get_best_parameters()
print(best_parameters, values)

print("------history------")
print(ax_client.generation_strategy.trials_as_df)
