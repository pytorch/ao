import sys
 
import torch
import torch.nn as nn
from torchao.quantization import quantize_
import random

from naive_intNwo import intN_weight_only

import copy
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from ax.service.ax_client import AxClient, ObjectiveProperties
import torch.multiprocessing as mp
from ax.modelbridge.cross_validation import cross_validate


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
            limit=limit
        )

    return result["results"]["wikitext"]["word_perplexity,none"]

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

# return evaluation results to complete BO trials
def eval(model, tokenizer, limit, fqn_to_config):
    return {
        "cal_PPL": (cal_wikitext_ppl(model, tokenizer, limit), 0.0),
        "model_size": (cal_model_size(model, fqn_to_config), 0.0),
    }

def load_model(repo_id, device):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to(
        device=device
    )
    return model, tokenizer

def define_parameter_list():

    # define the search space for all layers
    parameters_list = []

    for i in range(0, 3):  
        parameters_list.append(
            {
                "name": f"bitwidth.{i}.",
                "type": "fixed",
                "value_type": "int",
                "value": 5,
                "is_ordered": True,
                "sort_values": True,
            }
        )

        parameters_list.append(
            {
                "name": f"groupsize.{i}.",
                "type": "fixed",
                "value_type": "int",
                "value": 32,
                "is_ordered": True,
                "sort_values": True,
            }
        )

    for i in range(3, 30):  
        parameters_list.append(
            {
                "name": f"bitwidth.{i}.",
                "type": "choice",
                "value_type": "int",
                "values": [2,3,4,5,6,8],
                "is_ordered": True,
                "sort_values": True,
            }
        )

        parameters_list.append(
            {
                "name": f"groupsize.{i}.",
                "type": "choice",
                "value_type": "int",
                "values": [32, 64, 128, 256],
                "is_ordered": True,
                "sort_values": True,
            }
        )

    for i in range(30, 32):  
        parameters_list.append(
            {
                "name": f"bitwidth.{i}.",
                "type": "fixed",
                "value_type": "int",
                "value": 5,
                "is_ordered": True,
                "sort_values": True,
            }
        )
        parameters_list.append(
            {
                "name": f"groupsize.{i}.",
                "type": "fixed",
                "value_type": "int",
                "value": 32,
                "is_ordered": True,
                "sort_values": True,
            }
        )

    return parameters_list

# add initial search points based on the sensitivity score, need to automate this part
def get_initial_samples(num_initial=50):
    initial_points_set = []

    # auto sample the bit choices with random choice probability positive correlated to FIT score
    for _ in range(num_initial):
        initial_points = {}
        for i in range(0, 3):
            initial_points["bitwidth." + str(i) + "."] = 5
            initial_points["groupsize." + str(i) + "."] = 32

        for i in range(3, 18):
            if i in [5,6,7,10,11,12,16]:
                initial_points["bitwidth." + str(i) + "."] = random.choices([5, 4], [20, 80])[0]
                initial_points["groupsize." + str(i) + "."] = random.choices([32, 64], [30, 70])[0]
            else:
                initial_points["bitwidth." + str(i) + "."] = random.choices([5, 4], [30, 70])[0]
                initial_points["groupsize." + str(i) + "."] = random.choices([32, 64], [40, 60])[0]

        for i in range(18, 30):
            if i in [22,23,24]:
                initial_points["bitwidth." + str(i) + "."] = random.choices([5, 4, 3, 2], [20, 55, 20, 5])[0]
                initial_points["groupsize." + str(i) + "."] = random.choices([32, 64, 128, 256], [30, 40, 25, 5])[0]
            else:
                initial_points["bitwidth." + str(i) + "."] = random.choices([5, 4, 3, 2], [30, 55, 10, 5])[0]
                initial_points["groupsize." + str(i) + "."] = random.choices([32, 64, 128, 256], [40, 40, 15, 5])[0]
            
        for i in range(30, 32):
            initial_points["bitwidth." + str(i) + "."] = 5
            initial_points["groupsize." + str(i) + "."] = 32

        initial_points_set.append(initial_points)
    return initial_points_set

def run_sequential_BO(device, checkpoint, limit, num_initial, num_trials, model_size_constraint):

    parameters_list = define_parameter_list()
    initial_points_set = get_initial_samples(num_initial)

    #initialize ax_client
    constraint="model_size <= "+str(model_size_constraint)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters_list,
        name="test_quantize_BO",
        objectives={"cal_PPL": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_initial, # the number of trials to build generation strategy
        },
        outcome_constraints=[constraint],
    )

    history=[]
    trial_id = 0

    # add initial points into the BO trials
    for i in range(num_initial):

        ax_client.attach_trial(parameters=initial_points_set[i])

        m, tokenizer = load_model(checkpoint, device)
        quantize_by_fqn_to_config(m, device, initial_points_set[i])

        eval_results = eval(m, tokenizer, limit, initial_points_set[i])
        
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


    # run new BO trials
    for k_ in range(num_trials):
        parameters, trial_idx = ax_client.get_next_trial()
        
        m, tokenizer = load_model(checkpoint, device)

        quantize_by_fqn_to_config(m, device, parameters)

        eval_results = eval(m, tokenizer, limit, parameters)
        
        print("------------")
        print(trial_idx, parameters, eval_results)
        history.append((eval_results, parameters))
        
        ax_client.complete_trial(
            trial_index=trial_idx,
            raw_data=eval_results,
        )

        del m
        torch.cuda.empty_cache()

    stg_model = ax_client.generation_strategy.model
    cv_results = cross_validate(stg_model)
    print("----cv results----")
    print(cv_results)

    print("------Finish BO------")
    for h in history:
        print(h)

    print("------Best config------")
    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters, values)

    print("------history------")
    print(ax_client.generation_strategy.trials_as_df)

# Worker functio to perform BO trials on a specific GPU
def eval_in_parallel(gpu_id, checkpoint, limit, config, return_dict, proc_id, trial_id):

    model, tokenizer = load_model(checkpoint, f'cuda:{gpu_id}')
    parameters_list = define_parameter_list()

    print(f"Process {proc_id} on GPU {gpu_id} starts!")

    quantize_by_fqn_to_config(model=model, device=f'cuda:{gpu_id}', fqn_to_config=dict(config))

    eval_results = eval(model, tokenizer, limit, config)
        
    return_dict[proc_id] = (trial_id, config, eval_results)
    
    del model
    torch.cuda.empty_cache()


def run_parallel_BO(device, checkpoint, limit, num_initial, num_trials, model_size_constraint, gpu_list):

    parameters_list = define_parameter_list()
    initial_points_set = get_initial_samples(num_initial)

    #initialize ax_client
    constraint="model_size <= "+str(model_size_constraint)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters_list,
        name="test_quantize_BO",
        objectives={"cal_PPL": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_initial, # the number of trials to build generation strategy
        },
        outcome_constraints=[constraint],
    )

    gpu_list = [int(i) for i in gpu_list.split(",")]

    history=[]
    trial_id = 0

    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method("spawn", force=True)

    # add initial points into the BO trials
    for id in range(num_initial//len(gpu_list)):
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        # Start the worker processes
        for i, gpu_id in enumerate(gpu_list):
            ax_client.attach_trial(parameters=dict(initial_points_set[id*len(gpu_list)+i]))
            p = mp.Process(target=eval_in_parallel, args=(gpu_id, checkpoint, limit, initial_points_set[id*len(gpu_list)+i], return_dict, i, trial_id))
            trial_id += 1
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Print the results after all processes have finished
        print(return_dict)
        for i in range(len(gpu_list)):
            current_trial_id, config, eval_results = return_dict[i]
            history.append((eval_results, config))
            ax_client.complete_trial(trial_index=current_trial_id,raw_data=eval_results,)

    # run new BO trials
    for id in range(num_trials//len(gpu_list)):
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        # Start the worker processes
        for i, gpu_id in enumerate(gpu_list):
            parameters, trial_idx = ax_client.get_next_trial()
            parameter_tuple = []
            for k, v in parameters.items():
                parameter_tuple.append((k, v))            
            p = mp.Process(target=eval_in_parallel, args=(gpu_id, checkpoint, limit, parameter_tuple, return_dict, i, trial_idx))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Print the results after all processes have finished
        print(return_dict)
        for i in range(len(gpu_list)):
            current_trial_id, config, eval_results = return_dict[i]
            history.append((eval_results, config))
            ax_client.complete_trial(trial_index=current_trial_id,raw_data=eval_results,)

    print("------Finish BO------")
    for h in history:
        print(h)
    print("------Best config------")
    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters, values)

    print("------history------")
    print(ax_client.generation_strategy.trials_as_df)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for evaluation')
    parser.add_argument('--checkpoint', type=str, default="/home/hanxianhuang/ao/torchao/quantization/prototype/mixed_precision/checkpoints/meta-llama/Meta-Llama-3-8B", help='Path to load model')
    parser.add_argument('--limit', type=int, default=None, help='Number of eval samples to evaluate')
    parser.add_argument('--num_initial', type=int, default=50, help='Number of initial points sampled by sensitivity scores')
    parser.add_argument('--num_trials', type=int, default=150, help='Number of trials to run BO')
    parser.add_argument('--model_size_constraint', type=float, default=6.0, help='The model size constraint for BO')
    parser.add_argument('--multi_gpus', action='store_true', help="Use multi-processing to run evaluation on multi-gpus")
    parser.add_argument('--gpu_list', type=str, default="", help="A list of gpus to run evaluation, separated by comma, e.g., --gpu_lists=0,1,2,3")
    args = parser.parse_args()

    if not args.multi_gpus:
        run_sequential_BO(device=args.device, checkpoint=args.checkpoint, limit=args.limit, num_initial=args.num_initial, num_trials=args.num_trials, model_size_constraint=args.model_size_constraint)
    else:
        run_parallel_BO(device=args.device, checkpoint=args.checkpoint, limit=args.limit, num_initial=args.num_initial, num_trials=args.num_trials, model_size_constraint=args.model_size_constraint, gpu_list=args.gpu_list)
