import sys
 
import torch
import torch.nn as nn
from torchao.quantization import quantize_
import random
sys.path.append("/home/hanxianhuang/ao/torchao/quantization/prototype/mixed_precision")
from my_naive_intNwo import intN_weight_only

from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from ax.service.ax_client import AxClient, ObjectiveProperties
import torch.multiprocessing as mp
from ax.modelbridge.cross_validation import cross_validate

import os
import warnings

import torch.utils.data
import torchvision
import utils

from torchvision.transforms.functional import InterpolationMode
from torchvision import models

def define_parameter_list():
    # define the search space for all layers
    parameters_list = []

    parameters_list.append(
        {
            "name": f"bitwidthlayer_{0}.",
            "type": "fixed",
            "value_type": "int",
            "value": 8,
            "is_ordered": True,
            "sort_values": True,
        }
    )
    parameters_list.append(
        {
            "name": f"groupsizelayer_{0}.",
            "type": "fixed",
            "value_type": "int",
            "value": 32,
            "is_ordered": True,
            "sort_values": True,
        }
    )

    for i in range(1, 10):  
        if i==2 or i==3:
            parameters_list.append(
                {
                    "name": f"bitwidthlayer_{i}.",
                    "type": "choice",
                    "value_type": "int",
                    "values": [5,6,8],
                    "is_ordered": True,
                    "sort_values": True,
                }
            )

            parameters_list.append(
                {
                    "name": f"groupsizelayer_{i}.",
                    "type": "choice",
                    "value_type": "int",
                    "values": [32, 64],
                    "is_ordered": True,
                    "sort_values": True,
                }
            )
        else:
            parameters_list.append(
                {
                    "name": f"bitwidthlayer_{i}.",
                    "type": "choice",
                    "value_type": "int",
                    "values": [2,3,4,5,6,8],
                    "is_ordered": True,
                    "sort_values": True,
                }
            )

            parameters_list.append(
                {
                    "name": f"groupsizelayer_{i}.",
                    "type": "choice",
                    "value_type": "int",
                    "values": [32, 64, 128, 256],
                    "is_ordered": True,
                    "sort_values": True,
                }
            )

    for i in range(10, 12):  
        parameters_list.append(
            {
                "name": f"bitwidthlayer_{i}.",
                "type": "choice",
                "value_type": "int",
                "values": [5,6,8],
                "is_ordered": True,
                "sort_values": True,
            }
        )

        parameters_list.append(
            {
                "name": f"groupsizelayer_{i}.",
                "type": "choice",
                "value_type": "int",
                "values": [32, 64],
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
        initial_points["bitwidthlayer_" + str(0) + "."] = 8
        initial_points["groupsizelayer_" + str(0) + "."] = 32

        for i in range(1, 10):
            if i in [2,3]:
                initial_points["bitwidthlayer_" + str(i) + "."] = random.choices([5, 6], [80, 20])[0]
                initial_points["groupsizelayer_" + str(i) + "."] = random.choices([32, 64], [50, 50])[0]
            else:
                initial_points["bitwidthlayer_" + str(i) + "."] = random.choices([6, 5, 4], [30, 40, 30])[0]
                initial_points["groupsizelayer_" + str(i) + "."] = random.choices([32, 64, 128], [40, 40, 20])[0]

        for i in range(10, 12):
            initial_points["bitwidthlayer_" + str(i) + "."] = 8
            initial_points["groupsizelayer_" + str(i) + "."] = 32

        initial_points_set.append(initial_points)
    for i in initial_points_set:
        print(i)
    return initial_points_set

# return evaluation results to complete BO trials
def eval(model, limit, fqn_to_config, criterion, data_loader, device, args):
    return {
        'cal_acc': (utils.cal_acc(model, criterion, data_loader, device, print_freq=100, log_suffix="", args=None), 0.0),
        "model_size": (utils.cal_model_size(model, fqn_to_config), 0.0),
    }

def run_sequential_BO(args):

    # TODO: add default parameter list if not specified
    parameters_list = utils.load_parameters_from_json(args.parameters_list)
    initial_points_set = utils.load_initial_samples(args.initial_samples)
    num_BO_initial_samples =len(initial_points_set)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_test, test_sampler = utils.load_imagenet_data(args)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    num_classes = len(dataset_test.classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    #initialize ax_client
    constraint="model_size <= "+str(args.model_size_constraint)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters_list,
        name="test_quantize_BO",
        objectives={"cal_acc": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_BO_initial_samples, # the number of trials to build generation strategy
        },
        outcome_constraints=[constraint],
    )

    history=[]
    trial_id = 0

    # add initial points into the BO trials
    for i in range(num_BO_initial_samples):

        ax_client.attach_trial(parameters=initial_points_set[i])
        
        m = utils.load_vit_model(args.model_name, args.weights_name)#models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m = m.to(torch.bfloat16)
        m.to(device)
            
        utils.quantize_by_fqn_to_config(m, device, initial_points_set[i])

        eval_results = eval(m, args.limit, initial_points_set[i], criterion, data_loader_test, device, args)
        
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
    for k_ in range(args.num_trials):
        parameters, trial_idx = ax_client.get_next_trial()

        m = utils.load_vit_model(args.model_name, args.weights_name)#models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m = m.to(torch.bfloat16)
        m.to(device)
        
        utils.quantize_by_fqn_to_config(m, device, parameters)

        eval_results = eval(m, args.limit, parameters, criterion, data_loader_test, device, args)
        
        print("------------")
        print(trial_idx, parameters, eval_results)
        history.append((eval_results, parameters))
        
        ax_client.complete_trial(
            trial_index=trial_idx,
            raw_data=eval_results,
        )

        del m
        torch.cuda.empty_cache()

    #write BO search trial history to csv file
    utils.write_history_to_csv(history, args.history_output, ["cal_acc", "model_size", "quant_config"])

    print("------Best config------")
    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters, values)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--model_name", default="vit_b_16", type=str, help="model name")
    parser.add_argument("--weights_name", default="ViT_B_16_Weights.IMAGENET1K_V1", type=str, help="model weight name")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")

    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument('--num_trials', type=int, default=200, help='Number of trials to run BO')
    parser.add_argument('--model_size_constraint', type=float, default=0.50, help='The model size constraint for BO, in MB')
    parser.add_argument('--multi_gpus', action='store_true', help="Use multi-processing to run evaluation on multi-gpus")
    parser.add_argument('--gpu_list', type=str, default="", help="A list of gpus to run evaluation, separated by comma, e.g., --gpu_lists=0,1,2,3")
    parser.add_argument('--limit', type=int, default=2000, help='Number of samples to evaluate the model')
    parser.add_argument('--valdir', type=str, default="/home/andrewor/local/datasets/imagenet/val_blurred", help='The path to the validation dataset')
    parser.add_argument('--history_output', type=str, default="BO_acc_modelsize_output_ViT_b_16.csv", help="The csv file path to save the BO search trials")
    parser.add_argument('--parameters_list', type=str, default="./parameter_json/ViT_b_16_parameters.json", help="The json file path to save the parameters list for BO")
    parser.add_argument('--initial_samples', type=str, default="./initial_samples_json/ViT_b_16_initial_samples.json", help="The json file path to save the user-defined initial samples for BO")

    return parser

#TODO: merge this with BO_acc_modelsize.py
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_sequential_BO(args)
