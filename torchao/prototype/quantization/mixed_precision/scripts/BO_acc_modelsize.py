import random

import torch
import torch.multiprocessing as mp
from ax.service.ax_client import AxClient, ObjectiveProperties
from BO_acc_throughput import define_parameter_list
from utils import (
    cal_model_size,
    cal_wikitext_ppl,
    load_initial_samples,
    load_model,
    load_parameters_from_json,
    quantize_by_fqn_to_config,
    write_history_to_csv,
)


# return evaluation results to complete BO trials
def eval(model, tokenizer, num_PPL_eval_samples, fqn_to_config):
    return {
        "cal_PPL": (cal_wikitext_ppl(model, tokenizer, num_PPL_eval_samples), 0.0),
        "model_size": (cal_model_size(model, fqn_to_config), 0.0),
    }


# add initial search points based on the sensitivity score
# TODO: add random initial samples if no sensitivity prior
def get_initial_samples(num_BO_initial_samples=10):
    initial_points_set = []

    # auto sample the bit choices with random choice probability positive correlated to FIT score
    for _ in range(num_BO_initial_samples):
        initial_points = {}
        for i in range(0, 3):
            initial_points["bitwidth." + str(i) + "."] = 5
            initial_points["groupsize." + str(i) + "."] = 32

        for i in range(3, 18):
            if i in [5, 6, 7, 10, 11, 12, 16]:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [5, 4], [20, 80]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64], [30, 70]
                )[0]
            else:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [5, 4], [30, 70]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64], [40, 60]
                )[0]

        for i in range(18, 30):
            if i in [22, 23, 24]:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [5, 4, 3, 2], [20, 55, 20, 5]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64, 128, 256], [30, 40, 25, 5]
                )[0]
            else:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [5, 4, 3, 2], [30, 55, 10, 5]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64, 128, 256], [40, 40, 15, 5]
                )[0]

        for i in range(30, 32):
            initial_points["bitwidth." + str(i) + "."] = 5
            initial_points["groupsize." + str(i) + "."] = 32

        initial_points_set.append(initial_points)
    return initial_points_set


"""
This function will run BO trials sequentially on a single GPU.
Each time the BO gets one new trial, evaluates the trial on the GPU and return the evaluation results to update the BO.
One trial, one BO update.
TODO: refactor the sequential BO and parallel BO into a single function
"""


def run_sequential_BO(
    device,
    checkpoint,
    num_PPL_eval_samples,
    num_trials,
    model_size_constraint,
    history_output,
    parameters_list,
    initial_samples,
):
    # TODO: add default parameter list if not specified
    parameters_list = load_parameters_from_json(parameters_list)
    initial_points_set = load_initial_samples(initial_samples)
    num_BO_initial_samples = len(initial_points_set)

    # initialize ax_client
    constraint = "model_size <= " + str(model_size_constraint)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters_list,
        name="test_quantize_BO",
        objectives={"cal_PPL": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_BO_initial_samples,  # the number of trials to build generation strategy
        },
        outcome_constraints=[constraint],
    )

    history = []
    trial_id = 0

    # add initial points into the BO trials
    for i in range(num_BO_initial_samples):
        ax_client.attach_trial(parameters=initial_points_set[i])

        m, tokenizer = load_model(checkpoint, device)
        quantize_by_fqn_to_config(m, device, initial_points_set[i])

        eval_results = eval(m, tokenizer, num_PPL_eval_samples, initial_points_set[i])

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

        eval_results = eval(m, tokenizer, num_PPL_eval_samples, parameters)

        print("------------")
        print(trial_idx, parameters, eval_results)
        history.append((eval_results, parameters))

        ax_client.complete_trial(
            trial_index=trial_idx,
            raw_data=eval_results,
        )

        del m
        torch.cuda.empty_cache()

    # write BO search trial history to csv file
    write_history_to_csv(
        history, history_output, ["cal_PPL", "model_size", "quant_config"]
    )

    print("------Best config------")
    best_parameters, values = ax_client.get_best_parameters()
    print(values, best_parameters)


# Worker function to perform BO trials on a specific GPU
def eval_in_parallel(
    gpu_id, checkpoint, num_PPL_eval_samples, config, return_dict, proc_id, trial_id
):
    model, tokenizer = load_model(checkpoint, f"cuda:{gpu_id}")

    print(f"Process {proc_id} on GPU {gpu_id} starts!")

    quantize_by_fqn_to_config(
        model=model, device=f"cuda:{gpu_id}", fqn_to_config=dict(config)
    )

    eval_results = eval(model, tokenizer, num_PPL_eval_samples, config)

    return_dict[proc_id] = (trial_id, config, eval_results)

    del model
    torch.cuda.empty_cache()


"""
This function will run BO trials in parallel on multiple GPUs.
Each time the BO gets multiple new trials, evaluates the trials on the GPUs and return the evaluation results to update the BO.
Multiple trials, one BO update.
"""


def run_parallel_BO(
    device,
    checkpoint,
    num_PPL_eval_samples,
    num_trials,
    model_size_constraint,
    gpu_list,
    history_output,
    parameters_list,
    initial_samples,
):
    # TODO: add default parameter list if not specified
    parameters_list = define_parameter_list()
    initial_points_set = load_initial_samples(initial_samples)
    num_BO_initial_samples = len(initial_points_set)

    # initialize ax_client
    constraint = "model_size <= " + str(model_size_constraint)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters_list,
        name="test_quantize_BO",
        objectives={"cal_PPL": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_BO_initial_samples,  # the number of trials to build generation strategy
        },
        outcome_constraints=[constraint],
    )

    gpu_list = [int(i) for i in gpu_list.split(",")]

    history = []
    trial_id = 0

    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method("spawn", force=True)

    # add initial points into the BO trials
    for id in range(num_BO_initial_samples // len(gpu_list)):
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        # Start the worker processes
        for i, gpu_id in enumerate(gpu_list):
            ax_client.attach_trial(
                parameters=dict(initial_points_set[id * len(gpu_list) + i])
            )
            p = mp.Process(
                target=eval_in_parallel,
                args=(
                    gpu_id,
                    checkpoint,
                    num_PPL_eval_samples,
                    initial_points_set[id * len(gpu_list) + i],
                    return_dict,
                    i,
                    trial_id,
                ),
            )
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
            ax_client.complete_trial(
                trial_index=current_trial_id,
                raw_data=eval_results,
            )

    # run new BO trials
    for id in range(num_trials // len(gpu_list)):
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        # Start the worker processes
        for i, gpu_id in enumerate(gpu_list):
            parameters, trial_idx = ax_client.get_next_trial()
            parameter_tuple = []
            for k, v in parameters.items():
                parameter_tuple.append((k, v))
            p = mp.Process(
                target=eval_in_parallel,
                args=(
                    gpu_id,
                    checkpoint,
                    num_PPL_eval_samples,
                    parameter_tuple,
                    return_dict,
                    i,
                    trial_idx,
                ),
            )
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
            ax_client.complete_trial(
                trial_index=current_trial_id,
                raw_data=eval_results,
            )

    # write BO search trial history to csv file
    write_history_to_csv(
        history, history_output, ["cal_PPL", "model_size", "quant_config"]
    )

    print("------Best config------")
    best_parameters, values = ax_client.get_best_parameters()
    print(values, best_parameters)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bayesian optimization for mixed-precision quantization to optimize accuracy under model size constraint."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/Meta-Llama-3-8B",
        help="Path to load model",
    )
    parser.add_argument(
        "--num_PPL_eval_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate ppl",
    )
    parser.add_argument(
        "--num_trials", type=int, default=200, help="Number of trials to run BO"
    )
    parser.add_argument(
        "--model_size_constraint",
        type=float,
        default=6.0,
        help="The model size (GB) constraint for BO",
    )
    parser.add_argument(
        "--gpu_list",
        type=str,
        default="",
        help="A list of gpus to run evaluation, separated by comma, e.g., --gpu_lists=0,1,2,3",
    )
    parser.add_argument(
        "--history_output",
        type=str,
        default="BO_acc_modelsize_output.csv",
        help="The csv file path to save the BO search trials",
    )
    parser.add_argument(
        "--parameters_list",
        type=str,
        default="Llama3-8B_parameters.json",
        help="The json file path to save the parameters list for BO",
    )
    parser.add_argument(
        "--initial_samples",
        type=str,
        default="Llama3-8B_initial_samples.json",
        help="The json file path to save the user-defined initial samples for BO",
    )

    args = parser.parse_args()

    if args.gpu_list == "":
        run_sequential_BO(
            device=args.device,
            checkpoint=args.checkpoint,
            num_PPL_eval_samples=args.num_PPL_eval_samples,
            num_trials=args.num_trials,
            model_size_constraint=args.model_size_constraint,
            history_output=args.history_output,
            parameters_list=args.parameters_list,
            initial_samples=args.initial_samples,
        )
    else:
        run_parallel_BO(
            device=args.device,
            checkpoint=args.checkpoint,
            num_PPL_eval_samples=args.num_PPL_eval_samples,
            num_trials=args.num_trials,
            model_size_constraint=args.model_size_constraint,
            gpu_list=args.gpu_list,
            history_output=args.history_output,
            parameters_list=args.parameters_list,
            initial_samples=args.initial_samples,
        )
