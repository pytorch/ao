# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo.config
import torch._inductor.config
from ax.service.ax_client import AxClient, ObjectiveProperties
from transformers import AutoTokenizer
from utils import (
    cal_wikitext_ppl,
    load_initial_samples,
    load_model,
    load_parameters_from_json,
    quantize_by_fqn_to_config,
    write_history_to_csv,
)

import torchao
from torchao._models.llama.generate import (
    _load_model,
    decode_one_token,
    device_sync,
    encode_tokens,
    prefill,
)
from torchao._models.llama.model import Transformer, prepare_inputs_for_model
from torchao._models.llama.tokenizer import get_tokenizer

default_device = "cuda" if torch.cuda.is_available() else "cpu"


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            next_token, next_prob = next_token.clone(), next_prob.clone()
            input_pos += 1
            new_tokens.append(next_token)
            callback(new_tokens[-1])
            new_probs.append(next_prob)
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    callback=lambda x: x,
    kv_cache_quantization: bool = False,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    device = prompt.device
    T = prompt.numel()

    # calculate how many tokens to generate based on max_new_tokens and model's upper bound (block_size)
    max_seq_length = (
        min(T + max_new_tokens, model.config.block_size) if not interactive else 350
    )
    new_tokens = max_seq_length - T

    # full prompt+output will be stored in seq
    seq = torch.empty(max_seq_length, dtype=prompt.dtype, device=device)
    seq[:T] = prompt.view(-1)

    # setup model caches
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
        if kv_cache_quantization:
            from model import AffineQuantizedKVCache

            from torchao.quantization.quant_api import (
                _replace_with_custom_fn_if_matches_filter,
            )

            _replace_with_custom_fn_if_matches_filter(
                model,
                AffineQuantizedKVCache.from_float,
                lambda x, y: isinstance(x, torchao._models.llama.model.KVCache),
            )

    # format model input
    x, input_pos = prepare_inputs_for_model(prompt, max_new_tokens)

    # execute prefill
    next_token = prefill(model, x, input_pos, **sampling_kwargs).clone()
    seq[T] = next_token

    # execute token generation
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )
    seq[T + 1 :] = torch.cat(generated_tokens)

    return seq


def cal_throughput(
    model,
    tokenizer,
    device,
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("/tmp/Meta-Llama-3-8B/model.pth"),
    quantization: Optional[str] = None,
    kv_cache_quantization: bool = False,
    save: bool = False,
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    precision=torch.bfloat16,
    write_result: Optional[Path] = None,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    B_INST, E_INST = "[INST]", "[/INST]"

    torchao.quantization.utils.recommended_inductor_config_setter()

    is_chat = "chat" in str(checkpoint_path)

    device_sync(device=device)  # MKG

    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)

    if compile:
        print("Compiling Model")
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        if i == 0:
            torch.cuda.reset_peak_memory_stats()
        device_sync(device=device)  # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        import contextlib

        if i != num_samples - 1 or not profile:
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                kv_cache_quantization=kv_cache_quantization,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0

        if interactive:
            print()
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)

    tokpersec = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    print("tokpersec", tokpersec)
    return tokpersec


# return evaluation results to complete BO trials
def eval(model4ppl, model4tp, tokenizer, device, num_PPL_eval_samples, fqn_to_config):
    return {
        "cal_PPL": (cal_wikitext_ppl(model4ppl, tokenizer, num_PPL_eval_samples), 0.0),
        "cal_throughput": (
            cal_throughput(model=model4tp, tokenizer=tokenizer, device=device),
            0.0,
        ),
    }


# TODO: make it into a yaml or json file to enable users specify their custom model formats
def define_parameter_list():
    # define the search space for all layers
    parameters_list = []

    for i in range(0, 3):
        parameters_list.append(
            {
                "name": f"bitwidth.{i}.",
                "type": "fixed",
                "value_type": "int",
                "value": 8,
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
                "values": [2, 3, 4, 5, 6, 8],
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
                "value": 8,
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


# add initial search points based on the sensitivity score
# TODO: add default parameter list if not specified
def get_initial_samples(num_BO_initial_samples=10):
    initial_points_set = []

    # auto sample the bit choices with random choice probability positive correlated to FIT score
    for _ in range(num_BO_initial_samples):
        initial_points = {}
        for i in range(0, 3):
            initial_points["bitwidth." + str(i) + "."] = 8
            initial_points["groupsize." + str(i) + "."] = 32

        for i in range(3, 18):
            if i in [5, 6, 7, 10, 11, 12, 16]:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [8, 6, 5, 4], [25, 2, 2, 71]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64], [40, 60]
                )[0]
            else:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [8, 6, 5, 4], [30, 2, 2, 66]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64], [50, 50]
                )[0]

        for i in range(18, 30):
            if i in [22, 23, 24]:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [8, 6, 5, 4], [10, 2, 2, 86]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64, 128, 256], [35, 45, 10, 10]
                )[0]
            else:
                initial_points["bitwidth." + str(i) + "."] = random.choices(
                    [8, 6, 5, 4], [20, 2, 2, 76]
                )[0]
                initial_points["groupsize." + str(i) + "."] = random.choices(
                    [32, 64, 128, 256], [30, 40, 25, 5]
                )[0]

        for i in range(30, 32):
            initial_points["bitwidth." + str(i) + "."] = 8
            initial_points["groupsize." + str(i) + "."] = 32

        initial_points_set.append(initial_points)

    return initial_points_set


"""
This function will run BO trials sequentially on a single GPU.
Each time the BO gets one new trial, evaluates the trial on the GPU and return the evaluation results to update the BO.
One trial, one BO update.
"""


def run_sequential_BO(
    device,
    checkpoint_path,
    repo_id,
    num_PPL_eval_samples,
    num_trials,
    ppl_constraint,
    args,
):
    """
    currently use the loader and benchmark code from torchao/_models/llama/generate,
    and use lm_eval for ppl evaluation
    """
    # load tokenizers
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)
    device_sync(device=device)  # MKG
    tokenizer4tp = get_tokenizer(tokenizer_path, checkpoint_path)
    tokenizer4ppl = AutoTokenizer.from_pretrained(repo_id)

    # initialize parameters
    # TODO: add default parameter list if not specified
    parameters_list = load_parameters_from_json(args.parameters_list)

    # sample initial points
    # TODO(future PR): fix me
    initial_samples = []
    initial_points_set = load_initial_samples(initial_samples)
    num_BO_initial_samples = len(initial_points_set)

    # initialize BO experiment
    constraint = "cal_PPL <= " + str(ppl_constraint)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters_list,
        name="test_quantize_BO",
        objectives={"cal_throughput": ObjectiveProperties(minimize=False)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_BO_initial_samples  # the number of trials to build generation strategy
        },
        outcome_constraints=[constraint],
    )

    history = []
    trial_id = 0

    # add initial points into the BO trials
    for i in range(num_BO_initial_samples):
        ax_client.attach_trial(parameters=initial_points_set[i])

        # evaluate throuput of quantized model under torch.compile()
        model4tp = _load_model(checkpoint_path, device, torch.bfloat16)
        quantize_by_fqn_to_config(
            model=model4tp, device=device, fqn_to_config=initial_points_set[i]
        )
        tp = cal_throughput(model=model4tp, tokenizer=tokenizer4tp, device=device)
        del model4tp
        torch.cuda.empty_cache()

        # evaluate ppl of quantized model
        model4ppl = load_model(repo_id, device)
        quantize_by_fqn_to_config(
            model=model4ppl, device=device, fqn_to_config=initial_points_set[i]
        )
        ppl = cal_wikitext_ppl(model4ppl, tokenizer4ppl, num_PPL_eval_samples)
        del model4ppl
        torch.cuda.empty_cache()

        eval_results = {
            "cal_PPL": (ppl, 0.0),
            "cal_throughput": (tp, 0.0),
        }

        print("------------")
        print(trial_id, initial_points_set[i], eval_results)

        history.append((eval_results, initial_points_set[i]))
        ax_client.complete_trial(
            trial_index=trial_id,
            raw_data=eval_results,
        )
        trial_id += 1

    # run new BO trials
    for k_ in range(num_trials):
        parameters, trial_idx = ax_client.get_next_trial()

        # evaluate throuput of quantized model under torch.compile()
        model4tp = _load_model(checkpoint_path, device, torch.bfloat16)
        quantize_by_fqn_to_config(
            model=model4tp, device=device, fqn_to_config=initial_points_set[i]
        )
        tp = cal_throughput(model=model4tp, tokenizer=tokenizer4tp, device=device)
        del model4tp
        torch.cuda.empty_cache()

        # evaluate ppl of quantized model
        model4ppl = load_model(repo_id, device)
        quantize_by_fqn_to_config(
            model=model4ppl, device=device, fqn_to_config=initial_points_set[i]
        )
        ppl = cal_wikitext_ppl(model4ppl, tokenizer4ppl, num_PPL_eval_samples)
        del model4ppl
        torch.cuda.empty_cache()

        eval_results = {
            "cal_PPL": (ppl, 0.0),
            "cal_throughput": (tp, 0.0),
        }

        print("------------")
        print(trial_idx, parameters, eval_results)

        history.append((eval_results, parameters))

        ax_client.complete_trial(
            trial_index=trial_idx,
            raw_data=eval_results,
        )

    # write BO search trial history to csv file
    write_history_to_csv(
        history, args.history_output, ["cal_PPL", "cal_throughput", "quant_config"]
    )

    print("------Best config------")
    best_parameters, values = ax_client.get_best_parameters()
    print(values, best_parameters)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bayesian optimization for mixed-precision quantization to optimize inference speed under model accuracy constraint."
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for evaluation"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("/tmp/Meta-Llama-3-8B/model.pth"),
        help="Model checkpoint path for model.pth.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=Path("/tmp/Meta-Llama-3-8B"),
        help="Model repo id.",
    )
    parser.add_argument(
        "--num_PPL_eval_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate ppl",
    )
    parser.add_argument(
        "--num_trials", type=int, default=150, help="Number of trials to run BO"
    )
    parser.add_argument(
        "--ppl_constraint", type=float, default=7.5, help="The ppl constraint for BO"
    )
    parser.add_argument(
        "--multi_gpus",
        action="store_true",
        help="Use multi-processing to run evaluation on multi-gpus",
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
        default="BO_acc_speed_output.csv",
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
    run_sequential_BO(
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        num_PPL_eval_samples=args.num_PPL_eval_samples,
        num_trials=args.num_trials,
        ppl_constraint=args.ppl_constraint,
        args=args,
    )
