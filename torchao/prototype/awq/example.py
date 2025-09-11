# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.prototype.awq import (
    AWQConfig,
)
from torchao.quantization import (
    quantize_,
)


# adapted from: https://github.com/mit-han-lab/llm-awq/blob/main/awq/entry.py#L255
def get_calib_dataset(tokenizer=None, n_samples=100, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    samples = []
    n_tokens = n_samples * block_size
    n_run = n_tokens
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run -= len(line_encoded)
        if n_run <= n_samples:
            break

    cat_samples = torch.cat(samples, dim=1)
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_samples)
    ]


# from https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/eval_model.py
def wiki2_eval(
    model, tokenizer, sequence_length, stride=512, verbose=True, device="cuda"
):
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    encodings["input_ids"] = encodings["input_ids"].to(device)

    lls, t = [], []
    for i in tqdm(
        range(0, encodings["input_ids"].size(1), stride), disable=not verbose
    ):
        begin_loc = max(i + stride - sequence_length, 0)
        end_loc = min(i + stride, encodings["input_ids"].size(1))
        trg_len = end_loc - i
        input_ids = encodings["input_ids"][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore context

        t1 = time.time()
        with torch.no_grad():
            log_likelihood = model(input_ids, labels=target_ids).loss * trg_len
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if device.startswith("xpu"):
            torch.xpu.synchronize()
        t2 = time.time()
        t.append((t2 - t1))
        lls.append(log_likelihood)

        del input_ids, target_ids

    ppl = float(torch.exp(torch.stack(lls).sum() / end_loc))
    pred_time = sum(t) / len(t)
    if verbose:
        print("perplexity", ppl)
        print("time", str(pred_time) + "  sec")

    return {"perplexity": ppl, "prediction_time": pred_time}


# adapted from Hicham Badri (@mobicham)
def benchmark(model, tokenizer, max_length, tasks=None, device="cuda"):
    import lm_eval
    import numpy as np

    model.eval()
    model.config.use_cache = False
    try:
        lm_eval.tasks.initialize_tasks()
    except:
        pass
    model_eval = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    eval_batch_size = 1  # 8
    if tasks is None:
        tasks = [
            "PPL",
            "truthfulqa_mc2",
            "winogrande",
            "arc_challenge",
            "hellaswag",
            "gsm8k",
            "mmlu",
            "bbh",
        ]
    results = {}
    if "PPL" in tasks:
        results["perplexity"] = wiki2_eval(
            model, tokenizer, 512, verbose=True, device=device
        )
    ############################################
    if "truthfulqa_mc2" in tasks:
        for task in [("truthfulqa_mc2", 0)]:
            tag, fewshot = task
            results[tag] = lm_eval.evaluator.simple_evaluate(
                model_eval, tasks=[tag], num_fewshot=fewshot, batch_size=eval_batch_size
            )["results"]
            print(tag, results[tag])
    if "winogrande" in tasks:
        for task in [("winogrande", 5)]:
            tag, fewshot = task
            results[tag] = lm_eval.evaluator.simple_evaluate(
                model_eval, tasks=[tag], num_fewshot=fewshot, batch_size=eval_batch_size
            )["results"]
            print(tag, results[tag])
    if "arc_challenge" in tasks:
        for task in [("arc_challenge", 25)]:
            tag, fewshot = task
            results[tag] = lm_eval.evaluator.simple_evaluate(
                model_eval, tasks=[tag], num_fewshot=fewshot, batch_size=eval_batch_size
            )["results"]
            print(tag, results[tag])

    # ############################################
    if "hellaswag" in tasks:
        for task in [("hellaswag", 10)]:
            tag, fewshot = task
            results[tag] = lm_eval.evaluator.simple_evaluate(
                model_eval, tasks=[tag], num_fewshot=fewshot, batch_size=eval_batch_size
            )["results"]
            print(tag, results[tag])
    if "gsm8k" in tasks:
        for task in [("gsm8k", 5)]:
            tag, fewshot = task
            results[tag] = lm_eval.evaluator.simple_evaluate(
                model_eval, tasks=[tag], num_fewshot=fewshot, batch_size=eval_batch_size
            )["results"]
            print(tag, results[tag])
    # ############################################

    if "mmlu" in tasks:
        # MMLU
        results_mmlu = {}
        for task in [("mmlu", 5)]:
            tag, fewshot = task
            results_mmlu[tag] = lm_eval.evaluator.simple_evaluate(
                model_eval, tasks=[tag], num_fewshot=fewshot, batch_size=eval_batch_size
            )["results"]
            print(tag, results_mmlu[tag])

        mmlu_list = "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
        mmlu_list = [l.replace("hendrycksTest-", "") for l in mmlu_list.split(",")]
        results_mmlu = results_mmlu["mmlu"]

        k = []
        for r in results_mmlu:
            if np.any([(l in r) for l in mmlu_list]):
                k.append(results_mmlu[r]["acc,none"])

        assert len(k) == 57
        print("MMLU avg acc", np.mean(k))

        results["mmlu"] = np.mean(k)
    if "bbh" in tasks:
        for task in [("leaderboard_bbh", 3)]:
            tag, fewshot = task
            results[tag] = lm_eval.evaluator.simple_evaluate(
                model_eval, tasks=[tag], num_fewshot=fewshot, batch_size=eval_batch_size
            )["results"]
            print(tag, results[tag])
            results["bbh"] = results[tag]

    return results


def quantize_and_eval(
    repo_id: str,
    quant: str,
    tasks: list[str],
    max_seq_length: int,
    calibration_limit: int,
    validation_size: int,
    device: str,
    precision: torch.dtype,
    compile: bool,
    model_save_path: str,
    model_save_hf_hub_path: str,
):
    print(f"Loading model on {device}...")
    torch.manual_seed(34)
    t0 = time.time()
    # load any model with torch.nn.linear layers
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = (
        AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision)
        .eval()
        .to(device)
    )
    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    if quant.startswith("awq-int4wo"):
        group_size = int(quant.split("-")[2])
        print(f"running {quant} quantization with group size {group_size}")
        # TODO: this is temporary, we'll be using Int4WeightOnlyConfig soon
        from torchao.quantization import Int4WeightOnlyConfig

        base_config = Int4WeightOnlyConfig(group_size=group_size)
        print(f"running {quant} prepare and calibrate")
        t0 = time.time()
        quant_config = AWQConfig(base_config, step="prepare")

        quantize_(
            model,
            quant_config,
        )
        from torchao._models._eval import TransformerEvalWrapper

        TransformerEvalWrapper(
            model=model.to(device),
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            device=device,
        ).run_eval(
            tasks=tasks,
            limit=calibration_limit,
        )

        print(f"time for prepare and calibration: {time.time() - t0:.02f} seconds")
        print(f"running {quant} convert")
        t0 = time.time()
        quant_config = AWQConfig(base_config, step="convert")
        quantize_(model, quant_config)
        print(f"time for convert: {time.time() - t0:.02f} seconds")
        quant_config = AWQConfig(base_config, step="prepare_for_loading")
        model.config.quantization_config = TorchAoConfig(quant_config)

    elif quant.startswith("int4wo"):
        group_size = int(quant.split("-")[1])
        print(f"running {quant} quantization with group size {group_size}")
        # TODO: enable after migration: https://github.com/pytorch/ao/issues/2752
        # use_hqq = "hqq" in quant
        base_config = Int4WeightOnlyConfig(group_size=group_size, version=2)
        quantize_(model, base_config)

    if model_save_path is not None:
        print(f"Saving model to {model_save_path}")
        torch.save(model, model_save_path)

    if model_save_hf_hub_path is not None:
        print("pushing model to hub:", model_save_hf_hub_path)
        model.push_to_hub(model_save_hf_hub_path, safe_serialization=False)
        tokenizer.push_to_hub(model_save_hf_hub_path)

    if compile:
        model = torch.compile(model)

    return benchmark(model, tokenizer, max_seq_length, tasks=tasks, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with the specified parameters."
    )

    # Optional arguments with default values
    parser.add_argument("--repo", type=str, help="Repository ID of the model.")
    parser.add_argument(
        "--quant",
        type=str,
        help="Quantization method. Options are either awq-int4wo-<group_size>, or int4wo-<group_size>.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        help="Task to benchmark model on. Either PPL or QA",
        default=["PPL"],
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=10,
        help="Number of samples to use for calibration. Default is 10.",
    )
    parser.add_argument(
        "--validation_size", type=int, default=1, help="Validation size. Default is 1."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the evaluation on. Default is 'cuda'.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        help="Precision type. Default is 'bfloat16'.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length of examples to calibrate and evaluate model on. Default is 2048",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Flag to indicate if compilation is required.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to store the scale values.",
    )
    parser.add_argument(
        "--model_save_hf_hub_path",
        type=str,
        default=None,
        help="Huggingface hub path to store the quantized model and tokenizer.",
    )

    args = parser.parse_args()

    # Convert precision argument to torch dtype
    precision_dtype = getattr(torch, args.precision, torch.bfloat16)
    result = quantize_and_eval(
        args.repo,
        args.quant,
        args.tasks,
        args.max_seq_length,
        args.calibration_limit,
        args.validation_size,
        args.device,
        args.precision,
        args.compile,
        args.model_save_path,
        args.model_save_hf_hub_path,
    )

    print(f"{args.quant} Results: {result}")
