import argparse
import os
import time
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.prototype.smoothquant import (
    SmoothQuantConfig,
    SmoothQuantObservedLinear,
    insert_smooth_quant_observer_,
)
from torchao.quantization import quantize_


def get_calib_dataset(tokenizer=None, n_samples=100, block_size=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
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


def wiki2_eval(
    model, tokenizer, sequence_length, stride=512, verbose=True, device="cuda"
):
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    print("Loading dataset")
    t0 = time.time()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    print(f"Time to load dataset: {time.time() - t0:.02f} seconds")

    encodings["input_ids"] = encodings["input_ids"].to(device)

    print("Running evaluation")
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
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        t.append((t2 - t1))
        lls.append(log_likelihood)

        del input_ids, target_ids

    ppl = float(torch.exp(torch.stack(lls).sum() / end_loc))
    pred_time = sum(t) / len(t)
    if verbose:
        print("perplexity", ppl)
        print("time", str(pred_time) + " sec/it")

    return {"perplexity": ppl, "prediction_time": pred_time}


def benchmark(model, tokenizer, max_length, tasks=None, device="cuda"):
    model.eval()
    model.config.use_cache = False
    if tasks is None:
        tasks = ["PPL"]
    results = {}
    if "PPL" in tasks:
        results["perplexity"] = wiki2_eval(
            model, tokenizer, 512, verbose=True, device=device
        )
    return results


def wikitext2_ppl(
    model_id: str,
    alpha: Optional[float],
    quant_mode: str,
    calibration_size: int,
    device: str,
    precision: torch.dtype,
    sequence_length: int,
    compile: bool,
    model_load_path: str,
    model_save_path: str,
):
    print(f"Loading model on {device}...")
    torch.manual_seed(34)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_load_path is not None and os.path.exists(model_load_path):
        print(f"Loading quantized model from {model_load_path}")
        t0 = time.time()
        model = torch.load(model_load_path, weights_only=False).to(device)
        print(f"Time to load quantized model: {time.time() - t0:.02f} seconds")
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=precision)
            .eval()
            .to(device)
        )
        print(f"Time to load model: {time.time() - t0:.02f} seconds")
        print("running calibration")
        t0 = time.time()
        # insert observers to find average magnitude and calculate scales
        insert_smooth_quant_observer_(model, alpha, quant_mode)
        calibration_data = get_calib_dataset(
            tokenizer=tokenizer, n_samples=calibration_size, block_size=sequence_length
        )
        for batch in calibration_data:
            model(batch.to(device))
            batch.to("cpu")
        print(f"time for calibration: {time.time() - t0:.02f} seconds")

        is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
        print(f"running SmoothQuant with {quant_mode} quantization")
        t0 = time.time()
        quantize_(model, SmoothQuantConfig(), is_observed_linear)
        print(f"time for quantization: {time.time() - t0:.02f} seconds")
        if model_save_path is not None:
            print(f"Saving quantized model to {model_save_path}")
            t0 = time.time()
            torch.save(model, model_save_path)
            print(f"Time to save quantized model: {time.time() - t0:.02f} seconds")
    if compile:
        model = torch.compile(model, dynamic=True)

    return benchmark(model, tokenizer, sequence_length, tasks=["PPL"], device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with the specified parameters."
    )

    # Optional arguments with default values
    parser.add_argument(
        "--model-id", "-m", type=str, help="Repository ID of the model."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="The alpha hyperparameter for SmoothQuant.",
    )
    parser.add_argument(
        "--quant-mode", type=str, help="Quantization mode, either static or dynamic."
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=10,
        help="Number of samples to use for calibration. Default is 10.",
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
        "--seq_len",
        type=int,
        default=512,
        help="Length of examples to calibrate and evaluate model on. Default is 512",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Flag to indicate if compilation is required.",
    )
    parser.add_argument(
        "--model-load-path",
        type=str,
        default=None,
        help="Path to load quantized model. If this is provided, "
        "the model will be loaded from this path instead of quantizing the model.",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default=None,
        help="Path to store quantized model.",
    )
    parser.add_argument(
        "--disable-smooth-quant",
        action="store_true",
        help="Run conventional dynamic or static quantization for testing or debugging.",
    )

    args = parser.parse_args()

    # Convert precision argument to torch dtype
    precision_dtype = getattr(torch, args.precision, torch.bfloat16)
    ppl = wikitext2_ppl(
        args.model_id,
        None if args.disable_smooth_quant else args.alpha,
        args.quant_mode,
        args.calibration_samples,
        args.device,
        args.precision,
        args.seq_len,
        args.compile,
        args.model_load_path,
        args.model_save_path,
    )
