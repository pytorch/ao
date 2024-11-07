import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from safetensors.torch import load_file  # Import safetensors loader

from torchao._models.llama.model import prepare_inputs_for_model
from torchao.quantization import int4_weight_only, quantize_
from torchao._models._eval import MultiTensorInputRecorder
from torchao.dtypes import MarlinSparseLayout
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.GPTQ_MT import Int4WeightOnlyGPTQQuantizer, MultiTensor

# ENVS
model_name = 'meta-llama/Meta-Llama-3-8B'
model_name = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
device = "cuda"
calibration_limit = 4


def get_model_and_tokenizer(precision: torch.dtype = torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(dtype=precision, device=device)
    return model, tokenizer


def get_data(tokenizer: AutoTokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    return data


def eval(model, tokenizer):
    data = get_data(tokenizer)

    max_length = model.config.max_length
    stride = 512
    seq_len = data.input_ids.size(1)

    # Iterate over the sequence with a sliding window
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def baseline(model, tokenizer):
    return model


def int4wo_gptq(model, tokenizer):
    groupsize = 128
    calibration_tasks = ["wikitext"]
    calibration_seq_length = 100
    pad_calibration_inputs = False
    print("Recording inputs")
    inputs = MultiTensorInputRecorder(
        tokenizer,
        calibration_seq_length,
        prepare_inputs_for_model,
        pad_calibration_inputs,
        model.config.vocab_size,
        device="cpu"
    ).record_inputs(
        calibration_tasks,
        calibration_limit,
    ).get_inputs()
    
    print("Quantizing model")
    quantizer = Int4WeightOnlyGPTQQuantizer(group_size=groupsize, device=device)
    model = quantizer.quantize(model, inputs).to(device)
    print("Model quantized")
    
    return model


def int4wo(model, tokenizer):
    print("Quantizing model")
    quantize_(model, int4_weight_only())
    print("Model quantized")
    return model


def int4wo_sparse(model, tokenizer):
    print("Quantizing model")
    apply_fake_sparsity(model)  # Simple sparsity just to simulate it
    quantize_(model, int4_weight_only())
    print("Model quantized")
    return model


def int4wo_sparse_marlin(model, tokenizer):
    print("Quantizing model")
    quantize_(model.half(), int4_weight_only(layout=MarlinSparseLayout()))
    print("Model quantized")
    return model


def build_plot(results):
    fig, ax = plt.subplots()
    for method, ppl in results.items():
        ax.bar(method, ppl, label=method, color='firebrick', edgecolor='black') 

    # Sets background as gray
    ax.set_facecolor('lightgrey')

    ax.set_title("Perplexity of different quantization methods", fontsize=17)
    ax.set_ylabel("Perplexity", fontsize=15)
    ax.set_xlabel("Method", fontsize=15)
    plt.tight_layout()
    plt.savefig("fig1.png")
    plt.close('all')


def run_eval(fn: str):
    model, tokenizer = get_model_and_tokenizer()
    model = fn(model, tokenizer)
    result = eval(model, tokenizer)
    return result


method_fn_map = {
    "baseline": baseline,
    "int4wo": int4wo,
    "int4wo_gptq": int4wo_gptq,
    "int4wo_sparse": int4wo_sparse,
    "int4wo_sparse_marlin": int4wo_sparse_marlin,
}

# Put this in args
selected_methods = ["int4wo_gptq"]  # ["baseline", "int4wo", "int4wo_sparse", "int4wo_sparse_marlin"]


if __name__ == "__main__":
    total_results = {}

    for method in selected_methods:
        print(f"\n Running method: {method}\n")
        result = run_eval(method_fn_map[method])
        total_results[method] = result

    build_plot(total_results)
    