import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from safetensors.torch import load_file  # Import safetensors loader

from torchao._models.llama.model import prepare_inputs_for_model
from torchao.quantization import int4_weight_only, quantize_
from torchao.dtypes import MarlinSparseLayout
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.GPTQ_MT import Int4WeightOnlyGPTQQuantizer, MultiTensor

import lm_eval
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict


# ENVS
model_name = 'meta-llama/Meta-Llama-3-8B'
model_name = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
device = "cuda"
calibration_limit = 4

class InputRecorder(eval_wrapper):
    def __init__(
        self,
        tokenizer,
        calibration_seq_length,
        input_prep_func=None,
        pad_calibration_inputs=False,
        vocab_size=32000,
        pad_token=0,
        device="cpu",
    ):
        try:
            super().__init__()
        except TypeError:
            # lm_eval 0.4.2 removed the default init
            super().__init__("gpt2", device="cpu")

        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self.vocab_size = vocab_size
        self._max_seq_length = calibration_seq_length
        self.calibration_seq_length = calibration_seq_length

        self.input_prep_func = (
            input_prep_func if input_prep_func is not None
            else lambda x: (x,)
        )

        self.pad_calibration_inputs = pad_calibration_inputs
        self.pad_token = pad_token

        self.inputs = []

    @property
    def eot_token_id(self):
        try:
            return self.tokenizer.eos_id()
        except:
            return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        tokens = self.tokenizer.encode(string)
        if hasattr(self.tokenizer, "bos_id"):
            try:
                tokens = [self.tokenizer.bos_id()] + tokens
            except:
                tokens = [self.tokenizer.bos_id] + tokens
        return tokens

    def tok_decode(self, tokens):
        decoded = self.tokenizer.decode(tokens)
        return decoded

    def add_input(self, args):
        self.inputs.append(args)

    def record_inputs(
        self,
        calibration_tasks,
        calibration_limit,
    ):
        try:
            lm_eval.tasks.initialize_tasks()
        except:
            pass

        task_dict = get_task_dict(calibration_tasks)
        print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

        evaluate(
            self,
            task_dict,
            limit=calibration_limit,
        )
        return self

    def get_inputs(self):
        return self.inputs

    def _model_call(self, inps):
        inps = inps.squeeze(0)
        T = len(inps)
        if (
            # can't use inputs that are too short when padding disabled
            (T < self.calibration_seq_length and not self.pad_calibration_inputs)
            or
            # can't use inputs that actually use token we use for padding
            (self.pad_calibration_inputs and self.pad_token in inps)
        ):
            # give random output
            return torch.randn(
                (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
            )

        # pad or truncate to the right size
        if T >= self.calibration_seq_length:
            inps = inps[: self.calibration_seq_length]
        else:
            inps = F.pad(inps, (self.pad_token, self.calibration_seq_length - T))

        inps = inps.unsqueeze(0)
        model_in = self.input_prep_func(inps)

        self.add_input(model_in)

        # output `something` with correct shape to keep eval going
        return torch.randn(
            (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
        )

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


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
    blocksize = 128
    percdamp = 0.01
    groupsize = 64
    calibration_tasks = ["wikitext"]
    calibration_seq_length = 100
    input_prep_func = prepare_inputs_for_model
    pad_calibration_inputs = False
    print("Recording inputs")
    inputs = InputRecorder(
            tokenizer,
            calibration_seq_length,
            input_prep_func,
            pad_calibration_inputs,
            model.config.vocab_size,
            device="cpu",
        ).record_inputs(
            calibration_tasks,
            calibration_limit,
        ).get_inputs()
    print("Inputs recorded")
    quantizer = Int4WeightOnlyGPTQQuantizer(
            blocksize,
            percdamp,
            groupsize,
        )
    
    multi = [MultiTensor([ inp for inp, _ in inputs]), MultiTensor([ inds for _, inds in inputs])]
    print("Quantizing model")
    model = quantizer.quantize(model, multi).cuda()
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
    