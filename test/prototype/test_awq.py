from copy import deepcopy
import torch
import torch.nn.functional as F
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
from torchao.prototype.awq.api import ObservedLinear, insert_awq_observer, awq_quant
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time


def simple_test():
    class ToyLinearModel(torch.nn.Module):
        def __init__(self, m=512, n=256, k=128):
            super().__init__()
            self.linear1 = torch.nn.Linear(m, n, bias=False)
            self.linear2 = torch.nn.Linear(n, k, bias=False)
            self.linear3 = torch.nn.Linear(k, 1, bias=False)

        def example_inputs(self, batch_size, sequence_length=10, dtype=torch.bfloat16, device="cuda"):
            return [torch.randn(1, sequence_length, self.linear1.in_features, dtype=dtype, device=device) for j in range(batch_size)]

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x
        

    device = ("cuda")
    dataset_size = 1000
    dtype = torch.bfloat16
    l1,l2,l3 = 512,256,128
    m = ToyLinearModel(l1,l2,l3).eval().to(dtype).to(device)
    m_bf16 = deepcopy(m)

    dataset = m.example_inputs(dataset_size,  dtype=dtype, device=device)
    calibration_data = dataset[:100]
    bf16_out = torch.cat([m_bf16(i.squeeze(0)) for i in dataset], dim=0)


    m_int4wo = deepcopy(m)
    quantize_(m_int4wo, int8_weight_only())
    int4wo_out = torch.cat([m_int4wo(i.squeeze(0)) for i in dataset])

    # calibrate
    quant_dtype = "int4"
    group_size = 128
    insert_awq_observer(m, quant_dtype, group_size, dtype, device)
    for example in calibration_data:
        m(example.to(device))
    # print('calibrated')

    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)
    scales = []
    quantize_(m, awq_quant(quant_dtype = quant_dtype, scale_list=scales, group_size = group_size), is_observed_linear)
    print(scales)
    awq_out = torch.cat([m(i.squeeze(0)) for i in dataset])
    # m = torch.compile(m, fullgraph=True)
    # compare accuracy
    awq_err = torch.sum(torch.abs(awq_out - bf16_out)).sum().item() / dataset_size
    int4wo_err = torch.sum(torch.abs(int4wo_out - bf16_out)).sum().item() / dataset_size
    print(f"AWQ error: {awq_err}")
    print(f"Int4WO error: {int4wo_err}")



def create_batches_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def get_calib_dataset(tokenizer=None, n_samples=512, block_size=512, device="cuda"):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    # dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
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
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return torch.cat([
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ], dim=0)

def pile_eval(repo_id, quant, calibrate_size=100, eval_size=1000, device="cuda", precision=torch.bfloat16, max_length=2048, compile=False):
    print("Loading model ...")
    torch.manual_seed(34)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision).to(device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    # print(model)

    if quant.startswith("awq"):
        quant_dtype = quant.split("-")[1]
        print(f"running {quant} calibration")
        t0 = time.time()
        quant_dtype = quant.split("-")[1]
        group_size = 128 if quant_dtype == "int4" else -1
        insert_awq_observer(model, quant_dtype, group_size, precision, device)
        calibration_data = get_calib_dataset(tokenizer=tokenizer, n_samples=calibrate_size)

        model(calibration_data.to(device))

        print(f"time for calibration: {time.time() - t0:.02f} seconds")
        is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)
        t0 = time.time()
        # print(model)
        quantize_(model, awq_quant(quant_dtype=quant_dtype, group_size = group_size), is_observed_linear)
        print(f"time for quantization: {time.time() - t0:.02f} seconds")

    elif quant=="int8":
        print("running int8 quantization")
        quantize_(model, int8_weight_only())
        # print(model)

    elif quant=="int4":
        print("running int4 quantization")
        quantize_(model, int4_weight_only())
        # print(model)
    if compile:
        model = torch.compile(model)

    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // max_length
    model = model.eval()
    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * max_length) : ((i + 1) * max_length)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * max_length) : ((i + 1) * max_length)
        ][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * max_length
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_length))
    print(f"Perplexity: {ppl.item():.5f}")
    return ppl
    # int4: 29.75957
    # real-awq-int4: 28.9590
    # awq-int4:


parser = argparse.ArgumentParser(description="Evaluate a model with the specified parameters.")
    
# Positional arguments
parser.add_argument("repo", type=str, help="Repository ID of the model.")
parser.add_argument("quant", type=str, help="Quantization method or file path.")

# Optional arguments with default values
parser.add_argument("--calibrate_size", type=int, default=100, help="Calibration size. Default is 100.")
parser.add_argument("--eval_size", type=int, default=1000, help="Evaluation size. Default is 1000.")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on. Default is 'cuda'.")
parser.add_argument("--precision", type=str, default="bfloat16", help="Precision type. Default is 'bfloat16'.")
parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for evaluation. Default is 2048.")
parser.add_argument("--compile", action="store_true", help="Flag to indicate if compilation is required.")

args = parser.parse_args()

# Convert precision argument to torch dtype
precision_dtype = getattr(torch, args.precision, torch.bfloat16)

pile_eval(
    repo_id=args.repo,
    quant=args.quant,
    calibrate_size=args.calibrate_size,
    eval_size=args.eval_size,
    device=args.device,
    precision=precision_dtype,
    max_length=args.max_length,
    compile=args.compile
)
