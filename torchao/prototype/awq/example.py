import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time
from torchao.prototype.awq.api import insert_awq_observer_, ObservedLinear, awq_uintx
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only, Int4WeightOnlyGPTQQuantizer


# adapted from: https://github.com/mit-han-lab/llm-awq
def get_calib_dataset(tokenizer=None, n_samples=512, device="cuda"):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    block_size=512
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

    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    return torch.cat([
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ], dim=0)

def eval(model, tokenizer, max_length):
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // max_length
    model = model.eval()
    # calculate perplexity
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
    
    return ppl

def wikitext2_ppl(repo_id: str, quant: str, calibrate_size: int =100, group_size: int = 128, device="cuda", precision=torch.bfloat16, max_length=2048, compile=False):
    print("Loading model ...")
    torch.manual_seed(34)
    t0 = time.time()
    # load any model with torch.nn.linear layers
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision).to(device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    if quant.startswith("awq"):
        quant_dtype = quant.split("-")[1]
        quant_dtype = getattr(torch, quant_dtype, torch.bfloat16)
        print(f"running {quant_dtype} calibration")
        t0 = time.time()
        
        # insert observers to find average magnitude and calculate scales
        insert_awq_observer_(model, quant_dtype=quant_dtype, group_size=group_size)
        calibration_data = get_calib_dataset(tokenizer=tokenizer, n_samples=calibrate_size)
        model(calibration_data.to(device))
        print(f"time for calibration: {time.time() - t0:.02f} seconds")

        # use awq_quant() to apply awq quantization
        is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)
        t0 = time.time()
        quantize_(model, awq_uintx(quant_dtype=quant_dtype, group_size = group_size), is_observed_linear)
        print(f"time for quantization: {time.time() - t0:.02f} seconds")
    elif quant=="gptq":
        calibration_data = get_calib_dataset(tokenizer=tokenizer, n_samples=calibrate_size)
        quantizer = Int4WeightOnlyGPTQQuantizer()
        model = quantizer.quantize(model, calibration_data).to(device)
    elif quant=="int8":
        print("running int8 quantization")
        quantize_(model, int8_weight_only())

    elif quant=="uint4":
        print("running int4 quantization")
        quantize_(model, int4_weight_only(group_size=64))

    if compile:
        model = torch.compile(model)

    return eval(model, tokenizer, max_length)


parser = argparse.ArgumentParser(description="Evaluate a model with the specified parameters.")
    

# Optional arguments with default values
parser.add_argument("repo", type=str, help="Repository ID of the model.")
parser.add_argument("quant", type=str, help="Quantization method or file path.",choices=["uint4", "int8","gptq"])
parser.add_argument("--calibration_size", type=int, default=100, help="Calibration size. Default is 100.")
parser.add_argument("--group_size", type=int, default=128, help="Group size to use for weights. Default is '128'")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on. Default is 'cuda'.")
parser.add_argument("--precision", type=str, default="bfloat16", help="Precision type. Default is 'bfloat16'.")
parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for evaluation. Default is 2048.")
parser.add_argument("--compile", action="store_true", help="Flag to indicate if compilation is required.")

args = parser.parse_args()

# Convert precision argument to torch dtype
precision_dtype = getattr(torch, args.precision, torch.bfloat16)

# awq = wikitext2_ppl(
#     repo_id=args.repo,
#     quant="awq-"+args.quant,
#     calibrate_size=args.calibration_size,
#     group_size= args.group_size,
#     device=args.device,
#     precision=precision_dtype,
#     max_length=args.max_length,
#     compile=args.compile
# )

aqt = wikitext2_ppl(
    repo_id=args.repo,
    quant=args.quant,
    calibrate_size=args.calibration_size,
    group_size= args.group_size,
    device=args.device,
    precision=precision_dtype,
    max_length=args.max_length,
    compile=args.compile
)
# print(f"AWQ Perplexity: {awq.item():.5f}")
print(f"Affine quantized Perplexity: {aqt.item():.5f}")