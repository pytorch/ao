import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time
from torchao.prototype.awq import insert_awq_observer_, AWQObservedLinear, awq_uintx,
from torchao.quantization import quantize_, int4_weight_only, uintx_weight_only


# adapted from: https://github.com/mit-han-lab/llm-awq
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
    return [cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_samples)]

def wiki2_eval(model, tokenizer, sequence_length):
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    testenc = testenc.input_ids
    nsamples = 100
    model = model.eval()
    # calculate perplexity
    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, i : i + sequence_length].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        batch = batch.to("cpu")
        lm_logits = lm_logits.to("cpu")
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, i : i + sequence_length][:, 1:].to("cpu")
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * sequence_length
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * sequence_length))
    
    return ppl

def wikitext2_ppl(
        repo_id: str,
        quant: str, 
        calibration_size: int =100, 
        validation_size:int=100, 
        group_size: int = 128, 
        device="cuda", 
        precision=torch.bfloat16, 
        sequence_length=2048, 
        compile=False,
        model_save_path=None):
    print(f"Loading model on {device}...")
    torch.manual_seed(34)
    t0 = time.time()
    # load any model with torch.nn.linear layers
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision).eval().to(device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    if quant.startswith("awq"):
        quant_dtype = quant.split("-")[1]
        quant_dtype = getattr(torch, quant_dtype, torch.bfloat16)
        print(f"running {quant_dtype} calibration")
        t0 = time.time()
        
        # insert observers to find average magnitude and calculate scales
        insert_awq_observer_(model,validation_size, sequence_length, quant_dtype=quant_dtype, group_size=group_size)
        calibration_data = get_calib_dataset(tokenizer=tokenizer, n_samples=calibration_size, block_size=sequence_length)
        for batch in calibration_data:
            model(batch.to(device))
            batch.to("cpu")
        print(f"time for calibration: {time.time() - t0:.02f} seconds")

        print(f"running {quant_dtype} quantization")
        # use awq_uintx() to apply awq quantization
        is_observed_linear = lambda m, fqn: isinstance(m, AWQObservedLinear)
        t0 = time.time()
        quantize_(model, awq_uintx(quant_dtype=quant_dtype, group_size = group_size), is_observed_linear)
            
        print(f"time for quantization: {time.time() - t0:.02f} seconds")
        if model_save_path is not None:
            print(f"Saving model to {model_save_path}")
            torch.save(model, model_save_path)
    elif quant=="int4":
        print("running int4 quantization")
        quantize_(model, int4_weight_only(group_size=group_size))

    if compile:
        model = torch.compile(model)

    return wiki2_eval(model, tokenizer, 1024)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with the specified parameters.")
        

    # Optional arguments with default values
    parser.add_argument("repo", type=str, help="Repository ID of the model.")
    parser.add_argument("quant", type=str, help="Quantization method. Options are either int4 or awq-uintx where x is [1..8]")
    parser.add_argument("--calibration_samples", type=int, default=10, help="Number of samples to use for calibration. Default is 10.")
    parser.add_argument("--validation_size", type=int, default=1, help="Validation size. Default is 1.")
    parser.add_argument("--group_size", type=int, default=128, help="Group size to use for weights. Default is '128'")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on. Default is 'cuda'.")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision type. Default is 'bfloat16'.")
    parser.add_argument("--seq_len", type=int, default=512, help="Length of examples to calibrate/evaluate model on. Default 512")
    parser.add_argument("--compile", action="store_true", help="Flag to indicate if compilation is required.")
    parser.add_argument("--model_save_path", type=str, default=None, help="Path to store the scale values.")

    args = parser.parse_args()

    # Convert precision argument to torch dtype
    precision_dtype = getattr(torch, args.precision, torch.bfloat16)
    ppl = wikitext2_ppl(
        repo_id=args.repo,
        quant=args.quant,
        calibration_size=args.calibration_samples,
        validation_size=args.validation_size,
        group_size= args.group_size,
        device=args.device,
        precision=precision_dtype,
        sequence_length=args.seq_len,
        compile=args.compile,
        scale_store_path=args.model_save_path
    )

    print(f"{args.quant} Perplexity: {ppl.item():.5f}")