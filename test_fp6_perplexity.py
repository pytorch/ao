# adapted from https://huggingface.co/docs/transformers/perplexity

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
from torchao.quantization.subclass import Fp6WeightOnlyQuantizedLinearWeight

dtype = "fp32"  # fp32, fp16, or fp6

device = "cuda"
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, trust_remote_code=True)
if dtype != "fp32":
    model.half()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

if dtype == "fp6":
    modules = list(model.named_modules())
    for name, module in tqdm(modules, desc="Converting weight to FP6"):
        if isinstance(module, torch.nn.Linear):
            try:
                fp6_weight = Fp6WeightOnlyQuantizedLinearWeight.from_float(module.weight.detach())
                module.weight = torch.nn.Parameter(fp6_weight, requires_grad=False)
            except Exception as e:
                print(f"Unable to convert {name}.weight to FP6. {e}")  # typically LM head

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.max_length
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        nll = model(input_ids, labels=target_ids).loss

    nlls.append(nll)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)
