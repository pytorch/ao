import torch
from torchao import quantize_
from torchao.quantization import int4_weight_only
from torchao.dtypes import MarlinSparseLayoutType
from transformers import AutoTokenizer, LlamaForCausalLM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
name = "meta-llama/Llama-2-7b-hf"
token = "your token"

model = LlamaForCausalLM.from_pretrained(name, torch_dtype=torch.float16, token=token).to(device)
tokenizer = AutoTokenizer.from_pretrained(name, token=token)

prompt = "Hey, are you conscious? Can you talk to me? I'm"
inputs = tokenizer(prompt, return_tensors="pt")

# Quantize
quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))

# Generate
ids = inputs.input_ids.to(device)
generate_ids = model.generate(ids, max_length=30)
out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)
