import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

model_id = "Qwen/Qwen3-8B"

from torchao.quantization import Int4WeightOnlyConfig
quant_config = Int4WeightOnlyConfig(group_size=128, use_hqq=True)
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Push to hub
USER_ID = "liangel"
MODEL_NAME = model_id.split("/")[-1]
save_to = f"{USER_ID}/{MODEL_NAME}-INT4"
print("about to push to hub")
quantized_model.push_to_hub(save_to, safe_serialization=False)
tokenizer.push_to_hub(save_to)

# Manual Testing
prompt = "Hey, are you conscious? Can you talk to me?"
messages = [
    {
        "role": "system",
        "content": "",
    },
    {"role": "user", "content": prompt},
]
templated_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("Prompt:", prompt)
print("Templated prompt:", templated_prompt)
inputs = tokenizer(
    templated_prompt,
    return_tensors="pt",
).to("cuda")
generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
output_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Response:", output_text[0][len(prompt):])
