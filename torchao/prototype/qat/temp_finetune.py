from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer
import torch
from datasets import load_dataset

quantization_config = Mxfp4Config(dequantize=True)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

# The dataset has a "messages" column with entries like:
#   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<think>...</think>\n..."}]
# SFTTrainer natively handles a "messages" column by applying the tokenizer's chat template.

# Train/eval split
dataset = dataset.train_test_split(test_size=0.05, seed=42)

# Training configuration
training_args = SFTConfig(
    output_dir="./gpt-oss-20b-reasoning-sft",
    max_steps=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    max_length=4096,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train()

# Save the final model
trainer.save_model("./gpt-oss-20b-reasoning-sft/final")
tokenizer.save_pretrained("./gpt-oss-20b-reasoning-sft/final")
