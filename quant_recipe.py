# Quantization
import lm_eval
import torch
from lm_eval import evaluator
from lm_eval.utils import (
    make_table,
)
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

model_id = "microsoft/Phi-4-mini-instruct"
ENABLE_GPTQ = False
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    ZeroPointDomain,
    quantize_,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id, padding_side="right")

embedding_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int4, # torch.int8
    granularity=PerGroup(32), #PerAxis(0),
    mapping_type=MappingType.ASYMMETRIC,
    zero_point_domain=ZeroPointDomain.INT,
    # scale_dtype=torch.float32,
)
linear_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(128),
    weight_mapping_type=MappingType.SYMMETRIC,
    weight_zero_point_domain=ZeroPointDomain.NONE,
    # weight_scale_dtype=torch.bfloat16,
)

quantize_(model, embedding_config, lambda m, fqn: isinstance(m, torch.nn.Embedding))

if not ENABLE_GPTQ:
    quantize_(
        model,
        linear_config,
    )
else:
    # This needs work
    assert False, "GPTQ is not set up yet"
    calibration_tasks = ["hellaswag"]
    calibration_seq_length = 80
    calibration_limit = 1000
    pad_calibration_inputs = True
    from torchao._models._eval import QwenMultiTensorInputRecorder
    from torchao._models.llama.model import prepare_inputs_for_model
    from torchao.quantization.GPTQ_MT import Int4WeightOnlyGPTQQuantizer

    # Assert Int4WeightOnlyConfig compatibility
    # TODO: check if this is identical to Int4WeightOnlyGPTQQuantizer (other than dynamic activation bit)
    assert linear_config.weight_dtype == torch.int4
    assert isinstance(linear_config.weight_granularity, PerGroup)
    groupsize = linear_config.granularity.group_size
    assert linear_config.weight_zero_point_domain == ZeroPointDomain.NONE
    assert linear_config.weight_mapping_type == MappingType.ASYMMETRIC

    device = "cuda"
    assert groupsize in [
        32,
        64,
        128,
        256,
    ], f"int4wo groupsize needs to be one of [32,64,128,256] but got {groupsize}"
    inputs = (
        QwenMultiTensorInputRecorder(
            tokenizer,
            calibration_seq_length,
            prepare_inputs_for_model,
            pad_calibration_inputs,
            model.config.vocab_size,
            device="cpu",
        )
        .record_inputs(
            calibration_tasks,
            calibration_limit,
        )
        .get_inputs()
    )
    quantizer = Int4WeightOnlyGPTQQuantizer(group_size=groupsize, device=device)
    model = quantizer.quantize(model, kw_inputs=inputs).to(device)

# prompt = "Hey, are you conscious? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

messages = [
    {
        "role": "system",
        "content": "You are a medieval knight and must provide explanations to modern people.",
    },
    {"role": "user", "content": "How should I explain the Internet?"},
]
inputs = tokenizer(
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ),
    return_tensors="pt",
).to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])

# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
lm_eval_model = lm_eval.models.huggingface.HFLM(pretrained=model, batch_size=64)
results = evaluator.simple_evaluate(
    lm_eval_model, tasks=["hellaswag"], device="cuda", batch_size="auto"
)
print(make_table(results))
