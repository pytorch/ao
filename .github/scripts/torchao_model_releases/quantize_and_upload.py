# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from huggingface_hub import ModelCard, get_token, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    PerAxis,
    PerGroup,
    PerRow,
)


def _get_username():
    token = get_token()
    username = whoami(token=token)["name"]
    return username


def _untie_weights_and_save_locally(model_id):
    untied_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from transformers.modeling_utils import find_tied_parameters

    if getattr(
        untied_model.config.get_text_config(decoder=True), "tie_word_embeddings"
    ):
        setattr(
            untied_model.config.get_text_config(decoder=True),
            "tie_word_embeddings",
            False,
        )

    untied_model._tied_weights_keys = []
    untied_model.lm_head.weight = torch.nn.Parameter(
        untied_model.lm_head.weight.clone()
    )

    print("tied weights:", find_tied_parameters(untied_model))

    MODEL_NAME = model_id.split("/")[-1]
    # save locally
    save_to_local_path = f"{MODEL_NAME}-untied-weights"
    untied_model.save_pretrained(save_to_local_path)
    tokenizer.save_pretrained(save_to_local_path)
    return save_to_local_path


MODEL_CARD = """---
base_model: {base_model}
tags:
- transformers
- torchao
- {model_type}
license: apache-2.0
language:
- en
---

# {quant} {base_model} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Quantized from Model :** {base_model}
- **Quantization Method :** {quant}

{server_inference_recipe}

{mobile_inference_recipe}

# Quantization Recipe

Install the required packages:
```Shell
pip install git+https://github.com/huggingface/transformers@main
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126
pip install torch
pip install accelerate
```

{untie_embedding_recipe}

Use the following code to get the quantized model:
```Py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

model_id = "{base_model}"
model_to_quantize = "{untied_model}"

{quant_code}
quantized_model = AutoModelForCausalLM.from_pretrained(model_to_quantize, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Push to hub
USER_ID = "YOUR_USER_ID"
MODEL_NAME = model_id.split("/")[-1]
save_to = f"{{USER_ID}}/{{MODEL_NAME}}-{quant}"
quantized_model.push_to_hub(save_to, safe_serialization=False)
tokenizer.push_to_hub(save_to)

# Manual Testing
prompt = "Hey, are you conscious? Can you talk to me?"
messages = [
    {{
        "role": "system",
        "content": "",
    }},
    {{"role": "user", "content": prompt}},
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
```

Note: to `push_to_hub` you need to run
```Shell
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
and use a token with write access, from https://huggingface.co/settings/tokens

# Model Quality
We rely on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate the quality of the quantized model. Here we only run on mmlu for sanity check.

| Benchmark                        |                |                           |
|----------------------------------|----------------|---------------------------|
|                                  | {base_model}   | {quantized_model}         |
| mmlu                             | To be filled   | To be filled                      |


<details>
<summary> Reproduce Model Quality Results </summary>

Need to install lm-eval from source:
https://github.com/EleutherAI/lm-evaluation-harness#install

## baseline
```Shell
lm_eval --model hf --model_args pretrained={base_model} --tasks mmlu --device cuda:0 --batch_size 8
```

## {quant}
```Shell
export MODEL={quantized_model}
lm_eval --model hf --model_args pretrained=$MODEL --tasks mmlu --device cuda:0 --batch_size 8
```
</details>



{server_peak_memory_usage}


{server_model_performance}

{mobile_export_to_executorch}

# Paper: TorchAO: PyTorch-Native Training-to-Serving Model Optimization
The model's quantization is powered by **TorchAO**, a framework presented in the paper [TorchAO: PyTorch-Native Training-to-Serving Model Optimization](https://huggingface.co/papers/2507.16099).

**Abstract:** We present TorchAO, a PyTorch-native model optimization framework leveraging quantization and sparsity to provide an end-to-end, training-to-serving workflow for AI models. TorchAO supports a variety of popular model optimization techniques, including FP8 quantized training, quantization-aware training (QAT), post-training quantization (PTQ), and 2:4 sparsity, and leverages a novel tensor subclass abstraction to represent a variety of widely-used, backend agnostic low precision data types, including INT4, INT8, FP8, MXFP4, MXFP6, and MXFP8. TorchAO integrates closely with the broader ecosystem at each step of the model optimization pipeline, from pre-training (TorchTitan) to fine-tuning (TorchTune, Axolotl) to serving (HuggingFace, vLLM, SGLang, ExecuTorch), connecting an otherwise fragmented space in a single, unified workflow. TorchAO has enabled recent launches of the quantized Llama 3.2 1B/3B and LlamaGuard3-8B models and is open-source at this https URL .

# Resources
*   **Official TorchAO GitHub Repository:** [https://github.com/pytorch/ao](https://github.com/pytorch/ao)
*   **TorchAO Documentation:** [https://docs.pytorch.org/ao/stable/index.html](https://docs.pytorch.org/ao/stable/index.html)


# Disclaimer
PyTorch has not performed safety evaluations or red teamed the quantized models. Performance characteristics, outputs, and behaviors may differ from the original models. Users are solely responsible for selecting appropriate use cases, evaluating and mitigating for accuracy, safety, and fairness, ensuring security, and complying with all applicable laws and regulations.

Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the licenses the models are released under, including any limitations of liability or disclaimers of warranties provided therein.
"""


_int4_quant_code = """
from torchao.quantization import Int4WeightOnlyConfig
quant_config = Int4WeightOnlyConfig(group_size=128, use_hqq=True)
quantization_config = TorchAoConfig(quant_type=quant_config)
"""

_fp8_quant_code = """
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
quant_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
quantization_config = TorchAoConfig(quant_type=quant_config)
"""

_int8_int4_quant_code = """
from torchao.quantization.quant_api import (
    IntxWeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    ModuleFqnToConfig,
)
from torchao.quantization.granularity import PerGroup, PerAxis
embedding_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int8,
    granularity=PerAxis(0),
)
linear_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),
    weight_scale_dtype=torch.bfloat16,
)
quant_config = ModuleFqnToConfig({{"_default": linear_config, "model.embed_tokens": embedding_config}})
quantization_config = TorchAoConfig(quant_type=quant_config, include_embedding=True, untie_embedding_weights=True, modules_to_not_convert=[])
"""

_server_inference_recipe = """
# Inference with vLLM
Install vllm nightly and torchao nightly to get some recent changes:
```
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install torchao
```

## Serving
Then we can serve with the following command:
```Shell
# Server
export MODEL={quantized_model}
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve $MODEL --tokenizer $MODEL -O3
```

```Shell
# Client
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{{
  "model": "{quantized_model}",
  "messages": [
    {{"role": "user", "content": "Give me a short introduction to large language models."}}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 32768
}}'
```

Note: please use `VLLM_DISABLE_COMPILE_CACHE=1` to disable compile cache when running this code, e.g. `VLLM_DISABLE_COMPILE_CACHE=1 python example.py`, since there are some issues with the composability of compile in vLLM and torchao,
this is expected be resolved in pytorch 2.8.

# Inference with Transformers

Install the required packages:
```Shell
pip install git+https://github.com/huggingface/transformers@main
pip install torchao
pip install torch
pip install accelerate
```

Example:
```Py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{quantized_model}"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {{"role": "user", "content": prompt}}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```
"""

_server_peak_memory_usage = """
# Peak Memory Usage

## Results

| Benchmark        |                |                                |
|------------------|----------------|--------------------------------|
|                  | {base_model}   | {quantized_model}              |
| Peak Memory (GB) | To be filled   | To be filled (?% reduction)    |



<details>
<summary> Reproduce Peak Memory Usage Results </summary>

We can use the following code to get a sense of peak memory usage during inference:

```Py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

# use "{base_model}" or "{quantized_model}"
model_id = "{quantized_model}"
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

torch.cuda.reset_peak_memory_stats()

prompt = "Hey, are you conscious? Can you talk to me?"
messages = [
    {{
        "role": "system",
        "content": "",
    }},
    {{"role": "user", "content": prompt}},
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

mem = torch.cuda.max_memory_reserved() / 1e9
print(f"Peak Memory Usage: {{mem:.02f}} GB")
```

</details>
"""

_server_model_performance = """
# Model Performance

## Results (A100 machine)
| Benchmark (Latency)              |                |                          |
|----------------------------------|----------------|--------------------------|
|                                  | {base_model}   | {quantized_model}        |
| latency (batch_size=1)           | ?s          | ?s (?x speedup)    |

<details>
<summary> Reproduce Model Performance Results </summary>

## Setup

Get vllm source code:
```Shell
git clone git@github.com:vllm-project/vllm.git
```

Install vllm
```
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

Run the benchmarks under `vllm` root folder:

## benchmark_latency

### baseline
```Shell
export MODEL={base_model}
python benchmarks/benchmark_latency.py --input-len 256 --output-len 256 --model $MODEL --batch-size 1
```

### {quant}
```Shell
export MODEL={quantized_model}
VLLM_DISABLE_COMPILE_CACHE=1 python benchmarks/benchmark_latency.py --input-len 256 --output-len 256 --model $MODEL --batch-size 1
```

## benchmark_serving

We benchmarked the throughput in a serving environment.

Download sharegpt dataset:

```Shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```



Other datasets can be found in: https://github.com/vllm-project/vllm/tree/main/benchmarks

Note: you can change the number of prompts to be benchmarked with `--num-prompts` argument for `benchmark_serving` script.

### baseline
Server:
```Shell
export MODEL={base_model}
vllm serve $MODEL --tokenizer $MODEL -O3
```

Client:
```Shell
export MODEL={base_model}
python benchmarks/benchmark_serving.py --backend vllm --dataset-name sharegpt --tokenizer $MODEL --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --model $MODEL --num-prompts 1
```

### {quant}
Server:
```Shell
export MODEL={quantized_model}
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve $MODEL --tokenizer $MODEL -O3 --pt-load-map-location cuda:0
```

Client:
```Shell
export MODEL={quantized_model}
python benchmarks/benchmark_serving.py --backend vllm --dataset-name sharegpt --tokenizer $MODEL --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --model $MODEL --num-prompts 1
```
</details>
"""


# Mobile Specific recipes

_mobile_inference_recipe = """
# Running in a mobile app
(TODO: pte file name generation)
The [pte file](https://huggingface.co/{quantized_model}/blob/main/qwen3-4B-INT8-INT4-1024-cxt.pte) can be run with ExecuTorch on a mobile phone.  See the [instructions](https://pytorch.org/executorch/main/llm/llama-demo-ios.html) for doing this in iOS.
On iPhone 15 Pro, the model runs at (to be filled) tokens/sec and uses (to be filled) Mb of memory.

TODO: attach image
"""
_untie_embedding_recipe = """
## Untie Embedding Weights
We want to quantize the embedding and lm_head differently.  Since those layers are tied, we first need to untie the model:

```Py
from transformers import (
  AutoModelForCausalLM,
  AutoProcessor,
  AutoTokenizer,
)
import torch

model_id = "{base_model}"
untied_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(untied_model)
from transformers.modeling_utils import find_tied_parameters
print("tied weights:", find_tied_parameters(untied_model))
if getattr(untied_model.config.get_text_config(decoder=True), "tie_word_embeddings"):
    setattr(untied_model.config.get_text_config(decoder=True), "tie_word_embeddings", False)

untied_model._tied_weights_keys = []
untied_model.lm_head.weight = torch.nn.Parameter(untied_model.lm_head.weight.clone())

print("tied weights:", find_tied_parameters(untied_model))

USER_ID = "YOUR_USER_ID"
MODEL_NAME = model_id.split("/")[-1]
save_to = f"{{USER_ID}}/{{MODEL_NAME}}-untied-weights"

# save locally (we use this in the recipe)
save_to_local_path = f"{{MODEL_NAME}}-untied-weights"
untied_model.save_pretrained(save_to_local_path)
tokenizer.save_pretrained(save_to_local_path)


# or push to hub
untied_model.push_to_hub(save_to)
tokenizer.push_to_hub(save_to)
```

Note: to `push_to_hub` you need to run
```Shell
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
and use a token with write access, from https://huggingface.co/settings/tokens

## Quantization
"""

_mobile_export_to_executorch = """
# Exporting to ExecuTorch

We can run the quantized model on a mobile phone using [ExecuTorch](https://github.com/pytorch/executorch).
Once ExecuTorch is [set-up](https://pytorch.org/executorch/main/getting-started.html), exporting and running the model on device is a breeze.

We first convert the [quantized checkpoint](https://huggingface.co/{quantized_model}/blob/main/pytorch_model.bin) to one ExecuTorch's LLM export script expects by renaming some of the checkpoint keys.
The following script does this for you.  We have uploaded the converted checkpoint [pytorch_model_converted.bin](https://huggingface.co/{quantized_model}/blob/main/pytorch_model_converted.bin) for convenience.
```Shell
python -m executorch.examples.models.qwen3.convert_weights $(huggingface-cli download {quantized_model}) pytorch_model_converted.bin
```

Once the checkpoint is converted, we can export to ExecuTorch's pte format with the XNNPACK delegate.
The below command exports with a max_seq_length/max_context_length of 1024, but it can be changed as desired.

(TODO: pte file name, model config path, model name auto generation)
```Shell
PARAMS="executorch/examples/models/qwen3/4b_config.json"
python -m executorch.examples.models.llama.export_llama \
  --model "qwen3-4b" \
  --checkpoint "pytorch_model_converted.bin" \
  --params "$PARAMS" \
  -kv \
  --use_sdpa_with_kv_cache \
  -d fp32
  -X \
  --metadata '{{"get_bos_id":199999, "get_eos_ids":[200020,199999]}}' \
  --max_seq_length 1024 \
  --max_context_length 1024 \
  --output_name="qwen3-4b-INT8-INT4-1024-cxt.pte"
```

After that you can run the model in a mobile app (see [Running in a mobile app](#running-in-a-mobile-app)).
"""


def quantize_and_upload(model_id, quant):
    _int8_int4_linear_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
        weight_scale_dtype=torch.bfloat16,
    )
    _int8_int4_embedding_config = IntxWeightOnlyConfig(
        weight_dtype=torch.int8,
        granularity=PerAxis(0),
    )
    quant_to_config = {
        "FP8": Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        "INT4": Int4WeightOnlyConfig(group_size=128),
        "INT8-INT4": ModuleFqnToConfig(
            {
                "_default": _int8_int4_linear_config,
                "model.embed_tokens": _int8_int4_embedding_config,
            }
        ),
    }

    quant_to_quant_code = {
        "FP8": _fp8_quant_code,
        "INT4": _int4_quant_code,
        "INT8-INT4": _int8_int4_quant_code,
    }

    assert quant in quant_to_config, f"Unsupported quant option: {quant}"
    quant_config = quant_to_config[quant]

    model_to_quantize = model_id
    if quant == "INT8-INT4":
        model_to_quantize = _untie_weights_and_save_locally(model_to_quantize)

    quantization_config = TorchAoConfig(quant_type=quant_config)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_to_quantize,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    username = _get_username()

    MODEL_NAME = model_id.split("/")[-1]
    save_to = f"{username}/{MODEL_NAME}-{quant}"
    untied_model_path = 'f"{{MODEL_NAME}}-untied-weights"'
    is_mobile = quant == "INT8-INT4"
    quantized_model_id = save_to
    # model card
    content = MODEL_CARD.format(
        username=username,
        base_model=model_id,
        quantized_model=quantized_model_id,
        model_type=quantized_model.config.model_type,
        quant=quant,
        quant_code=quant_to_quant_code[quant],
        # server specific recipes
        server_inference_recipe=""
        if is_mobile
        else _server_inference_recipe.format(quantized_model=quantized_model_id),
        server_peak_memory_usage=""
        if is_mobile
        else _server_peak_memory_usage.format(
            base_model=model_id, quantized_model=quantized_model_id
        ),
        server_model_performance=""
        if is_mobile
        else _server_model_performance.format(
            base_model=model_id, quantized_model=quantized_model_id, quant=quant
        ),
        # mobile specific recipes
        untied_model=untied_model_path if is_mobile else model_id,
        untie_embedding_recipe=_untie_embedding_recipe if is_mobile else "",
        mobile_inference_recipe=_mobile_inference_recipe.format(
            quantized_model=quantized_model_id
        )
        if is_mobile
        else "",
        mobile_export_to_executorch=_mobile_export_to_executorch.format(
            quantized_model=quantized_model_id
        )
        if is_mobile
        else "",
    )
    card = ModelCard(content)

    # Push to hub
    quantized_model.push_to_hub(quantized_model_id, safe_serialization=False)
    tokenizer.push_to_hub(quantized_model_id)
    card.push_to_hub(quantized_model_id)

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
    print("Response:", output_text[0][len(prompt) :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with the specified parameters."
    )
    parser.add_argument(
        "--model_id", type=str, help="Huggingface hub model ID of the model."
    )
    parser.add_argument(
        "--quant",
        type=str,
        help="Quantization method. Options are FP8, INT4, INT8_INT4, AWQ-INT4",
    )
    args = parser.parse_args()
    quantize_and_upload(args.model_id, args.quant)
