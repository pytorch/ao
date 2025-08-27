(torchao_hf_integration)=
# Integration with HuggingFace: Architecture and Usage Guide

```{contents}
:local:
:depth: 2
```

(configuration-system)=
## Configuration System

(huggingface-model-configuration)=
### 1. HuggingFace Model Configuration

TorchAO quantization is configured through the model's `config.json` file:

```json
{
  "model_type": "llama",
  "quant_type": {
    "default": {
      "_type": "Int4WeightOnlyConfig",
      "_data": {
        "group_size": 128,
        "use_hqq": true
      }
    }
  }
}
```

(torchao-configuration-classes)=
### 2. TorchAO Configuration Classes

All quantization methods inherit from `AOBaseConfig`:

```python
from torchao.core.config import AOBaseConfig
from torchao.quantization import Int4WeightOnlyConfig

# Example configuration
config = Int4WeightOnlyConfig(
    group_size=128,
    use_hqq=True,
)
assert isinstance(config, AOBaseConfig)
```

```{note}
All quantization configurations inherit from {class}`torchao.core.config.AOBaseConfig`, which provides serialization and validation capabilities.
```

(module-level-configuration)=
### 3. Module-Level Configuration

For granular control, use `ModuleFqnToConfig`:

```python
from torchao.quantization import ModuleFqnToConfig, Int4WeightOnlyConfig, Int8WeightOnlyConfig

config = ModuleFqnToConfig({
    "model.layers.0.self_attn.q_proj": Int4WeightOnlyConfig(group_size=64),
    "model.layers.0.self_attn.k_proj": Int4WeightOnlyConfig(group_size=64),
    "model.layers.0.mlp.gate_proj": Int8WeightOnlyConfig(),
    "_default": Int4WeightOnlyConfig(group_size=128)  # Default for other modules
})
```

(usage-examples)=
## Usage Examples

First, install the required packages.

```bash
pip install git+https://github.com/huggingface/transformers@main
pip install torchao
pip install torch
pip install accelerate
```

(quantizing-models-huggingface)=
### 1. Quantizing Models with HuggingFace Integration

Below is an example of using `Float8DynamicActivationInt4WeightConfig` on the Llama-3.2-1B model.

```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization import Float8DynamicActivationInt4WeightConfig

# Create quantization configuration
quantization_config = TorchAoConfig(
    quant_type=Float8DynamicActivationInt4WeightConfig(group_size=128, use_hqq=True)
)

# Load and automatically quantize the model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

After we quantize the model, we can save it.

```python
# Save quantized model (see Serialization section below for safe_serialization details)
model.push_to_hub("your-username/Llama-3.2-1B-int4", safe_serialization=False)
```

Here is another example using `Float8DynamicActivationFloat8WeightConfig` on the Phi-4-mini-instruct model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

model_id = "microsoft/Phi-4-mini-instruct"

# Create quantization configuration
quant_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and automatically quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
```

Just like the previous example, we can now save the quantized model.

```python
# Save quantized model (see Serialization section below for safe_serialization details)
USER_ID = "YOUR_USER_ID"
MODEL_NAME = model_id.split("/")[-1]
save_to = f"{USER_ID}/{MODEL_NAME}-float8dq"
quantized_model.push_to_hub(save_to, safe_serialization=False)

# Save tokenizer
tokenizer.push_to_hub(save_to)
```

```{seealso}
For more information on quantization configs, see {class}`torchao.quantization.Int4WeightOnlyConfig`, {class}`torchao.quantization.Float8DynamicActivationInt4WeightConfig`, and {class}`torchao.quantization.Int8WeightOnlyConfig`.
```

```{note}
For more information on supported quantization and sparsity configurations, see [HF-Torchao Docs](https://huggingface.co/docs/transformers/main/en/quantization/torchao).
```

(serving-with-vllm)=
### 2. Serving with VLLM

```{note}
For more information on serving and inference with VLLM, please refer to [Integration with VLLM: Architecture and Usage Guide](https://docs.pytorch.org/ao/main/torchao_vllm_integration.html) and [(Part 3) Serving on vLLM, SGLang, ExecuTorch](https://docs.pytorch.org/ao/main/serving.html) for a full end-to-end tutorial.
```

(Inference-with-HuggingFace-Transformers)=
### 3. Inference with HuggingFace Transformers

Recall how we can quantize models using HuggingFace Transformers.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)

model_path = "pytorch/Phi-4-mini-instruct-float8dq"

# Load and automatically quantize the model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

Now we can use the model for inference.

```python
from transformers import pipeline

# Simulate conversation between user and assistant
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

# Initialize HuggingFace pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Generate output
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

```{seealso}
For more examples and recommended quantization methods based on different hardwares (i.e. A100 GPU, H100 GPU, CPU), see [HF-Torchao Docs (Quantization Examples)](https://huggingface.co/docs/transformers/main/en/quantization/torchao#quantization-examples).
```

(Inference-with-HuggingFace-Diffuser)=
### 4. Inference with HuggingFace Diffuser

```bash
pip install git+https://github.com/huggingface/diffusers@main
```

Below is an example of how we can integrate with HuggingFace Diffusers.

```python
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig

model_id = "black-forest-labs/Flux.1-Dev"
dtype = torch.bfloat16

quantization_config = TorchAoConfig("int8wo")
transformer = FluxTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=dtype,
)
pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=dtype,
)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
image.save("output.png")
```

We can also use [torch.compile]() to speed up inference by adding this line of code after initializing the transformer.
```python
# In the above code, add the following after initializing the transformer
transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)
```

```{seealso}
Please refer to [HF-TorchAO-Diffuser Docs](https://huggingface.co/docs/diffusers/en/quantization/torchao) for more examples and benchmarking results.
```

```{note}
Refer [here](https://github.com/huggingface/diffusers/pull/10009) for time and memory results from a single H100 GPU.
```

(Supported-Quantization-Types)=
## Supported Quantization Types

Weight-only quantization stores the model weights in a specific low-bit data type but performs computation with a higher-precision data type, like `bfloat16`. This lowers the memory requirements from model weights but retains the memory peaks for activation computation.

Dynamic activation quantization stores the model weights in a low-bit dtype, while also quantizing the activations on-the-fly to save additional memory. This lowers the memory requirements from model weights, while also lowering the memory overhead from activation computations. However, this may come at a quality tradeoff at times, so it is recommended to test different models thoroughly.

Below are the supported quantization types.

| Category | Full Function Names
|:---------|:-------------------|
| Integer quantization    | `Int8DynamicActivationInt4WeightConfig`<br>`Int8DynamicActivationIntxWeightConfig`<br>`Int4DynamicActivationInt4WeightConfig`<br> `Int4WeightOnlyConfig`<br> `Int8WeightOnlyConfig`<br>`Int8DynamicActivationInt8WeightConfig`|
| Floating point quantization   | `Float8DynamicActivationInt4WeightConfig`<br>`Float8WeightOnlyConfig`<br>`Float8DynamicActivationFloat8WeightConfig`<br> `Float8DynamicActivationFloat8SemiSparseWeightConfig`<br> `Float8StaticActivationFloat8WeightConfig` |
| Integer X-bit quantization | `IntxWeightOnlyConfig` |
| Floating point X-bit quantization   | `FPXWeightOnlyConfig` |
| Unsigned Integer Quanization   | `GemliteUIntXWeightOnlyConfig` <br> `UIntXWeightOnlyConfig`|

```{note}
For full definitions of the above types, please see [here](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py)
```


(Serialization)=
## Serialization

To serialize a quantized model in a given dtype, first load the model with the desired quantization dtype and then save it using the `save_pretrained()` method.


**Using Transformers**:
```python
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

quant_config = Int8WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config
)
# save the quantized model
output_dir = "llama-3.1-8b-torchao-int8"
quantized_model.save_pretrained(output_dir, safe_serialization=False)
```

**Using Diffusers**:
```python
import torch
from diffusers import AutoModel, TorchAoConfig

quantization_config = TorchAoConfig("int8wo")
transformer = AutoModel.from_pretrained(
    "black-forest-labs/Flux.1-Dev",
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)
transformer.save_pretrained("/path/to/flux_int8wo", safe_serialization=False)
```


To load a serialized quantized model, use the `from_pretrained()` method.

**Using Transformers**:
```python
# reload the quantized model
reloaded_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(reloaded_model.device.type)

output = reloaded_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Using Diffusers**:
```python
import torch
from diffusers import FluxPipeline, AutoModel

transformer = AutoModel.from_pretrained("/path/to/flux_int8wo", torch_dtype=torch.bfloat16, use_safetensors=False)
pipe = FluxPipeline.from_pretrained("black-forest-labs/Flux.1-Dev", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.0).images[0]
image.save("output.png")
```

(safetensors-support)=
### SafeTensors Support

**Current Status**: TorchAO quantized models cannot yet be serialized with safetensors due to tensor subclass limitations. When saving quantized models, you must use `safe_serialization=False`.

```python
# don't serialize model with Safetensors
output_dir = "llama3-8b-int4wo-128"
quantized_model.save_pretrained("llama3-8b-int4wo-128", safe_serialization=False)
```

**Workaround**: For production use, save models with `safe_serialization=False` when pushing to HuggingFace Hub.

**Future Work**: The TorchAO team is actively working on safetensors support for tensor subclasses. Track progress [here](https://github.com/pytorch/ao/issues/2338) and [here](https://github.com/pytorch/ao/pull/2881)
