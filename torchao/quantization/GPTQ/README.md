# GPTQ

GPTQ is a quantization technique that improves accuracy for various forms of quantization as introduced in the paper: https://arxiv.org/abs/2210.17323

In general GPTQ requires a model, a quantization technique and calibration data. GPTQ then optimizes the quantization parameters and quantized weights so they are more accurate accross the calibration data.

## API

The api for this technique is as follows:


```python
from torchao.quantization import MultiTensorInputRecorder, Int4WeightOnlyGPTQQuantizer

model = get_model() # user provided function

# first gather inputs
input_recorder = MultiTensorInputRecorder()

# Use lm-eval for calibration
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

original_forward = model.forward
model.forward = lambda *args, **kwargs: (input_recorder(*args, **kwargs), original_forward(*args, **kwargs))[1]
lm_obj = HFLM(pretrained=model, tokenizer=tokenizer)
simple_evaluate(model=lm_obj, tasks=["wikitext"], num_fewshot=0, limit=10)
model.forward = original_forward

# Perform GPTQ with collected inputs
quantizer = Int4WeightOnlyGPTQQuantizer(groupsize=64)
args, kwargs = input_recorder.get_recorded_args_and_kwargs()
quantizer.quantize(model, *args, **kwargs)
```

Important notes:
1) `input_recorder`, `quantizer.quantize` and `model` all take the same type of input. If you pass kwargs to the model, use them consistently across all three.
2) the GPTQ process can take a significant period of time depending on the size of the model and the size of the calibration set.
3) We currently only support int4 weight only quantization for GPTQ though this framework can be relatively easily extended to other techniques.

## Results

We tested the GPTQ implementation using the llama model in torchao/_models/llama with 10 calibration samples from lm_eval.

| Technique:     | Llama-2-7b-chat-hf | Meta-Llama-3-8B |
|----------------|--------------------|-----------------|
| bf16           |             12.245 |           7.441 |
| int4wo-64      |             12.876 |           8.316 |
| gptq-int4wo-64 |             12.523 |           8.026 |

In practice we find that GPTQ recovers ~1/2 to 1/3 of the perplexity lost compared to quantizing directly to int4.
You can reproduce these numbers using `python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization gptq-int4wo-64 --calibration_limit 10`
