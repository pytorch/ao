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
for i in range(calibration_limit):
    args = get_next_input() # user provided function
    input_recorder(*args) # compare to model(*args)
     # note: can do input_recorder(*args, **kwargs) if needed

# then perform GPTQ
quantizer = Int4WeightOnlyGPTQQuantizer() # quantization parameters like group_size can be set here
args = input_recorder.get_recorded_inputs() # use get_recorded_args_and_kwargs if necessary
quantizer.quantize(model, *args)
# model is now quantized and can be saved, compiled or run

args = get_next_input()
out = model(*args)
```

important notes:
1) `input_recorder`, `quantizer.quantize` and `model` all take the same type of input. If you pass in kwargs to the model like `model(*args, **kwargs)` you'll need to do `input_recorder(*args, **kwargs)` and `quantizer.quantize(model, *args, **kwargs)`
2) the GPTQ process can take a significant period of time depending on the size of the model and the size of the calibration set.
3) We currently only support int4 weight only quantization for GPTQ though this framework can be relatively easily extended to other techniques.


In many cases users use lm_eval to get calibration data. We also have an input recorder that integrates directly with lm_eval. This is equivalent to using lm_eval but setting your model to be a MultiTensorInputRecorder.

```python
from torchao._models._eval import LMEvalInputRecorder

args = (
    LMEvalInputRecorder(
        get_tokenizer(), # tokenizer
        calibration_seq_length,
        prepare_inputs_for_model, # optional function that transforms the input, e.g. constructing the indices tensor
        get_tokenizer_vocab_size(),
        pad_calibration_inputs, # boolean to allow padding
    )
    .record_inputs(
        calibration_tasks,
        calibration_limit,
    )
    .get_recorded_inputs()
)


```

## Results

We tested the GPTQ implementation using the llama model in torchao/_models/llama with 10 calibration samples from lm_eval.

| Technique:     | Llama-2-7b-chat-hf | Meta-Llama-3-8B |
|----------------|--------------------|-----------------|
| bf16           |             12.245 |           7.441 |
| int4wo-64      |             12.876 |           8.316 |
| gptq-int4wo-64 |             12.523 |           8.026 |

In practice we find that GPTQ recovers ~1/2 to 1/3 of the perplexity lost compared to quantizing directly to int4.
You can reproduce these numbers using `python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization gptq-int4wo-64 --calibration_limit 10`
