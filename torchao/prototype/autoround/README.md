# Auto-Round

Auto-Round is an advanced quantization algorithm designed for low-bit LLM inference. It leverages [sign gradient descent](https://arxiv.org/abs/1905.12938) to fine-tune rounding values and minmax values of weights. This approach competes impressively with recent methods without introducing any additional inference overhead while using low tuning costs. This module provides the end-to-end examples to quantize floating-point models to low-bit and integration with torchao's `quantize_` API and low-bit kernels.

## Usage

### Quick Start

```python
python autoround_llm.py -m /model/name/or/path
```

This script allows you to apply `Auto-Round` on a given model directly, more configurations options are list below:

| Argument                           |Default                     | Description                                                       |
|------------------------------------|----------------------------|-------------------------------------------------------------------|
| `model_name_or_path`               |`"facebook/opt-125m"`       | Pretrained model name or path                                     |
| `dataset_name`                     | `"NeelNanda/pile-10k"`     | Dataset name for calibration                                      |
| `iters`                            | 200                        | Number of steps for optimizing each block                         |
| `bits`                             | 4                          | Number of bits for quantization                                   |
| `batch_size`                       | 8                          | Batch size for calibration                                        |
| `nsamples`                         | 128                        | Number of samples for calibration process                         |
| `seqlen`                           | 2048                       | Sequence length for each samples                                  |
| `group_size`                       | 128                        | Group size for quantization                                       |
| `gradient_accumulate_steps`        | 1                          | Number of steps for accumulating gradients <br> before performing the backward pass |
| `quant_lm_head`                    | `False`                    | Whether to quantize the `lm_head`                                 |
| `use_optimized_layer_output`       | `False`                    | Whether to use optimized layer output as input for the next layer |
| `compile_optimization_process`     | `False`                    | Whether to compile the optimization process                       |
| `model_device`                     | `"cuda"`                   | Device for loading the float model (choices: `cpu`, `cuda`)       |


> [!NOTE]
> Before running, ensure you have installed the `auto-round` with `pip install -r requirements.txt`.


### Detailed Usage

`Auto-Round` is a calibration-based quantization algorithm. The flow involves three main steps: 1) insert hooks to the modules you want to quantize, 2) Wrap the calibration data with `MultiTensor` and run the model, 3) Replace the optimized weight with `AffineQuantizedTensor` to select the appropriate low-bit kernel.

> [!NOTE]
> To learn more about the flow and `MultiTensor`, please refer to [this example](https://github.com/pytorch/ao/blob/main/tutorials/calibration_flow/gptq_like.py).

#### Step 1: Prepare the Model
```python
model = ...  # Load your model
model_device = next(model.parameters()).device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a function to identify target modules for quantization.
# For example, to apply Auto-Round to all decoder layers and the `lm-head` in a Llama model:
decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
# Prepare the model for Auto-Round
from torchao.prototype.autoround.core import prepare_model_for_applying_auto_round_

prepare_model_for_applying_auto_round_(
    model,
    is_target_module=is_target_module,
    bits=4,
    group_size=128,
    iters=200,
    device=device,
)
```
> [!NOTE]
> To avoid OOM issues, load the model on CPU, and set `device` to `'cuda'`.

#### Step 2: Apply Optimization
Wrap all inputs as a `MultiTensor` to track all calibration data for optimized modules:

```python
input_ids_lst = []
for data in dataloader:
    input_ids_lst.append(data["input_ids"].to(model_device))

multi_t_input_ids = MultiTensor(input_ids_lst)
# The optimization is applied during the forward pass
out = model(multi_t_input_ids)
```
#### Step 3: Finalize Quantization
After obtaining optimized `zero_point` and `scale` values, create the `AffineQuantizedTensor`
for each target weight to select the right low-bits kernel.

```python
from torchao.prototype.autoround.core import apply_auto_round

quantize_(model, apply_auto_round(), is_target_module)
```

## End-to-End Results
### [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
|                  | Avg.   | Mmlu   | Piqa   | Winogrande | Hellaswag | Lambada_openai |
| ---------------- | ------ | ------ | ------ | ---------- | --------- | -------------- |
| bf16             | 0.7080 | 0.6783 | 0.8003 | 0.7403     | 0.5910    | 0.7303         |
| torchao-int4wo   | 0.6883 | 0.6363 | 0.7938 | 0.7348     | 0.5784    | 0.6980         |
| autoround-4bit   | 0.6996 | 0.6669 | 0.7916 | 0.7285     | 0.5846    | 0.7262         |
| autoround-4bit*  | 0.7010 | 0.6621 | 0.7976 | 0.7316     | 0.5847    | 0.7291         |

### [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
|                  | Avg.   | Mmlu   | Piqa   | Winogrande | Hellaswag | Lambada_openai |
| ---------------- | ------ | ------ | ------ | ---------- | --------- | -------------- |
| bf16             | 0.6881 | 0.6389 | 0.7840 | 0.7222     | 0.5772    | 0.7184         |
| torchao-int4wo   | 0.6728 | 0.5939 | 0.7737 | 0.7222     | 0.5612    | 0.7132         |
| autoround-4bit   | 0.6796 | 0.6237 | 0.7758 | 0.7198     | 0.5664    | 0.7122         |
| autoround-4bit*  | 0.6827 | 0.6273 | 0.7737 | 0.7348     | 0.5657    | 0.7120         |


### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
|                  | Avg.   | Mmlu   | Piqa   | Winogrande | Hellaswag | Lambada_openai |
| ---------------- | ------ | ------ | ------ | ---------- | --------- | -------------- |
| bf16             | 0.6347 | 0.4647 | 0.7644 | 0.6606     | 0.5770    | 0.7070         |
| torchao-int4wo   | 0.6252 | 0.4427 | 0.7617 | 0.6654     | 0.5674    | 0.6889         |
| autoround-4bit   | 0.6311 | 0.4548 | 0.7606 | 0.6614     | 0.5717    | 0.7072         |
| autoround-4bit*  | 0.6338 | 0.4566 | 0.7661 | 0.6646     | 0.5688    | 0.7130         |

> [!NOTE]
> - `torchao-int4wo` quantizes the model to 4 bits with a group size of 128 (`Int4WeightOnlyConfig(group_size=128, version=1)`) while leaving the `lm-head` unquantized. <br>
> - `auto-round-4bit` uses the deafult configuration from [quick start](#quick-start). <br>
> - `auto-round-4bit*` follows the same settings as `auto-round-4bit`, but with `gradient_accumulate_steps=2` and `batch_size=4`, which accumulating two batches(4 samples per batch) before performing the backward pass. <br>
> - To reproduce results, run `eval_autoround.py` with `AO_USE_DETERMINISTIC_ALGORITHMS=1`.


## Credits

- Paper: https://arxiv.org/abs/2309.05516
- Authors: [Intel® Neural Compressor Team](https://github.com/intel/neural-compressor)
