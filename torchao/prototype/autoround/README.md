# Auto-Round

Auto-Round is an advanced quantization algorithm designed for low-bit LLM inference. It leverages [sign gradient descent](https://arxiv.org/abs/1905.12938) to fine-tune rounding values and minmax values of weights. This approach competes impressively with recent methods without introducing any additional inference overhead while using low tuning costs. This module provides the end-to-end examples to quantize floating-point models to low-bit and integration with torchao's `quantize_` API and low-bit kernels.

## Usage

### Quick Start

```python
python autoround_llm.py -m /model/name/or/path
```


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
|                 | Avg.    | Mmlu   | Piqa   | Winogrande | Hellaswag | Lambada_openai |
| --------------  | ------- | ------ | ------ | ---------- | --------- | -------------- |
| bf16            | 0.7080  | 0.6783 | 0.8003 | 0.7403     | 0.5910    | 0.7303         |
| auto-round-4bit | 0.6988  | 0.6533 | 0.7949 | 0.7372     | 0.5837    | 0.7250         |
| torchao-int4wo  | 0.6883  | 0.6363 | 0.7938 | 0.7348     | 0.5784    | 0.6980         |

### [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
|                 | Avg.    | Mmlu   | Piqa   | Winogrande | Hellaswag | Lambada_openai |
| --------------  | ------- | ------ | ------ | ---------- | --------- | -------------- |
| bf16            | 0.6881  | 0.6389 | 0.7840 | 0.7222     | 0.5772    | 0.7184         |
| auto-round-4bit | 0.6818  | 0.6232 | 0.7862 | 0.7230     | 0.5661    | 0.7105         |
| torchao-int4wo  | 0.6728  | 0.5939 | 0.7737 | 0.7222     | 0.5612    | 0.7132         |


### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
|                 | Avg.    | Mmlu   | Piqa   | Winogrande | Hellaswag | Lambada_openai |
| --------------  | ------- | ------ | ------ | ---------- | --------- | -------------- |
| bf16            | 0.6347  | 0.4647 | 0.7644 | 0.6606     | 0.577     | 0.7070         |
| auto-round-4bit | 0.6327  | 0.4534 | 0.7590 | 0.6661     | 0.5706    | 0.7143         |
| torchao-int4wo  | 0.6252  | 0.4427 | 0.7617 | 0.6654     | 0.5674    | 0.6889         |

> [!NOTE]
> - `auto-round-4bit` represents the following configuration: `bits=4`, `iters=200`, `seqlen=2048`, `train_bs=8`, `group_size=128`, and `quant_lm_head=False`. <br>
> - `torchao-int4wo` represents `int4_weight_only(group_size=128)` and `quant_lm_head=False`.
> - If the model includes operations without a deterministic implementation (such as Flash Attention), the results may differ slightly.


## Credits

- Paper: https://arxiv.org/abs/2309.05516
- Authors: [Intel® Neural Compressor Team](https://github.com/intel/neural-compressor)
