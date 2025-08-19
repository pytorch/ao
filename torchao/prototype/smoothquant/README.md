# SmoothQuant quantization
This is a native PyTorch implementation of the algorithm described in [this paper](https://arxiv.org/abs/2211.10438).

In this implementation, weights are smoothed (equalized) and quantized to int8 during quantization. Activations are smoothed and quantized to int8 at runtime. Quantization is done either dynamically or statically. If activations are dynamically quantized, qparams (i.e., scales) are found at runtime while qparams are found during quantization for static quantization. For dynamic quantization, activations are quantized per token. And for static quantization, activations are quantized per tensor. Generally, dynamic quantization produces better accuracy while static quantization has better latency. In both cases, weights and activations are symmetrically quantized.

## Quick start
Run the example code with
```bash
python example.py --repo MODEL_ID --quant smoothquant --device <cuda or cpu>
# An example
python example.py --repo meta-llama/Llama-2-7b-hf --quant smoothquant --device cuda
```

To use the `torch.compile` for speedup, add `--compile`.
```bash
python example.py --repo MODEL_ID --quant smoothquant --device <cuda or cpu> --compile
```

To save a quantized model for reuse, specify `--model_save_path`
```bash
python example.py --repo MODEL_ID --quant smoothquant --device <cuda or cpu> --model_save_path ./quantized_model.pt
```

For different alpha values, use `smoothquant-0.7` format:
```bash
python example.py --repo MODEL_ID --quant smoothquant-0.7 --device cuda
```


## Usage of API
The following APIs are provided:
- SmoothQuantConfig
- SmoothQuantStep
- quantize_

`SmoothQuantConfig` configures applying SmoothQuant to each linear layer of the model. Use it with `torchao.quantization.quantize_`. For example:
```python
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.prototype.smoothquant.core import SmoothQuantStep
from torchao.quantization import quantize_
from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig

# Step 1: Prepare - insert observers
base_config = Int8DynamicActivationInt8WeightConfig()
quant_config = SmoothQuantConfig(
    base_config=base_config,
    step=SmoothQuantStep.PREPARE,
    alpha=0.5,
)
quantize_(model, quant_config)

# Step 2: Calibration
for data in calibration_dataset:
    model(data)

# Step 3: Convert
quant_config.step = SmoothQuantStep.CONVERT
quantize_(model, quant_config)
```

## Benchmark
Running the example with `torch.compile` on a NVIDIA A10G GPU.
### meta-llama/Llama-2-7b-hf
Perplexity
| Quant Method | alpha=0.25 | alpha=0.5 | alpha=0.75 | alpha=None* |
|-|-|-|-|-|
| SmoothQuant | - | - | - | - |

Note*: Conventional quantization without SmoothQuant

### meta-llama/Meta-Llama-3-8B
Perplexity
| Quant Method | alpha=0.25 | alpha=0.5 | alpha=0.75 | alpha=None* |
|-|-|-|-|-|
| SmoothQuant | - | - | - | - |

**Environment**
- AWS g5.12xlarge instance
- torch==2.6.0.dev20241017+cu124
- python==3.12.6
