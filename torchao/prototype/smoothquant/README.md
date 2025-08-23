# SmoothQuant quantization

This is a native PyTorch implementation of the algorithm described in [this paper](https://arxiv.org/abs/2211.10438) with TorchAO Quantization APIs.

$$
Smoothing factor: s_{j} = \frac{max(|X_{j})^\alpha}{max(|W_{j}|) ^(1-\alpha)}, \ j=1, 2, \dots, C_{i}
$$

In this implementation, weights are smoothed (equalized) and quantized to int8 during quantization. Activations are smoothed and quantized to int8 at runtime. Quantization is done either dynamically or statically. For dynamic quantization, activations are quantized per token. And for static quantization, activations are quantized per tensor.

## Quick start

Run the example code with

```bash
python example.py --repo <MODEL_ID>  --device <cuda/cpu>
# An example
python example.py --repo meta-llama/Llama-2-7b-chat-hf  --alpha 0.8
```

To use the `torch.compile` for speedup, add `--compile`.
```bash
python example.py --repo <MODEL_ID>  --compile
```

To save a quantized model for reuse, specify `--model_save_path`

```bash
python example.py --repo <MODEL_ID> --model_save_path ./quantized_model.pt
```

## Usage of API

`SmoothQuantConfig` configures applying SmoothQuant to each linear layer of the model. Use it with `torchao.quantization.quantize_`. For example:

```python
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.prototype.smoothquant.core import SmoothQuantStep
from torchao.quantization import quantize_
from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig

# Step 1: Prepare - insert observers
quant_config = SmoothQuantConfig(
    base_config=Int8DynamicActivationInt8WeightConfig(),
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

## Benchmarks

Evaluation perplexity numbers were calculated using the script in `smoothquant/example.py`. For Llama-2-7b-chat-hf, performance benchmarks were calculated using the `torchao/_models/llama/generate.py` script and run on a 1xA100 48GB PCIe interface.

| Model              | Quantization | Tokens/sec | Throughput (GB/sec) | Peak Mem (GB) | Model Size |Perplexity |
|--------------------|--------------|------------|---------------------|---------------|---------------|-----------------|
| Llama-2-7b-chat-hf | int8dq | 539 | 1,331 | 14.3 | 7.97 | 6.98 |
|                    | int8wo |  -  |  -  |  -  |  -  |    |
|                    | float8wo |  -  |  -  |  -  |  -  |    |
|                    | float8dq |  -  |  -  |  -  |  -  |    |

> Note: vLLM becnchmark will be introduced in forsesable future. See https://github.com/pytorch/ao/issues/2815 for more information.
