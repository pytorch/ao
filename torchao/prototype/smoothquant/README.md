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

All experiments use the `meta-llama/Llama-2-7b-chat-hf` model with max sequence length (SeqLen) 512 and calibration limit 128 on a 1xH100 80GB HBM2 instance. For comprehensive benchmarking, we compare three cases: 1. origin, 2. W4A8-dynamic, 3. SmoothQuant (W4A8-dynamic)

### Benchmark Results

| Precision | Quantization | Perplexity | Tokens/sec | PPL Change | Speed Change |
|-----------|--------------|------------|------------|------------|--------------|
| float16   | -                  | 6.93       | 625        | -          | -           |
| bfloat16  | -                  | 6.93       | 667        | -          | -           |
| bfloat16* | -                  | 6.93       | 27    🐌   | -          | -           |
| float16   | A4W8-dynamic       | 7.35       | 1,016      | +6.07%     | +39.51%   |
| bfloat16  | A4W8-dynamic       | 7.29       | 981        | +5.21%     | +37.46%   |
| float16   | A4W8-dynamic**     | 7.03       | 1,003      | **+0.83%** | +39.39%   |
| bfloat16  | A4W8-dynamic**     | 7.03       | 1,108      | **+1.39%** | +41.07%   |
| bfloat16* | A4W8-dynamic**     | 6.92       | 3          | -0.18%     | -768.29% 🐌 |

> *Used with `torch.compile`
> **Used with SmoothQuant

### Key Findings

- **Speed Improvement**: Most configurations show 35-40% speed improvement with SmoothQuant
- **Quality Trade-off**: Slight perplexity increase (~1-1.4%) in most cases
- **Compilation Impact**: Using `--compile` flag significantly degrades performance (768% slower)
- **Best Configuration**: `bfloat16` without `--compile` provides optimal balance

> Note: Unlike AWQ, this benchmark isn't computed using the script in `vllm/benchmarks` or `lm_eval`. vLLM benchmark will be introduced in foreseeable future. See https://github.com/pytorch/ao/issues/2815 for more information.
