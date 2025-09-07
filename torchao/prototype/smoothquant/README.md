# SmoothQuant quantization

This is a native PyTorch implementation of the algorithm described in [this paper](https://arxiv.org/abs/2211.10438) with TorchAO Quantization APIs.

$$
Smoothing factor: s_{j} = \frac{max(|X_{j})^\alpha}{max(|W_{j}|) ^(1-\alpha)}, \ j=1, 2, \dots, C_{i}
$$

In this implementation, weights are smoothed (equalized) and quantized to int8 during quantization. Activations are smoothed and quantized to int8 at runtime. Quantization is done either dynamically or statically. For dynamic quantization, activations are quantized per token. And for static quantization, activations are quantized per tensor.

## Quick start

Run the example code with

```bash
python example.py --model <MODEL_ID>  --device <cuda/cpu>
# An example
python example.py --model meta-llama/Llama-2-7b-chat-hf
```

To save a quantized model for reuse, specify `--model_save_path`

```bash
python example.py --model <MODEL_ID> --model_save_path ./model_smoothquant.pt
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

All experiments use the `meta-llama/Llama-2-7b-chat-hf` model with max sequence length (SeqLen) 512 and calibration limit 128 on a 1xH100 80GB HBM2 instance. For comprehensive benchmarking, we compare three cases: 1. origin, 2. W8A8, 3. SmoothQuant (W8A8).

### Benchmark Results

Result shows SmoothQuant with W8A8 slightly increase perplexity, reducing latency 33.82%. Since tinygemm kernel only uses bfloat16 inputs, Tokens/sec decreases for float16 input.

| Precision dtype | Quantization | Perplexity | Tokens/sec | PPL Change | Speed Change |
|-----------|--------------|------------|------------|------------|--------------|
| bfloat16  |  -             | 6.93       | 667        |  -         |  -          |
| bfloat16* |  -             | 6.93       | 27    🐌   |  -         |  -          |
| bfloat16  | W8A8-dynamic   | 7.35       | 1,967      | +6.07%     | +33.89%     |
| bfloat16  | W8A8-dynamic** | 7.03       | **1,972**  | **+1.39%** | **+33.82%**     |
| float16   |  -             | 6.93       | 625        |  -         |  -          |
| float16   | W8A8-dynamic   | 7.29       | 523        | +5.21%     | -19.42%     |
| float16   | W8A8-dynamic** | 6.94       | 516        | **+0.21%** | -21.23%     |
| bfloat16* | W8A8-dynamic** | 6.92       | 3        🐌  | -0.18%     | -768.29%  |

> *Used with `torch.compile`, **Used with **SmoothQuant**

### Key Findings

- **Speed Improvement**: Most configurations show 35-40% speed improvement with both W8A8 and SmoothQuant-W8A8
- **Quality Trade-off**: Slight perplexity increase (~1-1.4%) in most cases
- **Compilation Impact**: Using `--compile` flag significantly degrades performance (768% slower)
- **Best Configuration**: `bfloat16` without `--compile` provides optimal balance

> Note: Unlike AWQ, this benchmark isn't computed using the script in `vllm/benchmarks` or `lm_eval`. vLLM benchmark will be introduced in foreseeable future. See https://github.com/pytorch/ao/issues/2815 for more information.
