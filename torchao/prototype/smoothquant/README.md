# SmothQuant quantization
This is a native PyTorch implementation of the algorithm described in [this paper](https://arxiv.org/abs/2211.10438).

In this implementation, weights are smoothed (equalized) and quantized to int8 during quantization. Activations are smoothed and quantized to int8 at runtime. Quantization is done either dynamically or statically. If activations are dynamically quantized, qparams (i.e., scales) are found at runtime while qparams are found during quantization for static quantization. For dynamic quantization, activations are quantized per token. And for static quantization, activations are quantized per tensor. Generally, dynamic quantization produces better accuracy while static quantization has better latency. In both cases, weights and activations are symmetrically quantized.

## Quick start
Run the example code with
```bash
python example.py -m MODLE_ID --device=<cuda or cpu> --quant-mode=<dynamic or static>
# An example
python example.py -m meta-llama/Llama-2-7b-hf --device=cuda --quant-mode=dynamic
```
To use the `torch.compile` for speedup, add `--compile`. You may want to export `TORCHINDUCTOR_FREEZING=1` for even better performance.
```bash
TORCHINDUCTOR_FREEZING=1 python example.py -m MODLE_ID --device=<cuda or cpu> --quant-mode=<dynamic or static> --compile
```
To save a quantized model for reuse, specify `--model-save-path`
```bash
python example.py -m MODLE_ID --device=<cuda or cpu> --quant-mode=<dynamic or static> --model-save-path ./quantized_model.pt
```
And load it by `--model-load-path`
```bash
python example.py -m MODLE_ID --device=<cuda or cpu> --quant-mode=<dynamic or static> --model-load-path ./quantized_model.pt
```


## Usage of API
The following APIs are provided:
- insert_smooth_quant_observer_
- smooth_quant
- save_smooth_quant_recipe (advanced)
- load_smooth_quant_recipe (advanced)

`insert_smooth_quant_observer_` inserts observers into the model to be quantized. For example:
```python
insert_smooth_quant_observer_(model, alpha=0.5, quant_mode="dynamic")
```
After insertion, run the model for calibration on a certain dataset or (advanced) load a recipe.

`smooth_quant` applies SmoothQuant to each linear layer of the model. Use it by calling `torchao.quantization.quantize_`. For example:
```python
from torchao.prototype.smoothquant import SmoothQuantObservedLinear
is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
torchao.quantization.quantize_(model, smooth_quant(), is_observed_linear)
```
`is_observed_linear` is a filter so that we only quantize observed linear layers.

(Advanced) `save_smooth_quant_recipe` and `load_smooth_quant_recipe` saves or loads a recipe for a model.

A recipe contains smoothing factors and quantization parameters of weights and activation for all linear layers that are to be quantized. For advanced users, these parameters can be saved and modified somehow to produce better accuray, e.g., different alpha for different layers. Users can even leave some linear layers unquantized by deleting these layers in the recipe. Such modifications can be published as a recipe. By loading the recipe, it can be reused and calibration is no longer needed.

To save a recipe, users should insert observers and run calibration first. For example,
```python
insert_smooth_quant_observer_(model, alpha=0.5, quant_mode="dynamic")
for data in dataset_for_calibration:
    model(data)
save_smooth_quant_recipe(model, "./smooth_quant_recipe.json")
```
To load a recipe, users should insert observers first. For example,
```python
insert_smooth_quant_observer_(model)
load_smooth_quant_recipe(model, "./smooth_quant_recipe.json")
```

## Benchmark
Running the example with `torch.compile` on a NVIDIA A10G GPU.
### meta-llama/Llama-2-7b-hf
Perplexity
| Quant Method | alpha=0.25 | alpha=0.5 | alpha=0.75 | alpha=None* |
|-|-|-|-|-|
| Dynamic | 8.1872 | 7.4257 | 7.2518 | 7.5509 |
| Static | 43.8051 | 11.2984 | 7.5791 | 19.5050 |

Note*: Conventional quantization without SmoothQuant

### meta-llama/Meta-Llama-3-8B
Perplexity
| Quant Method | alpha=0.25 | alpha=0.5 | alpha=0.75 | alpha=None* |
|-|-|-|-|-|
| Dynamic | 21.2475 | 8.8288 | 9.6514 | 8.3574 |
| Static | 301.7118 | 18.0617 | 10.8343 | 278.9819 |

Note*: Conventional quantization without SmoothQuant

### Test method
**Commands**
```bash
# dynamic quant
TORCHINDUCTOR_FREEZING=1 python example.py -m <model_id> --device=cuda --quant-mode=dynamic --compile
# static quant
TORCHINDUCTOR_FREEZING=1 python example.py -m <model_id> --device=cuda --quant-mode=static --compile
```
Use `--alpha` to specify the alpha parameter. Add `--disable-smooth-quant` to run quantization without SmoothQuant.

**Environment**
- AWS g5.12xlarge instance
- torch==2.6.0.dev20241017+cu124
- python==3.12.6
