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
- insert_smooth_quant_observer
- smooth_quant
- save_smooth_quant_recipe (advanced)
- load_smooth_quant_recipe (advanced)

`insert_smooth_quant_observer` inserts observers into the model to be quantized. For example:
```python
insert_smooth_quant_observer(model, alpha=0.5, quant_mode="dynamic")
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

A recipe contains smoothing factors and quantization parameters of weights and activation for all linear layers that are to be quantized. For advanced users, these parameters can be saved and modified somehow to produce better accuray, e.g., different alpha for different layres. Users can even leave some linear layers unquantized by deleting these layers in the recipe. Such modifications can be published as a recipe. By loading the recipe, it can be reused and calibration is no longer needed.

To save a recipe, users should insert observers and run calibration first. For example,
```python
insert_smooth_quant_observer(model, alpha=0.5, quant_mode="dynamic")
for data in dataset:
    model(data)
save_smooth_quant_recipe(model, "./smooth_quant_recipe.json")
```
To load a recipe, users should insert observers first. For example,
```python
insert_smooth_quant_observer(model)
load_smooth_quant_recipe(model, "./smooth_quant_recipe.json")
```

## Benchmark
Running the example case with `torch.compile` on a NVIDIA A10 GPU.
### meta-llama/Llama-2-7b-hf
| Quant Method | Perplexity | Latency (sec/it) |
|-|-|-|
| SmoothQuant dynamic | 7.4318 | 0.6874 |
| SmoothQuant static | 424.995 | 0.4560 |
| AWQ UINT4 Group size 128 | 7.4837 | 1.0018 |

### meta-llama/Meta-Llama-3-8B
| Quant Method | Perplexity | Latency (sec/it) |
|-|-|-|
| SmoothQuant dynamic | 8.8274 | 0.9008 |
| SmoothQuant static | 124.2236 | 0.5537 |
| AWQ UINT4 Group size 128 | 8.7087 | 1.2868 |

Note:
1. Static quantization needs tuning on recipe and alpha to get good accuracy. So, the data here is for reference only.
2. AWQ's calibration runs on `mit-han-lab/pile-val-backup`, validation split while SmoothQuant's calibration runs on `wikitext/wikitext-2-raw-v1`, test split.
