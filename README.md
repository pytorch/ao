# torchao: PyTorch Architecture Optimization

**Note: This repository is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an github issue**

The `torchao` package allows you to quantize and prune your models using native PyTorch.

The repo hosts both
1. lower precision [dtypes](./torchao/dtypes) such as nf4, uint4
2. Quantization [algorithms](./torchao/quantization) such as dynamic quant, smoothquant
3. Sparsity [algorithms](./torchao/sparsity) such as Wanda

## Success stories
Our kernels have has been used to achieve SOTA inference performance on

1. Image segmentation modelss with [sam-fast](pytorch.org/blog/accelerating-generative-ai)
2. Language models with [gpt-fast](pytorch.org/blog/accelerating-generative-ai-2)
3. Diffusion models with [sd-fast](pytorch.org/blog/accelerating-generative-ai-3)


## Installation

**Note: this library makes liberal use of several new features in pytorch, its recommended to use it with the current pytorch nightly if you want full feature coverage. If not, the subclass APIs may not work, though the module swap api's will still work.**

1. From PyPI:
```Shell
pip install torchao
```

2. From Source:

```Shell
git clone https://github.com/pytorch-labs/ao
cd ao
pip install -e .
```

## Examples

Typically quantization algorithms will have different schemes for how the activation and weights are quantized so A16W8 for instance means the activations are quantized to 16 bits wheras the weights are quantized to 8 bits. Trying out different quantization schemes in `torchao` is generally a 1 line change.

### Autoquantization

The `autoquant` api can be used to quickly and accurately quantize your model. When used as in the example below, the api first identifies the shapes
of the activations that the different linear layers see, it then benchmarks these shapes across different types of quantized and non-quantized layers in order to pick the fastest one, attempting to take into account fusions where possible. Finally once the best class is found for each layer, it swaps the linear. Currently this api chooses between no quantization, int8 dynamic quantization and int8 weight only quantization for each layer.

```python
import torch
import torchao

# inductor settings which improve torch.compile performance for quantized modules
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True

# Plug in your model and example input
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
input = torch.randn(32,32, dtype=torch.bfloat16, device='cuda')

# perform autoquantization
torchao.autoquant(model, (input))

# compile the model to improve performance
model = torch.compile(model, mode='max-autotune')
model(input)
```


### A8W8 Dynamic Quantization

```python
# Fuse the int8*int8 -> int32 matmul and subsequent mul op avoiding materialization of the int32 intermediary tensor
torch._inductor.config.force_fuse_int_mm_with_mul = True
from torchao.quantization import quant_api
# convert linear modules to quantized tensor subclasses
quant_api.change_linear_weights_to_int8_dqtensors(model)
```

### A16W8 WeightOnly Quantization

```python
from torchao.quantization import quant_api
quant_api.change_linear_weights_to_int8_woqtensors(model)
```

This technique works best when the torch._inductor.config.use_mixed_mm option is enabled. This avoids dequantizing the weight tensor before the matmul, instead fusing the dequantization into the matmul, thereby avoiding materialization of a large floating point weight tensor.


### A16W4 WeightOnly Quantization

```python
from torchao.quantization import quant_api
quant_api.change_linear_weights_to_int4_woqtensors(model)
```

Note: The quantization error incurred by applying int4 quantization to your model can be fairly significant, so using external techniques like GPTQ may be necessary to obtain a usable model.


### A8W8 Dynamic Quantization with Smoothquant

We've also implemented a version of [smoothquant](https://arxiv.org/abs/2211.10438) with the same GEMM format as above. Due to requiring calibration, the API is more complicated.

Example

```Python
import torch
from torchao.quantization.smoothquant import swap_linear_with_smooth_fq_linear, smooth_fq_linear_to_inference

# Fuse the int8*int8 -> int32 matmul and subsequent mul op avoiding materialization of the int32 intermediary tensor
torch._inductor.config.force_fuse_int_mm_with_mul = True

# plug in your model
model = get_model()

# convert linear modules to smoothquant
# linear module in calibration mode
swap_linear_with_smooth_fq_linear(model)

# Create a data loader for calibration
calibration_data = get_calibration_data()
calibration_dataset = MyDataset(calibration_data)
calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=True)

# Calibrate the model
model.train()
for batch in calibration_loader:
    inputs = batch
    model(inputs)

# set it to inference mode
smooth_fq_linear_to_inference(model)

# compile the model to improve performance
model = torch.compile(model, mode='max-autotune')
model(input)
```

## Sharp edges

1. While these techniques are designed to improve model performance, in some cases the opposite can occur. This is because quantization adds additional overhead to the model that is hopefully made up for by faster matmuls (dynamic quantization) or loading weights faster (weight-only quantization). If your matmuls are small enough or your non-quantized perf isn't bottlenecked by weight load time, these techniques may reduce performance.
2. Use the PyTorch nightlies so you can leverage [tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) which is preferred over older module swap based methods because it doesn't modify the graph and is generally more composable and flexible.


## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
