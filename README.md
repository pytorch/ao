# torchao

**Note: This repository is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an github issue or reach out. We'd love to hear about how you're using the APIs.**

The torchao package contains apis and workflows used to apply AO techniques like quantization and pruning to models using only native pytorch.

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
python setup.py install
```

Verify Installation:

```Shell
pip list | grep torchao
```

Expected Output
```Shell
torchao                            0.0.1                   <install dir>
```

## Usage

Relevant APIs can be found in torchao.quantization.quant_api

Note: While these techniques are designed to improve model performance, in some cases the opposite can occur.
This is because quantization adds additional overhead to the model that is hopefully made up for by faster matmuls (dynamic quantization) or loading weights faster (weight-only quantization). If your matmuls are small enough or your non-quantized perf isn't bottlenecked by weight load time, these techniques may reduce performance.

The following apis use quantized [tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor). By taking a linear op/module and replacing the original weight with a q-tensor subclass, we're able to convert it into a quantized version of the op. Upon replacement, these q-tensor subclasses quantize the original weight and override the dispatch for linear ops to instead use the subclass' _quantized_op method.

This tensor subclass method of quantization is preferred over older module swap based methods because it doesn't modify the graph and is generally more composable and flexible.

### A8W8 Dynamic Quantization

The `change_linear_weights_to_int8_dqtensors` function converts the linear weights in a model to a quantized tensor subclass `Int8DynamicallyQuantizedLinearWeight`. In practice this
converts the floating point linear matmul of the original linear op to a dynamically quantized linear matmul.

Example

```Python
import torch
from torchao.quantization import quant_api

# some user model and example input
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
input = torch.randn(32,32, dtype=torch.bfloat16, device='cuda')

# convert linear modules to quantized linear modules
quant_api.change_linear_weights_to_int8_dqtensors(model)

# compile the model to improve performance
model = torch.compile(model, mode='max-autotune')
model(input)
```

This technique works best when the torch._inductor.config.force_fuse_int_mm_with_mul option is enabled. This allows fusion of the int8*int8 -> int32 matmul and subsequent mul op, thereby avoiding materialization of the int32 intermediary tensor.


### A16W8 WeightOnly Quantization

The `change_linear_weights_to_int8_woqtensors` function converts the linear weights in a model to a quantized tensor subclass `Int8WeightOnlyQuantizedLinearWeight`. In practice this
converts the floating point linear matmul of the original linear op to a weight only quantized linear matmul

Example

```Python
# some user model and example input
...

# convert linear modules to quantized linear modules
quant_api.change_linear_weights_to_int8_woqtensors(model)

# compile the model to improve performance
...
```

This technique works best when the torch._inductor.config.use_mixed_mm option is enabled. This avoids dequantizing the weight tensor before the matmul, instead fusing the dequantization into the matmul, thereby avoiding materialization of a large floating point weight tensor.


### A16W4 WeightOnly Quantization

The `change_linear_weights_to_int4_woqtensors` function converts the linear weights in a model to a quantized tensor subclass `Int4WeightOnlyQuantizedLinearWeight`. In practice this
converts the floating point linear matmul of the original linear op to a weight only quantized linear matmul

Example

```Python
# some user model and example input
...

# convert linear modules to quantized linear modules
quant_api.change_linear_weights_to_int4_woqtensors(model)

# compile the model to improve performance
...
```

The quantization error incurred by applying int4 quantization to your model can be fairly significant, so using external techniques like GPTQ may be necessary to obtain a usable model.

## Other APIs

### Module Swap APIs

The `apply_dynamic_quant` and `apply_weight_only_int8_quant` apis can be used in the same formula as above to achieve dynamic and weight-only quantization using module swaps instead of quantized tensor subclasses.

### A8W8 Dynamic Quantization with Smoothquant

We've also implemented a version of [smoothquant](https://arxiv.org/abs/2211.10438) with the same GEMM format as above.
Due to requiring calibration, the API is slightly more complicated and currently only exists with a module swap api.

Example

```Python
import torch
from torchao.quantization.smoothquant import swap_linear_with_smooth_fq_linear, smooth_fq_linear_to_inference

# some user model
model = get_model()

# convert linear modules to smoothquant
# linear module in calibration mode
swap_linear_with_smooth_fq_linear(model)

# calibration
for i in range(calibration_amount):
    input = get_input()
    model(input)

# set it to inference mode
smooth_fq_linear_to_inference(model)

# compile the model to improve performance
model = torch.compile(model, mode='max-autotune')
model(input)
```

like the other dynamic quantization apis, the torch._inductor.config.force_fuse_int_mm_with_mul option may significantly improve performance if enabled.

## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
