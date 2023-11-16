# torchao

**Note: This repository is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an github issue or reach out. We'd love to hear about how you're using the APIs.**

The torchao package contains apis and workflows used to apply AO techniques like quantization and pruning to models using only native pytorch.

## Installation

clone repository and install package:

```
git clone https://github.com/pytorch-labs/ao
cd ao
python setup.py install
```

verify installation:

```
pip list | grep torchao
```

should show
```
torchao                            0.0.1                   <install dir>
```

## Usage

Relevant APIs can be found in torchao.quantization.quant_api

Note: While these techniques are designed to improve model performance, in some cases the opposite can occur.
This is because quantization adds additional overhead to the model that is hopefully made up for by faster matmuls (dynamic quantization) or loading weights faster (weight-only quantization). If your matmuls are small enough or your non-quantized perf isn't bottlenecked by weight load time, these techniques may reduce performance.

### A8W8 Dynamic Quantization

Similar to the weight only api above, the `apply_dynamic_quant` function swaps all
linear modules to dynamically quantized quantized linear modules.

Example

```

# some user model and example input
...

# convert linear modules to quantized linear modules
quant_api.apply_dynamic_quant(model)

# compile the model to improve performance
...
```

This technique works best when the torch._inductor.config.force_fuse_int_mm_with_mul option is enabled. This allows fusion of the int8*int8 -> int32 matmul and subsequent mul op, thereby avoiding materialization of the int32 intermediary tensor.

### A16W8 WeightOnly Quantization

The `apply_weight_only_int8_quant` function swaps all
linear modules to weight-only quantized linear modules.

Example

```
import torch
from torchao.quantization import quant_api

# some user model and example input
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
input = torch.randn(32,32, dtype=torch.bfloat16, device='cuda')

# convert linear modules to quantized linear modules
quant_api.apply_weight_only_int8_quant(model)

# compile the model to improve performance
torch.compile(model, mode='max-autotune')
model(input)
```

This technique works best when the torch._inductor.config.use_mixed_mm option is enabled. This avoids dequantizing the weight tensor before the matmul, instead fusing the dequantization into the matmul, thereby avoiding materialization of a large floating point weight tensor.

## Other APIs

### A8W8 Dynamic Quantization by subclasses

You can use [tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) to do dynamic quantization with the `change_linear_weights_to_dqtensors` function using the exact same formula as above. This avoids modifying the graph and can be more composable with
other techniques.

### A8W8 Dynamic Quantization with Smoothquant

We've also implemented a version of [smoothquant](https://arxiv.org/abs/2211.10438) with the same GEMM format as above.
Due to requiring calibration, the API is slightly more complicated

Example

```
import torch
from torchao.smoothquant import swap_linear_with_smooth_fq_linear, smooth_fq_linear_to_inference

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
torch.compile(model, mode='max-autotune')
model(input)
```

## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
