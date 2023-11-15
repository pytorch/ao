# torchao

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

## Other APIs

### A8W8 Dynamic Quantization by subclasses

You can use [tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) to do dynamic quantization with the `change_linear_weights_to_dqtensors` function using the exact same formula as above. This avoids modifying the graph and can be more composable with
other techniques.

### A8W8 Dynamic Quantization with Smoothquant

We've also implemented a version of [smoothquant](https://arxiv.org/abs/2211.10438) with the same GEMM format as above.
Due to requiring calibraiton the API is slightly more complicated

Example

```
import torch
from torchao.smoothquant import swap_linear_with_smooth_fq_linear, smooth_fq_linear_to_inference

# some user model
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)

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
