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

Example

```
import torch
from torchao.quantization import quant_api

# some user model
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
# some example input
input = torch.randn(32,32, dtype=torch.bfloat16, device='cuda')

# convert linear modules to quantized linear modules
# insert quantization method/api of choice
quant_api.apply_weight_only_int8_quant(model)
# quant_api.apply_dynamic_quant(model)
# quant_api.change_linear_weights_to_dqtensors(model)

# compile the model to improve performance
torch.compile(model, mode='max-autotune')
model(input)
```

### A16W8 WeightOnly Quantization

## License

`torchao` is released under the [BSD 3](https://github.com/pytorch-labs/ao/blob/main/LICENSE) license.
