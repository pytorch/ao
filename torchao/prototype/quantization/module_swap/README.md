# Module Swap Quantization (prototype)

This is an alternative to quantization based on tensor subclasses,
bypassing the entire `AffineQuantizedTensor` stack for simplicity.
Quantized modules supported today include:

```
torch.nn.Linear -> QuantizedLinear
torch.nn.Embedding -> QuantizedEmbedding
```

Within each of these quantized modules, the user can specify different
quantization settings to quantize the weights and the activations
separately. For example:

```
quantized_linear = QuantizedLinear(...)
quantized_linear.input_quantization = IntQuantizer(...)
quantized_linear.weight_quantization = CodeBookQuantizer(...)
```

The current entry point API is `quantize_module_swap`, which takes
in a `QuantizationRecipe` and performs module swap on the model,
applying the configured quantizers to weights and activations on
the swapped quantized modules. However, **this API is highly subject
to change and will be replaced by `quantize_` in the future**.
Example usage:

```
import torch
import torch.nn as nn
from torchao.prototype.quantization.module_swap import (
    quantize_module_swap,
    QuantizationRecipe,
)

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, 64)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


model = MyModel()
recipe = QuantizationRecipe(
    embedding_quantization=True,
    embedding_bits=4
)
model = quantize_module_swap(model, recipe)
```

```
>>> model
MyModel(
  (embedding): QuantizedEmbedding(
    10, 64
    (weight_quantizer): IntQuantizer()
  )
)
>>> x = torch.randint(0, 10, (10, 64))
>>> model(x)
tensor([[[-0.0000,  1.7221,  0.6888,  ...,  0.5700, -0.5700, -0.8550],
         ...
         [ 1.2896, -0.0000,  0.3224,  ..., -0.5430, -1.9005,  0.5430]]],
       grad_fn=<EmbeddingBackward0>)
```
