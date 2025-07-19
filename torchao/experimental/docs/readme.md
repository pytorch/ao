# TorchAO experimental

TorchAO experimental contains lowbit ARM CPU and Metal kernels for linear and
embedding ops.

## Building ARM CPU kernels

To build torch ops that use the lowbit kernels, run
`sh build_torchao_ops.sh <aten|executorch>` from torchao/experimental.

For example, to build ATen ops, run `sh build_torchao_ops.sh aten` (this
requires PyTorch). Similarly, to build the ExecuTorch ops, run
`sh build_torchao_ops executorch` (this requires ExecuTorch).

After running the script, the op libraries will be in

```
cmake-out/lib/libtorchao_ops_aten.{dylib|so} # ATen op library
cmake-out/lib/libtorchao_ops_executorch.a # ExecuTorch op library
```

## Quantizing models

Once the ATen ops are built, you can quantize PyTorch models with them. The
quantized models can be run in eager model, compiled, used with AOTI, or
exported. The exported models can be lowered to ExecuTorch.

```python
import torch
torch.ops.load_library("cmake-out/lib/libtorchao_ops_aten.dylib") # make sure this path is correct on your machine
from torchao.experimental.quant_api import Int8DynActIntxWeightLinearQuantizer, IntxWeightEmbeddingQuantizer

my_model = Model()

embedding_quantizer = IntxWeightEmbeddingQuantizer(
    device="cpu",
    precision=torch.float32,
    bitwidth=2, # bitwidth to quantize embedding weights to (values 1-7 are supported)
    groupsize=32, # groupsize for embedding weights (any multiple of 32 is supported)
)
quantized_model = embedding_quantizer.quantize(my_model)


linear_quantizer = Int8DynActIntxWeightLinearQuantizer(
    device="cpu",
    precision=torch.float32,
    bitwidth=4, # bitwidth to quantize linear weights to (values 1-7 are supported)
    groupsize=256, # groupsize for quantization (any multiple of 16 is supported)
    has_weight_zeros=False, # whether to quantize weights with scales and zeros, or scales-only
)
quantized_model = linear_quantizer.quantize(quantized_model)
```

If you get stuck on the above steps, working examples for both linear and
embedding are in
torchao/experimental/tests/test_linear_8bit_act_xbit_weight_quantizer.py and
torchao/experimental/tests/test_embedding_xbit_quantizer.py. For example,
running `python tests/test_linear_8bit_act_xbit_weight_quantizer.py` loads the
ops, creates a toy model, quantizes the model, and runs it in eager, compile,
AOTI, and exports the model.

### Subclass API

For linear, you can also use the new subclass API in torchao. First install the
kernels by running the following command from the ao directory. (Note: takeshis
will only install the kernels if run on a Mac with Apple Silicon.)

```
USE_CPP=1 pip install .
```

Once the kernels are installed, you can quantize your model as follows:

```python
from torchao.dtypes import PlainLayout
from torchao.experimental.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    PackedLinearInt8DynamicActivationIntxWeightLayout,
)
from torchao.experimental.quant_api import (
    int8_dynamic_activation_intx_weight,
)
from torchao.quantization.granularity import (
    PerGroup,
    PerRow,
)
from torchao.quantization.quant_api import quantize_

my_model = Model()

quantize_(
    my_model,
    int8_dynamic_activation_intx_weight(
        weight_dtype=torch.int4,
        granularity=PerGroup(256), # PerRow() is also supported
        has_weight_zeros=False,
        layout=PackedLinearInt8DynamicActivationIntxWeightLayout(), # PlainLayout() is also supported, but much slower on CPU
    ),
)

If you get stuck, consult
`torchao/experimental/tests/test_packed_linear_int8_dynamic_activation_intx_weight_layout.py`
for a working example.

## Available in torchchat

TorchAO experimental kernels are
[available in torchchat](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels),
PyTorch's solution for running LLMs locally. Torchchat integration uses similar
steps to above.
