# MoE Quantization

Our goal with this prototype implementation of moe quantization is to enable usage of existing linear quantization techniques for moe quantization. While it would likely be more performant to use a fused kernel for quantized moe, by decomposing the moe operation into a sequence of linear operations, we can utilize the existing tools and UX that work for lienar quantization and apply them to moe.

Examples of the usage of these apis can be found in both the llama4_quant.py and ao/torchao/_models/mixtral-moe/generate.py

## Quantization API

The API for moe quantization is very similar to linear quantization, given a moe module that is decomposed into linear operations, is quantizable and compilable. In practice this requires us to use the modules found in quantizable_moe_modules.py or something similar. Once this change has been made the API is as follows for a few different quantization techniques:

```python

from torchao.prototype.moe_quant.utils import cond_ffn_filter,
from torchao.quantization.quant_api import quantize_, Int8WeightOnlyConfig

quantize_(model, MoEQuantConfig(Int8WeightOnlyConfig()), filter_fn=cond_ffn_filter)
model=torch.compile(model, mode="reduce-overhead")
# you can also use fullgraph=True for single token inference
```

This api is the same as for normal linear quantization but with a specific filter function. This works for several different quantization techniques where the quantized tensor subclass has been adapted to work with 3D tensors. Specifically this means Int8WeightOnlyConfig, Int4WeightOnlyConfig, Int4WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig, and Int8DynamicActivationInt8WeightConfig. It should be noted that due to the requirements on minimum tensor input size (>16), Int8DynamicActivationInt8WeightConfig is best used for expert choice moe rather than token choice which is what the rest of the framework in this folder supports.


## Alternative Quantization API

To make the above api work, each tensor subclass had to be edited to work as 3D tensors. However the only ops we actually need to support are a few indexing and slicing ops on the 0th dimension, the majority of the work was removing hard coded assumptions about the tensor dimensionality. This means its possible to instead create a new tensor subclass that pretends to be a 3D tensor by storing a series of 2D tensors and simulating the slicing and indexing ops until eventually just returning the singular desired 2D quantized tensor subclass. This can be achieved using the alternative api by changing the fake_extra_dim_tensor flag of the MoEQuantConfig:

```python

from torchao.prototype.moe_quant.utils import cond_ffn_filter, MoEQuantConfig, UseFakeExtraDimTensor
from torchao.quantization.quant_api import quantize_, Int8DynamicActivationIntxWeightConfig

config = MoEQuantConfig(
    Int8DynamicActivationIntxWeightConfig(),
    # this is the only difference from the above api
    use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE,
)

quantize_(model, , filter_fn=cond_ffn_filter)
model=torch.compile(model, mode="reduce-overhead")
```

It should also be noted that the default value for use_fake_extra_dim_tensor is AS_FALLBACK which means that it will try to use the base method but if not, will use the more general but less performant fake_extra_dim_tensor method.

While this approach turns out to not be especially performant, it does allow for slightly better memory characteristics since all the tensors are held seperately and aren't actually modified or indexed. It is flexible enough to work with all of the existing linear quantization techniques that make use of quantized tensor subclasses without any changes being made to those classes. It is compilable though neither single token nor multi token inference works with fullgraph compilation.

## Model API

In practice the moe implementations of known models tend to not be easy to quantize and even of those that are, they are often either compiled with many graph breaks or impossible to torch.compile at all.

The modules in the quantizable_moe_modules.py file were carefully written to satisfy both of those necessary characteristics but to apply moe quantization to your own model, it will require first a module swap from the existing MoE module type, to these more flexible ones. While there isn't a one size fits all way to do this, an example of how it was done for huggingface's llama4 implementation can be found in llama4_quant.py which can be seen as a proof of concept.
