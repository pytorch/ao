# Quantization
Typically quantization algorithms will have different schemes for how the activation and weights are quantized so A16W8 for instance means the activations are quantized to 16 bits wheras the weights are quantized to 8 bits. Trying out different quantization schemes in `torchao` is generally a 1 line change. Note: exact APIs are not stable, we may change them in the future.

## Benchmarks
Benchmarks and evaluation are gathered using the scripts for [generation](../_models/llama/generate.py) and [eval](../_models/llama/eval.py). Evaluation was done using the lm_eval library for tasks/data. The models used were meta-llama/Llama-2-7b-chat-hf and meta-llama/Meta-Llama-3-8B.
### CUDA backend |  NVIDIA-A100-80GB GPU
| Model       | Technique               | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ----------------------- | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (bfloat16)         | 12.212              |  107.38       | 1418.93                 | 13.88            | 13.21           |
|             | int8dq                  | 12.262              |    9.61       |   63.67                 |  8.61            |  6.62           |
|             | int8wo                  | 12.204              |  170.83       | 1131.18                 |  8.95            |  6.62           |
|             | fp6                     | 12.369              |  117.89       |  584.57                 |  6.52            |  4.96           |
|             | int4wo-64               | 12.843              |  201.14       |  751.42                 |  4.87            |  3.74           |
|             | int4wo-64-GPTQ          | 12.527              |  201.14       |  751.42                 |  4.87            |  3.74           |
|             | autoquant-int4hqq       | 12.825              |  209.19       |  804.32                 |  4.89            |  3.84           |
| Llama-3-8B  | Base (bfloat16)         |  7.441              |   95.64       | 1435.54                 | 16.43            | 15.01           |
|             | int8dq                  |  7.581              |    8.61       |   64.75                 |  9.24            |  7.52           |
|             | int8wo                  |  7.447              |  153.03       | 1150.80                 | 10.42            |  7.52           |
|             | fp6                     |  7.661              |  161.58       |  910.02                 |  7.72            |  5.63           |
|             | int4wo-64               |  8.316              |  180.80       |  763.33                 |  6.88            |  4.22           |
|             | int4wo-64-GPTQ          |  7.921              |  180.80       |  763.33                 |  6.88            |  4.22           |
|             | autoquant-int4hqq       |  8.110              |  188.41       |  800.58                 |  7.14            |  4.25           |
### XPU backend
| Model       | Technique               | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ----------------------- | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (bfloat16)         | NA              |  42.20       | 557.71                 | 13.89            | 13.21           |
|             | int8dq                  | NA              |    9.87       |   65.35                 |  14.60            |  6.62           |
|             | int8wo                  | NA              |  66.24       | 438.61                 |  14.60            |  6.62



### CUDA backend | NVIDIA-H100 GPU
| Model         | Technique               | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| -----------   | ----------------------- | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-3.1-8B  | Base (bfloat16)         |  7.54               |  126.90       | 1904.75                 | 16.75            | 15.01           |
|               | int8wo                  |  7.56               |  198.85       | 1495.41                 | 11.05            |  7.52           |
|               | int4wo-64               |  8.44               |  241.39       | 1019.14                 |  7.08            |  4.22           |
|               | float8wo                |  7.60               |  178.46       | 1339.93                 | 12.09            |  7.51           |
|               | float8dq (PerTensor)    |  7.62               |  116.40       |  873.58                 | 11.14            |  7.51           |
|               | float8dq (Per Row)      |  7.61               |  154.63       | 1161.47                 | 11.14            |  7.51           |

### XPU backend | Intel-Max1100
| Model         | Technique               | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| -----------   | ----------------------- | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-3-8.1B  | Base (bfloat16)         |  7.441              |   40.36       | 605.77                 | 16.35            | 15.01           |
|             | int8dq                  |  7.581              |    13.60       |   102.28                 |  18.69            |  7.52           |
|             | int8wo                  |  7.447              |  59.49       | 447.27                 | 18.60            |  7.52


Benchmarks and evaluation for model meta-llama/Meta-Llama-3.1-8B are gathered using [generation](../_models/llama/generate.py) and [eval](../_models/llama/eval.py). Evaluation was done using the lm_eval library for tasks/data.

note: Int8 dynamic quantization works best on compute bound models like [SAM](https://github.com/pytorch-labs/segment-anything-fast) whereas Llama with batchsize=1 tends to be memory bound, thus the rather low performance.

For int4 we make heavy use of [tinygemm](https://github.com/pytorch/ao/blob/cb3bd8c674f2123af232a0231b5e38ddafa756a8/torchao/dtypes/aqt.py#L526) of `torch.ops.aten._weight_int4pack_mm` to bitpack into a layout optimized for tensor cores

And a quick crash course on inference quantization to help parse the above table. Int4 quantization is an ambiguous term because there's the dtype in which a layer is represented and then the dtype in which the computation is done. For example, if you're using Weight-Only (wo) int4 quantization that means that the layer will be upcasted to a larger dtype like fp16 so an int4 matrix multiplication is defined as `F.linear(input, weight.to(input.dtype))`. Dynamic quantization (DQ) primarily targets activations, enabling on-the-fly quantization from higher precision formats like bf16 to lower precision formats such as int8. This process, when supported by hardware, allows for direct computation, such as performing `F.linear(input, weight)`. Naive quantization algorithms are also notoriously sensitive to outliers so we also typically set a group size that applies a scale factor per group of 64 elements in the case of `int4wo-64`.

## Evaluation

You can also use the EleutherAI [LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) to directly evaluate models
quantized with post training quantization, by following these steps:

1. Quantize your model with a [post training quantization strategy](#post-training-quantization).
2. Save your model to disk or upload to huggingface hub ([instructions]( https://huggingface.co/docs/transformers/main/en/quantization/torchao?torchao=manual#serialization)).
3. [Install](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install) lm-eval.
4. Run an evaluation. Example:

```bash
lm_eval --model hf --model_args pretrained=${HF_USER}/${MODEL_ID} --tasks hellaswag --device cuda:0 --batch_size 8
```

Check out the lm-eval [usage docs](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#basic-usage) for more details.

## Autoquantization

Autoquantization is a tool to automatically determine the best way to apply quantization to your model by comparing the performance of each quantization technique to each layer for the input types and shapes you care about.

```python
import torch
import torchao
from torchao.quantization import  DEFAULT_INT4_AUTOQUANT_CLASS_LIST

# Plug in your model and example input
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
input = torch.randn(32,32, dtype=torch.bfloat16, device='cuda')
use_autoquant_default = True

if use_autoquant_default:
    # perform autoquantization and torch.compile with default settings
    model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
elif not use_autoquant_default:
    # perform autoquantization and torch.compile with int4 support
    model = torchao.autoquant(torch.compile(model, mode='max-autotune'), qtensor_class_list=DEFAULT_INT4_AUTOQUANT_CLASS_LIST)

# pass in an input which is used in order to pick fastest quantization operations
# and apply torch compilation.
model(input)
```

When used as in the example above, when the `autoquant` api is called alongside torch.compile, autoquant sets up the model so that when its run on the next input, the autoquantization and torch.compile processes leave you with a heavily optimized model.

When `model(input)` is called, (under the hood) the tool does a preliminary run with the input where each linear layer keeps track of the different shapes and types of activations that it sees. Once the preliminary run is complete, the next step is to check each linear layer and benchmark the tracked shapes for different types of quantization techniques in order to pick the fastest one, attempting to take into account fusions where possible. Finally once the best class is found for each layer, the next step is to apply the necessary quantization technique to each layer, before finally allowing the normal `torch.compile` process to occur on the now quantized model. By default the api only uses int8 techniques, i.e. it chooses between no quantization, int8 dynamic quantization and int8 weight only quantization for each layer, though there is also an option add int4 quantization which can be used for maximum performance or to avoid perf regressions from `Int4WeightOnlyConfig()` since for certain (compute bound) regimes, int4 weight only quantization can be very slow.

Sometimes it is desirable to reuse a quantization plan that `autoquant` came up with. `torchao.quantization.AUTOQUANT_CACHE` is a dictionary holding autoquant's benchmark results. We can save it and restore it later, which will cause `autoquant` to choose the same quantization methods.

```python
import pickle
import torchao.quantization

# After the first forward pass (when quantization was done)
from torchao.quantization.autoquant import AUTOQUANT_CACHE
with open("quantization-cache.pkl", "wb") as f:
    pickle.dump(AUTOQUANT_CACHE, f)

# On load
from torchao.quantization.autoquant import AUTOQUANT_CACHE
with open("quantization-cache.pkl", "rb") as f:
    AUTOQUANT_CACHE.update(pickle.load(f))
```

## Quantization Techniques
While the above `autoquant` api tries multiple quantization techniques to find the best combination for your model, the techniques themselves can
be applied individually. While there are a large variety of quantization apis, the following techniques have been thoroughly tested and perform well for the metrics they seek to optimize. Each are examples of affine quantization

#### A16W4 WeightOnly Quantization

```python
# for torch 2.4+
from torchao.quantization import quantize_, Int4WeightOnlyConfig
group_size = 32

# you can enable [hqq](https://github.com/mobiusml/hqq/tree/master) quantization which is expected to improves accuracy through
# use_hqq flag for `Int4WeightOnlyConfig` quantization
use_hqq = False
quantize_(model, Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq))

# for torch 2.2.2 and 2.3
from torchao.quantization.quant_api import change_linear_weights_to_int4_woqtensors
change_linear_weights_to_int4_woqtensors(model)
```

Note: The quantization error incurred by applying int4 quantization to your model can be fairly significant, so using external techniques like GPTQ may be necessary to obtain a usable model.

#### A16W8 Int8 WeightOnly Quantization

```python
# for torch 2.4+
from torchao.quantization import quantize_, Int8WeightOnlyConfig
quantize_(model, Int8WeightOnlyConfig())

# for torch 2.2.2 and 2.3
from torchao.quantization.quant_api import change_linear_weights_to_int8_woqtensors
change_linear_weights_to_int8_woqtensors(model)
```

#### A8W8 Int8 Dynamic Quantization

```python
# for torch 2.4+
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
quantize_(model, Int8DynamicActivationInt8WeightConfig())

# for torch 2.2.2 and 2.3
from torchao.quantization.quant_api import change_linear_weights_to_int8_dqtensors
change_linear_weights_to_int8_dqtensors(model)
```

### A16W8 Float8 WeightOnly Quantization

```python
# for torch 2.5+
from torchao.quantization import quantize_, Float8WeightOnlyConfig
quantize_(model, Float8WeightOnlyConfig())
```

Supports all dtypes for original weight and activation. This API is only tested on H100. Hardware with CUDA compute capability 8.9 or greater is required.

#### A8W8 Float8 Dynamic Quantization with Tensorwise Scaling

```python
# for torch 2.4+
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
```

Supports all dtypes for original weight and activation. This API is only tested on H100. Hardware with CUDA compute capability 8.9 or greater is required.

### A8W8 Float8 Dynamic Quantization with Rowwise Scaling

```python
# for torch 2.5+
from torchao.quantization import quantize_, PerRow, Float8DynamicActivationFloat8WeightConfig
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
```

Per-row scaling is only supported for bfloat16 weight and activation. This API is only tested on H100. Hardware with CUDA compute capability 8.9 or greater is required.

#### A16W6 Floating Point WeightOnly Quantization

```python
# for torch 2.4+
from torchao.quantization import quantize_, FPXWeightOnlyConfig
quantize_(model, FPXWeightOnlyConfig(3, 2))
```

You can find more information [here](../dtypes/floatx/README.md). It should be noted where most other TorchAO apis and benchmarks have focused on applying techniques on top of a bf16 model, performance, fp6 works primarily with the fp16 dtype.

## Affine Quantization Details
Affine quantization refers to the type of quantization that maps from high precision floating point numbers to quantized numbers (low precision integer or floating point dtypes) with an affine transformation, i.e.: `quantized_val = high_precision_float_val / scale + zero_point` where `scale` and `zero_point` are quantization parameters for some granularity and based on some data (also some dtypes may not require a `zero_point`). Each of the techniques in the above section qualify as Affine Quantization.

### Quantization Primitives
We used to have different quantize and dequantize operators for quantization with different granularities. But in the end these can all be expressed with a `block_size` argument with different settings, so we unified existing quant primitives to `choose_qparams_affine`, `quantize_affine` and `dequantize_affine` that can represent symmetric/asymmetric per tensor/channel/token/channel_group quantization, this can be used to implement the unified quantized tensor subclass.

Note: these primitive ops supports two "types" of quantization, distinguished by whether `zero_point` is in floating point domain or integer domain. See docstrings for `choose_qparams` for more details.

### Quantized Tensor Subclass
We also have a unified quantized tensor subclass that implements how to get a quantized tensor from floating point tensor and what does it mean to call linear ops on an instance of the tensor, e.g. `F.linear` and `aten.addmm`, with this we could dispatch to different operators (e.g. `int4mm` op) based on device (cpu, cuda) and quantization settings (`int4`, `int8`) and also packing formats (e.g. format optimized for cpu int4 mm kernel)

#### Layouts
We extended the `layout` concept to represent different packing formats for a tensor. `AffineQuantizedTensor` supports `plain` and `tensor_core_tiled` layout. `plain` layout is used for workflows backing `Int8WeightOnlyConfig` and `Int8DynamicActivationInt8WeightConfig` and also as a default layout. `tensor_core_tiled` layout is used for workflows backing `Int4WeightOnlyConfig` quantization and is packing the weights in a format that is compatible with tinygemm [int4mm](https://github.com/pytorch/pytorch/blob/39357ba06f48cda7d293a4995aa5eba2a46598b5/aten/src/ATen/native/native_functions.yaml#L4138) kernels.

### Zero Point Domains
```ZeroPointDomain``` is used to control the data types of zero points. ```ZeroPointDomain.None``` means zero_point is None, ```ZeroPointDomain.FLOAT``` means zero_point is in the floating point domain and ```ZeroPointDomain.INT``` means integer domain. For detailed implementation of different zero point data types, refer to [the reference implementation](../../test/quantization/test_quant_primitives.py).
The following support matrix illustrates the relationship between layouts and zero point domains, which may be updated with backend changes:

|Layout|None(Symmetric)|Float|Int|
|------|---------------|-----|---|
|TensorCoreTiledLayout| Yes | Yes(Default) | No|
|Int4CPULayout | Yes | Yes(Default) | No |
|MarlinSparseLayout | No | No | Yes(Default) |


### Full Affine Quantization Flow Example
Let's use int4 weight only quantization that's targeting tinygemm int4 weight only quantized matmul
as an example:
```python
import torch
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.dtypes import to_affine_quantized_intx
import copy
from torchao.quantization.quant_api import (
    quantize_,
    Int4WeightOnlyConfig,
)

class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

dtype = torch.bfloat16
m = ToyLinearModel(1024, 1024, 1024).eval().to(dtype).to("cuda")
m_bf16 = copy.deepcopy(m)
example_inputs = m.example_inputs(dtype=dtype, device="cuda")

m_bf16 = torch.compile(m_bf16, mode='max-autotune')
# apply int4 weight only quant (compatible with tinygemm int4 weight only quant mm kernel in torchao)
group_size = 32
# only works for torch 2.4+
quantize_(m, Int4WeightOnlyConfig(group_size=group_size))
## If different zero_point_domain needed
# quantize_(m, Int4WeightOnlyConfig(group_size=group_size, zero_point_domain=ZeroPointDomain.FLOAT))

# temporary workaround for tensor subclass + torch.compile
# NOTE: this is only need for torch version < 2.5+
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5
from torchao.utils import unwrap_tensor_subclass
if not TORCH_VERSION_AT_LEAST_2_5:
    unwrap_tensor_subclass(m)
# compile the model to improve performance
m = torch.compile(m, mode='max-autotune')

# benchmark to see the speedup
from torchao.utils import benchmark_model

num_runs = 100
torch._dynamo.reset()
bf16_time = benchmark_model(m_bf16, num_runs, example_inputs)
print(f"bf16 mean time: {bf16_time}")
int4_time = benchmark_model(m, num_runs, example_inputs)
print(f"int4 weight only quantized mean time: {int4_time}")
print(f"speedup: {bf16_time / int4_time}")

# output (1xA100 GPU machine)
bf16 mean time: 71.457685546875
int4 weight only quantized mean time: 31.4580908203125
speedup: 2.2715200981216173
```

What we do underlying the APIs are roughly the following:
```python
from torchao.dtypes import to_affine_quantized_intx
def int8wo_quant(weight):
    return to_affine_quantized_intx(weight, MappingType.SYMMETRIC, (1, weight.shape[1]), torch.int8, eps=torch.finfo(torch.float32).eps, zero_point_dtype=torch.int64)

for module, name in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # optional filtering for module name, shape etc.
        m.weight = nn.Parameter(int8wo_quant(module.weight))

        # note: quantization for activation need to be applied after the weight quantization
        # quantization activation (needed by dynamic quantization)
        input_quant_func = int8wo_quant  # specify how input activation is quantized
        module.weight = nn.Parameter(to_linear_activation_quantized(module.weight, input_quant_func))
```

#### Workaround with `unwrap_tensor_subclass` for `export`, `AOTI` and `torch.compile`

If you are using pytorch 2.6 or before, you need to call `unwrap_tensor_subclass` before `torch.export.export` and `aot_compile`:
```
from torchao.utils import unwrap_tensor_subclass
m_unwrapped = unwrap_tensor_subclass(m)


# export
m = torch.export.export(m_unwrapped, example_inputs).module()

# aot_compile
torch._export.aot_compile(m_unwrapped, example_inputs)
```

If you are using pytorch 2.4 or before, you'll also need `unwrap_tensor_subclass` before calling `torch.compile` as well.

Note that the workaround is also required for `torch.compile` with `freezing` (`torch._inductor.config.freezing=True`) until https://github.com/pytorch/pytorch/pull/136265 is fixed.

## Other Available Quantization Techniques

### KV Cache Quantization
We've added kv cache quantization and other features in order to enable long context length (and necessarily memory efficient) inference.

In practice these features alongside int4 weight only quantization allow us to **reduce peak memory by ~55%**, meaning we can Llama3.1-8B inference with a **130k context length with only 18.9 GB of peak memory.** More details can be found [here](../../torchao/_models/llama/README.md#KV-Cache-Quantization-Memory-Efficient-Inference)

### Sparse-Marlin

Sparse-Marlin 2:4 is an optimized GPU kernel that extends the Mixed Auto-Regressive Linear (Marlin) dense kernel to support 4-bit quantized weights and 2:4 sparsity for extremely high performance.

| Model       | Technique               | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ----------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-3-8B  | Base (bfloat16)         |   95.64       | 1435.54                 | 16.43            | 15.01           |
|             | int8wo                  |  153.03       | 1150.80                 | 10.42            |  7.52           |
|             | int4wo-64               |  180.80       |  763.33                 |  6.88            |  4.22           |
|             | int4wo-64-sparse-marlin |  226.02       |  689.20                 |  5.32            |  3.05           |

More details can be found [here](../sparsity/README.md)

### Marlin QQQ

Marlin QQQ is an optimized GPU kernel that supports W4A8 mixed precision GEMM. For more details about Marlin QQQ, please refer to [paper](https://arxiv.org/pdf/2406.09904).

| Model       | Technique               | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ----------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (float16)          |  112.45       |  1486.00                | 13.93            | 13.21           |
|             | w4a8                    |  197.45       |  653.50                 | 4.79            |  3.31           |
|             | w4a8-g128               |  187.62       |  640.32                 | 4.82            |  3.41           |

### Gemlite Triton
Int4 and Int8 quantization using the [Gemlite Triton](https://github.com/mobiusml/gemlite) kernels. You can try it out with the `quantize_` api as above alongside the constructor `gemlite_uintx_weight_only`.  An example can be found in `torchao/_models/llama/generate.py`.

Note: we test on gemlite 0.4.1, but should be able to use any version after that, we'd recommend to use the latest release to get the most recent performance improvements.

### UINTx Quantization
We're trying to develop kernels for low bit quantization for intx quantization formats. While the current performance is not ideal, we're hoping to continue to iterate on these kernels to improve their performance.

| Model       | Technique               | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ----------------------- | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (bfloat16)         | 12.212              | 107.38        | 1418.93                 | 13.88            | 13.21           |
|             | uintx-4-64-hqq          | 12.775              |  50.99        |  200.08                 |  6.29            |  3.92           |
|             | uintx-2-8-hqq           | 24.500              |  40.25        |  265.95                 |  9.24            |  6.61           |
| Llama-3-8B  | Base (bfloat16)         |  7.441              |  95.64        | 1435.54                 | 16.43            | 15.01           |
|             | uintx-4-64-hqq          |  8.124              |  47.85        |  213.24                 | 11.85            |  4.46           |
|             | uintx-2-8-hqq           | 39.605              |  34.83        |  261.42                 | 14.99            |  7.51           |

You try can out these apis with the `quantize_` api as above alongside the config `UIntXWeightOnlyConfig`. An example can be found in  in `torchao/_models/llama/generate.py`.

### int8_dynamic_activation_intx_weight Quantization
We have kernels that do 8-bit dynamic quantization of activations and uintx groupwise quantization of weights.  These kernels are experimental and can only be run on a device with an ARM CPU (e.g., a Mac computers with Apple silicon).  The benchmarks below were run on an M1 Mac Pro, with 8 perf cores, and 2 efficiency cores, and 32GB of RAM.  In all cases, torch.compile was used.

| Model         | Technique                                        | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ------------- | -------------------------------------------------| --------------| ------------------------| ---------------- | ----------------|
| Llama-3.1-8B  | Base (bfloat16)                                  |  1.24         |  18.62                  |  NA              | 15.01           |
|               | int8_dynamic_activation_intx_weight-4-256-false  |  16.03        |  65.81                  |  NA              | 4.11            |
|               | int8_dynamic_activation_intx_weight-3-256-false  |  18.94        |  59.97                  |  NA              | 3.17            |

You can try out these apis with the `quantize_` api as above alongside the constructor `int8_dynamic_activation_intx_weight`.  An example can be found in `torchao/_models/llama/generate.py`.

### Codebook Quantization
The benchmarks below were run on a single NVIDIA-A6000 GPU.

| Model       | Technique               | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ----------------------- | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-3-8B  | Base (bfloat16)         |  7.590              |  32.36        |  485.71                 | 16.19            | 15.01           |
|             | codebook-4-64           |  9.533              |  1.73         |  8.62                   | 23.11            |  4.98           |
| Llama-3.1-8B| Base (bfloat16)         |  7.713              |  32.16        |  482.70                 | 16.35            | 15.01           |
|             | codebook-4-64           |  10.095             |  1.73         |  8.63                   | 23.11            |  4.98           |

You try can out these apis with the `quantize_` api as above alongside the constructor `codebook_weight_only` an example can be found in  in `torchao/_models/llama/generate.py`.

### GPTQ Quantization
We have a GPTQ quantization workflow that can be used to quantize a model to int4. More details can be found in [GPTQ](./GPTQ/README.md),
an example can be found in `torchao/_models/llama/eval.py`.

### Automatic Inductor Configuration

:warning: <em>This functionality is being migrated from the top level `quantize_` API to individual workflows, see https://github.com/pytorch/ao/issues/1715 for more details.</em>

The `quantize_` and `autoquant` apis now automatically use our recommended inductor configuration setings. You can mimic the same configuration settings for your own experiments by using the `torchao.quantization.utils.recommended_inductor_config_setter` to replicate our recommended configuration settings. Alternatively if you wish to disable these recommended settings, you can use the key word argument `set_inductor_config` and set it to false in the `quantize_` or `autoquant` apis to prevent assignment of those configuration settings. You can also overwrite these configuration settings after they are assigned if you so desire, as long as they are overwritten before passing any inputs to the torch.compiled model. This means that previous flows which referenced a variety of inductor configurations that needed to be set are now outdated, though continuing to manually set those same inductor configurations is unlikely to cause any issues.

## Notes

1. APIs have been hardware tested on A100 and T4(colab)
2. While these techniques are designed to improve model performance, in some cases the opposite can occur. This is because quantization adds additional overhead to the model that is hopefully made up for by faster matmuls (dynamic quantization) or loading weights faster (weight-only quantization). If your matmuls are small enough or your non-quantized perf isn't bottlenecked by weight load time, these techniques may reduce performance.
3. Use the PyTorch nightlies so you can leverage [tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) which is preferred over older module swap based methods because it doesn't modify the graph and is generally more composable and flexible.
