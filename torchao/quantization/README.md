# Quantization
Typically quantization algorithms will have different schemes for how the activation and weights are quantized so A16W8 for instance means the activations are quantized to 16 bits wheras the weights are quantized to 8 bits. Trying out different quantization schemes in `torchao` is generally a 1 line change. Note: exact APIs are not stable, we may change them in the future.

## Benchmarks
Benchmarks are run on a machine with a single A100 GPU using the script in _models/llama which generates text in a latency optimized way (batchsize=1), evaluation was done
Using the lm_eval. The models used were meta-llama/Llama-2-7b-chat-hf and meta-llama/Meta-Llama-3-8B.

| Model       | Technique          | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ------------------ | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (bfloat16)    | 12.212              |  105.14       | 1389.35                 | 13.88            | 13.21           |
|             | int8dq             | 12.262              |    9.20       |   60.93                 |  8.33            |  6.62           |
|             | int8wo             | 12.204              |  150.18       |  994.40                 |  8.95            |  6.62           |
|             | int4wo-64          | 12.843              |  199.86       |  746.66                 |  4.50            |  3.74           |
|             | int4wo-64-GPTQ     | 12.489              |  199.86       |  746.66                 |  4.50            |  3.74           |
|             | autoquant          | 12.204              |  159.22       | 1069.87                 |  8.91            |  6.72           |
| Llama-3-8B  | Base (bfloat16)    | N/A                 |   94.97       | 1425.55                 | 16.43            | 15.01           |
|             | int8dq             | N/A                 |    8.44       |   63.45                 |  8.98            |  7.52           |
|             | int8wo             | N/A                 |  139.76       | 1051.02                 | 10.42            |  7.52           |
|             | int4wo-64          | N/A                 |  179.44       |  757.60                 |  6.62            |  4.22           |
|             | autoquant          | N/A                 |  137.71       | 1037.74                 | 11.08            |  7.54           |

note: Int8 dynamic quantization works best on compute bound models like [SAM](https://github.com/pytorch-labs/segment-anything-fast) whereas Llama with batchsize=1 tends to be memory bound, thus the rather low performance.

## Autoquantization

The `autoquant` api can be used to quickly and accurately quantize your model. When used as in the example below, the api first identifies the shapes
of the activations that the different linear layers see, it then benchmarks these shapes across different types of quantized and non-quantized layers in order to pick the fastest one, attempting to take into account fusions where possible. Finally once the best class is found for each layer, it swaps the linear. Currently this api chooses between no quantization, int8 dynamic quantization and int8 weight only quantization for each layer.

```python
import torch
import torchao

# Plug in your model and example input
model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
input = torch.randn(32,32, dtype=torch.bfloat16, device='cuda')

# perform autoquantization and torch.compile
model = torchao.autoquant(torch.compile(model, mode='max-autotune'))

# pass in an input which is used in order to pick fastest quantization operations
# and apply torch compilation.
model(input)
```

Sometimes it is desirable to reuse a quantization plan that `autoquant` came up with. `torchao.quantization.AUTOQUANT_CACHE` is a dictionary holding autoquant's benchmark results. We can save it and restore it later, which will cause `autoquant` to choose the same quantization methods.

```python
import pickle
import torchao.quantization

# After the first forward pass (when quantization was done)
with open("quantization-cache.pkl", "wb") as f:
    pickle.dump(torchao.quantization.AUTOQUANT_CACHE)

# On load
with open("quantization-cache.pkl", "rb") as f:
    torchao.quantization.AUTOQUANT_CACHE.update(pickle.load(f))
```
## Affine Quantization
Affine quantization refers to the type of quantization that maps from floating point numbers to quantized numbers (typically integer) with an affine transformation, i.e.: `quantized_val = float_val / scale + zero_point` where `scale` and `zero_point` are quantization parameters for some granularity and based on some data.

### Quantization Primitives
We used to have different quantize and dequantize operators for quantization with different granularities. But in the end these can all be expressed with a `block_size` argument with different settings, so we unified existing quant primitives to `choose_qparams_affine`, `quantize_affine` and `dequantize_affine` that can represent symmetric/asymmetric per tensor/channel/token/channel_group quantization, this can be used to implement the unified quantized tensor subclass.

### Quantized Tensor Subclass
We also have a unified quantized tensor subclass that implements how to get a quantized tensor from floating point tensor and what does it mean to call linear ops on an instance of the tensor, e.g. `F.linear` and `aten.addmm`, with this we could dispatch to different operators (e.g. `int4mm` op) based on device (cpu, cuda) and quantization settings (`int4`, `int8`) and also packing formats (e.g. format optimized for cpu int4 mm kernel)

### Quantization Flow Example
Let's use int4 weight only quantization that's targeting tinygemm int4 weight only quantized matmul
as an example:
```python
import torch
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.dtypes import to_affine_quantized
import copy
from torchao.quantization.quant_api import (
    quantize,
    int4_weight_only,
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
m = quantize(m, int4_weight_only(group_size=group_size))

# compile the model to improve performance
m = torch.compile(m, mode='max-autotune')

# benchmark to see the speedup
from torchao.utils import benchmark_model

num_runs = 100
torch._dynamo.reset()
bf16_time = benchmark_model(m_bf16, num_runs, example_inputs[0])
print(f"bf16 mean time: {bf16_time}")
int4_time = benchmark_model(m, num_runs, example_inputs[0])
print(f"int4 weight only quantized mean time: {int4_time}")
print(f"speedup: {bf16_time / int4_time}")

# output (1xA100 GPU machine)
bf16 mean time: 71.457685546875
int4 weight only quantized mean time: 31.4580908203125
speedup: 2.2715200981216173
```

What we do underlying the APIs are roughly the following:
```
from torchao.dtypes import to_affine_quantized
def int8wo_quant(weight):
    return to_affine_quantized(weight, MappingType.SYMMETRIC, (1, weight.shape[1]), torch.int8, eps=torch.finfo(torch.float32).eps, zero_point_dtype=torch.int64)

for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        # optional filtering for module name, shape etc.
        m.weight = nn.Parameter(int8wo_quant(m.weight))

        # note: quantization for activation need to be applied after the weight quantization
        # quantization activation (needed by dynamic quantization)
        input_quant_func = int8wo_quant  # specify how input activation is quantized
        m.weight = nn.Parameter(to_linear_act_quantized(m.weight, input_quant_func))
```
The model/tensor subclass should also be compatible with AOTI and torch.export, currently we can support
`torch.export.export` and `torch.aot_compile`
```
# export
m = torch.export.export(m, example_inputs).module()
```

```
# aot_compile
torch._export.aot_compile(m, example_inputs)
```

### Automatic Inductor Configuration
The `quantize` and `autoquant` apis now automatically use our recommended inductor configuration setings. You can mimic the same configuration settings for your own experiments by using the `torchao.quantization.utils.recommended_inductor_config_setter` to replicate our recommended configuration settings. Alternatively if you wish to disable these recommended settings, you can use the key word argument `set_inductor_config` and set it to false in the `quantize` or `autoquant` apis to prevent assignment of those configuration settings. You can also overwrite these configuration settings after they are assigned if you so desire, as long as they are overwritten before passing any inputs to the torch.compiled model. This means that previous flows which referenced a variety of inductor configurations that needed to be set are now outdated, though continuing to manually set those same inductor configurations is unlikely to cause any issues.

### Other Available Quantization Techniques
#### A8W8 Dynamic Quantization

```python
# Fuse the int8*int8 -> int32 matmul and subsequent mul op avoiding materialization of the int32 intermediary tensor
torch._inductor.config.force_fuse_int_mm_with_mul = True

# for torch 2.4+
from torchao.quantization import quantize, int8_dynamic_activation_int8_weight
quantize(model, int8_dynamic_activation_int8_weight())

# for torch 2.2.2 and 2.3
from torchao.quantization.quant_api import change_linear_weights_to_int8_dqtensors
change_linear_weights_to_int8_dqtensors(model)
```

#### A16W8 WeightOnly Quantization

```python
# for torch 2.4+
from torchao.quantization import quantize, int8_weight_only
quantize(model, int8_weight_only())

# for torch 2.2.2 and 2.3
from torchao.quantization.quant_api import change_linear_weights_to_int8_woqtensors
change_linear_weights_to_int8_woqtensors(model)
```

This technique works best when the torch._inductor.config.use_mixed_mm option is enabled. This avoids dequantizing the weight tensor before the matmul, instead fusing the dequantization into the matmul, thereby avoiding materialization of a large floating point weight tensor.


#### A16W4 WeightOnly Quantization

```python
# for torch 2.4+
from torchao.quantization import quantize, int4_weight_only
quantize(model, int4_weight_only())

# for torch 2.2.2 and 2.3
from torchao.quantization.quant_api import change_linear_weights_to_int4_woqtensors
change_linear_weights_to_int4_woqtensors(model)
```

Note: The quantization error incurred by applying int4 quantization to your model can be fairly significant, so using external techniques like GPTQ may be necessary to obtain a usable model.

## (To be moved to prototype) A16W4 WeightOnly Quantization with GPTQ

```python
from torchao._models._eval import InputRecorder, TransformerEvalWrapper
from torchao.quantization.GPTQ import Int4WeightOnlyGPTQQuantizer
precision = torch.bfloat16
device = "cuda"
checkpoint_file_name = "../gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
checkpoint_path = Path(checkpoint_file_name)
model = Transformer.from_name(checkpoint_path.parent.name)
checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
model.load_state_dict(checkpoint, assign=True)
model = model.to(dtype=precision, device="cpu")
model.eval()
tokenizer_path = checkpoint_path.parent / "tokenizer.model"
assert tokenizer_path.is_file(), tokenizer_path
tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
    model_file=str(tokenizer_path)
)
blocksize = 128
percdamp = 0.01
groupsize = 128
calibration_tasks = ["wikitext"]
calibration_limit = 1
calibration_seq_length = 100
input_prep_func = prepare_inputs_for_model
pad_calibration_inputs = False

inputs = InputRecorder(
    tokenizer,
    calibration_seq_length,
    input_prep_func,
    pad_calibration_inputs,
    model.config.vocab_size,
    device="cpu",
).record_inputs(
    calibration_tasks,
    calibration_limit,
).get_inputs()

quantizer = Int4WeightOnlyGPTQQuantizer(
    blocksize,
    percdamp,
    groupsize,
)
model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)
model = quantizer.quantize(model, inputs).cuda()

```

## (To be deprecated) A8W8 Dynamic Quantization

```Python
from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer
quantizer = Int8DynActInt4WeightQuantizer(groupsize=128)
model = quantizer.quantize(model)
```

This is used in [ExecuTorch](https://github.com/pytorch/executorch) to quantize llama model right now.

## (To be moved to prototype) A8W8 Dynamic Quantization with Smoothquant

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

## Notes

1. APIs have been hardware tested on A100 and T4(colab)
2. While these techniques are designed to improve model performance, in some cases the opposite can occur. This is because quantization adds additional overhead to the model that is hopefully made up for by faster matmuls (dynamic quantization) or loading weights faster (weight-only quantization). If your matmuls are small enough or your non-quantized perf isn't bottlenecked by weight load time, these techniques may reduce performance.
3. Use the PyTorch nightlies so you can leverage [tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) which is preferred over older module swap based methods because it doesn't modify the graph and is generally more composable and flexible.
