# Quantization-Aware Training (QAT)

Quantization-Aware Training (QAT) refers to applying fake quantization during the
training or fine-tuning process, such that the final quantized model will exhibit
higher accuracies and perplexities. Fake quantization refers to rounding the float
values to quantized values without actually casting them to dtypes with lower
bit-widths, in contrast to post-training quantization (PTQ), which does cast the
quantized values to lower bit-width dtypes, e.g.:

```
# PTQ: x_q is quantized and cast to int8
# scale and zero point (zp) refer to parameters used to quantize x_float
# qmin and qmax refer to the range of quantized values
x_q = (x_float / scale + zp).round().clamp(qmin, qmax).cast(int8)

# QAT: x_fq is still in float
# Fake quantize simulates the numerics of quantize + dequantize
x_fq = (x_float / scale + zp).round().clamp(qmin, qmax)
x_fq = (x_fq - zp) * scale
```

## API

torchao currently supports two QAT schemes for linear layers:
- int8 per token dynamic activations + int4 per group weights
- int4 per group weights (using the efficient [int4 tinygemm kernel](https://github.com/pytorch/pytorch/blob/a672f6c84e318bbf455f13dfdd3fd7c68a388bf5/aten/src/ATen/native/cuda/int4mm.cu#L1097) after training)

QAT typically involves applying a transformation to your model before and after training.
In torchao, these are represented as the prepare and convert steps: (1) prepare inserts
fake quantize operations into linear layers, and (2) convert transforms the fake quantize
operations to actual quantize and dequantize operations after training, thereby producing
a quantized model (dequantize operations are typically fused with linear after lowering).
Between these two steps, training can proceed exactly as before.

![qat](images/qat_diagram.png)

To use QAT in torchao, apply the prepare step using the appropriate Quantizer before
training, then apply the convert step after training for inference or generation.
For example, on a single GPU:

```python
import torch
from torchtune.models.llama3 import llama3
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

# Smaller version of llama3 to fit in a single GPU
model = llama3(
    vocab_size=4096,
    num_layers=16,
    num_heads=16,
    num_kv_heads=4,
    embed_dim=2048,
    max_seq_len=2048,
).cuda()

# Quantizer for int8 dynamic per token activations +
# int4 grouped per channel weights, only for linear layers
qat_quantizer = Int8DynActInt4WeightQATQuantizer()

# Insert "fake quantize" operations into linear layers.
# These operations simulate quantization numerics during
# training without performing any dtype casting
model = qat_quantizer.prepare(model)

# Standard training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(10):
    example = torch.randint(0, 4096, (2, 16)).cuda()
    target = torch.randn((2, 16, 4096)).cuda()
    output = model(example)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Convert fake quantize to actual quantize operations
# The quantized model has the exact same structure as the
# quantized model produced in the corresponding PTQ flow
# through `Int8DynActInt4WeightQuantizer`
model = qat_quantizer.convert(model)

# inference or generate
```

Users can also leverage our integration with [torchtune](https://github.com/pytorch/torchtune)
and apply quantized-aware fine-tuning as follows:

```
tune run --nproc_per_node 8 qat_distributed --config llama3/8B_qat_full
```

For more detail, please refer to [this QAT tutorial](https://pytorch.org/torchtune/main/tutorials/qat_finetune.html).


## Evaluation Results

Evaluation was performed on 6-8 A100 GPUs (80GB each) using the torchtune QAT
integration described above. We fine-tune [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
on the [C4 dataset](https://huggingface.co/datasets/allenai/c4) (en subset)
for 5000 steps using a group size of 256 for the weights. Note that extensive
hyperparameter tuning may further improve these results.

Results for int8 per token dynamic activations + int4 per group weights, using a learning rate of 2e-5:

|                  | hellaswag<br>(acc) | hellaswag<br>(acc_norm) | wikitext<br>(word_perplexity) | wikitext<br>(byte_perplexity) | wikitext<br>(bits_per_byte) |
| ---------------- | ------ | ------ | ------ | ------ | ------ |
| No quantization  | 57.86% | 76.60% | 8.905  | 1.505  | 0.590  |
| PTQ              | 51.74% | 70.66% | 11.878 | 1.588  | 0.668  |
| QAT (quantized)  | 57.25% | 76.51% | 9.859  | 1.534  | 0.617  |
| PTQ degradation  | -6.11% | -5.94% | +2.973 | +0.083 | +0.078 |
| QAT degradation  | -0.61% | -0.21% | +0.947 | +0.029 | +0.027 |

Results for int4 per group weights, using a learning rate of 2e-6. For this quantization scheme, the
quantized path uses the more efficient [int4 tinygemm kernel](https://github.com/pytorch/pytorch/blob/a672f6c84e318bbf455f13dfdd3fd7c68a388bf5/aten/src/ATen/native/cuda/int4mm.cu#L1097).

|                  | hellaswag<br>(acc) | hellaswag<br>(acc_norm) | wikitext<br>(word_perplexity) | wikitext<br>(byte_perplexity) | wikitext<br>(bits_per_byte) |
| ---------------- | -------- | ------- | ------ | ------ | ------ |
| No quantization  | 57.16%  | 77.02% | 8.858  | 1.504  | 0.589  |
| PTQ              | 55.06%  | 74.24% | 10.311 | 1.547  | 0.630  |
| QAT (quantized)  | 55.86%  | 75.06% | 10.134 | 1.542  | 0.625  |
| PTQ degradation  | -2.10%  | -2.78% | +1.453 | +0.043 | +0.041 |
| QAT degradation  | -1.30%  | -1.96% | +1.276 | +0.038 | +0.036 |

For more details, please refer to [this blog post](https://pytorch.org/blog/quantization-aware-training).
