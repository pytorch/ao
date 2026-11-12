# Activation compression for sparse Squared-ReLU training

[pytorch/ao#1920](https://github.com/pytorch/ao/issues/1920) asks whether the
>85% activation sparsity of Squared-ReLU FFNs is better spent on compression
than on a 2:4 sparse matmul. The sparsity shows up while training, where the
autograd engine stores the activation for backward. Inference does not store
it, so compressing it does not shrink inference memory.

This prototype packs those saved tensors and unpacks them on the way back.

## Formats

`bitpack` keeps a bit per element plus the nonzero values, on the same device
as the activation. Bytes per element are `1/8 + density * element_size`:

| sparsity | fp32 / bf16-fp16 ratio |
| --- | --- |
| 50% | 1.9x / 1.8x |
| 85% | 5.5x / 4.7x |
| 95% | 12.3x / 8.9x |

`-0` is restored as `+0`. Every other finite value, and NaN/Inf payloads, round-trip exactly.

`zlib` is a byte codec in the same family as nvCOMP (LZ4, zstd, gdeflate). It is the portable way to time that style of compressor. It moves the tensor to CPU. nvCOMP is not a dependency; for exact zeros, bitpack is the better fit because the packed form never leaves the device and the ratio does not depend on how the zeros are arranged.

Dense tensors (weights, untrained pre-activations) are left alone. A tensor is packed only when it is a real floating tensor, large enough (`min_numel`, default 4096), and at least `min_sparsity` sparse (default 0.5). The packed form is also rejected when it is not actually smaller.

## Usage

```python
import torch
from torchao.prototype.activation_compression import compress_saved_activations

model = ...  # Squared-ReLU FFN, trained into the >85% sparse regime
x = ...

with compress_saved_activations(method="bitpack") as stats:
    loss = model(x)
loss.backward()

print(stats.dense_bytes, stats.stored_bytes, stats.ratio, stats.n_compressed)
```

`backward` may run outside the `with` block. The unpack function is stored with each saved tensor. While backward runs, one activation is dense again at a time; the memory that stays live between forward and backward is the packed set. Holding your own reference to the dense activation keeps the dense copy alive.

The hook counts every saved tensor, including dense weights. `stats.ratio` is therefore diluted by those weights. The activation-only ratio is what `measure_compression` reports.

## Benchmark

```bash
python benchmarks/prototype/activation_compression/benchmark_activation_compression.py
python benchmarks/prototype/activation_compression/benchmark_activation_compression.py \
  --shapes 8192x11008 --sparsities 0.85 0.9 0.95 --dtypes float32 bfloat16
```

The first run prints the activation-only ratio and compress/decompress time, then a Squared-ReLU FFN step with and without the hook. Ratio should track the table above. The extra step time is the overhead to weigh against the bytes saved, and it depends on the machine.

A CPU run on a 1024×4096 activation (`warmup=2`, `iters=5`) landed on the formula and was exact (`max abs error = 0`):

| method | dtype | sparsity | dense | packed | ratio | pack | unpack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bitpack | float32 | 0.85 | 16.00 MiB | 2.90 MiB | 5.52× | 4.2 ms | 3.4 ms |
| bitpack | bfloat16 | 0.85 | 8.00 MiB | 1.70 MiB | 4.71× | 4.0 ms | 2.9 ms |
| bitpack | float32 | 0.95 | 16.00 MiB | 1.30 MiB | 12.30× | 3.3 ms | 2.2 ms |
| bitpack | bfloat16 | 0.95 | 8.00 MiB | 0.90 MiB | 8.89× | 3.2 ms | 2.3 ms |
| zlib | float32 | 0.85 | 16.00 MiB | 3.65 MiB | 4.38× | 8.6 s | 2.0 s |
| zlib | bfloat16 | 0.85 | 8.00 MiB | 2.02 MiB | 3.97× | 4.3 s | 1.1 s |

zlib is the stand-in for an nvCOMP-style byte codec. It compresses almost as well as bitpack and costs seconds per activation on CPU, because the tensor has to be copied into a byte buffer and compressed there. Bitpack stays on device and is a few milliseconds for this shape.

On a tiny FFN (batch 32, dim 256, hidden 1024, measured sparsity 0.85) the hook packed 3 of 6 saved tensors. Stored bytes fell from 1.44 MiB to 1.13 MiB (1.27×) because the dense weights are saved too and are not packed. The step went from 0.22 ms to 0.85 ms with bitpack. That overhead is large only because the matmul itself is tiny; the absolute pack cost is the few milliseconds in the table. zlib made the same step take 251 ms.
