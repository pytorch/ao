## Fused DoRA Kernels

Fused DoRA layer implementation that reduces number of individual kernels from ~10 -> 5.

## Contents

- [Background](#background)
- [Optimization](#optimization)
- [Key Contributions](#key-contributions)
- [Usage](#usage)
- [Tests](#tests)
- [Benchmarks](#benchmarks)
- [Profiling](#profiling)

## Background

[DoRA](https://arxiv.org/abs/2402.09353) (weight-decomposed low-rank adaptation) is a variant of LoRA that decomposes the LoRA update into magnitude and vector components.

The DoRA layer is roughly as follows:

```python
    dora_out = (x @ base_weight.T + lora_out) * magnitude_scale
```

where:

```python
    lora_out = lora_B(lora_A(x))
    magnitude_scale = magnitude_vector / (base_weight + lora_B.weight @ lora_A.weight).norm(p=2, dim=1)
```

- `lora_A` and `lora_B` are `linear` layers with weight shapes `rank x in_features` and `out_features x rank`.
- `base_weight` is the weight of the frozen `linear` layer of shape `out_features x in_features`.
- `magnitude_vector` is initialized as the columnwise `2-norm` of the frozen weight (shape `out-features`).
- `x` are the inputs of shape `batch_size x seqlen x in_features`

## Optimization

After initial profiling, and as outlined above, the `DoRA` update layer requires multiple kernels.

In order of compute intensity:

- 4 GEMMs:
  - `x @ base_weight`
  - `lora_B(lora_A(x))`
  - `lora_B.weight @ lora_A.weight`
- 1 Reduction: `2-norm`
- 4 Elementwise: matrix-matrix additions (2) and broadcasted matrix-vector multiplications (2).

While `torch.compile` (and `CUDA` graphs) can partially mitigate the overhead of multiple small kernels and improve compute efficiency of individual kernels, there remains room for additional optimization by reordering the computations to facilitate fusions, and more importantly, exploiting the unique shapes of the GEMMs, thereby decreasing the number of kernel launches and increasing the compute intensity of each kernel.

## Key Contributions

**1 - Small K Fused Kernel**

Note that the `lora_B.weight @ lora_A.weight` has a specific shape, where `K << {M, N}`. That is, `lora_B.weight` is `out_features x lora_rank` and `lora_A.weight` is `lora_rank x in_features`.

Since `lora_rank` is typically `< 64` while `{in,out}-features` are typically `> 4096` (e.g., `Llama MLP / QKV projections`), this `GEMM` is inefficient, since each `CTA` loads a block, only to perform a few `MAC` iterations given small `K`.

Moreover, note that the result of this `GEMM` is not needed -- we only need the `2-norm` of this computation.

Combining these two observations, we can write a fused kernel where:

1. Each `CTA` computes an _entire_ row of the output matrix, with the key assumption that `BLOCK_K = K`. That is, each `CTA` does a single MAC iteration to compute a `BLOCK_M x BLOCK_N` output, then iterates across dimension `N`.
2. Since each block processes an entire row, we can now additionally fuse a grid-wise reduction along `axis=1` into the kernel. In this case, we can directly fold the `2-norm` computation into the `GEMM`.
3. As an added bonus, we can also include the `base_weight` elementwise addition and `magnitude_vector` multiplication into the `GEMM` epilogue.

Altogether, this allows us to fuse the following computation into a single kernel:

```python
    magnitude_scale = magnitude_vector / (base_weight + lora_B.weight @ lora_A.weight).norm(p=2, dim=1)
```

**2 - Fused Epilogue GEMM**

Additionally, instead of computing the base layer output before the `DoRA / LoRA` updates, we can compute the latter (`loRA layer` and `magnitude_scale`) first, and fold these into the epilogue of the base layer `GEMM`:

```python

    #DoRA / LoRA updates
    lora_out = lora_B(lora_A(x))
    magnitude_scale = magnitude_vector / (base_weight + lora_B.weight @ lora_A.weight).norm(p=2, dim=1)

    #This is now a single kernel
    final_out = (x @ base_weight.T + lora_out) * magnitude_scale
```

## Usage

The fused kernels can be used to implement `DoRA` / `QDoRA` layers.

A reference implementation is provided in `dora.dora_layer.DoRALinear`, which defines a base `QDoRA` linear layer (with a stub `dequantize` method) along with corresponding `BNBDoRALinear` and `HQQDoRALinear` subclasses, which override `dequantize` with their respective methods.

_Example_

```python
    import torch
    from bitsandbytes.nn import Linear4bit
    from torchao.prototypes.dora.dora_layer import BNBDoRALinear

    bs, seqlen = 1, 512
    dtype = torch.float16
    in_features, out_features, lora_rank = 4096, 4096, 16
    x = torch.randn(bs, seqlen, in_features, dtype=dtype, device="cuda")

    #Construct bitsnbytes QDoRA layer
    base_layer = Linear4bit(
            input_features=in_features,
            output_features=out_features,
            bias=False,
            quant_type="nf4",
            compute_dtype=dtype,
        ).cuda()
    base_layer.quant_state.dtype = base_layer.compute_dtype
    dora_layer = BNBDoRALinear(base_layer, lora_rank)

    #Run reference forward pass
    ref = dora_layer.forward(x)

    #Run fused forward pass
    fused_out = dora_layer.forward_fused(x)
```

See `test/test_dora_layer.py` and `benchmarks/dora_bench.py` for more detailed usage.

### Tests

See `test/dora/test*`, for correctness checks of the fused kernels and layers.

## Benchmarks

See `benchmarks/dora_bench.py`.

```python
python benchmarks/dora_bench.py --help
```

Run with flag `--kernel` set to one of `{dora-colnorm,dora-mm-epilogue}`, to benchmark the respective fused kernels against a reference `torch` / `torch.compile` implementation, or `--kernel=dora-full` to bench against the entire `DoRA` computation.

Additionally, passing either `--kernel={dora-bnb, dora-hqq}` will bench a reference `QDoRA` layer against their fused implementations.

## Profiling

The reference `DoRALinear` layer described above also has an instrumented forward pass with annotated regions for each of the `DoRA` ops.

An example script for running a profiled forward pass is provided in `dora/dora_profile.py`.

To run with `torch.profiler`:

```
python dora_profile.py
```

which outputs chrome trace to default folder `dora_profiles`.

To run with `nsys`:

```
nsys profile --capture_range=cudaProfilerApi ... python dora_profile.py --profiler=nsys
```

where `...` are other desired `nsys` options.

Note that `--capture_range=cudaProfilerApi` is required.
