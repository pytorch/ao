# torchao.float8

This is an early version of a library for accelerating training with float8 in native PyTorch
according to the recipes laid out in https://arxiv.org/pdf/2209.05433.pdf.
The codebase strives to stay small, easily hackable, debuggable with native PyTorch tooling,
and composable with key systems such as autograd, ```torch.compile``` and distributed.
With ``torch.compile`` on, initial results show
throughput speedups of up to 1.2x on small scale (8 GPUs) LLaMa pretraining jobs.

:warning: <em>See the [feature tracker](https://github.com/pytorch/ao/issues/556) for upcoming features.</em>

:warning: <em>Backwards compatibility is not guaranteed at this point. The codebase is in active development and
will change rapidly.</em>

# Single GPU User API

We provide three per-tensor scaling strategies: dynamic, delayed and static.  See https://arxiv.org/pdf/2209.05433.pdf, Section 4.3 for more details. These strategies are configurable separately for activations (`input`), weights (`weight`) and gradients (`grad_output`).

## float8 linear with dynamic scaling for `input`, `weight` and `grad_output`

This is the most accurate recipe as every tensor is scaled dynamically.

```python
from torchao.float8 import (
    convert_to_float8_training,
    precompute_float8_dynamic_scale_for_fsdp,
)

# create model
m = Model(...)

# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "output":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

# convert all `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(m, module_filter_fn=module_filter_fn)

# optional: use FSDP
model = FSDP(model, use_orig_params=True)

# optional: enable torch.compile for improved performance
m = torch.compile(m)

# toy training loop
for _ in range(N_ITER):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()

    # specific to fsdp2 + dynamic scaling, when fp8 all-gather is turned on
    # this method is optional but is highly recommended for performance
    # it calcuclates scales for all parameters in a single all-reduce
    precompute_float8_dynamic_scale_for_fsdp(model)

```

## float8 linear with delayed scaling

This is theoretically the most performant recipe as it minimizes memory reads.

```python
from torchao.float8 import (
    convert_to_float8_training,
    sync_float8_amax_and_scale_history,
    ScalingType,
)

# create model
m = Model(...)

# optional: configure for compatibility with FSDP. Note that workarounds
# gated with config.enable_amax_init and
# config.enable_pre_and_post_forward are needed for
# autocast + compile + FSDP + float8 to work
from torchao.float8 import Float8LinearConfig, ScalingType, CastConfig
config = Float8LinearConfig(
    enable_amax_init=False,  # only needed for autocast + compile + FSDP +  float8 delayed
    enable_pre_and_post_forward=False  # only needed for autocast + compile + FSDP +  float8 delayed
    cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
    cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
    cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
)

# convert all `torch.nn.Linear` modules to `Float8Linear`, specifying scaling
# type
convert_to_float8_training(
    m,
    config=config,
)

# optional: use FSDP
model = FSDP(model, use_orig_params=True)

# optional: enable torch.compile for improved performance
m = torch.compile(m)

# toy training loop
for _ in range(N_ITER):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()

    # specific to float8 with delayed scaling: separate step to sync scales/amaxes
    # in the future, this may move to a context manager
    sync_float8_amax_and_scale_history(model)

    optimizer.step()
```

# Multi GPU User API

We compose with the `DTensor` based [distributed APIs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html),
such as FSDP, TP and SP. Please see the [torchtitan](https://github.com/pytorch/torchtitan) repository for e2e examples
on using `torchao.float8` in a distributed setting.

# Performance

A common question about float8 training is "when is float8 linear faster vs bfloat16?".  Given the M, K, N of the forward pass through your linear, you can reference the table below for a microbenchmark based speedup estimate on NVIDIA H100:

<img width="805" alt="float8_speedup" src="https://github.com/user-attachments/assets/5c5f2817-7eb7-4cab-bd03-49fe70cd31a8">

Example 1 (small shapes):
* forward input tensor size 1024x2048, linear weight size 2048x1024; M, K, N = 1024, 2048, 1024
* benchmark speedup is 0.80
* recommendation: leave this linear in bfloat16, the shapes are too small to benefit from float8 compute

Example 2 (large shapes):
* forward input tensor size 4096x8192, linear weight size 8192x16384; M, K, N = 4096, 8192, 16384
* benchmark speedup is 1.39
* recommendation: enable float8 for this linear to get a speedup

To reproduce the raw data for table above, you can run the following script

```lang=shell
python benchmarks/float8/float8_roofline.py your_output_filename.csv --gemm_time_strategy benchmarks --shape_gen_name sweep
```

## Derivation

In a bf16 linear, assume all of the time is spent in gemms.  In a float8 linear, account for max_abs and casting overhead.  We want to know when

```
bf16_gemm_time > fp8_gemm_time + fp8_overhead_time
```

Or, equivalently,

```
bf16_gemm_time - fp8_gemm_time > fp8_overhead_time
```

There are three observations we can make about the formula above:
* LHS > 0 for large shapes, with the gemm speedup approaching 2x as M, K, N increase
* LHS < 0 for small shapes, on NVIDIA H100 + cuBLAS
* RHS > 0 for all shapes, bounded by memory bandwidth, framework overhead and compiler limitations

For small shapes, a combination of (2) and (3) leads to speedup < 1.  For medium shapes, (1) and (3) are of similar magnitude and the speedup depends on M, K, N and framework and compiler behavior.  For large shapes, (1) leads to speedup > 1.

## Scaling type vs speedup

Delayed scaling is theoretically faster than dynamic scaling because of reduced read/write traffic requirements.  Today, torch.compile has a couple of limitations (see the performance section of https://github.com/pytorch/ao/issues/556) which prevent us from reaching the optimal behavior for delayed scaling, so the observed performance of delayed scaling is close to that of dynamic scaling. As the torch.compile limitations are fixed, we expect delayed scaling to eventually become more performant compared to dynamic scaling.

## torch.compile behavior vs speedup

There are a couple of limitations in how torch.compile generates float8 scaling and casting kernels (see the performance section of https://github.com/pytorch/ao/issues/556).  As the limitations get resolved, we expect to reach improved performance.

# Testing

```bash
# run single-GPU unit tests
pytest test/float8/test_base.py

# run single-GPU compile tests
pytest test/float8/test_compile.py

# run single-GPU numerics integration tests
pytest test/float8/test_numerics_integration.py

# run a two-GPU integration test on FSDP
./test/float8/test_fsdp.sh

# run integration tests on the DTensor TP/SP integration
./test/float8/test_dtensor.sh

# run integration tests on the FSDP2 integration
python test/float8/test_fsdp2/test_fsdp2.py

# run all of these tests
./test/float8/test_everything.sh
```

# Benchmarking

```bash
# benchmark the torch._scaled_mm function on LLaMa 2 70B shapes
./benchmarks/float8/bench_matmul.py

# benchmark fw/bw of `Linear` and `Float8Linear` on LLaMa 2 70B shapes
# make sure to turn on torch.compile to get the best performance
./benchmarks/float8/bench_linear_float8.py -o ../tmp/test.txt --compile
```
