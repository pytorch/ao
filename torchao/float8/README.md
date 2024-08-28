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
