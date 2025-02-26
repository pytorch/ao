###TODO get blocksparse numbers
###update BC config stuff to add sparsity
###A4W4 cutlass kernel blurb?

# Highlights

We are excited to announce the 0.9.0 release of torchao! This release moves a number of sparsity techniques out of prototype including a new feature supermask, adds a cutlass kernel for 4 bit dynamic quantization and more!

### Block Sparsity promoted out of prototype (https://github.com/pytorch/ao/pull/1729, https://github.com/pytorch/ao/pull/1734)
We’ve promoted block sparsity out of torchao.prototype and made several performance improvements. 
You can accelerate your models with block sparsity as follows:

```python
from torchao.sparsity import sparsify, block_sparse_weight
sparsify_(model, block_sparse_weight(blocksize=64))
```

DETAILS ON HOW TO GENERATE THESE NUMBERS

##### Blocksparse Benchmarks
| `-q parameter` | Average tokens/sec | Average Bandwidth in GB/s | Peak Memory Usage in GB | Model Size in GB |
| :--- | ---: | ---: | ---: | ---: |
| 2:4 sparsity | 95.24 | 258.55 | 13.90 | 13.21 |
| `-q int8wo` | 155.31 | 1028.37 | 8.97 | 6.62 |
| `-q int4wo-32` | 186.70 | 774.98 | 5.31 | 4.15 |
| `-q int4wo-hqq` | 186.47 | 774.01 | 5.04 | 4.15 |
| `-q int8dq` | 49.64 | 328.72 | 9.44 | 6.62 |
| `-q w4a8-cutlass` (**tuned**) | 119.31 | 394.86 | 4.52 | 3.31 |

## BC Breaking

#### quantize_ configuration callables -> configs (https://github.com/pytorch/ao/pull/1595, https://github.com/pytorch/ao/pull/1694, https://github.com/pytorch/ao/pull/1696, https://github.com/pytorch/ao/pull/1697) 

We are migrating the way `quantize_` workflows are configured from callables (tensor subclass inserters) to direct configuration (config objects).  Motivation: align with the rest of the ecosystem, enable inspection of configs after instantiation, remove a common source of confusion.

**What is changing:**

Specifically, here is how the signature of `quantize_`'s second argument will change:

```python
#
# torchao v0.8.0 and before
#
def quantize(
    model: torch.nn.Module,
    apply_tensor_subclass: Callable[[torch.nn.Module], torch.nn.Module],
    ...,
): ...

#
# torchao v0.9.0
#
def quantize(
    model: torch.nn.Module,
    config: Union[AOBaseConfig, Callable[[torch.nn.Module], torch.nn.Module]],
    ...,
): ...

#
# torchao v0.10.0 or later (exact version TBD)
#
def quantize(
    model: torch.nn.Module,
    config: AOBaseConfig,
    ...,
): ...
```

1. the name of the second argument to `quantize_` changed from `apply_tensor_subclass` to `config`.  Since the vast majority of callsites today are passing in configuration with a positional argument, this change should not affect most people.
2. the type of the second argument to `quantize_` will change from `Callable[[torch.nn.Module], torch.nn.Module]` to `config: AOBaseConfig`, following a deprecation process detailed below.
3. for individual workflows, the user facing API name changed from snake case (`int8_weight_only`) to camel case (`Int8WeightOnlyConfig`).  All argument names for each config are kept as-is.  We will keep the old snake case names (`int8_weight_only`) around and alias them to the new names (`int8_weight_only = Int8WeightOnlyConfig`), to avoid breaking callsites.  We plan to keep the old names forever.  Here are all the workflow config name changes:

| old name (will keep working) | new name (recommended) |
| --- | --- |
| `int4_weight_only` | `Int4WeightOnlyConfig` |
| `float8_dynamic_activation_float8_weight` | `Float8DynamicActivationFloat8WeightConfig
` |
| `float8_static_activation_float8_weight` | `Float8StaticActivationFloat8WeightConfig` |
| `float8_weight_only` | `Float8WeightOnlyConfig` |
| `fpx_weight_only` | `FPXWeightOnlyConfig` |
| `gemlite_uintx_weight_only` | `GemliteUIntXWeightOnlyConfig` |
| `int4_dynamic_activation_int4_weight` | `Int4DynamicActivationInt4WeightConfig` |
| `int8_dynamic_activation_int4_weight` | `Int8DynamicActivationInt4WeightConfig` |
| `int8_dynamic_activation_int8_semi_sparse_weight` | n/a (deprecated) |
| `int8_dynamic_activation_int8_weight` | `Int8DynamicActivationInt8WeightConfig` |
| `int8_weight_only` | `Int8WeightOnlyConfig` |
| `uintx_weight_only` | `UIntXWeightOnlyConfig` |

Configuration for prototype workflows using `quantize_` will be migrated at a later time.

**How these changes can affect you:**
1. If you are a user of existing `quantize_` API workflows and are passing in config by a positional argument (`quantize_(model, int8_weight_only(group_size=128))`), **you are not affected**.  This syntax will keep working going forward.  You have the option to migrate your callsite to the new config name (`quantize_(model, Int8WeightOnlyConfig(group_size=128))` at your own pace.
2. If you are a user of existing `quantize_` API workflows and are passing in config by a keyword argument (`quantize_(model, tensor_subclass_inserter=int8_weight_only(group_size=128))`), your callsite will break.  You will need to change your callsite to `quantize_(model, config=int8_weight_only(group_size=128))`.  We don't expect many people to be in this bucket.
3. If you are a developer writing new workflows for the `quantize_` API, you will need to use the new configuration system.  Please see https://github.com/pytorch/ao/issues/1690 for details.
4. If you are a user of `sparsify_`, you are not affected for now and a similar change will happen in a future version of torchao.

This migration will be a two step process:
* in torchao v0.9.0, we will enable the new syntax while starting the deprecation process for the old syntax.
* in torchao v.0.10.0 or later, we will remove the old syntax

Please see https://github.com/pytorch/ao/issues/1690 for more details.



### Block Sparsity imports after moved out of prototype (https://github.com/pytorch/ao/pull/1734)

Before:

```python
from torchao.prototype.sparsity.superblock.blocksparse import block_sparse_weight
```

After:
```python
from torchao.sparsity import block_sparse_weight
```


## Deprecations

#### deprecation of the `set_inductor_config` argument of `quantize_` (https://github.com/pytorch/ao/pull/1716)

We are migrating the `set_inductor_config` argument of `quantize_` to individual workflows.  Motivation:
1. this functionality was intended for inference, and we don't want to expose it to future training workflows that we plan to add to `quantize_`.
2. higher level, this flag couples torchao workflows with torch.compile, which is not ideal.  We would rather keep these systems decoupled at the `quantize_` API level, with individual workflows opting in as needed.

##### Impact on users

* for torchao v0.9.0:: if you are passing in `set_inductor_config` to `quantize_`, your callsite will keep working with a deprecation warning.  We recommend that you migrate this option to your individual workflow.
* for a future version of torchao: the `set_inductor_config` argument will be removed from `quantize_`.

##### API changes

```python
# torchao v0.8.x
def quantize_(
    ...,
    set_inductor_config: bool = True,
    ...,
): ...

# torchao v.0.9.0
def quantize_(
    ...,
    set_inductor_config: Optional[bool] = None,
    ...,
):
    # if set_inductor_config != None, throw a deprecation warning
    # if set_inductor_config == None, set it to True to stay consistent with old behavior

# torchao v TBD (a future release)
def quantize_(
    ...,
):
    # set_inductor_config is removed from quantize_ and moved to relevant individual workflows
```

Please see https://github.com/pytorch/ao/issues/1715 for more details.

#### Deprecation warning for float8 training delayed and static scaling (https://github.com/pytorch/ao/pull/1681, https://github.com/pytorch/ao/issues/1680)
We plan to deprecate delayed and static scaling from torchao.float8 training codebase due to lack of real world use cases for delayed/static scaling (dynamic scaling is required for higher accuracy) and 
complexity tax for supporting these features.
* for torchao v0.9.0: add deprecation warning for delayed and static scaling
* for torchao v0.10.0: deprecate delayed and static scaling


## New Features

#### Supermask for improving accuracy for sparse models (https://github.com/pytorch/ao/pull/1729)

Supermask (https://pytorch.org/blog/speeding-up-vits/) is a technique for applying structured sparsity to neural networks using a learned mask. It works by learning a continuous mask (scores) that is applied element-wise to the weights of a neural network layer. To prepare a model for training you can use the following:

During inference, the binary mask is applied element-wise to the weights, pruning the weights that correspond to a 0 in the mask, resulting in a sparse network that can be efficiently computed.

```python
from torchao.sparsity import SupermaskLinear, block_sparse_weight
sparsify_(model, lambda x: SupermaskLinear.from_linear(x, block_size=64, sparsity_level=0.9)
# training here

# for inference speedup, collapse supermask back into a linear layer and then into a block sparse weight
sparsify_(model, lambda x: SupermaskLinear.to_linear(x, sparsity_level=0.9)
sparsify_(model, block_sparse_weight(blocksize=64))
```

#### Add CUTLASS-based W4A4 kernel (https://github.com/pytorch/ao/pull/1515) 

This kernel which adds support for 4 bit dynamic activation quantization + 4 bit weight quantization can be used as follows:

```python
from torchao.quantization import int4_dynamic_activation_int4_weight
quantize_(model, int4_dynamic_activation_int4_weight)
```

## Improvements

#### Early prototype MXFP8 and MXFP4 training and inference support for NVIDIA Blackwell GPUs

In torchao v0.9.0, we include **very early** support for training and inference on the NVIDIA Blackwell GPUs following the microscaling recipes from https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf, and backed by real MX gemms.

Here is how to use the current prototype APIs.  

:warning: <em> Note that torch.compile support is not fully there yet, there are no guarantees on performance at this time, and we expect to change these APIs rapidly as we iterate in future versions of torchao.  Please see https://github.com/pytorch/ao/issues/556 for more details. </em>

MX training

```python
from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_linear
from torchao.prototype.mx_formats.config import MXLinearConfig, MXGemmKernelChoice
from torchao.utils import is_sm_at_least_100

# early prototype: on MX-enabled hardware, you can use the real MX gemm backed by
# torchao's CUTLASS kernels. In the future, we will also add cuBLAS kernel support.
gemm_kernel_choice = MXGemmKernelChoice.EMULATED
if is_sm_at_least_100():
    gemm_kernel_choice = MXGemmKernelChoice.CUTLASS

m = torch.nn.Sequential(torch.nn.Linear(32, 32)).cuda()
config = MXLinearConfig(
    elem_dtype=torch.float8_e4m3fn, 
    block_size=32, 
    gemm_kernel_choice=gemm_kernel_choice,
)
swap_linear_with_mx_linear(m, config=config)

# training loop (not shown)
```

MX inference, weights are in MX and matmul is in high precision.

```python
from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_inference_linear
from torchao.prototype.mx_formats.config import MXLinearConfig

m = torch.nn.Sequential(torch.nn.Linear(32, 32)).cuda()
config = MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32)
swap_linear_with_mx_inference_linear(m, config=config)

# do inference (not shown)
```

The additional features for MX support in v0.9.0 were enabled by:
* Add mx_fp8_bf16 kernel (https://github.com/pytorch/ao/pull/1637)
* Support mixed MX element dtype in `mx_mm` function and `MXLinear`. (https://github.com/pytorch/ao/pull/1667)
* move block_size and elem_dtype into MXLinearConfig (https://github.com/pytorch/ao/pull/1689)
* hook up mxfp8 and mxfp4 CUTLASS kernels to MXLinear (https://github.com/pytorch/ao/pull/1713)
* add ceil and RNE rounding modes to the cast from fp32 to e8m0 (https://github.com/pytorch/ao/pull/1643)



### quantize_

* Clean up linear_int8_dynamic_activation_intx_weight_subclass (https://github.com/pytorch/ao/pull/1553)


### Experimental

* Q dq layout (https://github.com/pytorch/ao/pull/1642)
* Add support for kleidi AI quantization schemes (https://github.com/pytorch/ao/pull/1447)

### SAM2

* Add modal script extensions (https://github.com/pytorch/ao/pull/1500)
* Increase export usage, small perf improvements (https://github.com/pytorch/ao/pull/1673)
* Model experiments QoL improvements (https://github.com/pytorch/ao/pull/1683)
* Collect p90 latency statistics (https://github.com/pytorch/ao/pull/1703)

### Training

* Support power of 2 scaling factors in float8 training with rowwise scaling and use e4m3 in fwd and bwd pass (https://github.com/pytorch/ao/pull/1670)
* Float8 training: clean up recipe names (https://github.com/pytorch/ao/pull/1730)
* Float8 training: make the "config from recipe" API polished (https://github.com/pytorch/ao/pull/1731)
* Make FakeQuantizer expose useful config details when printed (https://github.com/pytorch/ao/pull/1717)

### Other

* Relax dtype requirements for int4 and float8 quants in autoquant (https://github.com/pytorch/ao/pull/1571)
* Update __init__.py to load experimental ops even if other C++ ops are not found (https://github.com/pytorch/ao/pull/1565)


## Bug Fixes

* Fix torch.intx support in FakeQuantizeConfig (https://github.com/pytorch/ao/pull/1544)
* Fix float related autoquant options (https://github.com/pytorch/ao/pull/1562)
* Fix #1559, sparsity instead of sparstiy (https://github.com/pytorch/ao/pull/1560)
* Fix `.item()` issue in running parallel evaluation for BO mixed precision (https://github.com/pytorch/ao/pull/1630)
* Add more stringent test for CPUOffloadOptimizer (https://github.com/pytorch/ao/pull/1650)
* Fix LR scheduler issue with CPU offload optimizer (https://github.com/pytorch/ao/pull/1649)
* Add int8 dynamic activation + int8 weight only test to TensorParallel (https://github.com/pytorch/ao/pull/1657)
* Fix compile issue for Marlin qqq on sm<8.0 (https://github.com/pytorch/ao/pull/1651)
* Fix use_hqq for int4_weight_only quantize (https://github.com/pytorch/ao/pull/1707)
* Unbreak float8 static quant tutorial (https://github.com/pytorch/ao
/pull/1709)
* Fix `DDP` with `nf4` (https://github.com/pytorch/ao/pull/1684)
* Fix tensor parallelism for float8 training with rowwise scaling (https://github.com/pytorch/ao/pull/1718)


## Performance

* Add workaround to reduce FSDP memory usage for float8 rowwise training  (https://github.com/pytorch/ao/pull/1629)

## Documentation

* Update supported dtypes for fp8 (https://github.com/pytorch/ao/pull/1573)
* Sparsity docs update (https://github.com/pytorch/ao/pull/1590)
* Sparsity getting started docs (https://github.com/pytorch/ao/pull/1592)
* Fix broken link on doc page (https://github.com/pytorch/ao/pull/1582)
* Add quick start guide for first time users (https://github.com/pytorch/ao/pull/1611)
* Update api_ref_dtypes docs (https://github.com/pytorch/ao/pull/1610)
* Add module swap -> tensor subclass migration tutorial (https://github.com/pytorch/ao/pull/1596)
* Update docs to refer to version.html (https://github.com/pytorch/ao/pull/1631)
* Split contributor guide into quantization overview (https://github.com/pytorch/ao/pull/1618)
* Update api_ref_quantization docs (https://github.com/pytorch/ao/pull/1619)
* Migrate static quant tutorials to direct configuration (https://github.com/pytorch/ao/pull/1710)
* Update torchao READMEs with new configuration APIs (https://github.com/pytorch/ao/pull/1711)
* Update SAM2 README.md (https://github.com/pytorch/ao/pull/1735)
* Add rowwise scaling README.md entry for float8 training(https://github.com/pytorch/ao/pull/1733)

## Developers

* Ruff lint (https://github.com/pytorch/ao/pull/1646)
* Consolidate `ZeroPointDomain.NONE` & `None` zero point domains (https://github.com/pytorch/ao/pull/1556)
* Only run docs build in CI if docs have changed (https://github.com/pytorch/ao/pull/1589)
* Add separate quantization primitives for float8 (https://github.com/pytorch/ao/pull/1597)
* Add boiler plate code to Tensor subclass (https://github.com/pytorch/ao/pull/1663)
* Change TORCH_LIBRARY to TORCH_LIBRARY_FRAGMENT (https://github.com/pytorch/ao/pull/1645)
* Reformat C++ kernels (https://github.com/pytorch/ao/pull/1723)
* Add torchao/experimental CI test (https://github.com/pytorch/ao/pull/1586)

## New Contributors
* @jaewoosong made their first contribution in https://github.com/pytorch/ao/pull/1560
* @haodongucsb made their first contribution in https://github.com/pytorch/ao/pull/1630
* @nikhil-arm made their first contribution in https://github.com/pytorch/ao/pull/1447
* @ngc92 made their first contribution in https://github.com/pytorch/ao/pull/1650
* @balancap made their first contribution in https://github.com/pytorch/ao/pull/1667

**Full Changelog**: https://github.com/pytorch/ao/compare/v0.8.0...v0.9.0-rc1



