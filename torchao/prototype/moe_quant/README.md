# MoE Quantization

This prototype implementation enables quantization of Mixture of Experts (MoE) models using two complementary approaches:

1. **Grouped Matrix Multiplication (`_grouped_mm`)**: Leverages PyTorch's dedicated grouped MM kernels for optimal performance
2. **Linear Decomposition Fallback**: Decomposes MoE operations into linear operations when grouped MM is unavailable or quantized kernels don't exist

## Recent Updates

This implementation has been significantly refactored to prioritize PyTorch's optimized `_grouped_mm` kernels:

- **Primary `_grouped_mm` implementation**: Now uses PyTorch's improved grouped matrix multiplication without padding requirements
- **Enhanced module swapping**: Generic `MoEMapping` class for converting existing MoE implementations
- **Flexible execution modes**: Automatic selection between grouped MM and linear decomposition based on availability and tensor types
- **Improved quantization compatibility**: Better integration with existing quantized tensor subclasses

Examples of the usage of these APIs can be found in both the `llama4_quant.py` and `torchao/_models/mixtral-moe/generate.py`

## Quantization API

## Execution Modes

The `ExpertsAOQuantizable` module automatically selects the optimal execution strategy:

### 1. Grouped Matrix Multiplication (Primary)
```python
# Uses torch._grouped_mm for optimal performance
final_out = self._forward_grouped_mm(x, expert_indices, scores, up_proj, down_proj, act_fn)
```
- **Best performance**: Leverages optimized grouped MM kernels
- **No padding required**: Uses PyTorch's improved implementation
- **Automatic selection**: Used when `decompose_grouped_mm=False` (default)

### 2. Linear Decomposition (Fallback)
```python
# Falls back to linear operations when needed
if x.shape[0] > 1:  # Multi-token
    final_out = self._forward_multi_token_linear_decomposition(...)
else:  # Single token
    final_out = self._forward_single_token_linear_decomposition(...)
```
- **Quantization compatibility**: Works with all quantized tensor subclasses
- **Automatic fallback**: Used when `decompose_grouped_mm=True` or for quantized tensors

## API

### Supported Techniques

- **BFloat16**: 16-bit floating point inference using `torch._grouped_mm`
- **Float8DynamicActivationFloat8WeightConfig**: Float8 dynamic activation and weight quantization using `torch.scaled_grouped_mm`
- **Int8WeightOnlyConfig**: 8-bit weight-only quantization using linear decomposition
- **Int4WeightOnlyConfig**: 4-bit weight-only quantization using linear decomposition

### Basic Usage

Going forward the intended direction of TorchAO's MoE quantization will be for model owners to use torch._grouped_mm and then quantize the parameters directly similar to linear quantization without requiring a module swap.

However currently the existing space has a variety of implementations and so the expectage usage will be to swap to the AO Quantizable MoE module using the new `MoEMapping` to facilitate transfering the necessary information. As an example:

```python
from torchao.prototype.moe_quant.utils import MoEMapping, MoEQuantConfig
from torchao.quantization.quant_api import quantize_, Int4WeightOnlyConfig

moe_mapping = MoEMapping(
    target_module_type=Llama4TextMoe,
    router_fqn="router",
    top_k_fqn="top_k",
    up_proj_fqn="experts.gate_up_proj",
    up_proj_part2_fqn=None,
    down_proj_fqn="experts.down_proj",
    order_of_weight_indices=(0, 2, 1),
    act_fn_fqn="experts.act_fn",
    shared_expert_fqn="shared_expert",
    return_scores=True,
    decompose_grouped_mm=True, # change this to false if doing bf16 or fp8dq
)
base_config = Int4WeightOnlyConfig() # this can be set to None to just do the swap to the AO Quantizable module

config = MoEQuantConfig(base_config, moe_mapping)

def moe_filter(module, fqn):
    return isinstance(module, YourModelsMoEModuleClass)

quantize_(model, config, moe_filter)
model = torch.compile(model, mode="reduce-overhead", fullgraph=False) # can use fullgraph for grouped_mm or single_token inference
```

### Production Examples

- **Llama4**: Complete integration example in `llama4_quant.py`
- **Mixtral**: Full pipeline with benchmarking in `torchao/_models/mixtral-moe/generate.py`

Both examples demonstrate end-to-end workflows including model loading, conversion, quantization, and performance evaluation.

## Performance Notes

### Grouped MM vs Linear Decomposition

- **Grouped MM**: Provides optimal performance for multi-token MoE operations using PyTorch's dedicated kernels
- **Linear Decomposition**: Enables quantization compatibility for existing techniques and is generally faster for single-token inference

## Alternative Quantization API: FakeExtraDimTensor

For broader compatibility with existing quantization techniques, we provide an alternative approach using `FakeExtraDimTensor`. This method simulates 3D tensors by storing multiple 2D tensors and implementing slicing/indexing operations, enabling compatibility with all existing linear quantization techniques without modifications. This can be done using the same API as above but adding the option for use_fake_extra_dim_tensor

### Usage

```python
from torchao.prototype.moe_quant.utils import UseFakeExtraDimTensor

# Configure with FakeExtraDimTensor
config = MoEQuantConfig(...
    use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE,  # Key difference
)
```

### Configuration Options

- **`UseFakeExtraDimTensor.TRUE`**: Always use the fake tensor approach
- **`UseFakeExtraDimTensor.FALSE`**: Use the direct 3D tensor approach
- **`UseFakeExtraDimTensor.AS_FALLBACK`** (default): Try direct approach first, fallback to fake tensor if needed

### Trade-offs

**Benefits:**
- Compatible with all existing quantization techniques without modifications
- Better memory characteristics (tensors stored separately)
- Flexible and general approach

**Limitations:**
- Less performant than direct 3D tensor approach
- Fullgraph compilation not supported for single/multi-token inference
- Additional overhead from tensor simulation
