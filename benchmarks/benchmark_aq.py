"""Benchmarks for affine quantized tensor, this includes int8 dynamic quant, int8 weight only quant and int4 weight only quant APIs
"""
import torch
from torchao.quantization.subclass import (
    Int8WeightOnlyQuantizedLinearWeight,
    Int4WeightOnlyQuantizedLinearWeight,
)
from torchao.utils import (
    TORCH_VERSION_AFTER_2_4,
)
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
)
import copy

class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False).to(torch.float)
        self.linear2 = torch.nn.Linear(n, k, bias=False).to(torch.float)

    def example_inputs(self, batch_size=1, dtype=torch.float, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def _ref_change_linear_weights_to_int8_dqtensors(model, filter_fn=None, **kwargs):
    """
    The deprecated implementation for int8 dynamic quant API, used as a reference for
    numerics and performance
    """
    from torchao.quantization.quant_api import _in_features_greater_than_16
    from torchao.quantization.quant_api import _is_linear
    from torchao.quantization.quant_api import _get_subclass_inserter
    from torchao.quantization.subclass import Int8DynamicallyQuantizedLinearWeight

    if filter_fn is None:
        filter_fn = lambda *args: _is_linear(*args) and _in_features_greater_than_16(
            *args
        )

    _replace_with_custom_fn_if_matches_filter(
        model, _get_subclass_inserter(Int8DynamicallyQuantizedLinearWeight, enable_parametrization=False, **kwargs), filter_fn
    )

def _get_ref_change_linear_weights_to_woqtensors(deprecated_tenosr_subclass):
    def _ref_change_linear_weights_to_woqtensors(model, filter_fn=None, **kwargs):
        """
        The deprecated implementation for weight only quant API, used as a reference for
        numerics and performance
        """
        from torchao.quantization.quant_api import _is_linear
        from torchao.quantization.quant_api import _get_subclass_inserter

        filter_fn = kwargs.pop("filter_fn", _is_linear)

        _replace_with_custom_fn_if_matches_filter(
            model,
            _get_subclass_inserter(deprecated_tenosr_subclass, enable_parametrization=True, **kwargs),
            filter_fn,
        )

    return _ref_change_linear_weights_to_woqtensors

_ref_change_linear_weights_to_int8_woqtensors = _get_ref_change_linear_weights_to_woqtensors(Int8WeightOnlyQuantizedLinearWeight)
_ref_change_linear_weights_to_int4_woqtensors = _get_ref_change_linear_weights_to_woqtensors(Int4WeightOnlyQuantizedLinearWeight)


def _bench_quantized_tensor_subclass_perf(api, ref_api, kwargs=None):
    if kwargs is None:
        kwargs = {}

    m = ToyLinearModel(1024, 1024, 1024).eval().to(torch.bfloat16).to("cuda")
    m_ref = copy.deepcopy(m)
    # setting batch_size to 20 to be compatible with the kernel
    example_inputs = m.example_inputs(batch_size=20, dtype=torch.bfloat16, device="cuda")

    api(m, **kwargs)

    # reference
    ref_api(m_ref, **kwargs)

    res = m(*example_inputs)
    ref = m_ref(*example_inputs)

    assert torch.equal(res, ref)

    # perf comparison
    from torchao.utils import benchmark_model
    # warmup
    WARMUP = 5
    RUNS = 100
    m = torch.compile(m, mode='max-autotune', fullgraph=True)

    benchmark_model(m, WARMUP, example_inputs)
    elapsed_time = benchmark_model(m, RUNS, example_inputs)

    m_ref = torch.compile(m_ref, mode='max-autotune', fullgraph=True)
    benchmark_model(m_ref, WARMUP, example_inputs)
    ref_elapsed_time = benchmark_model(m_ref, RUNS, example_inputs)

    print(f"elapsed time: {elapsed_time}, ref elapsed time: {ref_elapsed_time}")
    assert elapsed_time < 1.05 * ref_elapsed_time

if __name__ == "__main__" and TORCH_VERSION_AFTER_2_4 and torch.cuda.is_available():
    from torchao.quantization.quant_api import change_linear_weights_to_int8_dqtensors
    _bench_quantized_tensor_subclass_perf(change_linear_weights_to_int8_dqtensors, _ref_change_linear_weights_to_int8_dqtensors)

    from torchao.quantization.quant_api import change_linear_weights_to_int8_woqtensors
    _bench_quantized_tensor_subclass_perf(change_linear_weights_to_int8_woqtensors, _ref_change_linear_weights_to_int8_woqtensors)

    kwargs = {"groupsize": 32}
    from torchao.quantization.quant_api import change_linear_weights_to_int4_woqtensors
    _bench_quantized_tensor_subclass_perf(change_linear_weights_to_int4_woqtensors, _ref_change_linear_weights_to_int4_woqtensors, kwargs)
