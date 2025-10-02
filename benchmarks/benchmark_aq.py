# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmarks for affine quantized tensor, this includes int8 dynamic quant, int8 weight only quant and int4 weight only quant APIs"""

import copy

import torch

from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    _replace_with_custom_fn_if_matches_filter,
    quantize_,
)


def _int8wo_api(mod, **kwargs):
    quantize_(mod, Int8WeightOnlyConfig(**kwargs), set_inductor_config=False)


def _int8da_int8w_api(mod, **kwargs):
    quantize_(
        mod,
        Int8DynamicActivationInt8WeightConfig(**kwargs),
        set_inductor_config=False,
    )


def _int4wo_api(mod, **kwargs):
    kwargs_copy = kwargs.copy()
    if "groupsize" in kwargs_copy:
        kwargs_copy["group_size"] = kwargs_copy["groupsize"]
        del kwargs_copy["groupsize"]
    quantize_(mod, Int4WeightOnlyConfig(**kwargs_copy), set_inductor_config=False)


class ToyLinearModel(torch.nn.Module):
    """Single linear for m * k * n problem size"""

    def __init__(
        self, m=64, n=32, k=64, has_bias=False, dtype=torch.float, device="cuda"
    ):
        super().__init__()
        self.m = m
        self.dtype = dtype
        self.device = device
        self.linear = torch.nn.Linear(k, n, bias=has_bias).to(
            dtype=self.dtype, device=self.device
        )

    def example_inputs(self):
        return (
            torch.randn(
                self.m, self.linear.in_features, dtype=self.dtype, device=self.device
            ),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


def _ref_change_linear_weights_to_int8_dqtensors(model, filter_fn=None, **kwargs):
    """
    The deprecated implementation for int8 dynamic quant API, used as a reference for
    numerics and performance
    """
    from torchao.quantization.quant_api import (
        _get_subclass_inserter,
        _is_linear,
    )
    from torchao.quantization.subclass import Int8DynamicallyQuantizedLinearWeight

    def _in_features_greater_than_16(mod, *args):
        return hasattr(mod, "in_features") and mod.in_features > 16

    if filter_fn is None:
        filter_fn = lambda *args: _is_linear(*args) and _in_features_greater_than_16(
            *args
        )

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(
            Int8DynamicallyQuantizedLinearWeight, enable_parametrization=False, **kwargs
        ),
        filter_fn,
    )


def _get_ref_change_linear_weights_to_woqtensors(deprecated_tenosr_subclass):
    def _ref_change_linear_weights_to_woqtensors(model, filter_fn=None, **kwargs):
        """
        The deprecated implementation for weight only quant API, used as a reference for
        numerics and performance
        """
        from torchao.quantization.quant_api import _get_subclass_inserter, _is_linear

        filter_fn = kwargs.pop("filter_fn", _is_linear)

        _replace_with_custom_fn_if_matches_filter(
            model,
            _get_subclass_inserter(
                deprecated_tenosr_subclass, enable_parametrization=True, **kwargs
            ),
            filter_fn,
        )

    return _ref_change_linear_weights_to_woqtensors


torch._dynamo.config.cache_size_limit = 50000


@torch.no_grad
def _bench_quantized_tensor_subclass_perf(api, ref_api, M, N, K, kwargs=None):
    if kwargs is None:
        kwargs = {}

    m = ToyLinearModel(
        M, N, K, has_bias=True, dtype=torch.bfloat16, device="cuda"
    ).eval()
    m_bf16 = copy.deepcopy(m)
    m_ref = copy.deepcopy(m)
    example_inputs = m.example_inputs()

    api(m, **kwargs)

    # reference
    ref_api(m_ref, **kwargs)

    res = m(*example_inputs)
    ref = m_ref(*example_inputs)

    assert torch.equal(res, ref)

    # perf comparison
    from torchao.utils import benchmark_model

    # warmup
    WARMUP = 20
    RUNS = 100

    torch._dynamo.reset()
    m_ref = torch.compile(m_ref, mode="max-autotune", fullgraph=True)
    benchmark_model(m_ref, WARMUP, example_inputs)
    ref_elapsed_time = benchmark_model(m_ref, RUNS, example_inputs)

    torch._dynamo.reset()
    m = torch.compile(m, mode="max-autotune", fullgraph=True)
    benchmark_model(m, WARMUP, example_inputs)
    elapsed_time = benchmark_model(m, RUNS, example_inputs)

    torch._dynamo.reset()
    m_bf16 = torch.compile(m_bf16, mode="max-autotune", fullgraph=True)
    benchmark_model(m_bf16, WARMUP, example_inputs)
    bf16_elapsed_time = benchmark_model(m_bf16, RUNS, example_inputs)

    print(
        f"{(M, N, K)}: elapsed time: {elapsed_time}, ref elapsed time: {ref_elapsed_time}, bf16 elapsed time: {bf16_elapsed_time}"
    )


if __name__ == "__main__" and torch.cuda.is_available():
    all_shapes = [
        (20, 2048, 2048),
    ]

    print("_int8da_int8w_api")

    for M, N, K in all_shapes:
        _bench_quantized_tensor_subclass_perf(
            _int8da_int8w_api, _int8da_int8w_api, M, N, K
        )

    print("_int8wo_api")

    for M, N, K in all_shapes:
        _bench_quantized_tensor_subclass_perf(_int8wo_api, _int8wo_api, M, N, K)

    print("_int4wo_api")
    kwargs = {"groupsize": 32, "version": 1}

    for M, N, K in all_shapes:
        _bench_quantized_tensor_subclass_perf(_int4wo_api, _int4wo_api, M, N, K, kwargs)
