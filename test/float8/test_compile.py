# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import random
import sys
import unittest
from dataclasses import replace
from io import StringIO

import pytest

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import torch
import torch.nn as nn
from torch._dynamo.test_case import TestCase as DynamoTestCase
from torch._dynamo.testing import CompileCounterWithBackend

from torchao.float8 import _prototype_register_float8_delayed_scaling_inductor_passes
from torchao.float8.config import (
    CastConfig,
    Float8LinearConfig,
    Float8LinearRecipeName,
    ScalingType,
    e4m3_dtype,
    recipe_name_to_linear_config,
)
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
    get_float8_layers,
    sync_float8_amax_and_scale_history,
)
from torchao.float8.float8_scaling_utils import (
    hp_tensor_to_float8_delayed,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig, ScaledMMConfig
from torchao.float8.float8_utils import config_has_stateful_scaling
from torchao.float8.stateful_float8_linear import StatefulFloat8Linear
from torchao.testing.float8.test_utils import get_test_float8_linear_config
from torchao.utils import is_fbcode


def _test_compile_base(
    backend: str,
    fullgraph: bool,
    config: Float8LinearConfig,
    dtype: torch.dtype,
):
    random.seed(0)
    torch.manual_seed(0)
    x_shape = (16, 16)
    linear_dtype = torch.bfloat16

    x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype).requires_grad_()
    x_ref = copy.deepcopy(x)
    m_ref = nn.Linear(16, 32, bias=True, device="cuda", dtype=linear_dtype)

    if config_has_stateful_scaling(config):
        m_fp8 = StatefulFloat8Linear.from_float(
            copy.deepcopy(m_ref),
            config,
        )
    else:
        m_fp8 = Float8Linear.from_float(
            copy.deepcopy(m_ref),
            config,
        )

    m_fp8 = torch.compile(m_fp8, backend=backend, fullgraph=fullgraph)
    m_ref = torch.compile(m_ref, backend=backend, fullgraph=fullgraph)
    y_fp8 = m_fp8(x)
    y_fp8.sum().backward()
    y_ref = m_ref(x_ref)
    y_ref.sum().backward()
    # TODO(future PR): can also test fp8 eager vs compile here with a tigher
    # tolerance
    torch.testing.assert_close(y_fp8, y_ref, atol=9.5e-2, rtol=9.5e-2)
    torch.testing.assert_close(
        m_fp8.weight.grad, m_ref.weight.grad, atol=2e-1, rtol=2e-1
    )
    torch.testing.assert_close(m_fp8.bias.grad, m_ref.bias.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize(
    "scaling_type_input", [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC]
)
@pytest.mark.parametrize(
    "scaling_type_weight",
    [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
)
@pytest.mark.parametrize(
    "scaling_type_grad_output",
    [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
)
@pytest.mark.parametrize("emulate", [False, True] if is_sm_at_least_89() else [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_eager_only(
    fullgraph,
    emulate: bool,
    scaling_type_input: ScalingType,
    scaling_type_weight: ScalingType,
    scaling_type_grad_output: ScalingType,
    dtype: torch.dtype,
):
    torch._dynamo.reset()
    config = get_test_float8_linear_config(
        scaling_type_input,
        scaling_type_weight,
        scaling_type_grad_output,
        emulate,
    )
    _test_compile_base(
        "eager",
        fullgraph,
        config,
        dtype,
    )


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("emulate", [False, True] if is_sm_at_least_89() else [True])
@pytest.mark.parametrize(
    "scaling_type_input", [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC]
)
@pytest.mark.parametrize(
    "scaling_type_weight",
    [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
)
@pytest.mark.parametrize(
    "scaling_type_grad_output",
    [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_aot_eager(
    fullgraph,
    emulate: bool,
    scaling_type_input: ScalingType,
    scaling_type_weight: ScalingType,
    scaling_type_grad_output: ScalingType,
    dtype: torch.dtype,
):
    torch._dynamo.reset()
    config = get_test_float8_linear_config(
        scaling_type_input,
        scaling_type_weight,
        scaling_type_grad_output,
        emulate,
    )
    _test_compile_base(
        "aot_eager",
        fullgraph,
        config,
        dtype,
    )


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("emulate", [False])
@pytest.mark.parametrize(
    "scaling_type_input", [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC]
)
@pytest.mark.parametrize(
    "scaling_type_weight",
    [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
)
@pytest.mark.parametrize(
    "scaling_type_grad_output",
    [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
)
@unittest.skipIf(
    not torch.cuda.is_available() or not is_sm_at_least_89(),
    "CUDA with float8 support not available",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_inductor_from_config_params(
    fullgraph,
    emulate: bool,
    scaling_type_input: ScalingType,
    scaling_type_weight: ScalingType,
    scaling_type_grad_output: ScalingType,
    dtype: torch.dtype,
):
    torch._dynamo.reset()
    config = get_test_float8_linear_config(
        scaling_type_input,
        scaling_type_weight,
        scaling_type_grad_output,
        emulate,
    )
    _test_compile_base(
        "inductor",
        fullgraph,
        config,
        dtype,
    )


# Note: there are now too many config combinations to test all of
# them, so this function factors out some of the recipes which are annoying
# to combine with the main testing function.
# TODO(future PR): make this cleaner.
@pytest.mark.parametrize(
    "recipe_name",
    [
        Float8LinearRecipeName.ROWWISE,
        Float8LinearRecipeName.ROWWISE_WITH_GW_HP,
    ],
)
@unittest.skipIf(
    not is_sm_at_least_90(), "CUDA with capability 9.0 or greater not available"
)
def test_inductor_from_recipe(recipe_name):
    torch._dynamo.reset()
    config = recipe_name_to_linear_config(recipe_name)
    fullgraph = True
    dtype = torch.bfloat16
    _test_compile_base(
        "inductor",
        fullgraph,
        config,
        dtype,
    )


class TestGraphBreaks(DynamoTestCase):
    class MockLinear(torch.nn.Module):
        def __init__(self, graph_break: bool):
            super().__init__()
            self.register_buffer("fp8_amax_x", torch.tensor(1.0))
            self.register_buffer("fp8_scale_x", torch.tensor(1.0))
            self.graph_break = graph_break

        def forward(self, x):
            x_fp8 = hp_tensor_to_float8_delayed(
                x,
                self.fp8_scale_x,
                e4m3_dtype,
                self.fp8_amax_x,
                LinearMMConfig(),
            )
            if self.graph_break:
                torch._dynamo.graph_break()
                x_hp = x_fp8.to_original_precision()
                return x_hp
            return x_fp8

    # TODO(future): figure out why the test below fails on CUDA capability 8.9
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_sm_at_least_90(),
        "CUDA with capability 9.0 or greater not available",
    )
    def test_float8_with_graph_break_in_the_middle(self):
        """Test that having Float8Tensor object at the boundary of a subgraph"""
        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=True).cuda()
        compiled_mod = copy.deepcopy(mod)
        compiled_mod = torch.compile(compiled_mod, backend=cnts)
        x = torch.randn(16, 16, device="cuda")
        y_eager = mod(x)
        y_compiled = compiled_mod(x)
        self.assertEqual(cnts.frame_count, 2, "Compiled graph should have 2 frames!")
        torch.testing.assert_close(y_eager, y_compiled)

    @unittest.skipIf(
        not torch.cuda.is_available() or not is_sm_at_least_89(),
        "CUDA with float8 support not available",
    )
    def test_float8_graph_input(self):
        """Test that having Float8Tensor object as a graph input"""

        def to_float(x):
            return x.to_original_precision()

        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=False).cuda()
        x = torch.randn(2, 2, device="cuda")
        compiled_to_float = torch.compile(to_float, backend=cnts)
        y = mod(x)
        y2_eager = to_float(y)
        y2_compiled = compiled_to_float(y)
        self.assertEqual(
            cnts.frame_count,
            1,
            "to_float was not compiled into 1 frame and likely encountered a skip!",
        )
        torch.testing.assert_close(y2_eager, y2_compiled)

    @unittest.skipIf(
        not torch.cuda.is_available() or not is_sm_at_least_89(),
        "CUDA with float8 support not available",
    )
    def test_float8_graph_output(self):
        """Test that having Float8Tensor object as a graph output works"""
        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=False).cuda()
        compiled_mod = torch.compile(mod, backend=cnts)
        x = torch.randn(16, 16, device="cuda")
        y_compiled = compiled_mod(x)

        self.assertEqual(cnts.frame_count, 1, "Compiled graph should have 1 frame!")
        tensors, ctx = y_compiled.__tensor_flatten__()
        for tensor in tensors:
            assert not isinstance(
                getattr(y_compiled, tensor), torch._subclasses.fake_tensor.FakeTensor
            ), "Float8Tensor should not contain any FakeTensors!"
        assert isinstance(
            y_compiled._orig_dtype, torch.dtype
        ), "Float8Tensor._orig_dtype should be a dtype but got {}".format(
            type(y_compiled._orig_dtype)
        )
        assert isinstance(
            y_compiled._linear_mm_config.output.emulate, bool
        ), "Float8Tensor._emulate should be a bool but got {}".format(
            type(y_compiled._linear_mm_config.output.emulate)
        )


@unittest.skipIf(
    not torch.cuda.is_available() or not is_sm_at_least_89(),
    "CUDA with float8 support not available",
)
def test_sync_amax_func():
    torch._dynamo.reset()
    cnts = CompileCounterWithBackend("inductor")
    module = torch.nn.Sequential(
        nn.Linear(16, 32, bias=True), nn.ReLU(), nn.Linear(32, 16, bias=True)
    )
    config = Float8LinearConfig(
        cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
        cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
        cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
    )
    float8_mod = convert_to_float8_training(
        module,
        config=config,
    )
    compiled_swap_func = torch.compile(sync_float8_amax_and_scale_history, backend=cnts)
    compiled_swap_func(float8_mod)
    assert cnts.frame_count == 1, "Compiled graph should have 1 frame!"


class capture_stderr(list):
    """
    Replace sys.stderr with a temporary StringIO
    """

    def __enter__(self):
        self.sys_stderr = sys.stderr
        self.stringio = StringIO()
        sys.stderr = self.stringio
        return self

    def __exit__(self, *args):
        self.append(str(self.stringio.getvalue()))
        del self.stringio
        sys.stderr = self.sys_stderr


@unittest.skipIf(
    not torch.cuda.is_available() or not is_sm_at_least_89(),
    "CUDA with float8 support not available",
)
def test_sync_amax_func_cuda_graph_success():
    torch._dynamo.reset()
    with capture_stderr() as stderr:
        my_module = nn.Sequential(
            nn.Linear(16, 32, bias=True), nn.ReLU(), nn.Linear(32, 16, bias=True)
        ).to("cuda")
        config = Float8LinearConfig(
            cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
            cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
            cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
        )
        convert_to_float8_training(
            my_module,
            config=config,
        )
        inpt = torch.randn(
            16, 16, device="cuda", dtype=torch.float32, requires_grad=True
        )
        sync_func = torch.compile(
            sync_float8_amax_and_scale_history, mode="reduce-overhead", fullgraph=True
        )
        fp8_layers = get_float8_layers(my_module)
        my_module(inpt)
        sync_func(my_module, fp8_layers)

    assert "skipping cudagraphs due to mutaton on input" not in stderr[0]


@unittest.skipIf(
    not is_sm_at_least_89(),
    "CUDA not available",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ],
)
@pytest.mark.parametrize(
    "round_scales_to_power_of_2",
    [
        True,
        False,
    ],
)
def test_dynamic_scale_numeric_parity(
    dtype: torch.dtype, round_scales_to_power_of_2: bool
):
    scaling_type_weight = ScalingType.DYNAMIC
    torch.manual_seed(42)
    hp_tensor1 = torch.randn(16, 16, device="cuda", dtype=dtype)
    hp_tensor2 = hp_tensor1.detach().clone()
    float8_config = Float8LinearConfig(
        cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    linear_mm_config = LinearMMConfig(
        # output
        ScaledMMConfig(
            False,
            float8_config.gemm_config_output.use_fast_accum,
            False,
            float8_config.pad_inner_dim,
        ),
        # grad_input
        ScaledMMConfig(
            False,
            float8_config.gemm_config_grad_input.use_fast_accum,
            False,
            float8_config.pad_inner_dim,
        ),
        # grad_weight
        ScaledMMConfig(
            False,
            float8_config.gemm_config_grad_weight.use_fast_accum,
            False,
            float8_config.pad_inner_dim,
        ),
    )
    float8_eager = hp_tensor_to_float8_dynamic(
        hp_tensor1,
        e4m3_dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    torch._dynamo.reset()
    float8_compile = torch.compile(hp_tensor_to_float8_dynamic)(
        hp_tensor2,
        e4m3_dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    assert torch.equal(float8_eager._scale, float8_compile._scale)
    assert torch.equal(float8_eager._data, float8_compile._data)


@unittest.skipIf(
    not is_sm_at_least_89() or not is_fbcode(),
    "CUDA with float8 support not available; or not on fbcode (the test needs be run with the latest pytorch package)",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_delayed_scaling_pattern_replacement(dtype: torch.dtype):
    from torch._inductor import config as inductor_config
    from torch._inductor import metrics

    inductor_config.loop_ordering_after_fusion = True

    def clear_all():
        metrics.reset()
        from torch._inductor.fx_passes.post_grad import (
            pass_patterns as post_grad_patterns_all,
        )

        post_grad_patterns_all[1].clear()
        post_grad_patterns_all[1].seen_patterns.clear()

    def compile_and_run_single_layer():
        random.seed(0)
        torch.manual_seed(0)
        x_shape = (2048, 3072)
        linear_dtype = dtype

        x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype).requires_grad_()
        m_ref = nn.Linear(3072, 2048, bias=True, device="cuda", dtype=linear_dtype)

        config = get_test_float8_linear_config(
            ScalingType.DELAYED,
            ScalingType.DELAYED,
            ScalingType.DELAYED,
            False,
        )

        config = replace(config, enable_amax_init=False)

        m_fp8 = StatefulFloat8Linear.from_float(
            copy.deepcopy(m_ref),
            config,
        )

        m_fp8 = torch.compile(m_fp8, backend="inductor", fullgraph=True)
        m_ref = torch.compile(m_ref, backend="inductor", fullgraph=True)

        y_fp8 = m_fp8(x)
        y_fp8.sum().backward()

        return m_fp8.weight.grad

    clear_all()
    ref_output = compile_and_run_single_layer()
    ref_count_kernel = metrics.generated_kernel_count

    clear_all()
    _prototype_register_float8_delayed_scaling_inductor_passes()
    new_output = compile_and_run_single_layer()
    new_count_kernel = metrics.generated_kernel_count

    torch.equal(ref_output, new_output)
    # With the pattern replacement workaround, amax reduction kernels for the 3 tensors (weight, activation, gradient) are fused.
    assert ref_count_kernel == new_count_kernel + 3


if __name__ == "__main__":
    pytest.main([__file__])
