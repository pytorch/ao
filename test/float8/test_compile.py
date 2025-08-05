# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import random
import sys
import unittest
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

from torchao.float8.config import (
    CastConfig,
    Float8LinearConfig,
    Float8LinearRecipeName,
    ScalingType,
    e4m3_dtype,
)
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_scaling_utils import (
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_training_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)
from torchao.testing.training.test_utils import get_test_float8_linear_config

from torchao.utils import auto_detect_device

_DEVICE = auto_detect_device()

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

    x = torch.randn(*x_shape, device=_DEVICE, dtype=linear_dtype).requires_grad_()
    x_ref = copy.deepcopy(x)
    m_ref = nn.Linear(16, 32, bias=True, device=_DEVICE, dtype=linear_dtype)

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
@pytest.mark.parametrize("scaling_type_input", [ScalingType.DYNAMIC])
@pytest.mark.parametrize(
    "scaling_type_weight",
    [ScalingType.DYNAMIC],
)
@pytest.mark.parametrize(
    "scaling_type_grad_output",
    [ScalingType.DYNAMIC],
)
@pytest.mark.parametrize("emulate", [False, True] if is_sm_at_least_89() else [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
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
@pytest.mark.parametrize("scaling_type_input", [ScalingType.DYNAMIC])
@pytest.mark.parametrize(
    "scaling_type_weight",
    [ScalingType.DYNAMIC],
)
@pytest.mark.parametrize(
    "scaling_type_grad_output",
    [ScalingType.DYNAMIC],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
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
@pytest.mark.parametrize("scaling_type_input", [ScalingType.DYNAMIC])
@pytest.mark.parametrize(
    "scaling_type_weight",
    [ScalingType.DYNAMIC],
)
@pytest.mark.parametrize(
    "scaling_type_grad_output",
    [ScalingType.DYNAMIC],
)
@unittest.skipIf(
    torch.cuda.is_available() and not is_sm_at_least_89(),
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
    torch.cuda.is_available() and not is_sm_at_least_90(), "CUDA with capability 9.0 or greater not available"
)
def test_inductor_from_recipe(recipe_name):
    torch._dynamo.reset()
    config = Float8LinearConfig.from_recipe_name(recipe_name)
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
            self.graph_break = graph_break

        def forward(self, x):
            x_fp8 = hp_tensor_to_float8_dynamic(
                x,
                e4m3_dtype,
                LinearMMConfig(),
            )
            if self.graph_break:
                torch._dynamo.graph_break()
                x_hp = x_fp8.to_original_precision()
                return x_hp
            return x_fp8

    # TODO(future): figure out why the test below fails on CUDA capability 8.9
    @unittest.skipIf(
        torch.cuda.is_available() and not is_sm_at_least_90(),
        "CUDA with capability 9.0 or greater not available",
    )
    def test_float8_with_graph_break_in_the_middle(self):
        """Test that having Float8TrainingTensor object at the boundary of a subgraph"""
        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=True).to(_DEVICE)
        compiled_mod = copy.deepcopy(mod)
        compiled_mod = torch.compile(compiled_mod, backend=cnts)
        x = torch.randn(16, 16, device=_DEVICE)
        y_eager = mod(x)
        y_compiled = compiled_mod(x)
        self.assertEqual(cnts.frame_count, 2, "Compiled graph should have 2 frames!")
        torch.testing.assert_close(y_eager, y_compiled)

    @unittest.skipIf(
        torch.cuda.is_available() and not torch.cuda.is_available() or not is_sm_at_least_89(),
        "CUDA with float8 support not available",
    )
    def test_float8_graph_input(self):
        """Test that having Float8TrainingTensor object as a graph input"""

        def to_float(x):
            return x.to_original_precision()

        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=False).to(_DEVICE)
        x = torch.randn(2, 2, device=_DEVICE)
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
        torch.cuda.is_available() and not torch.cuda.is_available() or not is_sm_at_least_89(),
        "CUDA with float8 support not available",
    )
    def test_float8_graph_output(self):
        """Test that having Float8TrainingTensor object as a graph output works"""
        cnts = CompileCounterWithBackend("inductor")
        mod = self.MockLinear(graph_break=False).to(_DEVICE)
        compiled_mod = torch.compile(mod, backend=cnts)
        x = torch.randn(16, 16, device=_DEVICE)
        y_compiled = compiled_mod(x)

        self.assertEqual(cnts.frame_count, 1, "Compiled graph should have 1 frame!")
        tensors, ctx = y_compiled.__tensor_flatten__()
        for tensor in tensors:
            assert not isinstance(
                getattr(y_compiled, tensor), torch._subclasses.fake_tensor.FakeTensor
            ), "Float8TrainingTensor should not contain any FakeTensors!"
        assert isinstance(y_compiled._orig_dtype, torch.dtype), (
            "Float8TrainingTensor._orig_dtype should be a dtype but got {}".format(
                type(y_compiled._orig_dtype)
            )
        )
        assert isinstance(y_compiled._linear_mm_config.output.emulate, bool), (
            "Float8TrainingTensor._emulate should be a bool but got {}".format(
                type(y_compiled._linear_mm_config.output.emulate)
            )
        )


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
    torch.cuda.is_available() and not is_sm_at_least_89(),
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
    hp_tensor1 = torch.randn(16, 16, device=_DEVICE, dtype=dtype)
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


if __name__ == "__main__":
    pytest.main([__file__])
