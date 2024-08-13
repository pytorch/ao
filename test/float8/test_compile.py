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

from torchao.utils import TORCH_VERSION_AFTER_2_4

if not TORCH_VERSION_AFTER_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import torch
import torch.nn as nn
from torchao.float8.config import CastConfig, Float8LinearConfig, ScalingType
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
    get_float8_layers,
    sync_float8_amax_and_scale_history,
)
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_delayed
from torchao.float8.float8_tensor import LinearMMConfig
from torchao.float8.float8_utils import e4m3_dtype

from torch._dynamo.test_case import TestCase as DynamoTestCase
from torch._dynamo.testing import CompileCounterWithBackend

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

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

    x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype)
    m_ref = nn.Linear(16, 32, bias=True, device="cuda", dtype=linear_dtype)

    m_fp8 = Float8Linear.from_float(
        copy.deepcopy(m_ref),
        config,
    )

    m_fp8 = torch.compile(m_fp8, backend=backend, fullgraph=fullgraph)
    m_ref = torch.compile(m_ref, backend=backend, fullgraph=fullgraph)
    y_fp8 = m_fp8(x)
    y_fp8.sum().backward()
    y_ref = m_ref(x)
    y_ref.sum().backward()
    torch.testing.assert_close(y_fp8, y_ref, atol=9.5e-2, rtol=9.5e-2)
    torch.testing.assert_close(
        m_fp8.weight.grad, m_ref.weight.grad, atol=2e-1, rtol=2e-1
    )
    torch.testing.assert_close(m_fp8.bias.grad, m_ref.bias.grad, atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize(
    "scaling_type_input", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize(
    "scaling_type_weight", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize(
    "scaling_type_grad_output", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize("emulate", [False, True] if is_cuda_8_9 else [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
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
    config = Float8LinearConfig(
        cast_config_input=CastConfig(scaling_type=scaling_type_input),
        cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
        cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
        emulate=emulate,
    )
    _test_compile_base(
        "eager",
        fullgraph,
        config,
        dtype,
    )


@pytest.mark.parametrize("fullgraph", [True])
@pytest.mark.parametrize("emulate", [False, True] if is_cuda_8_9 else [True])
@pytest.mark.parametrize(
    "scaling_type_input", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize(
    "scaling_type_weight", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize(
    "scaling_type_grad_output", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
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
    config = Float8LinearConfig(
        cast_config_input=CastConfig(scaling_type=scaling_type_input),
        cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
        cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
        emulate=emulate,
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
    "scaling_type_input", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize(
    "scaling_type_weight", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@pytest.mark.parametrize(
    "scaling_type_grad_output", [ScalingType.DELAYED, ScalingType.DYNAMIC]
)
@unittest.skipIf(not torch.cuda.is_available() or not is_cuda_8_9, "CUDA with float8 support not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_inductor(
    fullgraph,
    emulate: bool,
    scaling_type_input: ScalingType,
    scaling_type_weight: ScalingType,
    scaling_type_grad_output: ScalingType,
    dtype: torch.dtype,
):
    torch._dynamo.reset()
    config = Float8LinearConfig(
        cast_config_input=CastConfig(scaling_type=scaling_type_input),
        cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
        cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
        emulate=emulate,
    )
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

    @unittest.skipIf(not torch.cuda.is_available() or not is_H100, "CUDA with float8 support not available")
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

    @unittest.skipIf(not torch.cuda.is_available() or not is_cuda_8_9, "CUDA with float8 support not available")
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

    @unittest.skipIf(not torch.cuda.is_available() or not is_cuda_8_9, "CUDA with float8 support not available")
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


@unittest.skipIf(not torch.cuda.is_available() or not is_cuda_8_9, "CUDA with float8 support not available")
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


@unittest.skipIf(not torch.cuda.is_available() or not is_cuda_8_9, "CUDA with float8 support not available")
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


if __name__ == "__main__":
    pytest.main([__file__])
