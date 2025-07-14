# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause

# Owner(s): ["oncall: quantization"]

import functools
import platform
import unittest
from typing import Dict

import torch
import torch.nn as nn
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
)
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skipIfNoInductorSupport,
)
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo

import torchao.quantization.pt2e.quantizer.arm_inductor_quantizer as armiq
from torchao.quantization.pt2e.inductor_passes.arm import (
    _register_quantization_weight_pack_pass,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.arm_inductor_quantizer import (
    ArmInductorQuantizer,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, TORCH_VERSION_AT_LEAST_2_7


# ----------------------------------------------------------------------------- #
# Helper decorators                                                             #
# ----------------------------------------------------------------------------- #
def skipIfNoArm(fn):
    reason = "Quantized operations require Arm."
    if isinstance(fn, type):
        if platform.processor() != "aarch64":
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if platform.processor() != "aarch64":
            raise unittest.SkipTest(reason)
        return fn(*args, **kwargs)

    return wrapper


# ----------------------------------------------------------------------------- #
# Mini-models                                                                   #
# ----------------------------------------------------------------------------- #
class _SingleConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class _SingleLinear(nn.Module):
    def __init__(self, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=bias)

    def forward(self, x):
        return self.linear(x)


if TORCH_VERSION_AT_LEAST_2_5:
    from torch.export import export_for_training


# ----------------------------------------------------------------------------- #
# Base harness                                                                  #
# ----------------------------------------------------------------------------- #
class _ArmInductorPerTensorTestCase(QuantizationTestCase):
    def _test_quantizer(
        self,
        model: torch.nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        quantizer: ArmInductorQuantizer,
        expected_node_occurrence: Dict[torch._ops.OpOverload, int],
        expected_node_list=None,
        *,
        is_qat: bool = False,
        lower: bool = False,
    ):
        gm = export_for_training(model.eval(), example_inputs).module()

        gm = prepare_pt2e(gm, quantizer)
        gm(*example_inputs)
        gm = convert_pt2e(gm)

        if lower:
            # Register weight-pack pass (only affects per-tensor path; harmless otherwise)
            _register_quantization_weight_pack_pass(per_channel=False)
            from torch._inductor.constant_folding import constant_fold
            from torch._inductor.fx_passes.freezing_patterns import freezing_passes

            gm.recompile()
            freezing_passes(gm, example_inputs)
            constant_fold(gm)
            gm(*example_inputs)

        self.checkGraphModuleNodes(
            gm,
            expected_node_occurrence={
                ns.call_function(k): v for k, v in expected_node_occurrence.items()
            },
            expected_node_list=[
                ns.call_function(n) for n in (expected_node_list or [])
            ],
        )


# ----------------------------------------------------------------------------- #
# Test-suite                                                                    #
# ----------------------------------------------------------------------------- #
@skipIfNoInductorSupport
@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_7, "Requires torch 2.7+")
class TestQuantizePT2EArmInductorPerTensor(_ArmInductorPerTensorTestCase):
    # ------------------------------------------------------------------ #
    # 1. Conv2d - per-tensor static PTQ                                  #
    # ------------------------------------------------------------------ #
    @skipIfNoArm
    def test_conv2d_per_tensor_weight(self):
        example_inputs = (torch.randn(2, 3, 16, 16),)
        q = ArmInductorQuantizer().set_global(
            armiq.get_default_arm_inductor_quantization_config(is_per_channel=False)
        )
        expected = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
        }
        self._test_quantizer(_SingleConv2d(), example_inputs, q, expected, lower=True)

    # ------------------------------------------------------------------ #
    # 2. Linear - per-tensor static PTQ                                  #
    # ------------------------------------------------------------------ #
    @skipIfNoArm
    def test_linear_per_tensor_weight(self):
        example_inputs = (torch.randn(4, 16),)
        q = ArmInductorQuantizer().set_global(
            armiq.get_default_arm_inductor_quantization_config(is_per_channel=False)
        )
        expected = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
        }
        self._test_quantizer(_SingleLinear(), example_inputs, q, expected, lower=True)

    # ------------------------------------------------------------------ #
    # 3. Linear - per-tensor **dynamic**                                 #
    # ------------------------------------------------------------------ #
    @skipIfNoArm
    def test_linear_dynamic_per_tensor_weight(self):
        example_inputs = (torch.randn(8, 16),)
        q = ArmInductorQuantizer().set_global(
            armiq.get_default_arm_inductor_quantization_config(
                is_dynamic=True, is_per_channel=False
            )
        )
        expected = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
        }
        self._test_quantizer(_SingleLinear(), example_inputs, q, expected, lower=True)

    # ------------------------------------------------------------------ #
    # 4. Conv2d - **per-channel** static PTQ                             #
    # ------------------------------------------------------------------ #
    @skipIfNoArm
    def test_conv2d_per_channel_weight(self):
        example_inputs = (torch.randn(2, 3, 16, 16),)
        q = ArmInductorQuantizer().set_global(
            armiq.get_default_arm_inductor_quantization_config(is_per_channel=True)
        )
        expected = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        self._test_quantizer(_SingleConv2d(), example_inputs, q, expected, lower=True)

    # ------------------------------------------------------------------ #
    # 5. Linear - **per-channel** static PTQ                             #
    # ------------------------------------------------------------------ #
    @skipIfNoArm
    def test_linear_per_channel_weight(self):
        example_inputs = (torch.randn(4, 16),)
        q = ArmInductorQuantizer().set_global(
            armiq.get_default_arm_inductor_quantization_config(is_per_channel=True)
        )
        expected = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        self._test_quantizer(_SingleLinear(), example_inputs, q, expected, lower=True)

    # ------------------------------------------------------------------ #
    # 6. Conv2d - **QAT** per-tensor                                    #
    # ------------------------------------------------------------------ #
    @skipIfTorchDynamo("slow under Dynamo")
    @skipIfNoArm
    def test_conv2d_qat_per_tensor_weight(self):
        example_inputs = (torch.randn(2, 3, 16, 16),)
        q = ArmInductorQuantizer().set_global(
            armiq.get_default_arm_inductor_quantization_config(is_qat=True)
        )
        expected = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
        }
        self._test_quantizer(
            _SingleConv2d(),
            example_inputs,
            q,
            expected,
            is_qat=True,
            lower=True,
        )

    # ------------------------------------------------------------------ #
    # 7. Linear - **dynamic + QAT** per-tensor                           #
    # ------------------------------------------------------------------ #
    @skipIfTorchDynamo("slow under Dynamo")
    @skipIfNoArm
    def test_linear_dynamic_qat_per_tensor_weight(self):
        example_inputs = (torch.randn(8, 16),)
        q = ArmInductorQuantizer().set_global(
            armiq.get_default_arm_inductor_quantization_config(
                is_dynamic=True, is_qat=True, is_per_channel=False
            )
        )
        expected = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
        }
        self._test_quantizer(
            _SingleLinear(),
            example_inputs,
            q,
            expected,
            is_qat=True,
            lower=True,
        )


if __name__ == "__main__":
    run_tests()
