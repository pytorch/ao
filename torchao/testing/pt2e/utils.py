# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from torch.ao.quantization.backend_config import (
    get_executorch_backend_config,
)
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.testing._internal.common_quantization import (
    NodeSpec,
    QuantizationTestCase,
)

from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if TORCH_VERSION_AT_LEAST_2_5:
    from torch.export import export_for_training


@unittest.skipIf(
    not TORCH_VERSION_AT_LEAST_2_5,
    "only works for torch 2.5+ since export_for_training is only supported after 2.5",
)
class PT2EQuantizationTestCase(QuantizationTestCase):
    """
    Base QuantizationTestCase for PT2 with some helper methods.
    """

    _MAP_TO_FX_TRACED_OPS = {
        torch.ops.quantized_decomposed.quantize_per_tensor: torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_channel: torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel: torch.ops.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    }

    def _test_quantizer(
        self,
        model,
        example_inputs,
        quantizer,
        expected_node_occurrence,
        expected_node_list=None,
        check_against_fx_quant=False,
        # TODO: remove the test if fx quant is removed from pytorch
        fx_qconfig_mapping=None,
        export_with_dynamic_shape=False,
        is_qat=False,
        is_debug_mode=False,
        training_ir_node_occurrence=None,
    ):
        # resetting dynamo cache
        torch._dynamo.reset()
        m_eager = model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        dynamic_shapes = tuple(
            {0: torch.export.Dim("dim")} if i == 0 else None
            for i in range(len(example_inputs))
        )
        m = export_for_training(
            m,
            example_inputs,
            dynamic_shapes=dynamic_shapes if export_with_dynamic_shape else None,
        ).module()

        if is_qat:
            m = prepare_qat_pt2e(m, quantizer)
        else:
            m = prepare_pt2e(m, quantizer)
        if is_debug_mode:
            print("prepared model:", m)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m)
        if is_debug_mode:
            print("quantized model", m)

        pt2_quant_output = m(*example_inputs)
        ns = NodeSpec
        node_occurrence = {
            ns.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        if expected_node_list is None:
            expected_node_list = []
        node_list = [ns.call_function(n) for n in expected_node_list]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )
        if check_against_fx_quant:
            qconfig_mapping = fx_qconfig_mapping
            backend_config = get_executorch_backend_config()
            m_copy = copy.deepcopy(m_eager)
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            m_fx(*example_inputs)
            m_fx = _convert_to_reference_decomposed_fx(
                m_fx, backend_config=backend_config
            )
            m_fx = export_for_training(
                m_fx,
                example_inputs,
                dynamic_shapes=dynamic_shapes if export_with_dynamic_shape else None,
            ).module()
            node_occurrence = {}
            for k, v in PT2EQuantizationTestCase._MAP_TO_FX_TRACED_OPS.items():
                if k in expected_node_occurrence:
                    node_occurrence[ns.call_function(v)] = expected_node_occurrence[k]
            if training_ir_node_occurrence is not None:
                node_occurrence = {
                    ns.call_function(k): v
                    for k, v in training_ir_node_occurrence.items()
                }
            self.checkGraphModuleNodes(m_fx, expected_node_occurrence=node_occurrence)
            fx_quant_output = m_fx(*example_inputs)
            self.assertEqual(fx_quant_output, pt2_quant_output)
        return m

    def _quantize(self, m, quantizer, example_inputs, is_qat: bool = False):
        # resetting dynamo cache
        torch._dynamo.reset()

        m = export_for_training(
            m,
            example_inputs,
        ).module()
        if is_qat:
            m = prepare_qat_pt2e(m, quantizer)
        else:
            m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        return m

    def _get_pt2e_quantized_linear(self, is_per_channel=False) -> torch.fx.GraphModule:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=is_per_channel
        )
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        return self._quantize(m, quantizer, example_inputs)
