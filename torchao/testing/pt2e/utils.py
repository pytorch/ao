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
from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    NodeSpec,
    QuantizationTestCase,
)
from torch.testing._internal.common_utils import TestCase

from torchao.quantization.pt2e import FROM_NODE_KEY
from torchao.quantization.pt2e._numeric_debugger import _extract_node_source_debug_info
from torchao.quantization.pt2e.graph_utils import bfs_trace_with_node_process
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_7


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
            strict=True,
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


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_7, "Requires torch 2.7+")
class PT2ENumericDebuggerTestCase(TestCase):
    """
    Base test case class for PT2E numeric debugger tests containing common utility functions
    for numeric debugging functionality.
    """

    def _assert_each_node_has_from_node_source(self, model) -> None:
        def _assert_node_has_from_node_source(node):
            if node.op == "placeholder" or node.op == "output":
                return
            self.assertIn(
                FROM_NODE_KEY,
                node.meta,
                f"Node {node} doesn't have from_node info",
            )

        bfs_trace_with_node_process(model, _assert_node_has_from_node_source)

    def _extract_from_node_source(self, model) -> dict[str, any]:
        from_node_source_map: dict[str, any] = {}

        def _extract_from_node_source_from_node(node):
            nonlocal from_node_source_map
            if (root_node_source := _extract_node_source_debug_info(node)) is not None:
                from_node_source_map[str(node)] = (
                    root_node_source.name,
                    root_node_source.graph_id,
                )

        bfs_trace_with_node_process(model, _extract_from_node_source_from_node)

        return from_node_source_map

    def _extract_from_node_source_with_prev_decomp_op(self, model) -> dict[str, any]:
        prev_decomp_op_to_from_node_source_map: dict[str, any] = {}

        def _extract_from_node_source_with_prev_decomp_op_from_node(node):
            nonlocal prev_decomp_op_to_from_node_source_map
            if FROM_NODE_KEY in node.meta and node.meta[FROM_NODE_KEY] is not None:
                prev_decomp_op = str(node.meta.get("nn_module_stack"))
                from_node_source = _extract_node_source_debug_info(node)
                if prev_decomp_op not in prev_decomp_op_to_from_node_source_map:
                    prev_decomp_op_to_from_node_source_map[prev_decomp_op] = (
                        from_node_source
                    )
                else:
                    assert (
                        prev_decomp_op_to_from_node_source_map[prev_decomp_op]
                        == from_node_source
                    ), (
                        f"Node {node} has different from_node info {from_node_source}"
                        f"than previous node sharing the same decomp op {prev_decomp_op}"
                    )

        bfs_trace_with_node_process(
            model, _extract_from_node_source_with_prev_decomp_op_from_node
        )
        return prev_decomp_op_to_from_node_source_map

    def assertNodeSourcesEqual(self, node_source_1, node_source_2):
        self.assertTrue(
            node_source_1.name == node_source_2.name
            and node_source_1.graph_id == node_source_2.graph_id
        )
