# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: quantization"]

import copy
import unittest
from collections import Counter

import torch
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase, run_tests

from torchao.quantization.pt2e import (
    FROM_NODE_KEY,
    compare_results,
    extract_results_from_loggers,
    prepare_for_propagation_comparison,
)
from torchao.quantization.pt2e._numeric_debugger import _generate_debug_handle_from_node
from torchao.quantization.pt2e.graph_utils import bfs_trace_with_node_process
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.testing.pt2e._xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_7

if TORCH_VERSION_AT_LEAST_2_7:
    from torch.export import export_for_training


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_7, "Requires torch 2.7+")
@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestNumericDebugger(TestCase):
    def _assert_each_node_has_debug_handle(self, model) -> None:
        def _assert_node_has_debug_handle(node):
            self.assertIn(
                FROM_NODE_KEY,
                node.meta,
                f"Node {node} doesn't have from_node info",
            )

        bfs_trace_with_node_process(model, _assert_node_has_debug_handle)

    def _extract_debug_handles(self, model) -> dict[str, int]:
        debug_handle_map: dict[str, int] = {}

        def _extract_debug_handles_from_node(node):
            nonlocal debug_handle_map
            if (dh := _generate_debug_handle_from_node(node)) is not None:
                debug_handle_map[str(node)] = dh

        bfs_trace_with_node_process(model, _extract_debug_handles_from_node)

        return debug_handle_map

    def _extract_debug_handles_with_prev_decomp_op(self, model) -> dict[str, int]:
        prev_decomp_op_to_debug_handle_map: dict[str, int] = {}

        def _extract_debug_handles_with_prev_decomp_op_from_node(node):
            nonlocal prev_decomp_op_to_debug_handle_map
            if FROM_NODE_KEY in node.meta:
                prev_decomp_op = str(node.meta.get("nn_module_stack"))
                debug_handle = _generate_debug_handle_from_node(node)
                if prev_decomp_op not in prev_decomp_op_to_debug_handle_map:
                    prev_decomp_op_to_debug_handle_map[prev_decomp_op] = debug_handle
                else:
                    assert (
                        prev_decomp_op_to_debug_handle_map[prev_decomp_op]
                        == debug_handle
                    ), f"Node {node} has different debug handle {debug_handle}"
                    "than previous node sharing the same decomp op {prev_decomp_op}"

        bfs_trace_with_node_process(
            model, _extract_debug_handles_with_prev_decomp_op_from_node
        )
        return prev_decomp_op_to_debug_handle_map

    @unittest.skip(
        "torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recom..."
    )
    def test_simple(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()
        self._assert_each_node_has_debug_handle(m)
        debug_handle_map = self._extract_debug_handles(m)

        self.assertEqual(len(set(debug_handle_map.values())), len(debug_handle_map))

    @unittest.skip("debug flow not working on model with conditional control flow")
    def test_control_flow(self):
        m = TestHelperModules.ControlFlow()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_debug_handle(m)
        debug_handle_map = self._extract_debug_handles(m)

        self.assertEqual(len(set(debug_handle_map.values())), len(debug_handle_map))

    def test_quantize_pt2e_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        debug_handle_map = self._extract_debug_handles(m)
        node_name_equip_with_output_observer = [
            "conv2d",
            "conv1d",
            "squeeze",
        ]
        res_counter = Counter(debug_handle_map.values())
        repeated_debug_handle_ids = [
            debug_handle_map[n_name] for n_name in node_name_equip_with_output_observer
        ]
        # 3 ids were repeated because we copy over the id from node to its output observer
        # torch.ops.aten.conv2d.default, torch.ops.aten.squeeze.dim and torch.ops.aten.conv1d.default
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 2)

        m(*example_inputs)
        m = convert_pt2e(m)
        self._assert_each_node_has_debug_handle(m)
        debug_handle_map = self._extract_debug_handles(m)
        res_counter = Counter(debug_handle_map.values())
        # same set of ids where repeated, because we copy over the id from observer/fake_quant to
        # quantize/dequantize node
        repeated_debug_handle_ids = [
            debug_handle_map[n_name] for n_name in node_name_equip_with_output_observer
        ]
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 3)

    def test_copy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = torch.export.export(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_debug_handle(m)
        debug_handle_map_ref = self._extract_debug_handles(m)

        ep_copy = copy.copy(ep)
        debug_handle_map = self._extract_debug_handles(ep_copy.module())

        self._assert_each_node_has_debug_handle(ep)
        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    def test_deepcopy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = torch.export.export(m, example_inputs, strict=True)

        debug_handle_map_ref = self._extract_debug_handles(ep.module())
        ep_copy = copy.deepcopy(ep)
        debug_handle_map = self._extract_debug_handles(ep_copy.module())

        self._assert_each_node_has_debug_handle(ep.module())
        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    @unittest.skip(
        "torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recom..."
    )
    def test_re_export_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_debug_handle(m)
        debug_handle_map_ref = self._extract_debug_handles(m)

        ep_reexport = export_for_training(m, example_inputs, strict=True)
        m_reexport = ep_reexport.module()

        self._assert_each_node_has_debug_handle(m_reexport)
        debug_handle_map = self._extract_debug_handles(m_reexport)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    @unittest.skip(
        "torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recom..."
    )
    def test_run_decompositions_same_handle_id(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_debug_handle(m)
        debug_handle_map_ref = self._extract_debug_handles(m)

        ep_copy = copy.copy(ep)
        ep_copy = ep_copy.run_decompositions()
        m_decomposed = ep_copy.module()

        self._assert_each_node_has_debug_handle(m_decomposed)
        debug_handle_map = self._extract_debug_handles(m_decomposed)

        # checking the map still has the same ids, the node may change
        self.assertEqual(
            set(debug_handle_map.values()), set(debug_handle_map_ref.values())
        )

    @unittest.skip(
        "torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recom..."
    )
    def test_run_decompositions_map_handle_to_new_nodes(self):
        test_models = [
            TestHelperModules.TwoLinearModule(),
            TestHelperModules.Conv2dThenConv1d(),
        ]

        for m in test_models:
            example_inputs = m.example_inputs()
            ep = export_for_training(m, example_inputs, strict=True)
            m = ep.module()

            self._assert_each_node_has_debug_handle(m)
            pre_decomp_to_debug_handle_map_ref = (
                self._extract_debug_handles_with_prev_decomp_op(m)
            )

            ep_copy = copy.copy(ep)
            ep_copy = ep_copy.run_decompositions()
            m_decomposed = ep_copy.module()
            self._assert_each_node_has_debug_handle(m_decomposed)
            pre_decomp_to_debug_handle_map = (
                self._extract_debug_handles_with_prev_decomp_op(m_decomposed)
            )

            # checking the map still has the same ids, the node may change
            self.assertEqual(
                pre_decomp_to_debug_handle_map, pre_decomp_to_debug_handle_map_ref
            )

    def test_prepare_for_propagation_comparison(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()
        m_logger = prepare_for_propagation_comparison(m)
        ref = m(*example_inputs)
        res = m_logger(*example_inputs)

        from torchao.quantization.pt2e._numeric_debugger import OutputLogger

        loggers = [m for m in m_logger.modules() if isinstance(m, OutputLogger)]
        self.assertEqual(len(loggers), 3)
        self.assertTrue("conv2d" in [logger.node_name for logger in loggers])
        self.assertEqual(res, ref)

    def test_extract_results_from_loggers(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()
        m_ref_logger = prepare_for_propagation_comparison(m)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        m_quant_logger = prepare_for_propagation_comparison(m)

        m_ref_logger(*example_inputs)
        m_quant_logger(*example_inputs)
        ref_results = extract_results_from_loggers(m_ref_logger)
        quant_results = extract_results_from_loggers(m_quant_logger)
        comparison_results = compare_results(ref_results, quant_results)
        for node_summary in comparison_results.values():
            if len(node_summary.results) > 0:
                self.assertGreaterEqual(node_summary.results[0].sqnr, 35)

    def test_extract_results_from_loggers_list_output(self):
        m = TestHelperModules.Conv2dWithSplit()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()
        m_ref_logger = prepare_for_propagation_comparison(m)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        m_quant_logger = prepare_for_propagation_comparison(m)

        m_ref_logger(*example_inputs)
        m_quant_logger(*example_inputs)
        ref_results = extract_results_from_loggers(m_ref_logger)
        quant_results = extract_results_from_loggers(m_quant_logger)
        comparison_results = compare_results(ref_results, quant_results)
        for node_summary in comparison_results.values():
            if len(node_summary.results) > 0:
                sqnr = node_summary.results[0].sqnr
                if isinstance(sqnr, list):
                    for sqnr_i in sqnr:
                        self.assertGreaterEqual(sqnr_i, 35)
                else:
                    self.assertGreaterEqual(sqnr, 35)

    def test_added_node_gets_unique_id(self) -> None:
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)

        ref_handles = self._extract_debug_handles(ep.module())
        ref_counter = Counter(ref_handles.values())

        for k, v in ref_counter.items():
            self.assertEqual(
                v,
                1,
                msg=f"For handle {k}, there were {v} nodes with that handle, but expected only 1",
            )

        # Now that we have unique ids, add a new node into the graph and re-generate
        # to make sure that the new node gets a unique id.
        last_node = next(iter(reversed(ep.graph.nodes)))
        with ep.graph.inserting_before(last_node):
            arg = last_node.args[0]
            self.assertIsInstance(arg, (list, tuple))
            arg = arg[0]
            # Add a function that only requires a single tensor input.
            n = ep.graph.call_function(torch.ops.aten.relu.default, args=(arg,))
            arg.replace_all_uses_with(n, lambda x: x != n)
        ep.graph_module.recompile()

        # Regenerate handles, make sure only the new relu node has a new id, and
        # it doesn't clash with any of the existing ids.

        m = ep.module()
        self._assert_each_node_has_debug_handle(m)
        handles_after_modification = self._extract_debug_handles(m)
        handles_counter = Counter(handles_after_modification.values())
        for name, handle in ref_handles.items():
            self.assertIn(name, handles_after_modification)
            # Check that handle was unchanged.
            self.assertEqual(handles_after_modification[name], handle)
            # Check that total count was unchanged.
            ref_count = ref_counter[handle]
            after_count = handles_counter[handle]
            self.assertEqual(
                after_count,
                ref_count,
                msg=f"For handle {handle}, there were {after_count} nodes with that handle, but expected only {ref_count}",
            )

        # Check for relu specifically. Avoid hardcoding the handle id since it
        # may change with future node ordering changes.
        self.assertNotIn(handles_after_modification["relu_default"], ref_counter)
        self.assertEqual(handles_counter[handles_after_modification["relu_default"]], 1)


if __name__ == "__main__":
    run_tests()
