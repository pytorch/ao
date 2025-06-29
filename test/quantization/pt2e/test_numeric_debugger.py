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
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests

from torchao.quantization.pt2e import (
    prepare_for_propagation_comparison,
)
from torchao.testing.pt2e.utils import PT2ENumericDebuggerTestCase
from torchao.utils import TORCH_VERSION_AT_LEAST_2_8

if TORCH_VERSION_AT_LEAST_2_8:
    from torch.export import export_for_training


@unittest.skipIf(
    not TORCH_VERSION_AT_LEAST_2_8, "Requires torch 2.8 and above, including nightly"
)
@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestNumericDebuggerInfra(PT2ENumericDebuggerTestCase):
    @unittest.skip(
        "torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recom..."
    )
    def test_simple(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()
        self._assert_each_node_has_from_node_source(m)
        from_node_source_map = self._extract_from_node_source(m)

        self.assertEqual(
            len(set(from_node_source_map.values())), len(from_node_source_map)
        )

    @unittest.skip("debug flow not working on model with conditional control flow")
    def test_control_flow(self):
        m = TestHelperModules.ControlFlow()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_from_node_source(m)
        from_node_source_map = self._extract_from_node_source(m)

        self.assertEqual(
            len(set(from_node_source_map.values())), len(from_node_source_map)
        )

    def test_copy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = torch.export.export(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_from_node_source(m)
        from_node_source_map_ref = self._extract_from_node_source(m)

        ep_copy = copy.copy(ep)
        from_node_source_map = self._extract_from_node_source(ep_copy.module())

        self._assert_each_node_has_from_node_source(ep)
        self.assertEqual(from_node_source_map, from_node_source_map_ref)

    def test_deepcopy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = torch.export.export(m, example_inputs, strict=True)

        from_node_source_map_ref = self._extract_from_node_source(ep.module())
        ep_copy = copy.deepcopy(ep)
        from_node_source_map = self._extract_from_node_source(ep_copy.module())

        self._assert_each_node_has_from_node_source(ep.module())
        self.assertEqual(from_node_source_map, from_node_source_map_ref)
        self.assertEqual(
            set(from_node_source_map.values()), set(from_node_source_map_ref.values())
        )

    @unittest.skip(
        "torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recom..."
    )
    def test_re_export_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_from_node_source(m)
        from_node_source_map_ref = self._extract_from_node_source(m)

        ep_reexport = export_for_training(m, example_inputs, strict=True)
        m_reexport = ep_reexport.module()

        self._assert_each_node_has_from_node_source(m_reexport)
        from_node_source_map = self._extract_from_node_source(m_reexport)

        self.assertEqual(from_node_source_map, from_node_source_map_ref)

    @unittest.skip(
        "torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recom..."
    )
    def test_run_decompositions_same_handle_id(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        self._assert_each_node_has_from_node_source(m)
        from_node_source_map_ref = self._extract_from_node_source(m)

        ep_copy = copy.copy(ep)
        ep_copy = ep_copy.run_decompositions()
        m_decomposed = ep_copy.module()

        self._assert_each_node_has_from_node_source(m_decomposed)
        from_node_source_map = self._extract_from_node_source(m_decomposed)

        # checking the map still has the same ids, the node may change
        self.assertEqual(
            set(from_node_source_map.values()), set(from_node_source_map_ref.values())
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

            self._assert_each_node_has_from_node_source(m)
            pre_decomp_to_from_node_source_map_ref = (
                self._extract_from_node_source_with_prev_decomp_op(m)
            )

            ep_copy = copy.copy(ep)
            ep_copy = ep_copy.run_decompositions()
            m_decomposed = ep_copy.module()
            self._assert_each_node_has_from_node_source(m_decomposed)
            pre_decomp_to_from_node_source_map = (
                self._extract_from_node_source_with_prev_decomp_op(m_decomposed)
            )

            # checking the map still has the same infos, the node may change
            self.assertEqual(
                pre_decomp_to_from_node_source_map,
                pre_decomp_to_from_node_source_map_ref,
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

    def test_added_node_gets_unique_id(self) -> None:
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)

        ref_from_node_source = self._extract_from_node_source(ep.module())
        ref_counter = Counter(ref_from_node_source.values())

        for k, v in ref_counter.items():
            self.assertEqual(
                v,
                1,
                msg=f"For from_node info {k}, there were {v} nodes with that info, but expected only 1",
            )

        # Now that we have unique infos, add a new node into the graph and re-generate
        # to make sure that the new node gets a unique info.
        last_node = next(iter(reversed(ep.graph.nodes)))
        with ep.graph.inserting_before(last_node):
            arg = last_node.args[0]
            self.assertIsInstance(arg, (list, tuple))
            arg = arg[0]
            # Add a function that only requires a single tensor input.
            n = ep.graph.call_function(torch.ops.aten.relu.default, args=(arg,))
            arg.replace_all_uses_with(n, lambda x: x != n)
        ep.graph_module.recompile()

        # Regenerate from_node info, make sure only the new relu node has a new info, and
        # it doesn't clash with any of the existing infos.

        m = ep.module()
        self._assert_each_node_has_from_node_source(m)
        from_node_source_after_modification = self._extract_from_node_source(m)
        from_node_source_counter = Counter(from_node_source_after_modification.values())
        for name, from_node_source in ref_from_node_source.items():
            self.assertIn(name, from_node_source_after_modification)
            # Check that from_node info was unchanged.
            self.assertEqual(
                from_node_source_after_modification[name], from_node_source
            )
            # Check that total count was unchanged.
            ref_count = ref_counter[from_node_source]
            after_count = from_node_source_counter[from_node_source]
            self.assertEqual(
                after_count,
                ref_count,
                msg=f"For from_node info {from_node_source}, there were {after_count} nodes with that info, but expected only {ref_count}",
            )

        # Check for relu specifically. Avoid hardcoding the from_node info since it
        # may change with future node ordering changes.
        self.assertNotIn(
            from_node_source_after_modification["relu_default"], ref_counter
        )
        self.assertEqual(
            from_node_source_counter[
                from_node_source_after_modification["relu_default"]
            ],
            1,
        )


if __name__ == "__main__":
    run_tests()
