# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4420.

constant_fold's cleanup pass deleted the underlying module attribute for a
get_attr node as soon as it found one dead get_attr node for that target -
even if another get_attr node with the same target was still live. That
left the graph with a live get_attr pointing at an attribute that no
longer existed, which graph.lint() rejects.
"""

import torch
from torch.fx import Graph, GraphModule
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.quantization.pt2e.constant_fold import constant_fold


class Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shared", torch.arange(4.0))


class TestConstantFold(TestCase):
    def test_shared_get_attr_target_survives_constant_fold(self):
        root = Root().eval()

        g = Graph()
        x = g.placeholder("x")

        # First get_attr is used only by a foldable constant expression, so
        # it becomes dead after constant folding.
        shared0 = g.get_attr("shared")
        folded = g.call_function(torch.ops.aten.neg.default, (shared0,))

        # Second get_attr references the same target but remains live.
        shared1 = g.get_attr("shared")
        live = g.call_function(torch.ops.aten.add.Tensor, (x, shared1))

        g.output((folded, live))
        gm = GraphModule(root, g)

        # Used to raise:
        #   RuntimeError: Node shared_1 target shared references
        #   nonexistent attribute shared of ...
        constant_fold(gm)

        # The live get_attr's target must still resolve.
        get_attr_nodes = gm.graph.find_nodes(op="get_attr")
        self.assertTrue(len(get_attr_nodes) >= 1)
        for node in get_attr_nodes:
            self.assertTrue(hasattr(gm, node.target))

        x_val = torch.ones(4)
        folded_out, live_out = gm(x_val)
        torch.testing.assert_close(folded_out, -root.shared)
        torch.testing.assert_close(live_out, x_val + root.shared)


if __name__ == "__main__":
    run_tests()
