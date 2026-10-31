# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: quantization"]
import copy
import unittest

import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase, run_tests

from torchao.quantization.pt2e.graph_utils import (
    collect_producer_nodes,
    find_sequential_partitions,
    get_equivalent_types,
    update_equivalent_types_dict,
)


class TestGraphUtils(TestCase):
    def test_include_functional_equivalent(self):
        class FunctionalConv(torch.nn.Module):
            def forward(self, x, weight):
                return torch.nn.functional.conv2d(x, weight)

        class ModuleConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

        x = torch.randn(1, 1, 8, 8)
        weight = torch.randn(1, 1, 3, 3)
        for model, inputs, exact_type in (
            (FunctionalConv(), (x, weight), torch.nn.functional.conv2d),
            (ModuleConv().eval(), (x,), torch.nn.Conv2d),
        ):
            gm = torchdynamo.export(
                model, aten_graph=True, assume_static_by_default=True
            )(*inputs).graph_module
            for partition_type in (torch.nn.Conv2d, torch.nn.functional.conv2d):
                with self.subTest(model=type(model), partition_type=partition_type):
                    self.assertEqual(
                        len(find_sequential_partitions(gm, [partition_type])), 1
                    )
                    self.assertEqual(
                        len(
                            find_sequential_partitions(
                                gm, [partition_type], include_functional_equivalent=True
                            )
                        ),
                        1,
                    )
                    self.assertEqual(
                        len(
                            find_sequential_partitions(
                                gm,
                                [partition_type],
                                include_functional_equivalent=False,
                            )
                        ),
                        int(partition_type is exact_type),
                    )

    def test_exact_module_and_functional_sequence(self):
        class MixedConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 3)

            def forward(self, x, weight):
                return torch.nn.functional.conv2d(self.conv(x), weight)

        gm = torchdynamo.export(
            MixedConv().eval(), aten_graph=True, assume_static_by_default=True
        )(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 3, 3)).graph_module
        types = [torch.nn.Conv2d, torch.nn.functional.conv2d]
        self.assertEqual(
            len(
                find_sequential_partitions(
                    gm, types, include_functional_equivalent=False
                )
            ),
            1,
        )
        with self.assertRaisesRegex(
            ValueError, "Each type in the sequence must be unique"
        ):
            find_sequential_partitions(gm, types)
        with self.assertRaisesRegex(
            ValueError, "Each type in the sequence must be unique"
        ):
            find_sequential_partitions(
                gm,
                [torch.nn.Conv2d, torch.nn.Conv2d],
                include_functional_equivalent=False,
            )

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_conv_bn_conv_relu(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                bn_out = self.bn1(self.conv1(x))
                relu_out = torch.nn.functional.relu(bn_out)
                return self.relu2(self.conv2(relu_out))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(  # noqa: F841©
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 1)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 1)

        def x():
            find_sequential_partitions(
                m,
                [
                    torch.nn.Conv2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.ReLU,
                    torch.nn.functional.conv2d,
                ],
            )

        self.assertRaises(ValueError, x)

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_conv_bn_relu(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                bn_out = self.bn1(x)
                return self.relu2(self.conv2(bn_out))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(  # noqa: F841
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 0)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.Conv2d]
        )
        self.assertEqual(len(fused_partitions), 1)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 0)

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_customized_equivalet_types_dict(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return torch.nn.functional.relu6(self.conv(x))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(  # noqa: F841
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        original_equivalent_types = copy.deepcopy(get_equivalent_types())
        customized_equivalent_types = copy.deepcopy(original_equivalent_types)
        customized_equivalent_types.append({torch.nn.ReLU6, torch.nn.functional.relu6})
        try:
            update_equivalent_types_dict(customized_equivalent_types)
            fused_partitions = find_sequential_partitions(
                m,
                [torch.nn.Conv2d, torch.nn.ReLU6],
            )
            self.assertEqual(len(fused_partitions), 1)
        finally:
            update_equivalent_types_dict(original_equivalent_types)

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_collect_producer_nodes_no_placeholder(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("weight", torch.ones(3))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.weight

        m = M().eval()
        example_inputs = (torch.randn(3),)

        m, guards = torchdynamo.export(  # noqa: F841
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        graph = m.graph
        add_nodes = [
            n for n in graph.nodes if n.op == "call_function" and "add" in str(n.target)
        ]

        # Check the add node, which should be None
        result = collect_producer_nodes(add_nodes[0])
        self.assertIsNone(result)

        # Check the input weight, which should not be None
        weight_node = add_nodes[0].args[1]
        result = collect_producer_nodes(weight_node)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    run_tests()
