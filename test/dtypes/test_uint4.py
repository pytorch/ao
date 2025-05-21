# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import unittest

import torch
from torch import nn
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torch.fx import (
    GraphModule,
    Node,
)
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
)
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
)

from torchao.dtypes.uintx.uint4_layout import (
    PerChannelSymmetricWeightUInt4Tensor,
    UInt4Tensor,
)
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.testing.utils import skip_if_rocm
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


def _apply_weight_only_uint4_quant(model):
    def fn(mod):
        mod.weight = torch.nn.Parameter(
            PerChannelSymmetricWeightUInt4Tensor.from_float(mod.weight),
            requires_grad=False,
        )
        return mod

    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: fn(mod),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )


@unittest.skip(
    "FAILED test/dtypes/test_uint4.py::TestUInt4::test_basic_tensor_ops - AttributeError: module 'torch' has no attribute 'uint4'"
)
class TestUInt4(QuantizationTestCase):
    def test_basic_tensor_ops(self):
        x = UInt4Tensor(
            torch.tensor(
                [
                    [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                    [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                    [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                ],
                dtype=torch.uint8,
            )
        )
        self.assertEqual(x.shape, (3, 16))
        # TODO: make sure this returns torch.uint4
        self.assertIs(x.dtype, torch.uint4)
        # making sure these works
        x.to(torch.uint8)
        expected = UInt4Tensor(
            torch.tensor(
                [
                    [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                ],
                dtype=torch.uint8,
            )
        )
        self.assertEqual(x[0:1, :], expected)
        expected = UInt4Tensor(
            torch.tensor(
                [
                    [0x23, 0x45],
                    [0x23, 0x45],
                    [0x23, 0x45],
                ],
                dtype=torch.uint8,
            )
        )
        self.assertEqual(x[:, 2:6], expected)
        torch.save(x, "uint4_tensor.pt")
        x = torch.load("uint4_tensor.pt")
        self.assertEqual(x[:, 2:6], expected)
        # only test locally
        # print("x:", x[0])

    @skip_if_rocm("ROCm enablement in progress")
    def test_gpu_quant(self):
        for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
            x = torch.randn(*x_shape)
            m = nn.Sequential(nn.Linear(4, 16))
            m(x)  # checking if it runs
            _apply_weight_only_uint4_quant(m)
            m(x)  # checking if it runs
            # sqnr = compute_error(y_ref, y_wo)
            opt = torch.compile(m, fullgraph=True, mode="max-autotune")
            # make sure it runs
            opt(x)

    @skip_if_rocm("ROCm enablement in progress")
    def test_pt2e_quant(self):
        from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
            QuantizationConfig,
        )

        class Uint4Observer(ObserverBase):
            def __init__(self, *args, **kwargs):
                # just faking a dtype here
                # TODO: make flow work with new dtypes
                super().__init__(dtype=torch.int8)

            def forward(self, x):
                return x

            def calculate_qparams(self, **kwargs):
                pass

            def convert(self, model: GraphModule, observer_node: Node):
                with model.graph.inserting_before(observer_node):
                    q_node = model.graph.call_function(
                        torch.ops.qtensors.quantize_per_tensor_uint4,
                        (observer_node.args[0], 1.0, 0),
                        {},
                    )
                    dq_node = model.graph.call_function(
                        torch.ops.qtensors.dequantize_per_tensor_uint4,
                        (q_node, 1.0, 0),
                        {},
                    )
                    observer_node.replace_all_uses_with(dq_node)
                    model.graph.erase_node(observer_node)

        from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
            _is_annotated,
            _mark_nodes_as_annotated,
        )

        class Int8ActUint4WeightQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                uint4_qspec = QuantizationSpec(
                    dtype=torch.uint4,
                    quant_min=0,
                    quant_max=2**4 - 1,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=Uint4Observer,
                )
                int8_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_symmetric,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=torch.ao.quantization.observer.default_weight_observer,
                )
                quantization_config = QuantizationConfig(
                    input_activation=int8_qspec,
                    weight=uint4_qspec,
                    bias=None,
                    output_activation=int8_qspec,
                )
                for n in model.graph.nodes:
                    if n.op != "call_function" or n.target not in [
                        torch.ops.aten.linear.default,
                    ]:
                        continue
                    linear_node = n

                    input_qspec_map = {}
                    input_act = linear_node.args[0]
                    assert isinstance(input_act, Node)
                    input_qspec_map[input_act] = quantization_config.input_activation

                    weight = linear_node.args[1]
                    assert isinstance(weight, Node)
                    input_qspec_map[weight] = quantization_config.weight

                    partition = [linear_node, linear_node.args[1]]

                    bias = linear_node.args[2] if len(linear_node.args) > 2 else None
                    if isinstance(bias, Node):
                        input_qspec_map[bias] = quantization_config.bias
                        partition.append(bias)

                    if _is_annotated(partition):
                        continue

                    linear_node.meta["quantization_annotation"] = (
                        QuantizationAnnotation(
                            input_qspec_map=input_qspec_map,
                            output_qspec=quantization_config.output_activation,
                            _annotated=True,
                        )
                    )
                    _mark_nodes_as_annotated(partition)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        quantizer = Int8ActUint4WeightQuantizer()
        node_occurrence = {
            # for weight
            torch.ops.qtensors.quantize_per_tensor_uint4: 1,
            torch.ops.qtensors.dequantize_per_tensor_uint4: 1,
            # for activation
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.qtensors.dequantize_per_tensor_uint4,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        example_inputs = (torch.randn(2, 4),)

        # _test_quantizer in PT2EQuantizationTestCase
        # resetting dynamo cache
        torch._dynamo.reset()
        m_eager = M().eval()

        # program capture
        m = copy.deepcopy(m_eager)
        if TORCH_VERSION_AT_LEAST_2_5:
            m = torch.export.texport_for_training(
                m,
                example_inputs,
            ).module()
        else:
            m = torch._export.capture_pre_autograd_graph(
                m,
                example_inputs,
            ).module()

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=False)
        m(*example_inputs)

        node_occurrence = {ns.call_function(k): v for k, v in node_occurrence.items()}
        node_list = [ns.call_function(n) for n in node_list]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )


if __name__ == "__main__":
    unittest.main()
