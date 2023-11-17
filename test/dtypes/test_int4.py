import torch
from torchao.dtypes.int4 import UInt4Tensor
import unittest
from unittest import TestCase, main
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer

from torch._export import capture_pre_autograd_graph
from torch._export import dynamic_dim
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
)
import copy


class TestInt4(QuantizationTestCase):
    def test_basic_tensor_ops(self):
        x = UInt4Tensor(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.uint8))
        self.assertTrue(x.shape, (3, 8))
        # making sure these works
        x.to(torch.uint8)
        expected = UInt4Tensor(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.uint8))
        self.assertTrue(x[0:1, :] == expected)
        expected = UInt4Tensor(torch.tensor([
            [0x23, 0x45],
            [0x23, 0x45],
            [0x23, 0x45],
        ], dtype=torch.uint8))
        self.assertTrue(x[:, 2:6] == expected)

    def test_gpu_quant(self):
        pass

    def test_aten_ir(self):
        # from torch.library import Library, impl
        # test_lib = Library("test_int4", "DEF")
        # test_lib.define("quantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor")
        # @impl(test_lib, "quantize_per_tensor_int4", "CompositeExplicitAutograd")
        # def quantize_per_tensor_int4(
        #     input: torch.Tensor,
        #     scale: float,
        #     zero_point: int,
        # ) -> torch.Tensor:
        #     inv_scale = 1.0 / scale
        #     return UInt4Tensor(torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 15).to(torch.uint8))

        # test_lib.define("dequantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor")
        # @impl(test_lib, "dequantize_per_tensor_int4", "CompositeExplicitAutograd")
        # def dequantize_per_tensor_int4(
        #     input: torch.Tensor,
        #     scale: float,
        #     zero_point: int,
        # ) -> torch.Tensor:
        #     return (input.to(torch.float32) - zero_point) * scale

        class QuantizePerTensorUInt4(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                input: torch.Tensor,
                scale: float,
                zero_point: int,
            ) -> torch.Tensor:
                inv_scale = 1.0 / scale
                return UInt4Tensor(torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 15).to(torch.uint8))

        class DeQuantizePerTensorUInt4(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                input: torch.Tensor,
                scale: float,
                zero_point: int,
            ) -> torch.Tensor:
                return (input.to(torch.float32) - zero_point) * scale

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3),)
        m = M().eval()
        m = capture_pre_autograd_graph(m, example_inputs)
        qop = QuantizePerTensorUInt4.apply
        dqop = DeQuantizePerTensorUInt4.apply
        for n in m.graph.nodes:
            if n.target == torch.ops.aten.add.Tensor:
                with m.graph.inserting_before(n):
                    # q = m.graph.call_function(torch.ops.test_int4.quantize_per_tensor_int4, (n.args[0], 1.0, 0), {})
                    # dq = m.graph.call_function(torch.ops.test_int4.dequantize_per_tensor_int4, (q, 1.0, 0), {})
                    q = m.graph.call_function(qop, (n.args[0], 1.0, 0), {})
                    dq = m.graph.call_function(dqop, (q, 1.0, 0), {})
                    n.replace_input_with(n.args[0], dq)
        m.recompile()
        print("m:", m)
        print(m(*example_inputs))

    # TODO: need more extension points from quant flow side
    @unittest.skip("need more extension points from quant flow side")
    def test_pt2e_quant(self):
        from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
            OP_TO_ANNOTATOR,
            QuantizationConfig,
        )

        class Int4ActQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                int4_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-2**3,
                    quant_max=2**3 - 1,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                int8_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_symmetric,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                quantization_config = QuantizationConfig(
                    input_activation=int8_qspec,
                    weight=int4_qspec,
                    bias=None,
                    output_activation=int8_qspec,
                )
                OP_TO_ANNOTATOR["conv"](model, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        quantizer = Int4ActQuantizer()
        node_occurrence = {
            # one for input of the first conv, one for output for the first conv
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        example_inputs = (torch.randn(1, 3, 3, 3),)

        # _test_quantizer in PT2EQuantizationTestCase
        # resetting dynamo cache
        export_with_dynamic_shape = False
        torch._dynamo.reset()
        m_eager = M().eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
            constraints=[dynamic_dim(example_inputs[0], 0)] if export_with_dynamic_shape else [],
        )

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)

        pt2_quant_output = m(*example_inputs)
        node_occurrence = {
            ns.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        if expected_node_list is None:
            expected_node_list = []
        node_list = [ns.call_function(n) for n in expected_node_list]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )

if __name__ == "__main__":
    main()
