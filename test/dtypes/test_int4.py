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
from torchao.quantization.utils import (
    compute_error,
)
from torchao.quantization.quant_api import (
    replace_with_custom_fn_if_matches_filter,
)
from torch.ao.quantization.observer import ObserverBase
from torch import nn
from torch.fx import (
    Node,
    GraphModule,
)
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
)
import copy

def _dynamically_quantize_per_channel_int4(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scale is the same dtype as the original tensor
    scale = torch.clamp(scale, min=eps).to(x.dtype)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scale/zp
    # reference: torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(0, 1) / scale
    x_round = torch.round(x_div)
    x_zp = x_round + zero_point
    x_zp = x_zp.transpose(0, 1)
    quant = torch.clamp(x_zp, quant_min, quant_max)
    if target_dtype == "int4":
        quant = UInt4Tensor.from_unpacked(quant.view(torch.bits8)).view(quant.size())
    else:
        quant = quant.to(target_dtype)

    return quant, scale, zero_point

class _WeightOnlyInt4QuantLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        w_int4 = kwargs.pop("w_int4")
        scales = kwargs.pop("scales")
        super().__init__(*args, **kwargs)
        self.w_int4 = w_int4
        self.scales = scales

    def forward(self, x):
        # if len(x.shape)<=2:
        #     y = torch.mm(x, self.w_int8.to(x.dtype)) * self.scales
        # else: # turn x into 2d tensor, then undo it for y
        x_view = x.view(-1, x.shape[-1])
        y = torch.mm(x_view, self.w_int4.to(torch.uint8).to(x.dtype)) * self.scales
        y = y.reshape(*x.shape[:-1], -1)
        if self.bias is not None:
            y += self.bias
        return y

    @classmethod
    def from_float(cls, mod):
        w_fp32 = mod.weight
        w_int4, scales, _zp = _dynamically_quantize_per_channel_int4(
            w_fp32, 0, 15, "int4"
        )
        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features,
            bias=mod.bias is not None,
            # w_int4=w_int4.t().contiguous(),
            w_int4=torch.ops.aten.transpose_copy(w_int4, 0, 1),
            scales=scales,
        )
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        del new_mod.weight
        new_mod.bias = mod.bias
        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

def _apply_weight_only_int4_quant(model):
    replace_with_custom_fn_if_matches_filter(
        model,
        _WeightOnlyInt4QuantLinear.from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )

from torch.library import Library, impl

test_lib = Library("test_int4", "DEF")
test_lib.define("quantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor")

@impl(test_lib, "quantize_per_tensor_int4", "CompositeExplicitAutograd")
def quantize_per_tensor_int4(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 15).to(torch.uint8).view(torch.bits8)

test_lib.define("dequantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor")
@impl(test_lib, "dequantize_per_tensor_int4", "CompositeExplicitAutograd")
def dequantize_per_tensor_int4(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    print("1", input.dtype)
    a = input.to(torch.uint8)
    print("2")
    a = a.to(torch.float32)
    print("3")
    a = a - zero_point
    print("4")
    a = a * scale
    print("5")
    return a
    # return (input.to(torch.uint8).to(torch.float32) - zero_point) * scale


class TestInt4(QuantizationTestCase):
    def test_basic_tensor_ops(self):
        x = UInt4Tensor(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.bits8))
        self.assertEqual(x.shape, (3, 16))
        # making sure these works
        x.to(torch.uint8)
        expected = UInt4Tensor(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.bits8))
        self.assertTrue(x[0:1, :] == expected)
        expected = UInt4Tensor(torch.tensor([
            [0x23, 0x45],
            [0x23, 0x45],
            [0x23, 0x45],
        ], dtype=torch.bits8))
        self.assertTrue(x[:, 2:6] == expected)

    def test_gpu_quant(self):
        for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
            x = torch.randn(*x_shape)
            m = nn.Sequential(nn.Linear(4, 16))
            y_ref = m(x)
            _apply_weight_only_int4_quant(m)
            y_wo = m(x)
            # sqnr = compute_error(y_ref, y_wo)
            opt = torch.compile(m, mode="max-autotune")
            # make sure it runs
            opt(x)

    def test_aten_ir(self):
        # class QuantizePerTensorUInt4(torch.autograd.Function):
        #     @staticmethod
        #     def forward(
        #         ctx,
        #         input: torch.Tensor,
        #         scale: float,
        #         zero_point: int,
        #     ) -> torch.Tensor:
        #         inv_scale = 1.0 / scale
        #         return UInt4Tensor(torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 15).to(torch.bits8))

        # class DeQuantizePerTensorUInt4(torch.autograd.Function):
        #     @staticmethod
        #     def forward(
        #         ctx,
        #         input: torch.Tensor,
        #         scale: float,
        #         zero_point: int,
        #     ) -> torch.Tensor:
        #         return (input.to(torch.float32) - zero_point) * scale

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3),)
        m = M().eval()
        m = capture_pre_autograd_graph(m, example_inputs)
        for n in m.graph.nodes:
            if n.target == torch.ops.aten.add.Tensor:
                with m.graph.inserting_before(n):
                    q = m.graph.call_function(torch.ops.test_int4.quantize_per_tensor_int4, (n.args[0], 1.0, 0), {})
                    dq = m.graph.call_function(torch.ops.test_int4.dequantize_per_tensor_int4, (q, 1.0, 0), {})
                    n.replace_input_with(n.args[0], dq)
        m.recompile()

    def test_pt2e_quant(self):
        from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
            OP_TO_ANNOTATOR,
            QuantizationConfig,
        )
        class int4_class():
            pass

        torch.int4 = int4_class()

        class Int4Observer(ObserverBase):
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
                        torch.ops.test_int4.quantize_per_tensor_int4, (observer_node.args[0], 1.0, 0), {})
                    dq_node = model.graph.call_function(
                        torch.ops.test_int4.dequantize_per_tensor_int4, (q_node, 1.0, 0), {})
                    observer_node.replace_all_uses_with(dq_node)
                    model.graph.erase_node(observer_node)

        from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
            _is_annotated,
            _mark_nodes_as_annotated,
        )

        class Int4WeightQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                int4_qspec = QuantizationSpec(
                    dtype=torch.int4,
                    quant_min=-2**3,
                    quant_max=2**3 - 1,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=Int4Observer,
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
                    weight=int4_qspec,
                    bias=None,
                    output_activation=int8_qspec,
                )
                for n in model.graph.nodes:
                    if n.op != "call_function" or n.target not in [
                        torch.ops.aten.conv1d.default,
                        torch.ops.aten.conv2d.default,
                    ]:
                        continue
                    conv_node = n

                    input_qspec_map = {}
                    input_act = conv_node.args[0]
                    assert isinstance(input_act, Node)
                    input_qspec_map[input_act] = quantization_config.input_activation

                    weight = conv_node.args[1]
                    assert isinstance(weight, Node)
                    input_qspec_map[weight] = quantization_config.weight

                    partition = [conv_node, conv_node.args[1]]

                    bias = conv_node.args[2] if len(conv_node.args) > 2 else None
                    if isinstance(bias, Node):
                        input_qspec_map[bias] = quantization_config.bias
                        partition.append(bias)

                    if _is_annotated(partition):
                        continue

                    conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=quantization_config.output_activation,
                        _annotated=True,
                    )
                    _mark_nodes_as_annotated(partition)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        quantizer = Int4WeightQuantizer()
        node_occurrence = {
            # for weight
            torch.ops.test_int4.quantize_per_tensor_int4: 1,
            torch.ops.test_int4.dequantize_per_tensor_int4: 1,
            # for activation
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.test_int4.dequantize_per_tensor_int4,
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
        )

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=False)

        pt2_quant_output = m(*example_inputs)
        node_occurrence = {
            ns.call_function(k): v for k, v in node_occurrence.items()
        }
        node_list = [ns.call_function(n) for n in node_list]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )

if __name__ == "__main__":
    main()
