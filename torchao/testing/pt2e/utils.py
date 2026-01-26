# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
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
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.inductor_utils import clone_preserve_strides_offset

import torchao
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e import FROM_NODE_KEY
from torchao.quantization.pt2e._numeric_debugger import _extract_node_source_debug_info
from torchao.quantization.pt2e.graph_utils import bfs_trace_with_node_process
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)
from torchao.utils import torch_version_at_least


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
        m = torch.export.export(
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
            m_fx = torch.export.export(
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


@unittest.skipIf(not torch_version_at_least("2.7.0"), "Requires torch 2.7+")
class PT2ENumericDebuggerTestCase(TestCase):
    """
    Base test case class for PT2E numeric debugger tests containing common utility functions
    for numeric debugging functionality.
    """

    def _assert_each_node_has_from_node_source(self, model) -> None:
        def _assert_node_has_from_node_source(node):
            if node.op == "placeholder" or node.op == "output":
                return

            # Handle guard nodes that don't have from_node metadata in newer PyTorch versions
            if FROM_NODE_KEY not in node.meta or node.meta[FROM_NODE_KEY] is None:
                # Guard nodes (like _guards_fn) created by newer PyTorch versions might not have from_node metadata
                # Skip these nodes as they are not part of the original user graph
                return

            # Check for nodes that are not part of the ExportedProgram.module().graph
            if (
                node.meta[FROM_NODE_KEY][-1].pass_name
                == "ExportedProgram.module().unlift()"
            ):
                # This node is not part of the ExportedProgram.module().graph, so it doesn't need debug info
                return

            # All other nodes should have from_node metadata
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


def get_default_quantizer(is_qat, is_dynamic):
    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        xiq.get_default_x86_inductor_quantization_config(
            is_qat=is_qat, is_dynamic=is_dynamic
        )
    )
    return quantizer


class FP8QDQLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias):
        super().__init__()
        self.qtype = torch.float8_e4m3fn
        self.weight = torch.randn((out_features, in_features)).to(self.qtype)
        self.weight_scale = 2.0
        self.scale = 2.0
        self.bias = None
        if has_bias:
            self.bias = torch.randn((out_features,))

    def forward(self, input):
        weight = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
            tensor=self.weight.data,
            scale=torch.tensor([self.weight_scale]),
            output_dtype=torch.float,
        )

        q_input = torch.ops.torchao.quantize_affine_float8_non_decomposed.default(
            tensor=input,
            scale=torch.tensor([self.scale]),
            float8_dtype=self.qtype,
        )
        dq_input = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
            tensor=q_input,
            scale=torch.tensor([self.scale]),
            output_dtype=torch.float,
        )

        out = torch.nn.functional.linear(dq_input, weight, self.bias)
        return out


class FP8QDQConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.qtype = torch.float8_e4m3fn
        self.weight = torch.randn(
            (out_channels, in_channels // groups, *kernel_size)
        ).to(self.qtype)
        self.weight_scale = 2.0
        self.scale = 2.0
        self.bias = None
        if bias:
            self.bias = torch.randn((out_channels,))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        weight = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
            tensor=self.weight.data,
            scale=torch.tensor([self.weight_scale]),
            output_dtype=torch.float,
        )
        q_input = torch.ops.torchao.quantize_affine_float8_non_decomposed.default(
            tensor=input,
            scale=torch.tensor([self.scale]),
            float8_dtype=self.qtype,
        )
        dq_input = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
            tensor=q_input,
            scale=torch.tensor([self.scale]),
            output_dtype=torch.float,
        )

        return torch.nn.functional.conv2d(
            dq_input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def qdq(input, scale):
    dtype = input.dtype
    q_input = torch.ops.torchao.quantize_affine_float8_non_decomposed.default(
        input,
        torch.tensor([scale]),
        torch.float8_e4m3fn,
    )
    dq_input = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
        q_input,
        torch.tensor([scale]),
        dtype,
    )
    return dq_input


def fp8_convert_(model):
    def generate_model_info(model):
        from collections import namedtuple

        mod_inst_info = namedtuple("ModInstInfo", ["name", "parent"])
        parent_child_mod_dict = {}

        def create_mod_info_recursion(parent):
            for name, mod in parent.named_children():
                parent_child_mod_dict[mod] = mod_inst_info(name=name, parent=parent)
                create_mod_info_recursion(mod)

        create_mod_info_recursion(model)
        return parent_child_mod_dict

    parent_child_mod_dict = generate_model_info(model)
    for name, mod in model.named_modules():
        mod_type_str = mod.__class__.__name__
        if mod_type_str not in ["Linear", "Conv2d"]:
            continue
        param = mod.weight
        xmax = torch.max(param)
        weight_scale = xmax / torch.finfo(torch.float8_e4m3fn).max
        mod.weight_scale = weight_scale
        q_param = torch.clamp(
            (param / weight_scale),
            torch.finfo(torch.float8_e4m3fn).min,
            torch.finfo(torch.float8_e4m3fn).max,
        ).to(torch.float8_e4m3fn)
        mod.weight.data = q_param
        if mod_type_str in ["Linear"]:
            patched_mod = FP8QDQLinear(mod.in_features, mod.out_features, False)
            patched_mod.bias = mod.bias
            patched_mod.weight_scale = weight_scale.item()
            patched_mod.weight.data = q_param
        elif mod_type_str in ["Conv2d"]:
            patched_mod = FP8QDQConv2d(
                mod.in_channels,
                mod.out_channels,
                mod.kernel_size,
                mod.stride,
                mod.padding,
                mod.dilation,
                mod.groups,
                False,
            )
            patched_mod.bias = mod.bias
            patched_mod.weight_scale = weight_scale.item()
            patched_mod.weight.data = q_param

        parent = parent_child_mod_dict[mod].parent
        name = parent_child_mod_dict[mod].name
        setattr(parent, name, patched_mod)


def _generate_qdq_quantized_model(
    mod,
    inputs,
    is_qat=False,
    is_dynamic=False,
    quantizer=None,
    is_fp8=False,
):
    maybe_no_grad = contextlib.nullcontext() if is_qat else torch.no_grad()
    with maybe_no_grad:
        if is_fp8:
            # fp8_convert_ not support dynamic and qat yet
            assert not is_dynamic
            assert not is_qat
            fp8_convert_(mod)
            return mod
        else:
            export_model = torch.export.export(mod, inputs, strict=True).module()
            quantizer = (
                quantizer if quantizer else get_default_quantizer(is_qat, is_dynamic)
            )
            prepare_model = (
                prepare_qat_pt2e(export_model, quantizer)
                if is_qat
                else prepare_pt2e(export_model, quantizer)
            )
            prepare_model(*inputs)
            torchao.quantization.pt2e.move_exported_model_to_eval(prepare_model)
            convert_model = convert_pt2e(prepare_model)
            return convert_model


def check_torch_compiled_model(
    model,
    example_inputs,
    kwargs=None,
    *,
    aoti=False,
    atol=None,
    rtol=None,
    fullgraph=True,
    inductor_configs=None,
    dynamic_shapes=None,
):
    """
    Utility function to check the results of torch compiled model
    by comparing the outputs of the original model and the compiled model.
    """
    kwargs = kwargs or {}
    torch._dynamo.reset()
    torch.manual_seed(0)

    ref_inputs = [clone_preserve_strides_offset(x) for x in example_inputs]
    ref_kwargs = kwargs
    ref_model = model
    correct = ref_model(*ref_inputs, **ref_kwargs)
    torch._inductor.metrics.reset()

    if aoti:
        with (
            torch.no_grad(),
            torch._export.config.patch(use_new_tracer_experimental=True),
        ):
            # strict=False needs extra migration work
            ep = torch.export.export(
                model,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=True,
                prefer_deferred_runtime_asserts_over_guards=True,
            )
            package_path = torch._inductor.aoti_compile_and_package(
                ep, inductor_configs=inductor_configs
            )
            compiled_model = torch._inductor.aoti_load_package(package_path)
    else:
        with torch.no_grad():
            compiled_model = torch.compile(model, fullgraph=fullgraph)
    actual = compiled_model(*example_inputs)
    torch.testing.assert_close(correct, actual, atol=atol, rtol=rtol)
