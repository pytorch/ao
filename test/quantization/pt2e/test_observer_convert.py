# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for how convert_pt2e turns observers into decomposed q/dq nodes."""

import pytest
import torch

from torchao.quantization.pt2e import FROM_NODE_KEY, observer
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torchao.testing.pt2e._xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

_QDQ_NAMESPACES = ("quantized_decomposed", "torchao")
_QDQ_NAME_PARTS = ("quantize", "choose_qparams", "convert_element_type")


class _ConvLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.linear = torch.nn.Linear(4 * 6 * 6, 8)

    def forward(self, x):
        x = self.conv(x)
        return self.linear(x.flatten(1))


class _LinearInputQuantizer(Quantizer):
    """Annotates each linear input activation with the given spec."""

    def __init__(self, qspec):
        super().__init__()
        self.qspec = qspec

    def annotate(self, model):
        for node in model.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.linear.default
            ):
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={node.args[0]: self.qspec},
                    _annotated=True,
                )
        return model

    def validate(self, model):
        pass


def _fp16_input():
    return _LinearInputQuantizer(
        QuantizationSpec(
            dtype=torch.float16,
            observer_or_fake_quant_ctr=observer.PlaceholderObserver.with_args(
                dtype=torch.float16
            ),
        )
    )


def _dynamic_symmetric_input():
    return _LinearInputQuantizer(
        QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=True,
            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
        )
    )


def _xnnpack(**config_kwargs):
    return XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(**config_kwargs)
    )


# name -> (quantizer factory, is_qat, dtype)
CASES = {
    "static_per_channel": (lambda: _xnnpack(is_per_channel=True), False, torch.float),
    "static_per_tensor": (lambda: _xnnpack(is_per_channel=False), False, torch.float),
    "dynamic_affine": (
        lambda: _xnnpack(is_per_channel=True, is_dynamic=True),
        False,
        torch.float,
    ),
    "dynamic_symmetric": (_dynamic_symmetric_input, False, torch.float),
    "qat": (lambda: _xnnpack(is_per_channel=True, is_qat=True), True, torch.float),
    "fp16": (_fp16_input, False, torch.float),
    "static_bf16": (lambda: _xnnpack(is_per_channel=False), False, torch.bfloat16),
}


def _convert_case(name):
    quantizer_factory, is_qat, dtype = CASES[name]
    torch.manual_seed(0)
    model = _ConvLinear().to(dtype)
    example_inputs = (torch.randn(2, 3, 8, 8, dtype=dtype),)
    exported = torch.export.export(model, example_inputs, strict=True).module()
    prepare = prepare_qat_pt2e if is_qat else prepare_pt2e
    prepared = prepare(exported, quantizer_factory())
    prepared(*example_inputs)
    return convert_pt2e(prepared)


def _is_qdq_target(target):
    name = str(target)
    return name.startswith(_QDQ_NAMESPACES) and any(
        part in name for part in _QDQ_NAME_PARTS
    )


def _format_arg(arg):
    if isinstance(arg, torch.fx.Node):
        return f"@{arg.target}" if arg.op == "get_attr" else f"%{arg.name}"
    if isinstance(arg, float):
        return repr(round(arg, 6))
    if isinstance(arg, (list, tuple)):
        return "[" + ", ".join(_format_arg(a) for a in arg) + "]"
    return str(arg)


def _qdq_signature(gm):
    """One line per q/dq node: target, args, kwargs, and whether it carries
    FROM_NODE_KEY provenance. get_attr args show their attribute name."""
    signature = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or not _is_qdq_target(node.target):
            continue
        args = ", ".join(_format_arg(a) for a in node.args)
        kwargs = "".join(f", {k}={_format_arg(v)}" for k, v in node.kwargs.items())
        provenance = " +from_node" if FROM_NODE_KEY in node.meta else ""
        signature.append(f"{node.target}({args}{kwargs}){provenance}")
    return signature


# Captured from convert_pt2e before observers owned their conversion.
EXPECTED = {
    "dynamic_affine": [
        "quantized_decomposed.dequantize_per_channel.default(@_frozen_param0, @_scale_0, @_zero_point_0, 0, -127, 127, torch.int8) +from_node",
        "quantized_decomposed.choose_qparams.tensor(%flatten, -128, 127, 0.000244, torch.int8)",
        "quantized_decomposed.quantize_per_tensor.tensor(%flatten, %getitem, %getitem_1, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.tensor(%quantize_per_tensor_tensor, %getitem, %getitem_1, -128, 127, torch.int8) +from_node",
    ],
    "dynamic_symmetric": [
        "quantized_decomposed.choose_qparams_symmetric.tensor(%flatten, -128, 127, 0.0, torch.int8)",
        "quantized_decomposed.quantize_per_tensor.tensor(%flatten, %getitem, %getitem_1, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.tensor(%quantize_per_tensor_tensor, %getitem, %getitem_1, -128, 127, torch.int8) +from_node",
    ],
    "fp16": [
        "quantized_decomposed.convert_element_type.no_fuse(%flatten, torch.float16)",
        "quantized_decomposed.convert_element_type.no_fuse(%convert_element_type_no_fuse, torch.float32)",
    ],
    "qat": [
        "quantized_decomposed.dequantize_per_channel.default(@_frozen_param0, @_scale_0, @_zero_point_0, 0, -127, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_channel.default(@_frozen_param1, @_scale_1, @_zero_point_1, 0, -127, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%x, 0.02366, 19, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default, 0.02366, 19, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%conv2d, 0.012277, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_1, 0.012277, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%flatten, 0.012277, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_2, 0.012277, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%linear, 0.004324, -8, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_3, 0.004324, -8, -128, 127, torch.int8) +from_node",
    ],
    "static_bf16": [
        "quantized_decomposed.dequantize_per_tensor.default(@_frozen_param0, 0.001515, 0, -127, 127, torch.int8, out_dtype=torch.bfloat16) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(@_frozen_param1, 0.000657, 0, -127, 127, torch.int8, out_dtype=torch.bfloat16) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%x, 0.02364, 19, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_2, 0.02364, 19, -128, 127, torch.int8, out_dtype=torch.bfloat16) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%conv2d, 0.01231, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_3, 0.01231, 16, -128, 127, torch.int8, out_dtype=torch.bfloat16) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%flatten, 0.01231, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_4, 0.01231, 16, -128, 127, torch.int8, out_dtype=torch.bfloat16) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%linear, 0.004318, -8, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_5, 0.004318, -8, -128, 127, torch.int8, out_dtype=torch.bfloat16) +from_node",
    ],
    "static_per_channel": [
        "quantized_decomposed.dequantize_per_channel.default(@_frozen_param0, @_scale_0, @_zero_point_0, 0, -127, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_channel.default(@_frozen_param1, @_scale_1, @_zero_point_1, 0, -127, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%x, 0.023649, 19, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default, 0.023649, 19, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%conv2d, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_1, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%flatten, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_2, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%linear, 0.004322, -7, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_3, 0.004322, -7, -128, 127, torch.int8) +from_node",
    ],
    "static_per_tensor": [
        "quantized_decomposed.dequantize_per_tensor.default(@_frozen_param0, 0.001512, 0, -127, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(@_frozen_param1, 0.000656, 0, -127, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%x, 0.023649, 19, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_2, 0.023649, 19, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%conv2d, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_3, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%flatten, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_4, 0.012294, 16, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.quantize_per_tensor.default(%linear, 0.004322, -7, -128, 127, torch.int8) +from_node",
        "quantized_decomposed.dequantize_per_tensor.default(%quantize_per_tensor_default_5, 0.004322, -7, -128, 127, torch.int8) +from_node",
    ],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_converted_graph_unchanged(name):
    assert _qdq_signature(_convert_case(name)) == EXPECTED[name]


def test_default_convert_detection():
    from torchao.quantization import MappingType, PerTensor
    from torchao.quantization.pt2e._affine_quantization import (
        AffineQuantizedMinMaxObserver,
    )
    from torchao.quantization.pt2e._decomposed_qdq import _uses_default_convert
    from torchao.quantization.pt2e.fake_quantize import (
        AffineFakeQuantize,
        FakeQuantize,
    )
    from torchao.quantization.pt2e.observer import MinMaxObserver

    class MyMinMax(MinMaxObserver):
        pass

    assert _uses_default_convert(MinMaxObserver())
    assert _uses_default_convert(MyMinMax())
    assert _uses_default_convert(FakeQuantize())
    assert not _uses_default_convert(
        AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )
    )


class _ObserverLeafTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        return isinstance(m, observer.ObserverBase) or super().is_leaf_module(
            m, module_qualified_name
        )


class _Observed(torch.nn.Module):
    def __init__(self, obs):
        super().__init__()
        self.obs = obs

    def forward(self, x):
        return self.obs(x) * 2


def test_observer_convert_uses_attr_prefix():
    obs = observer.PerChannelMinMaxObserver(ch_axis=0, dtype=torch.qint8)
    root = _Observed(obs)
    gm = torch.fx.GraphModule(root, _ObserverLeafTracer().trace(root))
    gm(torch.randn(4, 3))
    (obs_node,) = [n for n in gm.graph.nodes if n.op == "call_module"]

    obs.convert(gm, obs_node, attr_prefix="lin_input")

    get_attrs = [n.target for n in gm.graph.nodes if n.op == "get_attr"]
    assert any(t.startswith("lin_input_scale_") for t in get_attrs), get_attrs
    assert any(t.startswith("lin_input_zero_point_") for t in get_attrs), get_attrs
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert targets[:2] == [
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ]
    assert not [n for n in gm.graph.nodes if n.op == "call_module"]


def _convert_with(quantizer):
    torch.manual_seed(0)
    model = _ConvLinear()
    example_inputs = (torch.randn(2, 3, 8, 8),)
    exported = torch.export.export(model, example_inputs, strict=True).module()
    prepared = prepare_pt2e(exported, quantizer)
    prepared(*example_inputs)
    return convert_pt2e(prepared)


def test_convert_pt2e_goes_through_default_convert(monkeypatch):
    import torchao.quantization.pt2e._decomposed_qdq as decomposed_qdq

    observers = []
    real = decomposed_qdq._insert_decomposed_qdq

    def spy(observer, model, node, *, attr_prefix="", model_device=None):
        observers.append(type(observer).__name__)
        real(observer, model, node, attr_prefix=attr_prefix, model_device=model_device)

    monkeypatch.setattr(decomposed_qdq, "_insert_decomposed_qdq", spy)
    signature = _qdq_signature(_convert_case("static_per_channel"))
    assert signature == EXPECTED["static_per_channel"]
    assert "PerChannelMinMaxObserver" in observers, observers
    assert "HistogramObserver" in observers, observers


def test_custom_two_arg_convert_still_called():
    calls = []

    class TwoArgConvertObserver(observer.MinMaxObserver):
        def convert(self, model, observer_node):
            calls.append(observer_node.name)
            observer_node.replace_all_uses_with(observer_node.args[0])
            model.graph.erase_node(observer_node)

    gm = _convert_with(
        _LinearInputQuantizer(
            QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=TwoArgConvertObserver,
            )
        )
    )
    assert calls
    assert _qdq_signature(gm) == []


def test_unsupported_dtype_observer_is_removed():
    gm = _convert_with(
        _LinearInputQuantizer(
            QuantizationSpec(
                dtype=torch.float32,
                observer_or_fake_quant_ctr=observer.PlaceholderObserver,
            )
        )
    )
    observers = [
        n
        for n in gm.graph.nodes
        if n.op == "call_module" and n.target.startswith("activation_post_process")
    ]
    assert not observers
    assert _qdq_signature(gm) == []


def test_instance_patched_convert_still_called():
    calls = []

    def convert(model, observer_node):
        calls.append(observer_node.name)
        observer_node.replace_all_uses_with(observer_node.args[0])
        model.graph.erase_node(observer_node)

    class InstancePatchedObserver(observer.MinMaxObserver):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.convert = convert

    gm = _convert_with(
        _LinearInputQuantizer(
            QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=InstancePatchedObserver,
            )
        )
    )
    assert calls
    assert _qdq_signature(gm) == []
