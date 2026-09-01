# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassManager
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.multiprocessing.reductions import StorageWeakRef

from torchao.quantization.pt2e.qat_utils import _fold_conv_bn_qat, _fuse_conv_bn_qat
from torchao.quantization.pt2e.quantizer import (  # noqa: F401
    DuplicateDQPass,
    PortNodeMetaForQDQ,
    Quantizer,
)
from torchao.quantization.pt2e.utils import (
    _disallow_eval_train,
    _fuse_conv_bn_,
    _fuse_linear_bn_,
    _get_node_name_to_scope,
    get_arg,
)

from .constant_fold import constant_fold
from .convert import _convert_to_reference_decomposed_fx
from .prepare import prepare
from .reference_representation_rewrite import reference_representation_rewrite

__all__ = [
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "convert_pt2e",
]


def _reject_set_grad_enabled_subgraph(model: GraphModule, api_name: str) -> None:
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.higher_order.wrap_with_set_grad_enabled
        ):
            if api_name == "prepare_qat_pt2e":
                remediation = (
                    "Remove or disable the model's no-grad context before export because "
                    "quantization-aware training requires autograd."
                )
            else:
                remediation = (
                    "Export the model under a grad mode matching its forward method "
                    "(for example, use `with torch.no_grad():` for a forward method "
                    "decorated with `@torch.no_grad()`) before calling prepare_pt2e."
                )
            raise ValueError(
                f"{api_name} does not support wrap_with_set_grad_enabled subgraphs. "
                f"{remediation}"
            )


def prepare_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """Prepare a model for post training quantization

    Args:
      * `model` (torch.fx.GraphModule): a model captured by `torch.export.export` API.
      * `quantizer`: A backend specific quantizer that conveys how user want the
        model to be quantized. Tutorial for how to write a quantizer can be found here:
        https://pytorch.org/tutorials/prototype/pt2e_quantizer.html

    Return:
      A GraphModule with observer (based on quantizer annotation), ready for calibration

    Example::

        import torch
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
        from torchao.quantization.pt2e.quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

           def forward(self, x):
               return self.linear(x)

        # initialize a floating point model
        float_model = M().eval()

        # define calibration function
        def calibrate(model, data_loader):
            model.eval()
            with torch.no_grad():
                for image, target in data_loader:
                    model(image)

        # Step 1. program capture
        # NOTE: this API will be updated to torch.export API in the future, but the captured
        # result shoud mostly stay the same
        m = torch.export.export(m, *example_inputs).module()
        # we get a model with aten ops

        # Step 2. quantization
        # backend developer will write their own Quantizer and expose methods to allow
        # users to express how they
        # want the model to be quantized
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = prepare_pt2e(m, quantizer)

        # run calibration
        # calibrate(m, sample_inference_data)
    """
    torch._C._log_api_usage_once("torchao.quantization.pt2e.prepare_pt2e")
    _reject_set_grad_enabled_subgraph(model, "prepare_pt2e")
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    # TODO: check qconfig_mapping to make sure conv and bn are both configured
    # to be quantized before fusion
    # TODO: (maybe) rewrite this with subgraph_rewriter
    _fuse_conv_bn_(model)
    _fuse_linear_bn_(model)
    model = quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    model = prepare(
        model,
        node_name_to_scope,
        is_qat=False,
        obs_or_fq_callback=quantizer.prepare_obs_or_fq_callback,
    )
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    # Recursively prepare combine_fn subgraphs of scan ops.
    # This is done after the top-level prepare since prepare() does not
    # modify scan subgraphs (they are separate GraphModules), so ordering
    # does not matter here.
    for node in model.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.higher_order.scan:
            scan_combine_fn_node = node.args[0]
            assert isinstance(scan_combine_fn_node, Node)
            assert scan_combine_fn_node.op == "get_attr"
            assert isinstance(scan_combine_fn_node.target, str)
            scan_combine_fn = model.get_submodule(scan_combine_fn_node.target)
            prepared_scan_combine_fn = prepare_pt2e(scan_combine_fn, quantizer)
            setattr(model, scan_combine_fn_node.target, prepared_scan_combine_fn)
    # Recursively prepare body_fn subgraphs of while_loop ops.
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.higher_order.while_loop
        ):
            while_loop_body_fn_node = node.args[1]
            assert isinstance(while_loop_body_fn_node, Node)
            assert while_loop_body_fn_node.op == "get_attr"
            assert isinstance(while_loop_body_fn_node.target, str)
            while_loop_body_fn = model.get_submodule(while_loop_body_fn_node.target)
            prepared_while_loop_body_fn = prepare_pt2e(while_loop_body_fn, quantizer)
            setattr(model, while_loop_body_fn_node.target, prepared_while_loop_body_fn)
    return model


def prepare_qat_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """Prepare a model for quantization aware training

    Args:
      * `model` (torch.fx.GraphModule): see :func:`~torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e`
      * `quantizer`: see :func:`~torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e`

    Return:
      A GraphModule with fake quant modules (based on quantizer annotation), ready for
      quantization aware training

    Example::
        import torch
        from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
        from torchao.quantization.pt2e.quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

           def forward(self, x):
               return self.linear(x)

        # initialize a floating point model
        float_model = M().eval()

        # define the training loop for quantization aware training
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...

        # Step 1. program capture
        # NOTE: this API will be updated to torch.export API in the future, but the captured
        # result shoud mostly stay the same
        m = torch.export.export(m, *example_inputs).module()
        # we get a model with aten ops

        # Step 2. quantization
        # backend developer will write their own Quantizer and expose methods to allow
        # users to express how they
        # want the model to be quantized
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = prepare_qat_pt2e(m, quantizer)

        # run quantization aware training
        train_loop(prepared_model, train_loop)

    """
    torch._C._log_api_usage_once("torchao.quantization.pt2e.prepare_qat_pt2e")
    _reject_set_grad_enabled_subgraph(model, "prepare_qat_pt2e")
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    model = quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    # Perform fusion after annotate to avoid quantizing ops in the new
    # subgraph that don't need to be quantized
    # TODO: only fuse if conv and bn are both configured to be quantized
    _fuse_conv_bn_qat(model)
    model = prepare(
        model,
        node_name_to_scope,
        is_qat=True,
        obs_or_fq_callback=quantizer.prepare_obs_or_fq_callback,
    )
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model


_QUANT_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    torch.ops.torchao.quantize_affine,
]

_QUANTIZE_PER_TENSOR = torch.ops.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE_PER_TENSOR = torch.ops.quantized_decomposed.dequantize_per_tensor.default
_QParams = tuple[float, int, int, int, torch.dtype]


def _static_qparams(node: Node, target: torch._ops.OpOverload) -> _QParams | None:
    if node.target != target:
        return None
    return (
        get_arg(node, "scale", float),
        get_arg(node, "zero_point", int),
        get_arg(node, "quant_min", int),
        get_arg(node, "quant_max", int),
        get_arg(node, "dtype", torch.dtype),
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    signature = (tuple(tensor.shape), tensor.device, tensor.layout)
    if tensor.layout == torch.strided:
        return (*signature, tuple(tensor.stride()))
    return signature


def _fold_quantize_into_mutable_buffers(model: GraphModule) -> None:
    """Fold Q/DQ buffer boundaries with an explicit copy writeback."""
    named_buffers = dict(model.named_buffers(remove_duplicate=False))
    changed = False

    for target, buffer_value in named_buffers.items():
        buffers = model.graph.find_nodes(op="get_attr", target=target)
        if len(buffers) != 1:
            continue
        buffer = buffers[0]

        input_quants = [
            user for user in buffer.users if user.target == _QUANTIZE_PER_TENSOR
        ]
        if len(input_quants) != 1:
            continue
        input_quant = input_quants[0]
        input_qparams = _static_qparams(input_quant, _QUANTIZE_PER_TENSOR)
        assert input_qparams is not None
        input_dequants = [
            user
            for user in input_quant.users
            if _static_qparams(user, _DEQUANTIZE_PER_TENSOR) == input_qparams
        ]
        if len(input_dequants) != 1 or set(input_quant.users) != {input_dequants[0]}:
            continue
        input_dequant = input_dequants[0]

        copies = tuple(
            user
            for user in buffer.users
            if user.target == torch.ops.aten.copy_.default
            and user.args[0] is buffer
            and not user.users
        )
        if len(copies) != 1 or set(buffer.users) != {input_quant, copies[0]}:
            continue
        copy = copies[0]
        output_dequant = get_arg(copy, "src", Node)
        output_dequant_qparams = _static_qparams(output_dequant, _DEQUANTIZE_PER_TENSOR)
        if output_dequant_qparams is None:
            continue
        output_quant = get_arg(output_dequant, "input", Node)
        output_qparams = _static_qparams(output_quant, _QUANTIZE_PER_TENSOR)
        if output_qparams is None or output_dequant_qparams != output_qparams:
            continue
        output_dequants = tuple(
            user
            for user in output_quant.users
            if _static_qparams(user, _DEQUANTIZE_PER_TENSOR) == output_qparams
        )
        if not output_dequants or set(output_quant.users) != set(output_dequants):
            continue

        if not buffer_value.is_floating_point():
            continue
        storage_ref = StorageWeakRef(buffer_value.untyped_storage())
        aliases = [
            name
            for name, value in named_buffers.items()
            if name != target and StorageWeakRef(value.untyped_storage()) == storage_ref
        ]
        if aliases:
            raise ValueError(
                f"Cannot fold quantization into mutable buffer {target!r}; "
                f"it aliases registered buffers {sorted(aliases)}"
            )

        current_value = buffer.meta.get("val")
        if not isinstance(current_value, FakeTensor):
            raise ValueError(
                f"Cannot fold quantization into mutable buffer {target!r}; "
                "the buffer is missing FakeTensor metadata"
            )
        scale, zero_point, quant_min, quant_max, dtype = output_qparams
        with torch.utils._python_dispatch._disable_current_modes():
            quantized_buffer = _QUANTIZE_PER_TENSOR(
                buffer_value,
                scale,
                zero_point,
                quant_min,
                quant_max,
                dtype,
            )
        if _tensor_signature(quantized_buffer) != _tensor_signature(buffer_value):
            raise ValueError(
                f"Cannot fold quantization into mutable buffer {target!r}; "
                "quantization changed its shape, device, layout, or stride"
            )

        *prefix, attr = target.split(".")
        owner = model.get_submodule(".".join(prefix)) if prefix else model
        setattr(owner, attr, quantized_buffer)
        input_dequant.update_arg(0, buffer)
        for index, value in enumerate(output_qparams, 1):
            input_dequant.update_arg(index, value)
        copy.update_arg(1, output_quant)
        metadata_value = current_value.fake_mode.from_tensor(
            quantized_buffer, static_shapes=True
        )
        tensor_meta = _extract_tensor_metadata(metadata_value)
        for node in (buffer, copy):
            node.meta["val"] = metadata_value
            node.meta["tensor_meta"] = tensor_meta
        changed = True

    if changed:
        model.graph.eliminate_dead_code()
        model.graph.lint()
        model.recompile()


def _quant_node_constraint(n: Node) -> bool:
    """If there is any pure ops between get_attr and quantize op they will be const propagated
    e.g. get_attr(weight) -> transpose -> quantize -> dequantize*
    (Note: dequantize op is not going to be constant propagated)

    This filter is added because we don't want to constant fold the things that are not
    related to quantization
    """
    return n.op == "call_function" and n.target in _QUANT_OPS


def _is_torchao_prepared_do_not_use_outside_this_file(model):
    from torchao.quantization.pt2e.fake_quantize import (
        FakeQuantize as torchao_FakeQuantize,
    )
    from torchao.quantization.pt2e.observer import (
        AffineQuantizedObserverBase as torchao_AffineQuantizedObserverBase,
    )
    from torchao.quantization.pt2e.observer import ObserverBase as torchao_ObserverBase

    is_torch_ao_prepared = False
    is_torchao_prepared = False
    for _, m in model.named_modules():
        if (
            isinstance(m, torch.ao.quantization.fake_quantize.FakeQuantize)
            or isinstance(m, torch.ao.quantization.observer.ObserverBase)
            or isinstance(m, torch.ao.quantization.observer.AffineQuantizedObserverBase)
        ):
            is_torch_ao_prepared = True
        if (
            isinstance(m, torchao_FakeQuantize)
            or isinstance(m, torchao_ObserverBase)
            or isinstance(m, torchao_AffineQuantizedObserverBase)
        ):
            is_torchao_prepared = True

    if is_torch_ao_prepared:
        assert not is_torchao_prepared, (
            "Cannot be prepared using both torch.ao and torchao"
        )
    if is_torchao_prepared:
        assert not is_torch_ao_prepared, (
            "Cannot be prepared using both torch.ao and torchao"
        )

    return is_torchao_prepared


def convert_pt2e(
    model: GraphModule,
    use_reference_representation: bool = False,
    fold_quantize: bool = True,
    fold_quantize_into_mutable_buffers: bool = False,
) -> GraphModule:
    """Convert a calibrated/trained model to a quantized model

    Args:
      * `model` (torch.fx.GraphModule): calibrated/trained model
      * `use_reference_representation` (bool): boolean flag to indicate whether to produce referece representation or not
      * `fold_quantize` (bool): boolean flag for whether fold the quantize op or not
      * `fold_quantize_into_mutable_buffers` (bool): boolean flag for whether to fold
        quantize ops into mutable registered buffer storage. Requires `fold_quantize=True`.

    Returns:
        quantized model, either in q/dq representation or reference representation

    Example::

        # prepared_model: the model produced by `prepare_pt2e`/`prepare_qat_pt2e` and calibration/training
        # `convert_pt2e` produces a quantized model that represents quantized computation with
        # quantize dequantize ops and fp32 ops by default.
        # Please refer to
        # https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html#convert-the-calibrated-model-to-a-quantized-model
        # for detailed explanation of output quantized model
        quantized_model = convert_pt2e(prepared_model)

    """
    torch._C._log_api_usage_once("torchao.quantization.pt2e.convert_pt2e")
    if not isinstance(use_reference_representation, bool):
        raise ValueError(
            "Unexpected argument type for `use_reference_representation`, "
            f"please make sure you intend to pass argument {use_reference_representation} to convert_pt2e"
        )
    if fold_quantize_into_mutable_buffers and not fold_quantize:
        raise ValueError(
            "fold_quantize_into_mutable_buffers=True requires fold_quantize=True"
        )
    original_graph_meta = model.meta
    # Recursively convert combine_fn subgraphs of scan ops before the
    # top-level conversion, so that passes like DuplicateDQPass that
    # recursively lint child graphs won't encounter stale observer refs.
    for node in model.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.higher_order.scan:
            scan_combine_fn_node = node.args[0]
            assert isinstance(scan_combine_fn_node, Node)
            assert scan_combine_fn_node.op == "get_attr"
            assert isinstance(scan_combine_fn_node.target, str)
            scan_combine_fn = model.get_submodule(scan_combine_fn_node.target)
            converted_scan_combine_fn = convert_pt2e(
                scan_combine_fn,
                use_reference_representation=use_reference_representation,
                fold_quantize=fold_quantize,
                fold_quantize_into_mutable_buffers=fold_quantize_into_mutable_buffers,
            )
            setattr(model, scan_combine_fn_node.target, converted_scan_combine_fn)
    # Recursively convert body_fn subgraphs of while_loop ops.
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.higher_order.while_loop
        ):
            while_loop_body_fn_node = node.args[1]
            assert isinstance(while_loop_body_fn_node, Node)
            assert while_loop_body_fn_node.op == "get_attr"
            assert isinstance(while_loop_body_fn_node.target, str)
            while_loop_body_fn = model.get_submodule(while_loop_body_fn_node.target)
            converted_while_loop_body_fn = convert_pt2e(
                while_loop_body_fn,
                use_reference_representation=use_reference_representation,
                fold_quantize=fold_quantize,
                fold_quantize_into_mutable_buffers=fold_quantize_into_mutable_buffers,
            )
            setattr(model, while_loop_body_fn_node.target, converted_while_loop_body_fn)
    model = _convert_to_reference_decomposed_fx(model)
    model = _fold_conv_bn_qat(model)

    pm = PassManager([DuplicateDQPass()])
    model = pm(model).graph_module

    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module

    if fold_quantize:
        if fold_quantize_into_mutable_buffers:
            _fold_quantize_into_mutable_buffers(model)
        constant_fold(model, _quant_node_constraint)

    if use_reference_representation:
        model = reference_representation_rewrite(model)

    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model
