# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, TORCH_VERSION_AT_LEAST_2_7

if TORCH_VERSION_AT_LEAST_2_7:
    from .constant_fold import constant_fold

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassManager

from torchao.quantization.pt2e.qat_utils import _fold_conv_bn_qat, _fuse_conv_bn_qat
from torchao.quantization.pt2e.quantizer import (  # noqa: F401
    DuplicateDQPass,
    PortNodeMetaForQDQ,
    Quantizer,
)
from torchao.quantization.pt2e.utils import (
    _disallow_eval_train,
    _fuse_conv_bn_,
    _get_node_name_to_scope,
)

from .convert import _convert_to_reference_decomposed_fx
from .prepare import prepare
from .reference_representation_rewrite import reference_representation_rewrite

__all__ = [
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "convert_pt2e",
]


def prepare_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """Prepare a model for post training quantization

    Args:
      * `model` (torch.fx.GraphModule): a model captured by `torch.export.export_for_training` API.
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
        m = torch.export.export_for_training(m, *example_inputs).module()
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
    # We will temporarily make prepare_pt2e backward compatible with quantizers that configs, observers,
    # and fake quantizers from torch.ao instead of torchao
    if isinstance(quantizer, torch.ao.quantization.quantizer.quantizer.Quantizer):
        from torch.ao.quantization.quantize_pt2e import (
            prepare_pt2e as torch_prepare_pt2e,
        )

        return torch_prepare_pt2e(model, quantizer)

    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.prepare_pt2e")
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    # TODO: check qconfig_mapping to make sure conv and bn are both configured
    # to be quantized before fusion
    # TODO: (maybe) rewrite this with subgraph_rewriter
    _fuse_conv_bn_(model)
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
        m = torch.export.export_for_training(m, *example_inputs).module()
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
    # We will temporarily make prepare_qat_pt2e backward compatible with quantizers that configs, observers,
    # and fake quantizers from torch.ao instead of torchao
    if isinstance(quantizer, torch.ao.quantization.quantizer.quantizer.Quantizer):
        from torch.ao.quantization.quantize_pt2e import (
            prepare_qat_pt2e as torch_prepare_qat_pt2e,
        )

        return torch_prepare_qat_pt2e(model, quantizer)

    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.prepare_qat_pt2e")
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
]

# ops are only registered after 2.5
if TORCH_VERSION_AT_LEAST_2_5:
    _QUANT_OPS += [
        torch.ops.torchao.quantize_affine,
    ]


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
    from torchao.quantization.pt2e.observer import ObserverBase as torchao_ObserverBase
    from torchao.quantization.pt2e.observer import (
        AffineQuantizedObserverBase as torchao_AffineQuantizedObserverBase,
    )

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
) -> GraphModule:
    """Convert a calibrated/trained model to a quantized model

    Args:
      * `model` (torch.fx.GraphModule): calibrated/trained model
      * `use_reference_representation` (bool): boolean flag to indicate whether to produce referece representation or not
      * `fold_quantize` (bool): boolean flag for whether fold the quantize op or not

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
    # We will temporarily make convert_pt2e backward compatible with quantizers that configs, observers,
    # and fake quantizers from torch.ao instead of torchao
    if not _is_torchao_prepared_do_not_use_outside_this_file(model):
        from torch.ao.quantization.quantize_pt2e import (
            convert_pt2e as torch_convert_pt2e,
        )

        return torch_convert_pt2e(model, use_reference_representation, fold_quantize)

    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.convert_pt2e")
    if not isinstance(use_reference_representation, bool):
        raise ValueError(
            "Unexpected argument type for `use_reference_representation`, "
            f"please make sure you intend to pass argument {use_reference_representation} to convert_pt2e"
        )
    original_graph_meta = model.meta
    model = _convert_to_reference_decomposed_fx(model)
    model = _fold_conv_bn_qat(model)

    pm = PassManager([DuplicateDQPass()])
    model = pm(model).graph_module

    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module

    if fold_quantize and TORCH_VERSION_AT_LEAST_2_7:
        constant_fold(model, _quant_node_constraint)

    if use_reference_representation:
        model = reference_representation_rewrite(model)

    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model
