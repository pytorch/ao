# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs
import types

import torch
import torch.nn.functional as F

from torchao.utils import _assert_and_get_unique_device

__all__ = [
    "model_is_exported",
]

_EXPORTED_TRAINING_ATTR = "_exported_training"


class WrapperModule(torch.nn.Module):
    """Class to wrap a callable in an :class:`torch.nn.Module`. Use this if you
    are trying to export a callable.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        """Simple forward that just calls the ``fn`` provided to :meth:`WrapperModule.__init__`."""
        return self.fn(*args, **kwargs)


def model_is_exported(m: torch.nn.Module) -> bool:
    """
    Return True if the `torch.nn.Module` was exported, False otherwise
    (e.g. if the model was FX symbolically traced or not traced at all).
    """
    return isinstance(m, torch.fx.GraphModule) and any(
        "val" in n.meta for n in m.graph.nodes
    )


_DROPOUT_OPS = (
    torch.ops.aten.dropout.default,
    torch.ops.aten.dropout_.default,
)


def _replace_dropout(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    Switch dropout patterns in the model between train and eval modes.

    Dropout has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the dropout behavior between the two modes, so here we need to rewrite the aten
    dropout patterns manually to achieve the same effect.

    See https://github.com/pytorch/pytorch/issues/103681.
    """
    # Avoid circular dependencies
    from .utils import _get_aten_graph_module_for_pattern

    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()

    from torch.fx.subgraph_rewriter import replace_pattern_with_filters

    for inplace in [False, True]:

        def dropout_train(x):
            return F.dropout(x, p=0.5, training=True, inplace=inplace)

        def dropout_eval(x):
            return F.dropout(x, p=0.5, training=False, inplace=inplace)

        example_inputs = (torch.randn(1),)
        if train_to_eval:
            match_pattern = _get_aten_graph_module_for_pattern(
                WrapperModule(dropout_train),
                example_inputs,
            )
        else:
            match_pattern = _get_aten_graph_module_for_pattern(
                WrapperModule(dropout_eval),
                example_inputs,
            )

        def replacement_callback(match, original_graph, pattern_graph):
            # `ignore_literals=True` lets this pattern match dropout calls with
            # any `p`, not just the literal 0.5 used to build the pattern above.
            # Build the replacement using the *matched* node's actual `p` so we
            # don't silently overwrite the user's configured dropout rate with
            # 0.5. See https://github.com/pytorch/ao/issues/2980.
            (dropout_pattern_node,) = [
                n
                for n in pattern_graph.nodes
                if n.op == "call_function" and n.target in _DROPOUT_OPS
            ]
            matched_dropout_node = match.nodes_map[dropout_pattern_node]
            p = matched_dropout_node.args[1]
            target_training = not train_to_eval

            def dropout_replacement(x):
                return F.dropout(
                    x, p=p, training=target_training, inplace=inplace
                )

            return _get_aten_graph_module_for_pattern(
                WrapperModule(dropout_replacement),
                example_inputs,
            ).graph

        replace_pattern_with_filters(
            m,
            match_pattern,
            match_filters=[],
            ignore_literals=True,
            replacement_callback=replacement_callback,
        )
        m.recompile()


_BATCHNORM_OPS = (torch.ops.aten.batch_norm.default,)


def _replace_batchnorm(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    Switch batchnorm patterns in the model between train and eval modes.

    Batchnorm has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the batchnorm behavior between the two modes, so here we need to rewrite the aten
    batchnorm patterns manually to achieve the same effect.
    """
    # Avoid circular dependencies
    from .utils import _get_aten_graph_module_for_pattern

    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()

    def bn_train(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True
        )

    def bn_eval(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False
        )

    example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    device = _assert_and_get_unique_device(m)
    is_cuda = device is not None and device.type == "cuda"
    bn_train_aten = _get_aten_graph_module_for_pattern(
        WrapperModule(bn_train),
        example_inputs,
        is_cuda,
    )
    bn_eval_aten = _get_aten_graph_module_for_pattern(
        WrapperModule(bn_eval),
        example_inputs,
        is_cuda,
    )

    if train_to_eval:
        match_pattern = bn_train_aten
    else:
        match_pattern = bn_eval_aten

    def replacement_callback(match, original_graph, pattern_graph):
        # `ignore_literals=True` lets this pattern match batch_norm calls with
        # any momentum/eps, not just the (unset, i.e. torch defaults of 0.1 and
        # 1e-5) values used to build the pattern above. Build the replacement
        # using the *matched* node's actual momentum/eps/cudnn_enabled so we
        # don't silently overwrite the user's configured values.
        # See https://github.com/pytorch/ao/issues/2980.
        (bn_pattern_node,) = [
            n
            for n in pattern_graph.nodes
            if n.op == "call_function" and n.target in _BATCHNORM_OPS
        ]
        matched_bn_node = match.nodes_map[bn_pattern_node]
        # torch.ops.aten.batch_norm.default args:
        # (input, weight, bias, running_mean, running_var, training, momentum,
        #  eps, cudnn_enabled)
        _, _, _, _, _, _, momentum, eps, cudnn_enabled = matched_bn_node.args
        target_training = not train_to_eval

        def bn_replacement(
            x: torch.Tensor,
            bn_weight: torch.Tensor,
            bn_bias: torch.Tensor,
            bn_running_mean: torch.Tensor,
            bn_running_var: torch.Tensor,
        ):
            return torch.ops.aten.batch_norm.default(
                x,
                bn_weight,
                bn_bias,
                bn_running_mean,
                bn_running_var,
                target_training,
                momentum,
                eps,
                cudnn_enabled,
            )

        return _get_aten_graph_module_for_pattern(
            WrapperModule(bn_replacement),
            example_inputs,
            is_cuda,
        ).graph

    from torch.fx.subgraph_rewriter import replace_pattern_with_filters

    replace_pattern_with_filters(
        m,
        match_pattern,
        match_filters=[],
        ignore_literals=True,
        replacement_callback=replacement_callback,
    )
    m.recompile()


# TODO: expose these under this namespace?
def _move_exported_model_to_eval(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to eval mode.

    This is equivalent to model.eval() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing inference on the model.

    This call is idempotent; if the model is already in eval mode, nothing will happen.
    """
    is_training = getattr(model, _EXPORTED_TRAINING_ATTR, True)
    if not is_training:
        return model
    setattr(model, _EXPORTED_TRAINING_ATTR, False)
    _replace_dropout(model, train_to_eval=True)
    _replace_batchnorm(model, train_to_eval=True)
    return model


def _move_exported_model_to_train(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to train mode.

    This is equivalent to model.train() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing training on the model.

    This call is idempotent; if the model is already in train mode, nothing will happen.
    """
    is_training = getattr(model, _EXPORTED_TRAINING_ATTR, False)
    if is_training:
        return model
    setattr(model, _EXPORTED_TRAINING_ATTR, True)
    _replace_dropout(model, train_to_eval=False)
    _replace_batchnorm(model, train_to_eval=False)
    return model


def _allow_exported_model_train_eval(model: torch.fx.GraphModule):
    """
    Allow users to call `model.train()` and `model.eval()` on an exported model,
    but with the effect of changing behavior between the two modes limited to special
    ops only, which are currently dropout and batchnorm.

    Note: This does not achieve the same effect as what `model.train()` and `model.eval()`
    does in eager models, but only provides an approximation. In particular, user code
    branching on `training` flag will not function correctly in general because the branch
    is already specialized at export time. Additionally, other ops beyond dropout and batchnorm
    that have different train/eval behavior will also not be converted properly.
    """

    def _train(self, mode: bool = True):
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)

    def _eval(self):
        _move_exported_model_to_eval(self)

    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model
