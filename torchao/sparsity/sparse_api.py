# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.sparse import to_sparse_semi_structured

from torchao.core.config import AOBaseConfig
from torchao.prototype.sparsity.sparsifier.weight_norm_sparsifier import (
    WeightNormSparsifier,
)
from torchao.quantization.quant_api import (
    _is_linear,
    _linear_extra_repr,
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.sparsity.blocksparse import BlockSparseTensor
from dataclasses import dataclass

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
)
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)

from torchao.kernel.splitk_sparse_gemv import splitk_sparse_gemv
from torch.utils._python_dispatch import return_and_correct_aliasing


# Sparsity helper functions
def apply_fake_sparsity(model, **kwargs):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    """
    filter_fn = kwargs.pop("filter_fn", _is_linear)
    # torch.ao.pruning flow
    sparse_config = []
    for name, mod in model.named_modules():
        if filter_fn(mod, name):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


@dataclass
class BlockSparseWeightConfig(AOBaseConfig):
    blocksize: int = 64


# for bc
block_sparse_weight = BlockSparseWeightConfig


@register_quantize_module_handler(BlockSparseWeightConfig)
def _block_sparse_weight_transform(
    module: torch.nn.Module,
    config: BlockSparseWeightConfig,
):
    blocksize = config.blocksize
    new_weight = BlockSparseTensor.from_dense(module.weight, blocksize)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


class SemiSparseWeightConfig(AOBaseConfig):
    """
    Configuration for converting the weight of linear modules to semi-structured (2:4) sparsity
    """

    pass


# for bc
semi_sparse_weight = SemiSparseWeightConfig


@register_quantize_module_handler(SemiSparseWeightConfig)
def _semi_sparse_weight_transform(
    module: torch.nn.Module,
    config: SemiSparseWeightConfig,
) -> torch.nn.Module:
    new_weight = to_sparse_semi_structured(module.weight)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def sparsify_(
    model: torch.nn.Module,
    config: AOBaseConfig,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> torch.nn.Module:
    """Convert the weight of linear modules in the model with `apply_tensor_subclass`.
    This function is essentially the same as quantize, put for sparsity subclasses.

    Currently, we support three options for sparsity:
        - semi-structured (2:4) sparsity with `semi_sparse_weight`
        - int8 dynamic quantization + 2:4 sparsity with `layout=SemiSparseLayout`
        - int4 weight-only quantization + 2:4 sparsity with `layout=SparseMarlinLayout`

    Args:
        model (torch.nn.Module): input model
        config (AOBaseConfig): a workflow configuration object
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): function that takes a nn.Module instance and fully qualified name of the module, returns True if we want to apply the specified workflow to this module.

    **Example:**
    ::
            import torch
            import torch.nn as nn
            from torchao.sparsity import sparsify_

            def filter_fn(module: nn.Module, fqn: str) -> bool:
                return isinstance(module, nn.Linear)

            m = nn.Sequential(nn.Linear(32, 1024), nn.Linear(1024, 32))

            # for 2:4 sparsity
            from torchao.sparse_api import semi_sparse_weight
            m = sparsify_(m, semi_sparse_weight(), filter_fn)

            # for int8 dynamic quantization + 2:4 sparsity
            from torchao.dtypes import SemiSparseLayout
            m = quantize_(m, int8_dynamic_activation_int8_weight(layout=SemiSparseLayout), filter_fn)
    """
    handler = _QUANTIZE_CONFIG_HANDLER[type(config)]
    _replace_with_custom_fn_if_matches_filter(
        model,
        handler,
        _is_linear if filter_fn is None else filter_fn,
        extra_args=(config,),
    )


from torchao.utils import TorchAOBaseTensor


class ActivationSparseTensor(TorchAOBaseTensor):
    data: Optional[torch.Tensor]

    __slots__ = ["data"]

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        data: Optional[torch.Tensor],
        requires_grad: bool = False,
    ):
        assert data is not None
        kwargs = {
            "device": data.device,
            "dtype": data.dtype,
            "layout": data.layout,
            "requires_grad": requires_grad,
        }
        tensor = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]
        tensor.data = data
        return tensor

    def __repr__(self) -> str:  # type: ignore[override]
        assert hasattr(self, "shape")
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(self):
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (self.shape, self.requires_grad)
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta,
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape, requires_grad = tensor_meta
        return cls(
            shape=shape,
            data=inner_tensors.get("data", None),
            requires_grad=requires_grad,
        )

    @classmethod
    def from_dense(cls, weight):
        return cls(weight.shape, weight.data.t().contiguous().t(), requires_grad=False)

    def apply_fn_to_shard(self, func):
        return ActivationSparseTensor(
            shape=self.shape,
            data=func(self.data),
            requires_grad=self.requires_grad,
        )


# Subclass op dispatch registration
implements = ActivationSparseTensor.implements
aten = torch.ops.aten


@implements(
    [
        aten.detach.default,
        aten.slice.Tensor,
    ]
)
def _(func, types, args, kwargs):
    new_data = func(args[0].data, *args[1:], **kwargs)
    return ActivationSparseTensor(
        new_data.shape,
        data=new_data,
        requires_grad=False,
    )


@implements([aten.copy_.default])
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    self.data.copy_(src.data)
    return


@implements(torch.nn.functional.linear)
def sparse_activation_linear(func, types, args, kwargs):
    x_orig, w, bias = args
    print(x_orig.shape)
    assert bias is None
    x = x_orig.view(-1, x_orig.size(-1))
    # M = w.shape[0]
    # K = w.shape[1]

    if x.shape[0] == 1:
        x_relu = torch.square(torch.nn.functional.relu(x))
        res = torch.ops.torchao.splitk_sparse_gemv(x_relu, w.data)
        return res.view(*x_orig.shape[:-1], w.shape[0])
    else:
        x_orig_relu = torch.square(torch.nn.functional.relu(x_orig))
        return torch.nn.functional.linear(x_orig_relu, w.data, bias)


@dataclass
class ActivationSparseLinearConfig(AOBaseConfig):
    """
    Adds in acceleration for activation sparsity to linear layers for decode.

    Args:
        `activation_dtype`: data type for quantized activation tensor.
        `weight_dtype`: data type for quantized weight tensor.
    """

    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn


@register_quantize_module_handler(ActivationSparseLinearConfig)
def _(
    module: torch.nn.Module,
    config: ActivationSparseLinearConfig,
):
    new_weight = ActivationSparseTensor.from_dense(module.weight.data)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module
