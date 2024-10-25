# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
import torch.nn.functional as F

from torchao.quantization.unified import TwoStepQuantizer
from torchao.quantization.utils import get_group_qparams_symmetric
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.quantization.quant_primitives import TorchAODType
from .api import FakeQuantizeConfig
from .fake_quantizer import FakeQuantizer
from .utils import (
    _fake_quantize_per_channel_group,
    _get_qmin_qmax,
)


class FakeQuantizedEmbedding(torch.nn.Embedding):
    """
    General embedding layer with fake quantized weights.

    Specific target dtypes, granularity, schemes etc. are specified
    through separate configs for weights and activations.

    Example usage::

        weight_config = FakeQuantizeConfig(
            dtype=torch.int4,
            group_size=8,
            symmetric=True,
        )
        fq_embedding = FakeQuantizedEmbedding(5, 10, weight_config)
        fq_embedding(torch.LongTensor([3]))
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        weight_config: Optional[FakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            *args,
            **kwargs,
        )
        if weight_config is not None:
            self.weight_fake_quantizer = FakeQuantizer(weight_config)
        else:
            self.weight_fake_quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_fake_quantizer is not None:
            w = self.weight_fake_quantizer(self.weight)
        else:
            w = self.weight
        return F.embedding(
            x, w, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )


# ======================================
# |   Embedding int4 weight-only QAT   |
# ======================================

class Int4WeightOnlyEmbeddingQATQuantizer(TwoStepQuantizer):
    """
    Quantizer for performing QAT on a model, where embedding layers have
    int4 fake quantized grouped per channel weights.
    """

    def __init__(
        self,
        group_size: int = 256,
        scale_precision: torch.dtype = torch.float32,
        zero_point_precision: torch.dtype = torch.int32,
    ) -> None:
        super().__init__()
        self.bit_width = 4
        self.group_size: int = group_size
        self.scale_precision: torch.dtype = scale_precision
        self.zero_point_precision: torch.dtype = zero_point_precision

    def prepare(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
    ) -> torch.nn.Module:
        """
        Swap `nn.Embedding` modules with `Int4WeightOnlyQATEmbedding`.
        """
        def filter_fn(child: torch.nn.Module, cur_fqn:str) -> bool:
            return isinstance(child, torch.nn.Embedding)

        def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
            new_embedding = Int4WeightOnlyQATEmbedding(
                # nn.Embedding args
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                # quantization args
                group_size=self.group_size,
                scale_precision=self.scale_precision,
                zero_point_precision=self.zero_point_precision,
                device=child.weight.device,
            )
            # In distributed training, the model may be instantiated
            # on the meta device, in which case there is no need to
            # copy the weights, and doing so will result in an error
            if child.weight.device != torch.device("meta"):
                new_embedding.weight = child.weight
            return new_embedding

        _replace_with_custom_fn_if_matches_filter(model, replacement_fn, filter_fn)
        return model

    def convert(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
    ) -> torch.nn.Module:
        """
        Swap all `Int4WeightOnlyQATEmbedding` modules with `Int4WeightOnlyEmbedding`.
        """
        self._convert_helper(model)
        return model

    def _convert_helper(self, module: torch.nn.Module):
        """
        Helper function to recursively swap `Int4WeightOnlyQATEmbedding`
        modules with `Int4WeightOnlyEmbedding`
        """
        from torchao._executorch_ops import _quantized_decomposed_quantize_per_channel_group_wrapper
        for name, child in module.named_children():
            if isinstance(child, Int4WeightOnlyQATEmbedding):
                group_size = child.weight_fake_quantizer.config.group_size
                scale_precision = child.weight_fake_quantizer.config.scale_precision
                zero_point_precision = child.weight_fake_quantizer.config.zero_point_precision
                quantized_embedding = Int4WeightOnlyEmbedding(
                    # nn.Embedding args
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    padding_idx=child.padding_idx,
                    max_norm=child.max_norm,
                    norm_type=child.norm_type,
                    scale_grad_by_freq=child.scale_grad_by_freq,
                    sparse=child.sparse,
                    # quantization args
                    group_size=group_size,
                    scale_precision=scale_precision,
                    zero_point_precision=zero_point_precision,
                    device=child.weight.device,
                )
                setattr(module, name, quantized_embedding)

                # Load weights and qparams into quantized embedding
                (qmin, qmax) = _get_qmin_qmax(self.bit_width)
                (s, zp) = get_group_qparams_symmetric(child.weight, self.bit_width, group_size)
                q_weight = _quantized_decomposed_quantize_per_channel_group_wrapper(
                    child.weight, s, zp, qmin, qmax, torch.int8, group_size,
                )
                quantized_embedding.weight = q_weight
                quantized_embedding.scales = s
                quantized_embedding.zeros = zp
            else:
                self._convert_helper(child)


class Int4WeightOnlyQATEmbedding(FakeQuantizedEmbedding):
    """
    This module implements a embedding layer with int4 fake quantized
    grouped per channel weights.

    args:
        group_size: the number of elements in each quantized group for weights
        scale_precision: precision of per group scales
        zero_point_precision: precision of per group zero points
    """

    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        group_size: int = 32, 
        scale_precision: torch.dtype = torch.float32,
        zero_point_precision: torch.dtype = torch.int32,
        *args, 
        **kwargs,
    ):
        weight_config = FakeQuantizeConfig(
            dtype=TorchAODType.INT4,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=True,
            scale_precision=scale_precision,
            zero_point_precision=zero_point_precision,
        )
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight_config,
            *args,
            **kwargs,
        )

    def enable_fake_quant(self, enabled: bool = True):
        self.weight_fake_quantizer.enabled = enabled

    def disable_fake_quant(self):
        self.enable_fake_quant(False)


class Int4WeightOnlyEmbedding(torch.nn.Module):
    """
    This module implements a embedding layer with int4 quantized
    grouped per channel weights.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        group_size: int = 32,
        scale_precision: torch.dtype = torch.float32,
        zero_point_precision: torch.dtype = torch.int32,
        device: torch.device = None,
    ):
        super().__init__()

        # nn.Embedding args
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # quantization args
        self.bit_width = 4
        self.group_size = group_size
        self.scale_precision = scale_precision
        self.zero_point_precision = zero_point_precision

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.empty((num_embeddings, embedding_dim), dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "scale",
            torch.empty(
                (num_embeddings, embedding_dim // group_size),
                dtype=scale_precision,
                device=device,
            ),
        )
        self.register_buffer(
            "zero_point",
            torch.empty(
                (num_embeddings, embedding_dim // group_size),
                dtype=zero_point_precision,
                device=device,
            ),
        )

    def forward(self, x):
        from torchao._executorch_ops import _quantized_decomposed_dequantize_per_channel_group_wrapper
        qmin, qmax = _get_qmin_qmax(self.bit_width)
        w_dq = _quantized_decomposed_dequantize_per_channel_group_wrapper(
            self.weight,
            self.scale,
            self.zero_point,
            qmin,
            qmax,
            torch.int8,
            self.group_size,
            x.dtype,
        )
        return F.embedding(
            x, w_dq, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )
