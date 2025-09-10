# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from typing import Callable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization.fx._decomposed import (
    quantize_per_channel_group,
)

from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
    _is_kernel_library_loaded,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from torchao.quantization.granularity import Granularity, PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    quantize_,
)
from torchao.quantization.quant_primitives import _DTYPE_TO_BIT_WIDTH


class QuantizedEmbedding(nn.Module):
    def __init__(
        self,
        bit_width,
    ):
        super().__init__()
        self.bit_width = bit_width

    def quantize_and_pack_weights(self, weights, group_size, mapping_type):
        num_embeddings, embedding_dim = weights.shape

        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        embedding.weight = weights
        quantize_(
            embedding,
            IntxWeightOnlyConfig(
                weight_dtype=getattr(torch, f"int{self.bit_width}"),
                granularity=PerGroup(group_size) if group_size > 0 else PerAxis(0),
                mapping_type=mapping_type,
            ),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )

        weight_qvals = embedding.weight.qdata
        weight_scales = embedding.weight.scale
        weight_zeros = embedding.weight.zero_point

        assert weight_zeros is not None
        weight_scales = weight_scales.reshape(num_embeddings, -1)
        weight_zeros = weight_zeros.reshape(num_embeddings, -1).to(torch.int8)
        self.register_buffer(
            "packed_weight_qvals",
            getattr(torch.ops.torchao, f"_pack_embedding_{self.bit_width}bit")(
                weight_qvals.to(torch.int8)
            ),
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight_scales", weight_scales)
        self.register_buffer("weight_zeros", weight_zeros)

    def forward(self, x):
        shape = x.shape
        return getattr(torch.ops.torchao, f"_embedding_{self.bit_width}bit")(
            self.packed_weight_qvals,
            self.num_embeddings,
            self.embedding_dim,
            self.weight_scales,
            # embedding op requires weight_zeros be passed, even if they are all 0
            self.weight_zeros,
            x.reshape(-1),
        ).reshape(*shape, -1)


class QuantizedEmbeddingFallback(nn.Module):
    def __init__(
        self,
        bit_width,
    ):
        super().__init__()
        self.bit_width = bit_width

    def quantize_and_pack_weights(self, weights, group_size, mapping_type):
        self.embedding = torch.nn.Embedding(*weights.shape)
        self.embedding.weight = weights
        quantize_(
            self.embedding,
            IntxWeightOnlyConfig(
                weight_dtype=getattr(torch, f"int{self.bit_width}"),
                granularity=PerGroup(group_size) if group_size > 0 else PerAxis(0),
                mapping_type=mapping_type,
            ),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )

    def forward(self, x):
        return self.embedding(x)


class QuantizedSharedEmbedding(nn.Module):
    def __init__(self, bit_width, unembedding_packed_weights, group_size, n, k):
        super().__init__()
        self.bit_width = bit_width
        self.register_buffer("unembedding_packed_weights", unembedding_packed_weights)
        self.n = n
        self.k = k
        if group_size == -1:
            self.group_size = k
        else:
            self.group_size = group_size
        self.shared_embedding_op = getattr(
            torch.ops.torchao, f"_shared_embedding_{bit_width}bit"
        )

    def forward(self, x):
        shape = x.shape
        return self.shared_embedding_op(
            self.unembedding_packed_weights,
            self.group_size,
            self.n,
            self.k,
            x.reshape(-1),
        ).reshape(*shape, -1)


def _replace_embedding_with_quantized_embedding(
    module: nn.Module,
    kwargs={},
    fqn: str = "",
):
    group_size = kwargs.get("group_size", None)
    bit_width = kwargs.get("bit_width", None)
    use_fallback = kwargs.get("use_fallback", None)
    mapping_type = kwargs.get("mapping_type", None)

    assert not isinstance(module, nn.Embedding)
    for name, child in module.named_children():
        child_fqn = f"{fqn}.{name}" if fqn != "" else name

        if not isinstance(child, nn.Embedding):
            _replace_embedding_with_quantized_embedding(child, kwargs, child_fqn)
        else:
            assert child.weight.device == torch.device("cpu"), "Only CPU is supported"
            assert child.weight.dtype == torch.float32, "Only float32 is supported"

            if use_fallback:
                qembedding = QuantizedEmbeddingFallback(bit_width)
                setattr(module, name, qembedding)
                getattr(module, name).quantize_and_pack_weights(
                    child.weight,
                    group_size,
                    mapping_type,
                )
            else:
                assert _is_kernel_library_loaded(), (
                    "torchao kernel library is not loaded"
                )
                qembedding = QuantizedEmbedding(bit_width)
                setattr(module, name, qembedding)
                getattr(module, name).quantize_and_pack_weights(
                    child.weight,
                    group_size,
                    mapping_type,
                )


class EmbeddingQuantizer:
    def __init__(
        self,
        weight_dtype: torch.dtype = torch.int4,
        granularity: Granularity = PerAxis(0),
        mapping_type: MappingType = MappingType.ASYMMETRIC,
        use_fallback: bool = False,
    ):
        assert weight_dtype in [getattr(torch, f"int{i}") for i in range(1, 9)]
        bit_width = _DTYPE_TO_BIT_WIDTH[weight_dtype]

        if isinstance(granularity, PerGroup):
            group_size = granularity.group_size
        elif isinstance(granularity, PerAxis):
            assert granularity.axis == 0
            group_size = -1
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

        self.bit_width = bit_width
        self.group_size = group_size
        self.use_fallback = use_fallback
        self.mapping_type = mapping_type

    def quantize(self, model: nn.Module) -> nn.Module:
        _replace_embedding_with_quantized_embedding(
            model,
            kwargs={
                "group_size": self.group_size,
                "bit_width": self.bit_width,
                "use_fallback": self.use_fallback,
                "mapping_type": self.mapping_type,
            },
        )
        return model


def _get_fqns_with_filter(
    module: nn.Module,
    filter_fn: Callable[Tuple[str, nn.Module], bool],
    fqn: str,
    fqns: List[str],
):
    for name, child in module.named_children():
        child_fqn = f"{fqn}.{name}" if fqn != "" else name
        if filter_fn(child, child_fqn):
            fqns.append(child_fqn)
        else:
            _get_fqns_with_filter(child, filter_fn, child_fqn, fqns)


def get_fqns_with_filter(
    module: nn.Module, filter_fn: Callable[Tuple[str, nn.Module], bool]
) -> List[str]:
    fqns = []
    _get_fqns_with_filter(module, filter_fn, "", fqns)
    return fqns


class QuantizedLinear(nn.Module):
    def __init__(self, packed_weight, n, k, group_size, bit_width, bias):
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.n = n
        self.k = k
        self.group_size = group_size
        self.bit_width = bit_width
        self.bias = bias

    def _forward_2d(self, x):
        assert x.dim() == 2
        m, k = x.shape
        assert k == self.k
        return getattr(
            torch.ops.torchao, f"_linear_8bit_act_{self.bit_width}bit_weight"
        )(x, self.packed_weight, self.group_size, self.n, self.k)

    def forward(self, x):
        if x.dim() == 2:
            res = self._forward_2d(x)
        else:
            assert x.dim() >= 3
            lead_shape = x.shape[0:-2]
            m, k = x.shape[-2], x.shape[-1]
            assert k == self.k
            res = self._forward_2d(x.reshape(-1, k))
            res = res.reshape(*lead_shape, m, self.n)

        if self.bias is not None:
            res = res + self.bias
        return res


def get_parent_by_fqn(root: nn.Module, fqn: str):
    parts = fqn.split(".")
    if len(parts) == 1:
        # e.g. "fqn" → parent is root, child is "fqn"
        return root, parts[0]

    parent_fqn = ".".join(parts[:-1])
    child_name = parts[-1]
    parent = dict(root.named_modules()).get(parent_fqn, None)
    if parent is None:
        raise KeyError(f"Parent module {parent_fqn} not found in model")
    return parent, child_name


class SharedEmbeddingQuantizer:
    def __init__(
        self,
        weight_dtype: torch.dtype = torch.int4,
        granularity: Granularity = PerAxis(0),
        mapping_type: MappingType = MappingType.ASYMMETRIC,
    ):
        self.weight_dtype = weight_dtype
        self.granularity = granularity
        self.mapping_type = mapping_type

    def quantize(
        self,
        model: nn.Module,
        embedding_to_unembedding: Optional[Mapping[str, str]] = None,
    ):
        embedding_fqns = get_fqns_with_filter(
            model, lambda m, fqn: isinstance(m, nn.Embedding)
        )
        linear_fqns = get_fqns_with_filter(
            model, lambda m, fqn: isinstance(m, nn.Linear)
        )
        state_dict = model.state_dict()

        # If embedding_to_unembedding is not provided, automatically detect shared embeddings and unembeddings
        if embedding_to_unembedding is None:
            embedding_to_unembedding = {}
            for embedding_fqn in embedding_fqns:
                embedding_w = state_dict[embedding_fqn + ".weight"]
                for linear_fqn in linear_fqns:
                    linear_w = state_dict[linear_fqn + ".weight"]
                    if embedding_w.shape == linear_w.shape and torch.allclose(
                        embedding_w, linear_w
                    ):
                        print(
                            f"Found shared embedding {embedding_fqn} and unembedding {linear_fqn}"
                        )
                        if embedding_fqn not in embedding_to_unembedding:
                            embedding_to_unembedding[embedding_fqn] = linear_fqn
                        else:
                            raise ValueError(
                                f"Found multiple candidate unembeddings ({embedding_to_unembedding[embedding_fqn]}, {linear_fqn}) for embedding {embedding_fqn}.  This is not supported yet.  Please explicitly define the input embedding_to_unembedding."
                            )

        # Construct reverse mapping
        unembedding_to_embedding = {}
        for v, k in embedding_to_unembedding.items():
            if k not in unembedding_to_embedding:
                unembedding_to_embedding[k] = v
            else:
                raise ValueError(
                    f"Found multiple candidate embeddings ({unembedding_to_embedding[k]}, {v}) for unembedding {k}.  This is not supported yet."
                )

        # Check that embeddings are shared, embeddings are embeddings, and unembeddings are linear ops
        for embedding_fqn, unembedding_fqn in embedding_to_unembedding.items():
            assert embedding_fqn in embedding_fqns, (
                f"Embedding {embedding_fqn} is not found in model"
            )
            assert unembedding_fqn in linear_fqns, (
                f"Unembedding {unembedding_fqn} is not found in model"
            )
            assert torch.allclose(
                state_dict[embedding_fqn + ".weight"],
                state_dict[unembedding_fqn + ".weight"],
            ), (
                f"Embedding {embedding_fqn} does not share weights with unembedding {unembedding_fqn}"
            )

        # Quantize unembeddings
        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=self.weight_dtype,
                weight_granularity=self.granularity,
                weight_mapping_type=self.mapping_type,
                # Only universal layout is supported for shared embedding
                intx_packing_format="opaque_torchao_lowbit",
            ),
            filter_fn=lambda m, fqn: isinstance(m, nn.Linear)
            and fqn in list(embedding_to_unembedding.values()),
        )

        embedding_fqn_to_quantized_unembedding = {}
        for fqn, t in model.state_dict().items():
            if (
                fqn.endswith(".weight")
                and fqn[: -len(".weight")] in unembedding_to_embedding
            ):
                embedding_fqn = unembedding_to_embedding[fqn[: -len(".weight")]]
                embedding_fqn_to_quantized_unembedding[embedding_fqn] = t

        for embedding_fqn, unembedding_fqn in embedding_to_unembedding.items():
            weight = embedding_fqn_to_quantized_unembedding[embedding_fqn]
            n, k = weight.shape
            group_size = weight.block_size[1]
            packed_weight = weight.packed_weights
            bit_width = weight.bit_width

            # Set embedding
            parent, child_name = get_parent_by_fqn(model, embedding_fqn)
            child = getattr(parent, child_name)
            assert n == child.num_embeddings, (
                "num_embeddings must match n in shared_unembedding"
            )
            assert k == child.embedding_dim, (
                "embedding_dim must match k in shared_unembedding"
            )
            setattr(
                parent,
                child_name,
                QuantizedSharedEmbedding(
                    bit_width,
                    packed_weight,
                    group_size,
                    n,
                    k,
                ),
            )

            # Set unembedding
            parent, child_name = get_parent_by_fqn(model, unembedding_fqn)
            child = getattr(parent, child_name)
            if weight.packed_weights_has_bias:
                assert child.bias is None
            setattr(
                parent,
                child_name,
                QuantizedLinear(packed_weight, n, k, group_size, bit_width, child.bias),
            )


def _quantize(
    vals: torch.Tensor, group_size: int, nbit: int, has_weight_zeros: bool, signed=True
):
    assert nbit >= 1 and nbit <= 8
    if signed:
        qmin = -(1 << (nbit - 1))
        qmax = (1 << (nbit - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << nbit) - 1

    n, k = vals.shape
    vals = vals.reshape(-1, group_size)
    vmins, _ = torch.min(vals, axis=1)
    vmaxs, _ = torch.max(vals, axis=1)
    group_scales = (vmaxs - vmins) / (qmax - qmin)

    if not has_weight_zeros:
        group_zeros = torch.zeros_like(group_scales)
    else:
        group_zeros = qmin - torch.round(vmins / group_scales)

    vals = vals.reshape(n, k)
    group_scales = group_scales.reshape(n, -1)
    group_zeros = group_zeros.reshape(n, -1)

    group_qvals = quantize_per_channel_group(
        input=vals,
        scales=group_scales,
        zero_points=group_zeros,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.int8 if signed else torch.uint8,
        group_size=group_size,
    )

    if not has_weight_zeros:
        group_zeros = None

    return group_qvals, group_scales, group_zeros


class UIntxWeightOnlyQuantizedLinear(nn.Module):
    def __init__(
        self,
        pack_weight_op,
        linear_op,
    ):
        super().__init__()
        self._pack_weights_op = pack_weight_op
        self._linear_op = linear_op

    def quantize_and_pack_weights(self, weights, nbit, group_size):
        self.nbit = nbit
        self.group_size = group_size

        weight_qvals, weight_scales, weight_zeros = _quantize(
            weights, self.group_size, self.nbit, has_weight_zeros=True, signed=False
        )
        weight_zeros = -weight_zeros * weight_scales
        self.weight_scales = nn.Parameter(weight_scales, requires_grad=False)
        self.weight_zeros = nn.Parameter(weight_zeros, requires_grad=False)
        packed_weights = self._pack_weights_op(weight_qvals.cpu()).to(device="mps")
        self.packed_weights = nn.Parameter(packed_weights, requires_grad=False)

    def forward(self, x):
        assert x.dim() >= 2
        if x.dim() == 2:
            return self._linear_op(
                x,
                self.packed_weights,
                self.group_size,
                self.weight_scales,
                self.weight_zeros,
            )

        lead_shape = x.shape[0:-1]
        k = x.shape[-1]
        n = self.weight_scales.shape[0]
        return self._linear_op(
            x.reshape(-1, k),
            self.packed_weights,
            self.group_size,
            self.weight_scales,
            self.weight_zeros,
        ).reshape(*lead_shape, n)


# TODO(mcandales): Consolidate with _replace_linear_with_quantized_linear
def _replace_linear_with_quantized_linear_mps(module: nn.Module, kwargs={}):
    group_size = kwargs["group_size"]
    nbit = kwargs["nbit"]

    assert not isinstance(module, nn.Linear)
    assert nbit >= 1 and nbit <= 7

    for name, child in module.named_children():
        if not isinstance(child, nn.Linear):
            _replace_linear_with_quantized_linear_mps(child, kwargs)
        else:
            assert child.bias is None
            qlinear = UIntxWeightOnlyQuantizedLinear(
                pack_weight_op=getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit"),
                linear_op=getattr(
                    torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight"
                ),
            )
            setattr(module, name, qlinear)
            qlinear.quantize_and_pack_weights(child.weight, nbit, group_size)


class UIntxWeightOnlyLinearQuantizer:
    def __init__(
        self,
        device,
        precision,
        *,
        bitwidth: Optional[int] = None,
        groupsize: Optional[int] = None,
    ):
        if device != "mps":
            raise NotImplementedError(
                "Only device=mps is currently supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.device = device

        if precision not in [torch.float32, torch.float16, torch.bfloat16]:
            raise ValueError(
                "Only precisions float32, float16 & bfloat16 are supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.precision = precision

        if bitwidth is None:
            bitwidth = 4
            logger.warning(f"bitwidth not specified, defaulting to {bitwidth}.")
        if bitwidth not in range(1, 8):
            raise ValueError(
                "Only bitwidts 1 to 7 are supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.bitwidth = bitwidth

        if groupsize is None:
            groupsize = 128
            logger.warning(f"groupsize not specified, defaulting to {groupsize}.")
        if groupsize not in [32, 64, 128, 256]:
            raise ValueError(
                "Only groupsizes 32, 64, 128 & 256 are supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.groupsize = groupsize

    def quantize(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device).to(self.precision)
        _replace_linear_with_quantized_linear_mps(
            model,
            kwargs={
                "group_size": self.groupsize,
                "nbit": self.bitwidth,
            },
        )
        return model
