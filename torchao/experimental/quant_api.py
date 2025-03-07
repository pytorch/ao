# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.ao.quantization.fx._decomposed import (
    dequantize_per_channel_group,
    quantize_per_channel_group,
)

from torchao.quantization.granularity import PerGroup, PerRow
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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


class _Int8DynActIntxWeightQuantizedLinearNative(nn.Module):
    def __init__(
        self,
        pack_weight_op,
        linear_op,
    ):
        super().__init__()
        self._pack_weights_op = pack_weight_op
        self._linear_op = linear_op

    def quantize_and_pack_weights(self, weights, nbit, group_size, has_weight_zeros):
        self.nbit = nbit
        self.group_size = group_size
        self.has_weight_zeros = has_weight_zeros

        n, k = weights.shape

        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        # AOTI does not allow a tensor of size (n, 0), so we do (0, n)
        self._n = torch.empty(0, n, dtype=torch.int8)
        self._k = torch.empty(0, k, dtype=torch.int8)
        self._group_size = torch.empty(0, group_size, dtype=torch.int8)

        weight_qvals, weight_scales, weight_zeros = _quantize(
            weights, self.group_size, self.nbit, self.has_weight_zeros
        )
        if self.has_weight_zeros:
            self.packed_weights = self._pack_weights_op(
                weight_qvals,
                weight_scales.reshape(-1),
                weight_zeros.to(torch.int8).reshape(-1),
                self._group_size,
            )
        else:
            self.packed_weights = self._pack_weights_op(
                weight_qvals, weight_scales.reshape(-1), self._group_size
            )

    def forward(self, x):
        assert x.dim() >= 2
        if x.dim() == 2:
            return self._linear_op(
                x,
                self.packed_weights,
                self._group_size,
                self._n,
                self._k,
            )

        assert x.dim() >= 3
        lead_shape = x.shape[0:-2]
        m, k = x.shape[-2], x.shape[-1]
        n = self._n.shape[1]
        res = self._linear_op(
            x.reshape(-1, k), self.packed_weights, self._group_size, self._n, self._k
        )
        res = res.reshape(*lead_shape, m, n)
        return res


# Python-based reference implementation of Int8DynActLowbitWeightQuantizedLinear
# It is arithmetically equivalent to Int8DynActLowbitWeightQuantizedLinear
# This is used to test Int8DynActLowbitWeightQuantizedLinear, and as a fallback when
# Int8DynActLowbitWeightQuantizedLinear is not available
class _Int8DynActIntxWeightQuantizedLinearFallback(nn.Module):
    def __init__(self):
        super().__init__()

    def quantize_and_pack_weights(self, weights, nbit, group_size, has_weight_zeros):
        self.nbit = nbit
        self.group_size = group_size
        self.has_weight_zeros = has_weight_zeros

        self._n, self._k = weights.shape
        assert self._k % group_size == 0, "group_size must divide k"

        self.weight_qvals, self.weight_scales, self.weight_zeros = _quantize(
            weights, self.group_size, self.nbit, self.has_weight_zeros
        )

    def _forward_2d(self, x):
        assert x.dim() == 2

        _, k = self._n, self._k
        m, k_ = x.shape
        assert k_ == k

        weights_dequantized = dequantize_per_channel_group(
            w_int8=self.weight_qvals,
            scales=self.weight_scales,
            zero_points=(
                self.weight_zeros
                if self.has_weight_zeros
                else torch.zeros_like(self.weight_scales)
            ),
            quant_min=None,  # TODO: why is this an arg for this function
            quant_max=None,  # TODO: why is this an arg for this function
            dtype=None,  # TODO: why is this an arg for this function
            group_size=self.group_size,
            output_dtype=torch.float32,
        )

        activation_qvals, activation_scales, activation_zeros = _quantize(
            x, group_size=k, nbit=8, has_weight_zeros=True
        )
        activations_dequantized = dequantize_per_channel_group(
            w_int8=activation_qvals,
            scales=activation_scales,
            zero_points=activation_zeros,
            quant_min=None,  # TODO: why is this an arg for this function
            quant_max=None,  # TODO: why is this an arg for this function
            dtype=None,  # TODO: why is this an arg for this function
            group_size=k,
            output_dtype=torch.float32,
        )

        res = torch.matmul(activations_dequantized, weights_dequantized.transpose(1, 0))
        return res

    def forward(self, x):
        assert x.dim() >= 2
        if x.dim() == 2:
            return self._forward_2d(x)

        assert x.dim() >= 3
        lead_shape = x.shape[0:-2]
        m, k = x.shape[-2], x.shape[-1]
        n = self._n
        x = x.reshape(-1, m, k)

        res = [self._forward_2d(x[i, :, :]) for i in range(x.shape[0])]
        res = torch.stack(res)
        res = res.reshape(*lead_shape, m, n)
        return res


def _maybe_get_quantized_linear_native(nbit, has_weight_zeros):
    try:
        if nbit in [1, 2, 3, 4, 5, 6, 7, 8]:
            wzp_suffix = "" if has_weight_zeros else "0zp"
            return _Int8DynActIntxWeightQuantizedLinearNative(
                pack_weight_op=getattr(
                    torch.ops.torchao,
                    f"_pack_8bit_act_{nbit}bit{wzp_suffix}_weight",
                ),
                linear_op=getattr(
                    torch.ops.torchao, f"_linear_8bit_act_{nbit}bit{wzp_suffix}_weight"
                ),
            )
        else:
            logger.warning(
                f"_Int8DynActIntxWeightQuantizedLinearNative does not support: nbit={nbit}, has_weight_zeros={has_weight_zeros}."
            )
    except Exception as e:
        logger.warning(
            f"_Int8DynActIntxWeightQuantizedLinearNative raised an exception during initialization: {e}"
        )

    logger.warning(
        "Falling back to **slow** implementation _Int8DynActIntxWeightQuantizedLinearFallback."
    )
    return _Int8DynActIntxWeightQuantizedLinearFallback()


def _replace_linear_with_quantized_linear(module: nn.Module, kwargs={}):
    group_size = kwargs["group_size"]
    nbit = kwargs["nbit"]
    has_weight_zeros = kwargs["has_weight_zeros"]

    assert not isinstance(module, nn.Linear)
    assert nbit >= 1 and nbit <= 8

    for name, child in module.named_children():
        if not isinstance(child, nn.Linear):
            _replace_linear_with_quantized_linear(child, kwargs)
        else:
            assert child.bias is None
            qlinear = _maybe_get_quantized_linear_native(
                nbit=nbit, has_weight_zeros=has_weight_zeros
            )
            try:
                # The packing function may raise some error from the C++ layer (e.g., if group_size is unsupported)
                # so calling quantize_and_pack_weights can fail.  In this case, we still switch to fallback
                # implementation
                setattr(module, name, qlinear)
                getattr(module, name).quantize_and_pack_weights(
                    child.weight, nbit, group_size, has_weight_zeros
                )
            except Exception as e:
                if not isinstance(qlinear, _Int8DynActIntxWeightQuantizedLinearNative):
                    raise e
                logger.warning(
                    f"_Int8DynActIntxWeightQuantizedLinearNative raised an exception during quantize_and_pack_weights: {e}\n"
                    + "Falling back to **slow** implementation _Int8DynActIntxWeightQuantizedLinearFallback."
                )
                qlinear = _Int8DynActIntxWeightQuantizedLinearFallback()
                setattr(module, name, qlinear)
                getattr(module, name).quantize_and_pack_weights(
                    child.weight, nbit, group_size, has_weight_zeros
                )


class Int8DynActIntxWeightLinearQuantizer:
    def __init__(
        self,
        device,
        precision,
        *,
        bitwidth: Optional[int] = None,
        groupsize: Optional[int] = None,
        has_weight_zeros: Optional[bool] = None,
    ):
        if device != "cpu":
            raise NotImplementedError(
                "Only device=cpu is currently supported in Int8DynActIntxWeightLinearQuantizer"
            )
        else:
            self.device = device

        if precision != torch.float32:
            raise NotImplementedError(
                "Only precision=torch.float32 is currently supported in Int8DynActIntxWeightLinearQuantizer"
            )
        else:
            self.precision = precision

        if bitwidth is None:
            self.bitwidth = 4
            logger.warning(f"bitwidth not specified, defaulting to {self.bitwidth}.")
        else:
            self.bitwidth = bitwidth

        if groupsize is None:
            self.groupsize = 128
            logger.warning(f"groupsize not specified, defaulting to {self.groupsize}.")
        else:
            self.groupsize = groupsize

        if has_weight_zeros is None:
            self.has_weight_zeros = False
            logger.warning(
                f"has_weight_zeros not specified, defaulting to {self.has_weight_zeros}."
            )
        else:
            self.has_weight_zeros = has_weight_zeros

    def quantize(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device).to(self.precision)
        _replace_linear_with_quantized_linear(
            model,
            kwargs={
                "group_size": self.groupsize,
                "nbit": self.bitwidth,
                "has_weight_zeros": self.has_weight_zeros,
            },
        )
        return model


class _IntxWeightQuantizedEmbedding(nn.Module):
    def __init__(
        self,
        nbit,
        pack_weights_op,
        embedding_op,
    ):
        super().__init__()
        self.nbit = nbit
        self._pack_weights_op = pack_weights_op
        self._embedding_op = embedding_op

    def quantize_and_pack_weights(self, weights, group_size):
        self.group_size = group_size
        num_embeddings, embedding_dim = weights.shape

        weight_qvals, weight_scales, weight_zeros = _quantize(
            weights, self.group_size, self.nbit, has_weight_zeros=True
        )
        self.packed_weight_qvals = self._pack_weights_op(weight_qvals.to(torch.int8))
        self.num_embeddings = torch.empty(0, num_embeddings, dtype=torch.int8)
        self.embedding_dim = torch.empty(0, embedding_dim, dtype=torch.int8)
        self.weight_scales = weight_scales
        self.weight_zeros = weight_zeros.to(torch.int8)

    def forward(self, x):
        shape = x.shape
        return self._embedding_op(
            self.packed_weight_qvals,
            self.num_embeddings,
            self.embedding_dim,
            self.weight_scales,
            self.weight_zeros,
            x.reshape(-1),
        ).reshape(*shape, -1)


class _IntxWeightQuantizedEmbeddingFallback(nn.Module):
    def __init__(
        self,
        nbit,
    ):
        super().__init__()
        self.nbit = nbit

    def quantize_and_pack_weights(self, weights, group_size):
        self.group_size = group_size
        num_embeddings, embedding_dim = weights.shape

        weight_qvals, weight_scales, weight_zeros = _quantize(
            weights, self.group_size, self.nbit, has_weight_zeros=True
        )
        self.weight_qvals = weight_qvals.to(torch.int32)
        self.weight_scales = weight_scales
        self.weight_zeros = weight_zeros.to(torch.int32)

    def forward(self, x):
        shape = x.shape
        res = []
        for i in x:
            res.append(
                dequantize_per_channel_group(
                    w_int8=self.weight_qvals[i, :].reshape(1, -1),
                    scales=self.weight_scales[i, :].reshape(1, -1),
                    zero_points=self.weight_zeros[i, :].reshape(1, -1),
                    quant_min=None,  # TODO: why is this an arg for this function
                    quant_max=None,  # TODO: why is this an arg for this function
                    dtype=None,  # TODO: why is this an arg for this function
                    group_size=self.group_size,
                    output_dtype=torch.float32,
                ).reshape(-1)
            )
        return torch.stack(res).reshape(*shape, -1)


def _replace_embedding_with_quantized_embedding(module: nn.Module, kwargs={}):
    group_size = kwargs["group_size"]
    nbit = kwargs["nbit"]

    assert not isinstance(module, nn.Embedding)
    assert nbit >= 1 and nbit <= 8

    for name, child in module.named_children():
        if not isinstance(child, nn.Embedding):
            _replace_embedding_with_quantized_embedding(child, kwargs)
        else:
            try:
                qembedding = _IntxWeightQuantizedEmbedding(
                    nbit,
                    getattr(torch.ops.torchao, f"_pack_embedding_{nbit}bit"),
                    getattr(torch.ops.torchao, f"_embedding_{nbit}bit"),
                )
                setattr(module, name, qembedding)
                getattr(module, name).quantize_and_pack_weights(
                    child.weight, group_size
                )
            except Exception as e:
                logger.warning(
                    f"_IntxWeightQuantizedEmbedding raised an exception during quantize_and_pack_weights: {e}\n"
                    + "Falling back to **slow** implementation _IntxWeightQuantizedEmbeddingFallback."
                )
                qembedding = _IntxWeightQuantizedEmbeddingFallback(nbit)
                setattr(module, name, qembedding)
                getattr(module, name).quantize_and_pack_weights(
                    child.weight, group_size
                )


class IntxWeightEmbeddingQuantizer:
    def __init__(
        self,
        device,
        precision,
        *,
        bitwidth: Optional[int] = None,
        groupsize: Optional[int] = None,
    ):
        if device != "cpu":
            raise NotImplementedError(
                "Only device=cpu is currently supported in IntxWeightEmbeddingQuantizer"
            )
        else:
            self.device = device

        if precision != torch.float32:
            raise NotImplementedError(
                "Only precision=torch.float32 is currently supported in IntxWeightEmbeddingQuantizer"
            )
        else:
            self.precision = precision

        if bitwidth is None:
            self.bitwidth = 4
            logger.warning(f"bitwidth not specified, defaulting to {self.bitwidth}.")
        else:
            self.bitwidth = bitwidth

        if groupsize is None:
            self.groupsize = 128
            logger.warning(f"groupsize not specified, defaulting to {self.groupsize}.")
        else:
            self.groupsize = groupsize

    def quantize(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device).to(self.precision)
        _replace_embedding_with_quantized_embedding(
            model,
            kwargs={
                "group_size": self.groupsize,
                "nbit": self.bitwidth,
            },
        )
        return model


from dataclasses import dataclass

from torchao.core.config import AOBaseConfig
from torchao.dtypes.utils import Layout
from torchao.experimental.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    PackedLinearInt8DynamicActivationIntxWeightLayout,
    Target,
    to_affine_quantized_intx_experimental,
)
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.quant_api import (
    MappingType,
    to_affine_quantized_intx,
    ZeroPointDomain,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.quantization.utils import _get_per_token_block_size


@dataclass
class Int8DynamicActivationIntxWeightConfig(AOBaseConfig):
    """
    Configuration for dynamically quantizing activations with 8-bits and quantizing weights with a low-bit value.
    More specifically, activations are dynamically quantized to 8-bits in a channelwise manner with scales and zeros.
    Weights are quantized with scales and optionally zeros (controlled by has_weight_zeros) in a groupwise or channelwise
    manner using the number of bits specified by weight_dtype.

    args:
        weight_dtype: The dtype to use for weight quantization.  Must be torch.intx, where 1 <= x <= 8.
        granularity: The granularity to use for weight quantization.  Must be PerGroup or PerRow.
        has_weight_zeros: Whether or not to include zeros in the weight quantization.
        weight_mapping_type: The type of mapping to use for the weight quantization.  Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.
        act_mapping_type: The type of mapping to use for the activation quantization.  Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.
        layout: The layout to use for the packed weight tensor.  The layout does not affect the quantization numerically and different
            layouts will give similar results.  The following are available layouts:
            - PackedLinearInt8DynamicActivationIntxWeightLayout: This layout is optimized for CPU performance.
            - QDQLayout: This layout is designed for export to ExecuTorch
            - PlainLayout: This layout is a simple python-based layout.  It has low performance, but can be used
                when PackedLinearInt8DynamicActivationIntxWeightLayout is unavailable.
    """

    weight_dtype: torch.dtype = torch.int4
    granularity: Union[PerRow, PerGroup] = PerRow()
    has_weight_zeros: bool = False
    weight_mapping_type: MappingType = MappingType.ASYMMETRIC
    act_mapping_type: MappingType = MappingType.ASYMMETRIC
    layout: Layout = PackedLinearInt8DynamicActivationIntxWeightLayout()


# For BC
int8_dynamic_activation_intx_weight = Int8DynamicActivationIntxWeightConfig


@register_quantize_module_handler(Int8DynamicActivationIntxWeightConfig)
def _int8_dynamic_activation_intx_weight_transform(
    module: torch.nn.Module, config: Int8DynamicActivationIntxWeightConfig
) -> torch.nn.Module:
    weight = module.weight
    bias = module.bias
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    has_weight_zeros = config.has_weight_zeros
    weight_mapping_type = config.weight_mapping_type
    act_mapping_type = config.act_mapping_type
    layout = config.layout

    dtype_to_bit_width = {
        torch.int1: 1,
        torch.int2: 2,
        torch.int3: 3,
        torch.int4: 4,
        torch.int5: 5,
        torch.int6: 6,
        torch.int7: 7,
        torch.int8: 8,
    }
    if weight_dtype not in dtype_to_bit_width:
        raise ValueError(
            f"weight_dtype must be one of {list(dtype_to_bit_width.keys())}, got {weight_dtype}"
        )
    bit_width = dtype_to_bit_width[weight_dtype]

    if isinstance(granularity, PerGroup):
        group_size = granularity.group_size
    elif isinstance(granularity, PerRow):
        group_size = weight.shape[-1]
    else:
        raise ValueError(f"granularity must be PerGroup or PerRow, got {granularity}")

    scale_dtype = torch.float32
    tensor_impl_ctr_kwargs = None
    if isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout):
        # We need to create a new layout object for each module because when
        # granularity is PerRow, the layout objects cannot share the group_size
        layout = PackedLinearInt8DynamicActivationIntxWeightLayout(layout.target)
        layout.set_params(
            bit_width=bit_width,
            group_size=group_size,
            has_weight_zeros=has_weight_zeros,
            has_bias=False,
        )

        assert (
            weight.device == torch.device("cpu")
        ), "PackedLinearInt8DynamicActivationIntxWeightLayout requires weight.device=CPU"
        assert (
            weight.dtype == torch.float32
        ), "PackedLinearInt8DynamicActivationIntxWeightLayout requires weight.dtype=float32"
        assert (
            act_mapping_type == MappingType.ASYMMETRIC
        ), "PackedLinearInt8DynamicActivationIntxWeightLayout requires act_mapping_type=MappingType.ASYMMETRIC"

        tensor_impl_ctr_kwargs = {"bias": bias}

        if layout.target == Target.AUTO:
            # Check kernels are installed/loaded
            try:
                torch.ops.torchao._pack_8bit_act_4bit_weight
            except AttributeError:
                raise Exception(
                    "TorchAO experimental kernels are not loaded.  To install the kernels, run `USE_CPP=1 pip install .` from ao on a machine with an ARM CPU."
                    + " You can also set target to 'aten' if you are using ARM CPU."
                )
        elif layout.target == Target.ATEN:
            # TODO: long term, we want to disfavor this route for using KleidiAI in torchao
            # KleidiAI kernels are accessible via Target.AUTO if torchao is built
            # with TORCHAO_BUILD_KLEIDIAI=1.  The Target.AUTO route has the advantage
            # of it automatially dispatching to different kernel libaries based on the CPU
            # capability and the desired quantization
            assert (
                TORCH_VERSION_AT_LEAST_2_6
            ), "ATEN target requires torch version > 2.6.0"
            assert (
                torch.backends.kleidiai.is_available()
            ), "ATEN target requires torch.backends.kleidiai.is_available()"
            assert weight_dtype == torch.int4, "ATEN target only supports torch.int4"
            assert (
                not has_weight_zeros
            ), "ATEN target only supports has_weight_zeros=False"

            # KleidiAI groupwise kernel requires bfloat16 scale
            # Otherwise it falls back to a reference implementation
            if isinstance(granularity, PerGroup):
                scale_dtype = torch.bfloat16

    quant_min = -(1 << (bit_width - 1))
    quant_max = (1 << (bit_width - 1)) - 1

    weight = to_affine_quantized_intx_experimental(
        input_float=weight,
        mapping_type=weight_mapping_type,
        block_size=(1, group_size),
        target_dtype=torch.int32,
        quant_min=quant_min,
        quant_max=quant_max,
        eps=torch.finfo(torch.float32).eps,
        scale_dtype=scale_dtype,
        zero_point_dtype=torch.int8,
        preserve_zero=has_weight_zeros,
        zero_point_domain=(
            ZeroPointDomain.INT if has_weight_zeros else ZeroPointDomain.NONE
        ),
        _layout=layout,
        use_hqq=False,
        tensor_impl_ctr_kwargs=tensor_impl_ctr_kwargs,
    )

    # Note that PackedLinearInt8DynamicActivationIntxWeightLayout has dynamic activation quantization fused
    # with the kernel and it should not be applied separately
    if not isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout):
        activation_quant_func = lambda x: to_affine_quantized_intx(
            x,
            mapping_type=act_mapping_type,
            block_size=_get_per_token_block_size(x),
            target_dtype=torch.int32,
            quant_min=-128,  # lower bound of int8
            quant_max=127,  # upper bound of int8
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int32,
        )
        weight = to_linear_activation_quantized(weight, activation_quant_func)

    module.weight = torch.nn.Parameter(weight, requires_grad=False)

    # If bias was packed with weights, set bias to None on module
    if (
        isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout)
        and layout.has_bias
    ):
        module.bias = None

    return module


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
        weight_scales = torch.transpose_copy(weight_scales, 1, 0)
        weight_zeros = torch.transpose_copy(weight_zeros, 1, 0)
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
        n = self.weight_scales.shape[1]
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
