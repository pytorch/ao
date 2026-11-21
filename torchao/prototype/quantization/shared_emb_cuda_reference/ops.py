# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F

from torchao.prototype.quantization.embedding.api import (
    QuantizedLinear,
    QuantizedTiedEmbedding,
)
from torchao.quantization.qat.utils import _fake_quantize_per_token
from torchao.quantization.quant_primitives import MappingType, choose_qparams_affine
from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
    IntxOpaqueTensor,
)

_CUDA_LIBRARY: torch.library.Library | None = None


def _pack_weights(
    weight_qvals: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zeros: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
    target: str | None,
    *,
    weight_nbit: int,
) -> torch.Tensor:
    """Native CPU setup packing of already-quantized CUDA codes and qparams."""
    if target not in (None, "universal"):
        raise NotImplementedError(
            "CUDA reference packing only supports the universal format"
        )
    if weight_qvals.ndim != 2 or not 1 <= weight_nbit <= 8:
        raise ValueError("Expected a 2D INT1–INT8 code tensor")
    n, k = weight_qvals.shape
    if n < 1 or k < 1 or group_size < 16 or group_size % 16 or k % group_size:
        raise ValueError(
            "Universal groups must be multiples of 16 dividing K; per-channel passes the actual K"
        )
    pack = getattr(torch.ops.torchao, f"_pack_8bit_act_{weight_nbit}bit_weight")
    return pack(
        weight_qvals.to("cpu"),
        weight_scales.to("cpu"),
        weight_zeros.to("cpu") if weight_zeros is not None else None,
        group_size,
        bias.to("cpu") if bias is not None else None,
        "universal",
    )


class PreparedCUDALinear(torch.nn.Module):
    """Explicit CUDA QAT activation reference with setup-prepared FP32 weights."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        external_bias: torch.Tensor | None = None,
    ) -> None:
        if (
            weight.ndim != 2
            or weight.device.type != "cuda"
            or weight.dtype != torch.float32
        ):
            raise ValueError("Prepared weights must be a 2D CUDA FP32 tensor")
        for value in (bias, external_bias):
            if value is not None and (
                value.device != weight.device
                or value.dtype != torch.float32
                or value.shape != (weight.shape[0],)
            ):
                raise ValueError(
                    "Prepared biases must be FP32 [n] on the weights device"
                )
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("external_bias", external_bias)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if (
            activations.device.type != "cuda"
            or activations.device != self.weight.device
            or activations.dtype != torch.float32
            or self.weight.dtype != torch.float32
            or activations.ndim < 2
            or activations.shape[-1] != self.weight.shape[1]
        ):
            raise ValueError(
                "activations must be CUDA FP32 [...,k] on the prepared weights device"
            )
        k = self.weight.shape[1]
        flattened = activations.reshape(-1, k)
        scales, zeros = choose_qparams_affine(
            flattened,
            mapping_type=MappingType.ASYMMETRIC,
            # TorchAO annotates a one-axis tuple, but a block has one entry per axis.
            block_size=(1, k),  # pyrefly: ignore[bad-argument-type]
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
        )
        dequantized = _fake_quantize_per_token(flattened, scales, zeros, -128, 127)
        output = F.linear(dequantized, self.weight, self.bias)
        if self.external_bias is not None:
            output = output + self.external_bias
        return output.reshape(*activations.shape[:-1], self.weight.shape[0])


class PreparedCUDAEmbedding(torch.nn.Module):
    """CUDA lookup using the same setup-prepared FP32 table as its tied head."""

    def __init__(self, weight: torch.Tensor) -> None:
        if (
            weight.ndim != 2
            or weight.device.type != "cuda"
            or weight.dtype != torch.float32
        ):
            raise ValueError("Prepared weights must be a 2D CUDA FP32 tensor")
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if (
            indices.device.type != "cuda"
            or indices.device != self.weight.device
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.ndim < 1
            or self.weight.dtype != torch.float32
        ):
            raise ValueError(
                "indices must be CUDA int32/int64 on the prepared weights device"
            )
        return F.embedding(indices, self.weight)


def _prepare_weight(
    packed: torch.Tensor,
    group_size: int,
    n: int,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # The native CPU op owns header validation and universal INT1–INT8 decoding.
    # No original float weight, quantization, or Python byte codec is involved.
    codes, scales, zeros, bias = torch.ops.torchao._unpack_intx_weights(
        packed.detach().to("cpu"), group_size, n, k
    )
    if codes.dtype != torch.int8 or scales.dtype != torch.float32:
        raise RuntimeError("Native unpack must return int8 codes and FP32 scales")
    if codes.shape != (n, k) or scales.shape != (n, k // group_size):
        raise RuntimeError("Native unpack returned inconsistent code/qparam shapes")
    grouped = codes.to(device=device).to(torch.float32).reshape(n, -1, group_size)
    if zeros is not None:
        if zeros.dtype != torch.int8 or zeros.shape != scales.shape:
            raise RuntimeError("Native unpack must return int8 zeros matching scales")
        grouped = grouped - zeros.to(device=device).unsqueeze(-1)
    weight = (grouped * scales.to(device=device).unsqueeze(-1)).reshape(n, k)
    prepared_bias = bias.to(device=device) if bias is not None else None
    return weight, prepared_bias


def convert_opaque_modules_for_cuda_reference(
    model: torch.nn.Module, device: str | torch.device = "cuda"
) -> torch.nn.Module:
    """Unpack once during setup and replace packed modules with explicit CUDA buffers.

    The preparation-local cache is bounded by this model's converted modules.
    Equal packed bytes and geometry share one prepared table, including the tied
    lookup and output. It is discarded after setup; forwards never decode or
    ignore a packed argument, and no global pointer cache is used.
    """
    if _CUDA_LIBRARY is None:
        raise RuntimeError("Register CUDA setup packing before preparing modules")
    torch._C._dispatch_find_schema_or_throw("torchao::_unpack_intx_weights", "")
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "torchao::_unpack_intx_weights", "CPU"
    ):
        raise RuntimeError(
            "Native CPU universal unpack kernel is required during setup"
        )
    destination = torch.device(device)
    if destination.type != "cuda":
        raise ValueError("Prepared reference modules require a CUDA device")
    cache: list[
        tuple[torch.Tensor, int, int, int, torch.Tensor, torch.Tensor | None]
    ] = []
    for name, module in list(model.named_modules()):
        external_bias = None
        if isinstance(module, QuantizedLinear):
            packed = module.packed_weight
            n, k, group_size = module.n, module.k, module.group_size
            external_bias = module.bias
            linear = True
        elif isinstance(module, QuantizedTiedEmbedding):
            packed = module.unembedding_packed_weights
            n, k, group_size = module.n, module.k, module.group_size
            linear = False
        elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            weight = module.weight
            if not isinstance(weight, IntxOpaqueTensor):
                continue
            if not 1 <= weight.bit_width <= 8 or weight.dtype != torch.float32:
                raise NotImplementedError(
                    "Prepared references require INT1–INT8 weights and FP32 outputs"
                )
            packed = weight.packed_weights
            n, k = weight.shape
            group_size = weight.block_size[1]
            linear = isinstance(module, torch.nn.Linear)
            external_bias = module.bias if linear else None
        else:
            continue
        cpu_packed = packed.detach().to("cpu")
        prepared = None
        for previous, previous_group, previous_n, previous_k, value, bias in cache:
            if (group_size, n, k) == (previous_group, previous_n, previous_k) and (
                cpu_packed is previous or torch.equal(cpu_packed, previous)
            ):
                prepared = (value, bias)
                break
        if prepared is None:
            prepared = _prepare_weight(cpu_packed, group_size, n, k, destination)
            cache.append((cpu_packed, group_size, n, k, *prepared))
        value, bias = prepared
        if linear:
            external_bias = (
                external_bias.detach().to(destination)
                if external_bias is not None
                else None
            )
            replacement = PreparedCUDALinear(value, bias, external_bias)
        else:
            replacement = PreparedCUDAEmbedding(value)
        replacement.train(module.training)
        if name:
            model.set_submodule(name, replacement)
        else:
            model = replacement
    return model


def register_cuda_reference_ops() -> None:
    """Enable eight CUDA-to-CPU setup pack bridges; prepared forwards use PyTorch CUDA.

    CPU/default packed forward dispatch is unchanged. There are no CUDA packed
    linear/embedding registrations: native CPU unpack runs once during explicit
    module preparation, then CUDA activation Q/DQ, F.linear and F.embedding use
    the prepared buffers. This is neither an optimized CUDA kernel nor a PTE backend.
    """
    global _CUDA_LIBRARY
    if _CUDA_LIBRARY is not None:
        return
    names = [f"_pack_8bit_act_{nbits}bit_weight" for nbits in range(1, 9)]
    for name in names:
        torch._C._dispatch_find_schema_or_throw(f"torchao::{name}", "")
        if torch._C._dispatch_has_kernel_for_dispatch_key(f"torchao::{name}", "CUDA"):
            raise RuntimeError(f"torchao::{name} already has a CUDA implementation")
    library = torch.library.Library("torchao", "IMPL", "CUDA")
    for nbits, name in enumerate(names, 1):
        library.impl(name, partial(_pack_weights, weight_nbit=nbits))
    _CUDA_LIBRARY = library
