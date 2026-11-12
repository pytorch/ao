# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Exact compression of sparse floating-point activations.

Squared-ReLU FFNs produce activations that are mostly exact zeros (often
above 85% sparse; see pytorch/ao#1920). Two codecs are provided:

* ``bitpack`` stores a bit mask plus the nonzero values. It stays on the
  activation's device, is lossless for finite values (``-0`` is restored
  as ``+0``), and is the format to use while training.
* ``zlib`` is a byte-level codec, the same family as nvCOMP's LZ4, zstd,
  and gdeflate. It is a portable way to measure that overhead. Those
  codecs need a host or device round-trip; bitpack does not.

At density ``d`` and element size ``e`` bytes, bitpack uses
``1/8 + d * e`` bytes per element. At 85% sparsity that is about 5.5x for
fp32 and 4.7x for fp16/bf16.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import torch

COMPRESSION_METHODS = ("bitpack", "zlib")

# Elements per compression chunk. A multiple of 8 so concatenated chunk
# masks are a valid single bitstream. Bounds the temporary bool mask.
_BITPACK_CHUNK = 1 << 20
_BIT_WEIGHTS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass
class CompressedActivation:
    """Packed activation. ``payload`` is the only data kept for backward."""

    method: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    numel: int
    nnz: int
    payload: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.payload)


@dataclass
class CompressionMeasurement:
    """One compress / decompress timing on a single tensor."""

    method: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    sparsity: float
    dense_bytes: int
    compressed_bytes: int
    ratio: float
    compress_ms: float
    decompress_ms: float
    max_abs_error: float


def activation_sparsity(x: torch.Tensor) -> float:
    """Fraction of elements that compare equal to zero."""
    n = x.numel()
    if n == 0:
        return 0.0
    return float((x == 0).sum().item()) / n


def synthesize_sparse_squared_relu(
    shape: int | tuple[int, ...],
    sparsity: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Squared-ReLU activation with approximately ``sparsity`` exact zeros.

    Trained Squared-ReLU FFNs land in this regime. A randomly initialized
    network does not (ReLU of a centered pre-activation is ~50% sparse), so
    benchmarks shift the pre-activation to the requested sparsity.
    ``generator``, when set, must live on ``device``.
    """
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    device = torch.device(device)
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    if sparsity == 1.0:
        return torch.zeros(*shape, dtype=dtype, device=device)
    if sparsity == 0.0:
        pre = torch.rand(*shape, device=device, generator=generator) + 0.1
    else:
        # P(Normal(mu, 1) <= 0) = sparsity.
        mu = torch.special.ndtri(torch.tensor(1.0 - sparsity)).item()
        pre = torch.randn(*shape, device=device, generator=generator) + mu
    return torch.relu(pre).square().to(dtype=dtype)


def compress_activation(
    x: torch.Tensor,
    method: str = "bitpack",
    *,
    zlib_level: int = 1,
) -> CompressedActivation:
    """Losslessly pack a real floating-point tensor.

    ``zlib_level`` is zlib's compression level (``-1`` through ``9``) and is
    ignored by ``bitpack``. Level 1 favors speed, which is the right default
    when the packed tensor is unpacked on every backward.
    """
    _check_method(method)
    _check_zlib_level(zlib_level)
    _check_activation(x)
    with torch.no_grad():
        if method == "bitpack":
            return _compress_bitpack(x)
        return _compress_zlib(x, zlib_level)


def decompress_activation(packed: CompressedActivation) -> torch.Tensor:
    """Inverse of :func:`compress_activation`."""
    if not isinstance(packed, CompressedActivation):
        raise TypeError(
            f"expected CompressedActivation, got {type(packed).__name__}"
        )
    with torch.no_grad():
        if packed.method == "bitpack":
            return _decompress_bitpack(packed)
        if packed.method == "zlib":
            return _decompress_zlib(packed)
    raise ValueError(f"unknown compression method {packed.method!r}")


def measure_compression(
    x: torch.Tensor,
    method: str = "bitpack",
    *,
    warmup: int = 5,
    iters: int = 20,
    zlib_level: int = 1,
) -> CompressionMeasurement:
    """Time ``method`` on ``x`` and report the realized compression ratio.

    CUDA and MPS are synchronized so the times include the device work.
    ``ratio`` is dense bytes divided by packed bytes.
    """
    if warmup < 0 or iters < 1:
        raise ValueError(f"warmup must be >= 0 and iters >= 1, got {warmup}, {iters}")
    packed = compress_activation(x, method, zlib_level=zlib_level)
    restored = decompress_activation(packed)
    for _ in range(warmup):
        packed = compress_activation(x, method, zlib_level=zlib_level)
        decompress_activation(packed)
    _sync(x.device)

    start = time.perf_counter()
    for _ in range(iters):
        packed = compress_activation(x, method, zlib_level=zlib_level)
    _sync(x.device)
    compress_ms = (time.perf_counter() - start) * 1e3 / iters

    start = time.perf_counter()
    for _ in range(iters):
        restored = decompress_activation(packed)
    _sync(x.device)
    decompress_ms = (time.perf_counter() - start) * 1e3 / iters

    dense_bytes = x.numel() * x.element_size()
    compressed_bytes = packed.nbytes
    if x.numel() == 0 or torch.equal(restored, x):
        max_abs_error = 0.0
    else:
        max_abs_error = float((restored - x).abs().max().item())
    return CompressionMeasurement(
        method=method,
        shape=tuple(x.shape),
        dtype=str(x.dtype).removeprefix("torch."),
        device=str(x.device),
        numel=x.numel(),
        sparsity=activation_sparsity(x),
        dense_bytes=dense_bytes,
        compressed_bytes=compressed_bytes,
        ratio=(dense_bytes / compressed_bytes) if compressed_bytes else float("inf"),
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
        max_abs_error=max_abs_error,
    )


def _check_method(method: str) -> None:
    if method not in COMPRESSION_METHODS:
        raise ValueError(
            f"unknown compression method {method!r}; expected one of {COMPRESSION_METHODS}"
        )


def _check_zlib_level(level: int) -> None:
    if not -1 <= level <= 9:
        raise ValueError(f"zlib_level must be in [-1, 9], got {level}")


def _check_activation(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected a tensor, got {type(x).__name__}")
    if not x.is_floating_point() or x.is_complex():
        raise TypeError(
            f"activation compression expects a real floating tensor, got {x.dtype}"
        )
    if x.layout is not torch.strided:
        raise ValueError(f"expected a strided tensor, got layout {x.layout}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _compress_bitpack(x: torch.Tensor) -> CompressedActivation:
    flat = x.detach().contiguous().reshape(-1)
    numel = flat.numel()
    weights = torch.tensor(_BIT_WEIGHTS, dtype=torch.int32, device=flat.device)
    masks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    for start in range(0, numel, _BITPACK_CHUNK):
        part = flat[start : start + _BITPACK_CHUNK]
        mask = part != 0
        # Boolean indexing copies, so the packed values do not alias ``x``.
        value_chunks.append(part[mask])
        masks.append(_pack_bits(mask, weights))
    if value_chunks:
        values = torch.cat(value_chunks)
        mask_bits = torch.cat(masks)
    else:
        values = flat.new_empty(0)
        mask_bits = torch.empty(0, dtype=torch.uint8, device=flat.device)
    return CompressedActivation(
        method="bitpack",
        shape=tuple(x.shape),
        dtype=x.dtype,
        device=torch.device(x.device),
        numel=numel,
        nnz=values.numel(),
        payload=(mask_bits, values),
    )


def _pack_bits(mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Pack a 1D bool mask into uint8. Bit 0 of each byte is the earliest element."""
    numel = mask.numel()
    if numel == 0:
        return torch.empty(0, dtype=torch.uint8, device=mask.device)
    remainder = numel % 8
    if remainder:
        mask = torch.nn.functional.pad(mask, (0, 8 - remainder))
    bits = mask.view(-1, 8).to(torch.int32)
    return (bits * weights).sum(dim=1).to(torch.uint8)


def _unpack_bits(packed: torch.Tensor, numel: int) -> torch.Tensor:
    if numel == 0:
        return torch.empty(0, dtype=torch.bool, device=packed.device)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    bits = (packed.to(torch.int32).unsqueeze(1) >> shifts) & 1
    return bits.reshape(-1)[:numel].to(torch.bool)


def _decompress_bitpack(packed: CompressedActivation) -> torch.Tensor:
    mask_bits, values = packed.payload
    mask = _unpack_bits(mask_bits, packed.numel)
    out = torch.zeros(packed.numel, dtype=packed.dtype, device=packed.device)
    if values.numel():
        out[mask] = values
    return out.reshape(packed.shape)


def _cpu_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of ``t``, with no storage offset or padding."""
    t = t.detach()
    if (not t.is_contiguous()) or t.storage_offset() != 0:
        t = t.contiguous().clone() if t.storage_offset() != 0 else t.contiguous()
    t = t.cpu()
    if t.storage_offset() != 0:
        t = t.clone()
    nbytes = t.numel() * t.element_size()
    if nbytes == 0:
        return b""
    raw = bytes(t.untyped_storage())
    if len(raw) < nbytes:
        raise RuntimeError(
            f"storage has {len(raw)} bytes, tensor needs {nbytes}"
        )
    return raw[:nbytes]


def _tensor_from_bytes(
    data: bytes,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    numel = 1
    for size in shape:
        numel *= size
    if numel == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    restored = torch.frombuffer(bytearray(data), dtype=dtype).clone()
    if restored.numel() != numel:
        raise RuntimeError(
            f"decompressed {restored.numel()} elements, expected {numel}"
        )
    restored = restored.reshape(shape)
    if device.type != "cpu":
        restored = restored.to(device)
    return restored


def _compress_zlib(x: torch.Tensor, level: int) -> CompressedActivation:
    cpu = x.detach().contiguous().cpu()
    nnz = int((cpu != 0).sum().item()) if cpu.numel() else 0
    blob = zlib.compress(_cpu_bytes(cpu), level)
    payload = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    return CompressedActivation(
        method="zlib",
        shape=tuple(x.shape),
        dtype=x.dtype,
        device=torch.device(x.device),
        numel=x.numel(),
        nnz=nnz,
        payload=(payload,),
    )


def _decompress_zlib(packed: CompressedActivation) -> torch.Tensor:
    (payload,) = packed.payload
    blob = _cpu_bytes(payload)
    raw = zlib.decompress(blob)
    return _tensor_from_bytes(raw, packed.shape, packed.dtype, packed.device)
