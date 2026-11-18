# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Compression ratio and overhead for sparse Squared-ReLU activations.

Answers pytorch/ao#1920: how small do these activations get, and what does
the pack/unpack cost on a training step?

    python benchmarks/prototype/activation_compression/benchmark_activation_compression.py
    python benchmarks/prototype/activation_compression/benchmark_activation_compression.py \
        --device cuda --shapes 8192x11008 --sparsities 0.85 0.95
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from torchao.prototype.activation_compression import (
    activation_sparsity,
    compress_saved_activations,
    measure_compression,
    synthesize_sparse_squared_relu,
)
from torchao.prototype.activation_compression.codec import _sync


class SquaredReLUFFN(nn.Module):
    """FFN whose pre-activation bias targets ``sparsity`` under unit input std.

    ``up.weight`` is initialized at std ``1/sqrt(dim)``, so ``x @ W`` has std
    about 1 when ``x`` is standard normal. The bias is the normal quantile
    that makes ReLU drop ``sparsity`` of the units. That is the trained
    Squared-ReLU regime from pytorch/ao#1920, without running the training
    recipe.
    """

    def __init__(self, dim: int, hidden: int, sparsity: float = 0.85):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim, bias=False)
        nn.init.normal_(self.up.weight, std=dim**-0.5)
        nn.init.normal_(self.down.weight, std=hidden**-0.5)
        shift = torch.special.ndtri(torch.tensor(sparsity)).item()
        self.up.bias.data.fill_(-shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)).square())


def _parse_shape(text: str) -> tuple[int, ...]:
    parts = tuple(int(p) for p in text.lower().replace("*", "x").split("x"))
    if not parts or any(p <= 0 for p in parts):
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}")
    return parts


def _dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"unknown dtype {name!r}") from exc
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise argparse.ArgumentTypeError(f"{name!r} is not a floating dtype")
    return dtype


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):8.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):8.1f} KiB"
    return f"{n:8d} B"


def _print_codec_table(args: argparse.Namespace) -> None:
    header = (
        f"{'method':<8} {'dtype':<10} {'shape':<16} {'sparsity':>8} "
        f"{'dense':>12} {'packed':>12} {'ratio':>7} "
        f"{'pack ms':>8} {'unpack ms':>9} {'err':>8}"
    )
    print(header)
    print("-" * len(header))
    for shape in args.shapes:
        for sparsity in args.sparsities:
            for dtype in args.dtypes:
                activation = synthesize_sparse_squared_relu(
                    shape,
                    sparsity,
                    dtype=dtype,
                    device=args.device,
                    generator=torch.Generator(device=args.device).manual_seed(0),
                )
                for method in args.methods:
                    measured = measure_compression(
                        activation,
                        method,
                        warmup=args.warmup,
                        iters=args.iters,
                        zlib_level=args.zlib_level,
                    )
                    print(
                        f"{measured.method:<8} {measured.dtype:<10} "
                        f"{'x'.join(map(str, measured.shape)):<16} "
                        f"{measured.sparsity:8.3f} "
                        f"{_fmt_bytes(measured.dense_bytes)} "
                        f"{_fmt_bytes(measured.compressed_bytes)} "
                        f"{measured.ratio:7.2f} "
                        f"{measured.compress_ms:8.3f} "
                        f"{measured.decompress_ms:9.3f} "
                        f"{measured.max_abs_error:8.1e}"
                    )


def _time_step(fn, iters: int, device: torch.device) -> float:
    for _ in range(2):
        fn()
    _sync(device)
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start is not None:
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / iters
    import time

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1e3 / iters


def _print_training_step(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = args.dtypes[0]
    torch.manual_seed(0)
    model = SquaredReLUFFN(args.dim, args.hidden, sparsity=args.ffn_sparsity).to(
        device=device, dtype=dtype
    )
    x = torch.randn(args.batch, args.dim, device=device, dtype=dtype)
    with torch.no_grad():
        sparse = torch.relu(model.up(x)).square()
    print(
        f"\nSquared-ReLU FFN  batch={args.batch} dim={args.dim} "
        f"hidden={args.hidden} dtype={dtype} device={device}"
    )
    print(f"activation sparsity {activation_sparsity(sparse):.3f}")

    def dense_step():
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()

    dense_ms = _time_step(dense_step, args.iters, device)
    print(f"dense step     {dense_ms:8.3f} ms")

    for method in args.methods:

        def compressed_step(method=method):
            model.zero_grad(set_to_none=True)
            with compress_saved_activations(
                method=method,
                min_numel=1,
                min_sparsity=0.5,
                zlib_level=args.zlib_level,
            ) as stats:
                loss = model(x).square().mean()
            loss.backward()
            compressed_step.stats = stats

        compressed_ms = _time_step(compressed_step, args.iters, device)
        stats = compressed_step.stats
        overhead = (compressed_ms - dense_ms) / dense_ms * 100 if dense_ms else 0.0
        print(
            f"{method:<8} step {compressed_ms:8.3f} ms  "
            f"overhead {overhead:+6.1f}%  "
            f"saved tensors {stats.n_compressed}/{stats.n_saved} packed  "
            f"stored {_fmt_bytes(stats.stored_bytes).strip()} / "
            f"dense {_fmt_bytes(stats.dense_bytes).strip()}  "
            f"({stats.ratio:.2f}x)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=_parse_shape,
        default=[(1024, 4096)],
        help="Activation shapes, e.g. 1024x4096 8192x11008",
    )
    parser.add_argument(
        "--sparsities",
        nargs="+",
        type=float,
        default=[0.85, 0.95],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=_dtype,
        default=[torch.float32, torch.bfloat16],
    )
    parser.add_argument("--methods", nargs="+", default=["bitpack", "zlib"])
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ffn-sparsity", type=float, default=0.85)
    parser.add_argument(
        "--skip-step",
        action="store_true",
        help="Only run the activation microbenchmark",
    )
    args = parser.parse_args()
    _print_codec_table(args)
    if not args.skip_step:
        _print_training_step(args)


if __name__ == "__main__":
    main()
