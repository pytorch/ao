# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Low-overhead launch for FlyDSL @flyc.jit functions.

A bare ``launch(*args)`` pays ~57us of per-call host dispatch every time —
``inspect.Signature.bind`` + ``_resolve_and_make_cache_key`` (Protocol
introspection, DLPack) — dwarfing the ~10us device cost. ``flyc.compile``
(public API, used by aiter's FlyDSL ops) pre-resolves all of that once and
returns a ``CompiledFunction`` whose call does only ctypes-slot updates +
the JIT'd function pointer (~7us). Runtime scalars (M/grid) and the stream
are slots, so one ``CompiledFunction`` serves every M and every stream.

FlyDSL's IR is version-unstable, so a compile failure here must never break
the op: on any error we cache a ``None`` sentinel and fall back to the normal
``launch(*args)`` path for the lifetime of that launch fn.
"""

from __future__ import annotations

_compiled = {}  # launch_fn -> CompiledFunction | None (None = use slow path)


def fast_launch(launch_fn, *args):
    """Call ``launch_fn(*args)`` via a cached ``flyc.compile`` CompiledFunction.

    All args must be positional (``flyc.compile`` requires it) — pass the
    stream positionally, not as ``stream=``.
    """
    if launch_fn not in _compiled:
        try:
            import flydsl.compiler as flyc

            _compiled[launch_fn] = flyc.compile(launch_fn, *args)
        except Exception:
            _compiled[launch_fn] = None  # version skew / unsupported arg
    cf = _compiled[launch_fn]
    return cf(*args) if cf is not None else launch_fn(*args)
