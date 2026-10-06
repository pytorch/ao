# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Workaround for a Triton miscompile that produces spurious NaN/Inf in compiled
float8 backward kernels (e.g. FSDP2 float8 training with torch.compile).

Root cause: PyTorch inductor PR pytorch/pytorch#186933 changed the Triton codegen
for ``minimum``/``maximum`` from the ``tl.where``-based ``triton_helpers.{minimum,
maximum}`` to ``tl.{minimum,maximum}(a, b, tl.PropagateNan.ALL)``. The two forms
are numerically identical (both propagate NaN), but the ``PropagateNan.ALL`` form
lowers to PTX ``min.NaN``/``max.NaN`` instructions that make Triton mis-lower a
neighboring transposed, vectorized 1-byte (fp8) store. The stored value is
correct, but the transposed store writes garbage bytes that decode as NaN/Inf in
fp8. See https://github.com/triton-lang/triton/issues/11111.

This monkeypatch reverts the inductor min/max codegen to the numerically-identical
``tl.where`` form, which does not emit the ``min.NaN``/``max.NaN`` instructions and
so avoids tripping the Triton store bug. It is applied from
``convert_to_float8_training`` so it only affects callers who actually use the
float8 training product, not every ``import torchao``. The ``tl.where`` form is
always correct, so this is a no-op on numerics even once Triton fixes the
underlying bug.

Note: we must emit the ``tl.where`` expression *inline* rather than delegate to
``triton_helpers.{minimum,maximum}``. #186933's follow-up also rewrote those helper
functions to ``tl.{minimum,maximum}(..., propagate_nan=tl.PropagateNan.ALL)``, so
routing through them would still emit ``min.NaN``/``max.NaN`` and fail to work
around the bug (confirmed against torch 2.14.0.dev / triton 3.8.0).
"""

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def _patch_inductor_min_max_codegen() -> None:
    """Revert inductor's Triton min/max codegen to the ``tl.where`` form.

    Idempotent and defensive: if inductor internals have moved, this logs and
    returns rather than breaking ``import torchao.float8``.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from torch._inductor.codegen.triton import TritonOverrides
    except Exception as e:  # pragma: no cover - inductor internals moved
        logger.debug("float8: could not import TritonOverrides to patch: %s", e)
        return

    # Numerically-identical replacements for the post-pytorch/pytorch#186933
    # `tl.{minimum,maximum}(a, b, tl.PropagateNan.ALL)` codegen. `(a != a)`
    # propagates NaN from `a`; NaN in `b` is preserved because the comparison is
    # false, so `b` is selected. `a` and `b` are already-CSE'd operands, so
    # referencing `a` twice is free. This matches the pre-#186933 codegen
    # semantics without routing through the (now also NaN-propagating)
    # `triton_helpers.{minimum,maximum}` helpers.
    @staticmethod
    def minimum(a, b):
        return f"tl.where(({a} < {b}) | ({a} != {a}), {a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"tl.where(({a} > {b}) | ({a} != {a}), {a}, {b})"

    TritonOverrides.minimum = minimum
    TritonOverrides.maximum = maximum
    _PATCHED = True
    logger.debug(
        "float8: patched inductor Triton min/max codegen to work around "
        "https://github.com/triton-lang/triton/issues/11111"
    )
