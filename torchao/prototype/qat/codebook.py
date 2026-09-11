# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Codebook (lookup table) Quantization-Aware Training (QAT) support.

This module provides QAT support for CoreML-style codebook weight-only
quantization. The fake quantization numerics mirror the post-training
quantization (PTQ) flow in
:class:`~torchao.prototype.quantization.codebook_coreml.CodebookWeightOnlyConfig`.

The codebook (the k-means centroids / lookup table) is expensive to compute and
non-differentiable, so it is not recomputed on every training step. Instead it
is refreshed every ``refresh_interval`` steps; in between, only the cheap
nearest-centroid assignment against the cached codebook is recomputed (this part
is pure torch and runs on the weight's device). On a refresh step the fake
quantization is bit-for-bit identical to PTQ.
"""

from dataclasses import dataclass, field
from typing import List

import torch

from torchao.quantization.qat import FakeQuantizeConfigBase, FakeQuantizerBase
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _SUB_BYTE_UINT_BOUNDS,
)


@dataclass
class CodebookFakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for fake quantizing weights using CoreML-style codebook (lookup table)
    quantization, mirroring the numerics of
    :class:`~torchao.prototype.quantization.codebook_coreml.CodebookWeightOnlyConfig`.

    Args:
        dtype (torch.dtype): the logical dtype for the codes, one of
            [torch.uint1, ..., torch.uint8]. This determines the number of bits
            per code and hence the number of codebook entries (``2 ** nbits``).
        block_size (List[int]): granularity of quantization, specifying the
            dimensions of the tensor blocks that share the same lookup table.
            Must have length equal to the weight's rank (2). A value of ``-1``
            in a dimension means the entire dimension is used. See
            :func:`~torchao.prototype.quantization.codebook_coreml.choose_qparams_and_quantize_codebook_coreml`
            for details.
        refresh_interval (int): how often (in forward steps) to recompute the
            codebook via k-means. On these steps the fake quantization matches
            PTQ exactly. In between, the cached codebook is reused and only the
            (cheap) nearest-centroid assignment is recomputed against the current
            weight. ``refresh_interval=1`` recomputes the codebook every step
            (exact PTQ numerics every step, but expensive). Default is 100.
    """

    dtype: torch.dtype
    block_size: List[int] = field(default_factory=lambda: [-1, 1])
    refresh_interval: int = 100

    def __post_init__(self):
        supported_dtypes = list(_SUB_BYTE_UINT_BOUNDS.keys()) + [torch.uint8]
        if self.dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}', must be one of "
                f"torch.uint1 to torch.uint8"
            )
        if self.refresh_interval < 1:
            raise ValueError(
                f"refresh_interval must be >= 1, got {self.refresh_interval}"
            )


class CodebookFakeQuantizer(FakeQuantizerBase):
    """
    Generic module for applying codebook fake quantization to a weight tensor,
    matching the numerics of
    :class:`~torchao.prototype.quantization.codebook_coreml.CodebookQuantizedTensor`.

    On a refresh step, we compute the codebook (lookup table) and codes using the
    exact same op as the PTQ flow, so the output matches PTQ bit-for-bit. On other
    steps, we reuse the cached codebook and recompute only the nearest-centroid
    assignment against the current weight (pure torch, runs on the weight's
    device). Because codebook selection is not differentiable, we use a
    straight-through estimator so gradients flow to the original weight unchanged.
    """

    def __init__(self, config: CodebookFakeQuantizeConfig):
        super().__init__()
        self.config = config
        # Number of forward steps taken so far, and the cached codebook. The
        # cached codebook is registered as a (non-persistent) buffer so it moves
        # with the module across devices.
        # Number of forward steps taken and refreshes performed so far.
        self._step = 0
        self._num_refreshes = 0
        # The cached codebook is a persistent buffer so it is saved as part of the
        # checkpoint (state_dict) and restored on load. It is created lazily on
        # the first forward, so it starts as None; see _load_from_state_dict for
        # how it is materialized when loading a checkpoint into a fresh module.
        self.register_buffer("_codebook", None, persistent=True)
        torch._C._log_api_usage_once("torchao.quantization.qat.CodebookFakeQuantizer")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # The codebook buffer is created lazily (None until the first forward), so
        # a freshly constructed module has it as None and the default loader would
        # treat the checkpointed codebook as an unexpected key. Materialize it here
        # so the standard load path can copy the checkpointed values in.
        codebook_key = prefix + "_codebook"
        if self._codebook is None and codebook_key in state_dict:
            self._codebook = state_dict[codebook_key].clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Imported here to avoid a hard dependency on coremltools at import time
        from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
            choose_qparams_and_quantize_codebook_coreml,
            dequantize_codebook,
        )

        nbits = _DTYPE_TO_BIT_WIDTH[self.config.dtype]
        block_size = self.config.block_size

        # Refresh (re-run k-means) on the very first forward (no codebook yet) and
        # then every `refresh_interval` steps. The `_step > 0` guard means a
        # codebook restored from a checkpoint is used as-is on the next forward
        # rather than being immediately recomputed (or re-clustered at inference).
        refresh = self._codebook is None or (
            self._step > 0 and self._step % self.config.refresh_interval == 0
        )
        if refresh:
            # Recompute the codebook (and codes) via k-means, exactly as PTQ does.
            codebook, codes = choose_qparams_and_quantize_codebook_coreml(
                w,
                self.config.dtype,
                block_size,
            )
            codebook = codebook.to(w.device)
            codes = codes.to(w.device)
            self._codebook = codebook
            self._num_refreshes += 1
        else:
            # Reuse the cached codebook and only recompute the nearest-centroid
            # assignment against the current weight.
            from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
                assign_codebook_codes,
            )

            codebook = self._codebook.to(w.device)
            codes = assign_codebook_codes(w, codebook, block_size)

        self._step += 1

        # dequantize_codebook indexes the codebook using int32 codes, matching
        # CodebookQuantizedTensor.dequantize.
        codes = codes.to(torch.int32)
        dq = dequantize_codebook(
            codes,
            codebook,
            nbits,
            block_size,
            output_dtype=w.dtype,
        )

        # Straight-through estimator. The parentheses matter: `w - w.detach()` is
        # exactly zero elementwise, so the forward value equals `dq` bit-for-bit
        # while gradients still flow to `w` unchanged.
        return dq.detach() + (w - w.detach())
