# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for RoPE fusion pattern detection in the pre-grad custom pass.

These tests verify that rope_sdpa_fusion_pass correctly identifies each
RoPE pattern in the FX graph, independent of any GPU kernel or hardware.
"""

import contextlib
import io
import unittest
from functools import partial

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn

from torchao.prototype.attention.shared_utils.custom_ops import (
    register_fp8_attention_ops,
)
from torchao.prototype.attention.shared_utils.fusion_utils import rope_sdpa_fusion_pass


# Register test-only custom ops with dummy implementations.
def _dummy_rope_sdpa(q, k, v, cos, sin, **kwargs):
    B, S, H, D = q.shape
    return torch.zeros(B, H, S, D, dtype=q.dtype, device=q.device)


def _dummy_sdpa(q, k, v, **kwargs):
    return torch.zeros_like(q)


_ops = register_fp8_attention_ops("test_fusion", _dummy_rope_sdpa, _dummy_sdpa)


class PatternANeoXRoPE(nn.Module):
    """Pattern A: NeoX rotate_half RoPE -> transpose -> FP8 SDPA."""

    def forward(self, q, k, v, cos, sin):
        d_half = q.shape[-1] // 2
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        q_rot = torch.cat([-q[..., d_half:], q[..., :d_half]], dim=-1)
        q = q * cos + q_rot * sin

        k_rot = torch.cat([-k[..., d_half:], k[..., :d_half]], dim=-1)
        k = k * cos + k_rot * sin

        return _ops.fp8_sdpa_op(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )


class PatternBHuggingFaceRoPE(nn.Module):
    """Pattern B: transpose -> NeoX RoPE -> FP8 SDPA (HuggingFace-style)."""

    def forward(self, q, k, v, cos, sin):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        d_half = q.shape[-1] // 2
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, S, D] for BHSD layout
        sin = sin.unsqueeze(0).unsqueeze(1)

        q_rot = torch.cat([-q[..., d_half:], q[..., :d_half]], dim=-1)
        q = q * cos + q_rot * sin

        k_rot = torch.cat([-k[..., d_half:], k[..., :d_half]], dim=-1)
        k = k * cos + k_rot * sin

        return _ops.fp8_sdpa_op(q, k, v, is_causal=True)


class PatternCWanRoPE(nn.Module):
    """Pattern C: Wan-style indexed-write RoPE -> transpose -> FP8 SDPA."""

    def forward(self, q, k, v, freqs):
        out_q = torch.empty_like(q)
        out_q[..., 0::2] = (
            freqs[..., 0::2] * q[..., 0::2] - freqs[..., 1::2] * q[..., 1::2]
        )
        out_q[..., 1::2] = (
            freqs[..., 1::2] * q[..., 0::2] + freqs[..., 0::2] * q[..., 1::2]
        )
        out_q = out_q.type_as(q)

        out_k = torch.empty_like(k)
        out_k[..., 0::2] = (
            freqs[..., 0::2] * k[..., 0::2] - freqs[..., 1::2] * k[..., 1::2]
        )
        out_k[..., 1::2] = (
            freqs[..., 1::2] * k[..., 0::2] + freqs[..., 0::2] * k[..., 1::2]
        )
        out_k = out_k.type_as(k)

        return _ops.fp8_sdpa_op(
            out_q.transpose(1, 2),
            out_k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )


class TestRoPEFusionDetection(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self._old_pass = inductor_config.pre_grad_custom_pass

    def tearDown(self):
        inductor_config.pre_grad_custom_pass = self._old_pass
        torch._dynamo.reset()

    def _run_fusion_pass(self, model, *args):
        """Compile model with fusion pass, return captured stdout."""
        inductor_config.pre_grad_custom_pass = partial(
            rope_sdpa_fusion_pass,
            rope_sdpa_op=_ops.rope_sdpa_op,
            fp8_sdpa_op=_ops.fp8_sdpa_op,
            backend_name="TEST",
        )
        compiled = torch.compile(model)
        buf = io.StringIO()
        with torch.no_grad(), contextlib.redirect_stdout(buf):
            compiled(*args)
        return buf.getvalue()

    def _assert_fused(self, model, *extra_args):
        """Create BSHD inputs, run fusion pass, assert 1 node was fused."""
        B, S, H, D = 1, 32, 4, 64
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        output = self._run_fusion_pass(model, q, k, v, *extra_args)
        self.assertIn("1 fused with RoPE", output)

    def test_pattern_a_neox_rope(self):
        S, D = 32, 64
        self._assert_fused(PatternANeoXRoPE(), torch.randn(S, D), torch.randn(S, D))

    def test_pattern_b_huggingface_rope(self):
        S, D = 32, 64
        self._assert_fused(
            PatternBHuggingFaceRoPE(), torch.randn(S, D), torch.randn(S, D)
        )

    def test_pattern_c_wan_rope(self):
        S, D = 32, 64
        self._assert_fused(PatternCWanRoPE(), torch.randn(1, S, 1, D))


if __name__ == "__main__":
    unittest.main()
