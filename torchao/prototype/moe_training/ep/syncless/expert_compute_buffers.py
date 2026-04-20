# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Pre-allocated intermediate buffers for expert compute (forward/backward).

Companion to ``SavedActivationsBuffer``.  Holds all intermediate tensors
produced and consumed by the expert FFN kernels at ``max_output_rows``
capacity.  Allocated once at init; reused every iteration with zero
per-iteration ``torch.empty()`` calls.

This decouples buffer capacity (worst-case symmetric memory size) from
kernel work: grids are sized to the buffer, but kernels early-exit when
``row >= num_tokens``, making unused blocks cost ~0.
"""

import torch
import triton


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _blocked_scale_numel(rows: int, scale_cols: int) -> int:
    """Number of uint8 elements in a blocked (128, 4) scale layout."""
    BLOCK_ROWS, BLOCK_COLS = 128, 4
    padded_rows = _cdiv(rows, BLOCK_ROWS) * BLOCK_ROWS
    padded_cols = _cdiv(scale_cols, BLOCK_COLS) * BLOCK_COLS
    return padded_rows * padded_cols


class ExpertComputeBuffers:
    """Pre-allocated intermediate tensors for expert forward/backward.

    All tensors are allocated at ``max_output_rows`` capacity.  Kernels
    write only ``num_tokens`` rows per iteration and early-exit for
    the remainder.

    Args:
        max_output_rows: Buffer row capacity (must be divisible by 128).
        dim: Model dimension.
        hidden_dim: Expert hidden dimension.
        device: CUDA device.
    """

    def __init__(
        self,
        max_output_rows: int,
        dim: int,
        hidden_dim: int,
        device: torch.device,
    ):
        assert (
            max_output_rows % 128 == 0
        ), f"max_output_rows ({max_output_rows}) must be divisible by 128"
        self.max_output_rows = max_output_rows
        self.dim = dim
        self.hidden_dim = hidden_dim
        block_size = 32

        # ==================================================================
        # Forward intermediates
        # ==================================================================

        # silu_mul_fw_mxfp8 outputs
        self.h_e4m3 = torch.empty(
            max_output_rows,
            hidden_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.h_scales_blocked = torch.empty(
            _blocked_scale_numel(max_output_rows, hidden_dim // block_size),
            dtype=torch.uint8,
            device=device,
        )

        # triton_mx_block_rearrange_input_sym_mem_buffer output
        self.scale_a_blocked = torch.empty(
            _blocked_scale_numel(max_output_rows, dim // block_size),
            dtype=torch.uint8,
            device=device,
        )

        # ==================================================================
        # Backward intermediates
        # ==================================================================

        # mxfp8_quant_and_transpose outputs (for grad_out of shape (M, dim))
        M, N = max_output_rows, dim
        self.grad_out_e4m3 = torch.empty(
            M,
            N,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.grad_out_scales_blocked = torch.empty(
            _blocked_scale_numel(M, N // block_size),
            dtype=torch.uint8,
            device=device,
        )
        self.grad_out_t_e4m3 = torch.empty(
            N,
            M,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.grad_out_t_scales_blocked = torch.empty(
            _blocked_scale_numel(N, M // block_size),
            dtype=torch.uint8,
            device=device,
        )

        # silu_mul_bw outputs
        self.h_out = torch.empty(
            max_output_rows,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.grad_h13_out = torch.empty(
            max_output_rows,
            2 * hidden_dim,
            dtype=torch.bfloat16,
            device=device,
        )
