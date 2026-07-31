# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.dtensor_utils import (
    dtensor_from_local_like as _dtensor_from_local_like,
)
from torchao.prototype.blockwise_fp8_training.dtensor_utils import (
    is_dtensor as _is_dtensor,
)
from torchao.prototype.blockwise_fp8_training.dtensor_utils import (
    local_tensor as _local_tensor,
)
from torchao.prototype.blockwise_fp8_training.dtensor_utils import (
    replicate_like_dtensor as _replicate_like_dtensor,
)
from torchao.prototype.blockwise_fp8_training.dtensor_utils import (
    require_dtensor_not_sharded_on_dim as _require_dtensor_not_sharded_on_dim,
)
from torchao.prototype.blockwise_fp8_training.dtensor_utils import (
    require_dtensor_replicated as _require_dtensor_replicated,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _is_column_major,
    _is_row_major,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_lhs,
)
from torchao.prototype.moe_training.blockwise_fp8.grouped_mm_backend import (
    _select_fp8_blockwise_grouped_mm_backend,
)
from torchao.prototype.moe_training.utils import (
    conditional_nostrict_trace,
    pad_token_groups,
    unpad_token_groups,
)
from torchao.quantization.quantize_.common import KernelPreference

_TOKEN_GROUP_PADDING_REASON = "when padding token groups"


def _pad_token_groups_preserve_dtensor(
    input_act: torch.Tensor,
    group_end_offsets: torch.Tensor,
    *,
    alignment_size: int,
    kernel_preference: KernelPreference,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_dtensor(input_act) and not _is_dtensor(group_end_offsets):
        return pad_token_groups(
            input_act,
            group_end_offsets,
            alignment_size=alignment_size,
            kernel_preference=kernel_preference,
        )

    _require_dtensor_not_sharded_on_dim(
        input_act,
        "input_act",
        0,
        dim_name="token dimension",
        allow_partial=True,
        reason=_TOKEN_GROUP_PADDING_REASON,
    )
    _require_dtensor_replicated(
        group_end_offsets,
        "group_end_offsets",
        reason=_TOKEN_GROUP_PADDING_REASON,
    )
    local_padded, local_starts, local_ends = pad_token_groups(
        _local_tensor(input_act),
        _local_tensor(group_end_offsets),
        alignment_size=alignment_size,
        kernel_preference=kernel_preference,
    )
    padded = _dtensor_from_local_like(local_padded, input_act)
    if not _is_dtensor(padded):
        padded = _replicate_like_dtensor(padded, group_end_offsets)
    return (
        padded,
        _replicate_like_dtensor(local_starts, group_end_offsets, input_act),
        _replicate_like_dtensor(local_ends, group_end_offsets, input_act),
    )


def _unpad_token_groups_preserve_dtensor(
    padded_output: torch.Tensor,
    original_group_end_offsets: torch.Tensor,
    padded_group_start_offsets: torch.Tensor,
    num_tokens: int,
    *,
    alignment_size: int,
    kernel_preference: KernelPreference,
) -> torch.Tensor:
    if (
        not _is_dtensor(padded_output)
        and not _is_dtensor(original_group_end_offsets)
        and not _is_dtensor(padded_group_start_offsets)
    ):
        return unpad_token_groups(
            padded_output,
            original_group_end_offsets,
            padded_group_start_offsets,
            num_tokens,
            alignment_size=alignment_size,
            kernel_preference=kernel_preference,
        )

    _require_dtensor_not_sharded_on_dim(
        padded_output,
        "padded_output",
        0,
        dim_name="token dimension",
        allow_partial=True,
        reason=_TOKEN_GROUP_PADDING_REASON,
    )
    _require_dtensor_replicated(
        original_group_end_offsets,
        "original_group_end_offsets",
        reason=_TOKEN_GROUP_PADDING_REASON,
    )
    _require_dtensor_replicated(
        padded_group_start_offsets,
        "padded_group_start_offsets",
        reason=_TOKEN_GROUP_PADDING_REASON,
    )
    local_unpadded = unpad_token_groups(
        _local_tensor(padded_output),
        _local_tensor(original_group_end_offsets),
        _local_tensor(padded_group_start_offsets),
        num_tokens,
        alignment_size=alignment_size,
        kernel_preference=kernel_preference,
    )
    return _dtensor_from_local_like(local_unpadded, padded_output)


@conditional_nostrict_trace
def _to_fp8_blockwise_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    float8_dtype: torch.dtype = e4m3_dtype,
    block_size: int = 128,
    pad_token_groups_for_grouped_mm: bool = True,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
) -> torch.Tensor:
    """
    Differentiable FP8 blockwise grouped matrix multiplication.

    A has shape (M, K). B_t has shape (E, K, N), transposed and in
    per-expert column-major layout.
    """
    assert block_size == 128, "Only block_size=128 is supported"
    assert kernel_preference in (
        KernelPreference.AUTO,
        KernelPreference.EMULATED,
    ), "kernel_preference must be AUTO or EMULATED"
    return _Float8BlockwiseGroupedMM.apply(
        A,
        B_t,
        offs,
        out_dtype,
        float8_dtype,
        block_size,
        pad_token_groups_for_grouped_mm,
        kernel_preference,
    )


@conditional_nostrict_trace
def _to_fp8_blockwise_then_emulated_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    float8_dtype: torch.dtype = e4m3_dtype,
    block_size: int = 128,
    pad_token_groups_for_grouped_mm: bool = True,
) -> torch.Tensor:
    """
    Differentiable FP8 blockwise grouped matrix multiplication using an emulated GEMM.
    """
    return _to_fp8_blockwise_then_scaled_grouped_mm(
        A,
        B_t,
        offs,
        out_dtype,
        float8_dtype,
        block_size,
        pad_token_groups_for_grouped_mm,
        KernelPreference.EMULATED,
    )


class _Float8BlockwiseGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        float8_dtype: torch.dtype = e4m3_dtype,
        block_size: int = 128,
        pad_token_groups_for_grouped_mm: bool = True,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
    ) -> torch.Tensor:
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B_t must be 3D"
        assert offs is not None and offs.dtype == torch.int32, "offs must be int32"
        assert A.dtype in (torch.bfloat16, torch.float32), "A must be bf16 or fp32"
        assert B_t.dtype in (torch.bfloat16, torch.float32), "B_t must be bf16 or fp32"
        assert A.shape[-1] == B_t.shape[-2], (
            f"shape {A.shape} and {B_t.shape} are not compatible"
        )
        assert A.shape[-1] % block_size == 0 and B_t.shape[-1] % block_size == 0, (
            f"K and N must be divisible by block_size={block_size}"
        )
        assert _is_row_major(A), "A must be row-major"
        assert _is_column_major(B_t), "B_t must be per-expert column-major"

        # DTensor custom ops require every tensor argument to be a DTensor.
        # TP commonly shards only the expert weights, so lift replicated
        # activations/metadata onto the same mesh before dispatch.
        A = _replicate_like_dtensor(A, B_t)
        B_t = _replicate_like_dtensor(B_t, A)
        num_tokens = A.shape[0]
        offs = _replicate_like_dtensor(offs, A, B_t)
        padded_group_start_offsets = None
        padded_group_end_offsets = offs
        if pad_token_groups_for_grouped_mm:
            padded_A, padded_group_start_offsets, padded_group_end_offsets = (
                _pad_token_groups_preserve_dtensor(
                    A,
                    offs,
                    alignment_size=block_size,
                    kernel_preference=kernel_preference,
                )
            )
        else:
            padded_A = A

        local_group_end_offsets = _local_tensor(padded_group_end_offsets)
        local_original_group_end_offsets = _local_tensor(offs)
        local_padded_group_start_offsets = (
            _local_tensor(padded_group_start_offsets)
            if padded_group_start_offsets is not None
            else None
        )
        backend = _select_fp8_blockwise_grouped_mm_backend(
            kernel_preference,
            A,
            out_dtype,
            block_size,
            local_group_end_offsets,
            original_group_end_offsets=local_original_group_end_offsets
            if pad_token_groups_for_grouped_mm
            else None,
            padded_group_start_offsets=local_padded_group_start_offsets,
            num_rows=padded_A.shape[0],
        )

        A_fp8, A_scale = triton_fp8_blockwise_act_quant_lhs(
            padded_A.contiguous(),
            block_size=block_size,
            dtype=float8_dtype,
        )
        B_t_fp8, B_t_scale = backend.quantize_forward_rhs(
            B_t,
            block_size,
            float8_dtype,
        )
        out = backend.grouped_mm(
            A_fp8,
            B_t_fp8,
            A_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            B_t_scale,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            padded_group_end_offsets,
            out_dtype,
            block_size,
        )
        if pad_token_groups_for_grouped_mm:
            out = _unpad_token_groups_preserve_dtensor(
                out,
                offs,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=block_size,
                kernel_preference=kernel_preference,
            )

        ctx.save_for_backward(
            padded_A,
            B_t,
            offs,
            padded_group_start_offsets,
            padded_group_end_offsets,
        )
        ctx.out_dtype = out_dtype
        ctx.float8_dtype = float8_dtype
        ctx.block_size = block_size
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.kernel_preference = kernel_preference
        ctx.num_tokens = num_tokens
        ctx.backend = backend
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            padded_A,
            B_t,
            original_group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
        ) = ctx.saved_tensors
        out_dtype = ctx.out_dtype
        float8_dtype = ctx.float8_dtype
        block_size = ctx.block_size
        pad_token_groups_for_grouped_mm = ctx.pad_token_groups_for_grouped_mm
        kernel_preference = ctx.kernel_preference
        num_tokens = ctx.num_tokens
        backend = ctx.backend

        grad_output = _replicate_like_dtensor(grad_output, padded_A, B_t)
        if pad_token_groups_for_grouped_mm:
            padded_grad_output, _, _ = _pad_token_groups_preserve_dtensor(
                grad_output,
                original_group_end_offsets,
                alignment_size=block_size,
                kernel_preference=kernel_preference,
            )
        else:
            padded_grad_output = grad_output

        grad_output_fp8, grad_output_scale = triton_fp8_blockwise_act_quant_lhs(
            padded_grad_output.contiguous(),
            block_size=block_size,
            dtype=float8_dtype,
        )
        B_fp8, B_scale = backend.quantize_dgrad_rhs(
            B_t,
            block_size,
            float8_dtype,
        )
        grad_A = backend.grouped_mm(
            grad_output_fp8,
            B_fp8,
            grad_output_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            B_scale,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            padded_group_end_offsets,
            out_dtype,
            block_size,
        )
        if pad_token_groups_for_grouped_mm:
            grad_A = _unpad_token_groups_preserve_dtensor(
                grad_A,
                original_group_end_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=block_size,
                kernel_preference=kernel_preference,
            )

        grad_B = backend.wgrad(
            padded_grad_output,
            padded_A,
            padded_group_end_offsets,
            out_dtype,
            block_size,
            float8_dtype,
        )
        return (
            grad_A,
            grad_B.transpose(-2, -1),
            None,
            None,
            None,
            None,
            None,
            None,
        )
