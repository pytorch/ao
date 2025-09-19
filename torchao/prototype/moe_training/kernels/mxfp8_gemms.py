import logging

import torch

from torchao.prototype.moe_training.kernels.mxfp8 import (
    torch_to_blocked_2d_M_groups,
    torch_to_blocked_per_group_3d,
)

logger: logging.Logger = logging.getLogger(__name__)

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401
except Exception as e:
    logging.warning(
        f"fbgemm_gpu_genai package is required for this feature but import failed with exception: {e}"
        "Please install nightly builds of pytorch and fbgemm_gpu_genai build using this command and try again: "
        "pip3 install --force-reinstall --pre torch fbgemm-gpu-genai --index-url https://download.pytorch.org/whl/nightly/cu129"
        "If errors persist, please file a bug report."
    )

DEBUG = False


@torch.library.custom_op("torchao::fbgemm_mxfp8_grouped_mm_2d_3d", mutates_args={})
def fbgemm_mxfp8_grouped_mm_2d_3d(
    A_fp8: torch.Tensor,
    A_scales: torch.Tensor,
    B_fp8: torch.Tensor,
    B_scales: torch.Tensor,
    offs: torch.Tensor,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A_fp8.ndim == 2, "A_fp8 tensor must be 2D"
    assert B_fp8.ndim == 3, "B_fp8 tensor must be 3D"
    assert block_size == 32, "Only block_size=32 is supported"
    assert out_dtype == torch.bfloat16, "Only out_dtype=bfloat16 is supported"
    assert A_fp8.shape[-1] == B_fp8.shape[-1], "A_fp8 and B_fp8 must have same last dim"

    # Convert scales for each group to blocked format.
    Mg, K = A_fp8.shape
    A_scales_blocked, starting_row_after_padding = torch_to_blocked_2d_M_groups(
        A_scales, offs, K
    )
    B_scales_blocked = torch_to_blocked_per_group_3d(B_scales)

    # From this, we compute `group_sizes` and `starting_row_after_padding`:
    # group_sizes = [32, 32, 64]
    # starting_row_after_padding = [0, 32, 64, 128]
    zero = torch.tensor([0], dtype=offs.dtype, device=offs.device)
    group_sizes = torch.diff(offs, prepend=zero).to(torch.int64)

    # TODO: remove debug logging once prototype is more mature.
    _log_inputs(
        A_fp8,
        B_fp8,
        A_scales,
        A_scales_blocked,
        B_scales,
        B_scales_blocked,
        offs,
        group_sizes,
        starting_row_after_padding,
    )

    out = torch.ops.fbgemm.mx8mx8bf16_grouped_stacked(
        A_fp8,
        B_fp8,
        A_scales_blocked,
        B_scales_blocked,
        group_sizes,
        starting_row_after_padding=starting_row_after_padding,
    )
    return out


@fbgemm_mxfp8_grouped_mm_2d_3d.register_fake
def _fbgemm_mxfp8_grouped_mm_2d_3d_fake(
    A_fp8: torch.Tensor,
    A_scales: torch.Tensor,
    B_fp8: torch.Tensor,
    B_scales: torch.Tensor,
    offs: torch.Tensor,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A_fp8.ndim == 2, "A_fp8 tensor must be 2D"
    assert B_fp8.ndim == 3, "B_fp8 tensor must be 3D"
    assert out_dtype == torch.bfloat16, "Only out_dtype=bfloat16 is supported"
    assert A_fp8.shape[-1] == B_fp8.shape[-1], "A_fp8 and B_fp8 must have same last dim"
    mg, k = A_fp8.shape
    e, n, k = B_fp8.shape
    n_groups = offs.numel()
    assert n_groups == e, (
        "Size of `offs` (number of groups) must match first dim of `B_fp8`"
    )
    output = torch.empty((mg, n), dtype=torch.bfloat16, device=A_fp8.device)
    return output


def _log_inputs(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    A_scales: torch.Tensor,
    A_scales_blocked: torch.Tensor,
    B_scales: torch.Tensor,
    B_scales_blocked: torch.Tensor,
    offs: torch.Tensor,
    group_sizes: torch.Tensor,
    starting_row_after_padding: torch.Tensor,
):
    # TODO: figure out why python logging module is not behaving as expected,
    # when setting log level to DEBUG, it still doesn't print logger.debug lines.
    # Using this hack for now.
    if not DEBUG:
        return

    logger.info("offs: %s, dtype: %s", offs, offs.dtype)
    logger.info(
        "A_fp8.shape: %s, stride: %s, dtype: %s",
        A_fp8.shape,
        A_fp8.stride(),
        A_fp8.dtype,
    )
    logger.info(
        "B_fp8.shape: %s, stride: %s, dtype: %s",
        B_fp8.shape,
        B_fp8.stride(),
        B_fp8.dtype,
    )
    logger.info(
        "A_scales (non-blocked) shape: %s, stride: %s, dtype: %s",
        A_scales.shape,
        A_scales.stride(),
        A_scales.dtype,
    )
    logger.info(
        "A_scales_blocked.shape: %s, stride: %s, dtype: %s",
        A_scales_blocked.shape,
        A_scales_blocked.stride(),
        A_scales_blocked.dtype,
    )
    logger.info(
        "B_scales (non-blocked) shape: %s, stride: %s, dtype: %s",
        B_scales.shape,
        B_scales.stride(),
        B_scales.dtype,
    )
    logger.info(
        "B_scales_blocked.shape: %s, stride: %s, dtype: %s",
        B_scales_blocked.shape,
        B_scales_blocked.stride(),
        B_scales_blocked.dtype,
    )
    logger.info(
        "group_sizes: %s, stride: %s, dtype: %s",
        group_sizes,
        group_sizes.stride(),
        group_sizes.dtype,
    )
    logger.info(
        "starting_row_after_padding: %s, stride: %s, dtype: %s",
        starting_row_after_padding,
        starting_row_after_padding.stride(),
        starting_row_after_padding.dtype,
    )
