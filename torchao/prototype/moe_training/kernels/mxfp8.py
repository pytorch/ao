import logging

logger: logging.Logger = logging.getLogger(__name__)

import torch

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401
except Exception as e:
    logging.warning(
        f"fbgemm_gpu_genai package is required for this feature but import failed with exception: {e}"
        "Please install nightly builds of pytorch and fbgemm_gpu_genai build using this command and try again: "
        "pip3 install --force-reinstall --pre torch fbgemm-gpu-genai --index-url https://download.pytorch.org/whl/nightly/cu129"
        "If errors persist, please file a bug report."
    )


@torch.library.custom_op("torchao::fbgemm_mxfp8_grouped_mm_2d_3d", mutates_args={})
def fbgemm_mxfp8_grouped_mm_2d_3d(
    A_mx: torch.Tensor,
    A_scale: torch.Tensor,
    B_t_mx_dim1: torch.Tensor,
    B_t_scales_dim1: torch.Tensor,
    offs: torch.Tensor,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A_mx.ndim == 2, "A_mx tensor must be 2D"
    assert B_t_mx_dim1.ndim == 3, "B_t_mx_dim1 tensor must be 3D"
    assert block_size == 32, "Only block_size=32 is supported"
    assert out_dtype == torch.bfloat16, "Only out_dtype=bfloat16 is supported"

    # "A" and "offs" already have been padded so token group sizes along Mg are multiples of scaling block size (32).
    # e.g. offs = [32, 64, 128]
    # From this, we compute `group_sizes` and `starting_row_after_padding`:
    # group_sizes = [32, 32, 64]
    # starting_row_after_padding = [0, 32, 64, 128]
    zero = torch.tensor([0], dtype=offs.dtype, device=offs.device)
    group_sizes = torch.diff(offs, prepend=zero).to(torch.int64)
    starting_row_after_paddding = torch.cat((zero, offs))
    out = torch.ops.fbgemm.mx8mx8bf16_grouped_stacked(
        A_mx,  # (Mg, K)
        B_t_mx_dim1,  # (E, K, N)
        A_scale,  # (Mg, K//block_size)
        B_t_scales_dim1,  # (E, K//block_size, N)
        group_sizes,  # size of each token group, computed from end idx of each group (`offs`)
        starting_row_after_padding=starting_row_after_paddding,
    )
    return out


@fbgemm_mxfp8_grouped_mm_2d_3d.register_fake
def _fbgemm_mxfp8_grouped_mm_2d_3d_fake(
    A_mx: torch.Tensor,
    B_t_mx_dim1: torch.Tensor,
    A_scale: torch.Tensor,
    B_t_scales_dim1: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    assert A_mx.ndim == 2, "A_mx tensor must be 2D"
    assert B_t_mx_dim1.ndim == 3, "B_t_mx_dim1 tensor must be 3D"
    mg, k = A_mx.shape
    e, k, n = B_t_mx_dim1.shape
    n_groups = offs.numel()
    assert n_groups == e, (
        "Size of `offs` (number of groups) must match first dim of `B_t_mx_dim1`"
    )
    output = torch.empty((mg, n), dtype=torch.bfloat16, device=A_mx.device)
    return output
