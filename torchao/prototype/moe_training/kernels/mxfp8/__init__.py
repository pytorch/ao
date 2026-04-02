import torch
import torch.distributed._symmetric_memory as symm_mem

from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cuda_kernels_available,  # noqa: F401
    fused_pad_token_groups_cuda,  # noqa: F401
    fused_unpad_token_groups_cuda,  # noqa: F401
    mx_block_rearrange_2d_M_groups_cuda,  # noqa: F401
    mxfp8_quantize_2d_1x32_cutedsl,  # noqa: F401
    mxfp8_quantize_2d_32x1_cutedsl,  # noqa: F401
    mxfp8_quantize_cuda_3d,  # noqa: F401
    torch_pad_token_groups,  # noqa: F401
    torch_to_blocked_2d_K_groups,  # noqa: F401
    torch_to_blocked_2d_M_groups,  # noqa: F401
    torch_to_blocked_per_group_3d,  # noqa: F401
    torch_unpad_token_groups,  # noqa: F401
    triton_mx_block_rearrange_2d_K_groups,  # noqa: F401
    triton_mx_block_rearrange_2d_M_groups,  # noqa: F401
    triton_mx_block_rearrange_per_group_3d,  # noqa: F401
)


class EPBufferManager:
    """Manages reusable buffers for MXFP8 all-to-all operations across MoE layers."""

    def __init__(self):
        self.output = None
        self.output_scales = None
        self.max_output_rows_per_rank = None

    def preallocate_buffers(
        self,
        max_output_rows_per_rank: int,
        data_shape: tuple,
        scales_shape: tuple,
        data_dtype: torch.dtype = torch.float8_e4m3fn,
        scales_dtype: torch.dtype = torch.uint8,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Preallocate symmetric memory buffers for output data and scales.

        Args:
            max_output_rows_per_rank: Maximum output rows per rank (worst-case allocation)
            data_shape: Shape of data tensor (excluding first dimension)
            scales_shape: Shape of scales tensor (excluding first dimension)
            data_dtype: Data tensor dtype
            scales_dtype: Scales tensor dtype
            device: Device to allocate on
        """
        self.max_output_rows_per_rank = max_output_rows_per_rank

        # Allocate output data buffer
        self.output = symm_mem.empty(
            max_output_rows_per_rank,
            *data_shape,
            dtype=data_dtype,
            device=device,
        )

        # Allocate output scales buffer
        self.output_scales = symm_mem.empty(
            max_output_rows_per_rank,
            *scales_shape,
            dtype=scales_dtype,
            device=device,
        )

    def reset(self):
        """Clear all buffers (useful for testing or changing configs)."""
        self.__init__()


# Module-level singleton for buffer management
_default_buffer_manager = None


def get_buffer_manager():
    """Get the default buffer manager, creating it if necessary."""
    global _default_buffer_manager
    if _default_buffer_manager is None:
        _default_buffer_manager = EPBufferManager()
    return _default_buffer_manager
