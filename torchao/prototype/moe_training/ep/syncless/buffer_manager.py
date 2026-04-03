import torch
import torch.distributed._symmetric_memory as symm_mem


class EPBufferManager:
    """Manages reusable buffers for MXFP8 all-to-all operations across MoE layers."""

    def __init__(self):
        self.output = None
        self.output_scales = None
        self.max_output_rows_per_rank = None
        # Metadata for real data views
        self.output_expert_splits = None
        self.expert_padded_offsets = None
        self.real_data_size = None

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

    def set_real_data_metadata(
        self,
        output_expert_splits: torch.Tensor,
        expert_padded_offsets: torch.Tensor,
    ):
        """
        Set metadata for real data views after mxfp8_token_dispatch.

        Args:
            output_expert_splits: Tokens each expert received, shape (world_size, num_experts_per_rank)
            expert_padded_offsets: Starting offset for each expert, shape (num_experts_per_rank,)
        """
        self.output_expert_splits = output_expert_splits
        self.expert_padded_offsets = expert_padded_offsets

        # Compute total real data size (sum of all actual tokens across experts)
        # Keep as tensor to avoid device-to-host sync
        self.real_data_size = output_expert_splits.sum()

    def get_real_output_data(self) -> torch.Tensor:
        """
        Get view of real data only (zero-copy slice of overallocated buffer).

        Returns:
            Real data tensor, shape (real_data_size, dim)
        """
        if self.real_data_size is None:
            return self.output
        return self.output[: self.real_data_size]

    def get_real_output_scales(self) -> torch.Tensor:
        """
        Get view of real scales only (zero-copy slice of overallocated buffer).

        Returns:
            Real scales tensor, shape (real_data_size, scale_dim)
        """
        if self.real_data_size is None:
            return self.output_scales
        return self.output_scales[: self.real_data_size]

    def get_group_end_offsets(self) -> torch.Tensor:
        """
        Convert expert information to group_end_offsets format for mxfp8_grouped_mm.

        Returns:
            Cumulative token counts per expert, shape (num_experts_per_rank,)
        """
        if self.output_expert_splits is None:
            raise RuntimeError(
                "Real data metadata not set. Call set_real_data_metadata() first."
            )

        # Sum tokens across all source ranks for each expert
        # output_expert_splits: (world_size, num_experts_per_rank)
        tokens_per_expert = self.output_expert_splits.sum(
            dim=0
        )  # (num_experts_per_rank,)

        # Convert to cumulative offsets (group_end_offsets format)
        group_end_offsets = tokens_per_expert.cumsum(dim=0)

        return group_end_offsets

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
