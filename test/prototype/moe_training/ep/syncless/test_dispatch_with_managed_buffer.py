# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration test: unified buffer allocator + custom allocator + token dispatch.

Tests the pattern where a forward pass of token dispatch saves both the
original input (x) and the dispatch result into a unified GPU+CPU managed
buffer, using the custom allocator for sub-allocation offsets.

NOTE: In production, allocator offsets (GPU tensors) are passed directly to
compute kernels (e.g., mxfp8_grouped_gemm's out_offset parameter), which write
results into the buffer without any device-to-host sync. This test uses .item()
to read offsets for Python-level slicing since we don't have the actual compute
kernel integration — the .item() syncs are test-only and do NOT reflect the
production sync-free pattern.
"""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
    pytest.skip("Test requires CUDA build on SM100", allow_module_level=True)

try:
    from cuda.bindings import driver as cuda_drv
except ImportError:
    pytest.skip("Test requires cuda.bindings (cuda-python)", allow_module_level=True)

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.prototype.moe_training.ep.syncless.cuda_allocator import CUDAAllocator
from torchao.prototype.moe_training.ep.syncless.token_dispatch import (
    mxfp8_token_dispatch,
)
from torchao.prototype.moe_training.ep.syncless.unified_buffer_allocator import (
    _check,
    create_unified_buffer,
    free_unified_buffer,
)


def _get_granularity() -> int:
    """Return the recommended allocation granularity for the current GPU."""
    device_id = torch.cuda.current_device()
    prop = cuda_drv.CUmemAllocationProp()
    prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    granularity = _check(
        cuda_drv.cuMemGetAllocationGranularity(
            prop,
            cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    return granularity


@instantiate_parametrized_tests
class TestDispatchWithManagedBuffer(MultiProcessTestCase):
    """Integration test: token dispatch forward with activations saved in a
    unified GPU+CPU buffer managed by the custom allocator.

    Simulates the forward pass of an MoE layer where:
    1. A unified GPU+CPU buffer is created (like the SOL's StaticBuffers)
    2. A CUDAAllocator manages sub-allocation offsets within the buffer
    3. Token dispatch runs, routing tokens to expert-major layout
    4. The original input (x) and the dispatch result are saved into the
       managed buffer at offsets returned by the allocator
    5. The saved activations are verified to be intact
    6. The allocator offsets are freed (simulating backward cleanup)
    """

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    def _init_device(self):
        symm_mem.set_backend("NVSHMEM")

    def test_save_activations_in_managed_buffer(self):
        """Forward dispatch + save input and dispatch result into the unified buffer."""
        self._init_process()
        try:
            self._init_device()

            tokens_per_expert = 2
            experts_per_rank = 2
            total_local_tokens = tokens_per_expert * experts_per_rank
            dim = 128

            # --- Step 1: Create input tensor ---
            start_value = self.rank * total_local_tokens + 1
            input_tensor = torch.stack(
                [
                    torch.full(
                        (dim,),
                        float(start_value + i),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    for i in range(total_local_tokens)
                ]
            )

            # --- Step 2: Set up expert splits and run dispatch ---
            total_experts_global = self.world_size * experts_per_rank
            num_tokens_per_expert_global = generate_split_sizes(
                total_experts_global, total_local_tokens, self.device
            )
            expert_splits_per_rank = num_tokens_per_expert_global.view(
                self.world_size, -1
            )
            input_rank_level_splits = expert_splits_per_rank.sum(dim=1)

            total_global_tokens = total_local_tokens * self.world_size
            max_output_tokens_per_rank = (
                total_global_tokens + 128 * experts_per_rank * self.world_size
            )

            from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
                get_buffer_manager,
            )

            buffer_manager = get_buffer_manager()
            buffer_manager.preallocate_buffers(
                max_output_rows_per_rank=max_output_tokens_per_rank,
                data_shape=(dim,),
                scales_shape=(dim // 32,),
                data_dtype=torch.float8_e4m3fn,
                scales_dtype=torch.uint8,
                device=self.device,
            )

            (
                output_e4m3,
                output_scales_e8m0,
                output_rank_level_splits,
                output_expert_splits,
                expert_padded_offsets,
                all_expert_splits,
                _padded_tokens_per_expert,
            ) = mxfp8_token_dispatch(
                input_tensor,
                input_rank_level_splits,
                expert_splits_per_rank,
                dist.group.WORLD,
            )

            # --- Step 3: Allocate unified GPU+CPU buffer + custom allocator ---
            g = _get_granularity()
            row_bytes = dim * 2  # bf16 = 2 bytes per element
            max_rows = total_local_tokens + output_e4m3.shape[0] + 256
            total_bytes = max_rows * row_bytes
            gpu_bytes = ((total_bytes // g) + 1) * g

            buffer = create_unified_buffer(cpu_bytes=0, gpu_bytes=gpu_bytes)
            buf_2d = buffer.view(torch.bfloat16).view(-1, dim)
            total_rows = buf_2d.shape[0]

            allocator = CUDAAllocator(buffer, [total_rows])
            allocator.alloc(128)  # sentinel (match SOL pattern)

            # --- Step 4: Save input (x) into the managed buffer ---
            # In production, the compute kernel writes directly at the GPU-side
            # offset (e.g., mxfp8_grouped_gemm's out_offset parameter).
            # Here we use .item() for Python slicing — test-only sync.
            input_offset = allocator.alloc(total_local_tokens)
            input_off = input_offset.item()
            buf_2d[input_off : input_off + total_local_tokens].copy_(input_tensor)

            # --- Step 5: Save dispatch result into the managed buffer ---
            num_output_rows = output_e4m3.shape[0]
            output_offset = allocator.alloc(num_output_rows)
            output_off = output_offset.item()
            output_as_bytes = output_e4m3.view(torch.uint8)
            buf_bytes = buffer.view(torch.uint8)
            dst_byte_start = output_off * row_bytes
            src_bytes = output_as_bytes.numel()
            buf_bytes[dst_byte_start : dst_byte_start + src_bytes].copy_(
                output_as_bytes.view(-1)
            )

            # --- Step 6: Verify saved activations are intact ---
            saved_input = buf_2d[input_off : input_off + total_local_tokens]
            assert torch.equal(saved_input, input_tensor), (
                "Saved input does not match original"
            )

            saved_output_bytes = buf_bytes[dst_byte_start : dst_byte_start + src_bytes]
            assert torch.equal(saved_output_bytes, output_as_bytes.view(-1)), (
                "Saved dispatch output does not match"
            )

            # --- Step 7: Verify allocator stats ---
            stats = allocator.stats()
            expected_allocated = 128 + total_local_tokens + num_output_rows
            assert stats.sum_allocated[0].item() == expected_allocated, (
                f"Expected {expected_allocated} allocated, got {stats.sum_allocated[0].item()}"
            )

            # --- Step 8: Free in reverse order (simulating backward cleanup) ---
            # allocator.free() accepts GPU tensor offsets directly (no sync needed)
            allocator.free(output_offset)
            allocator.free(input_offset)

            stats = allocator.stats()
            assert stats.sum_allocated[0].item() == 128, (
                "Only sentinel should remain after freeing saved activations"
            )

            # --- Cleanup ---
            allocator.free_all()
            free_unified_buffer(buffer.data_ptr())

        finally:
            dist.destroy_process_group()

    def test_multiple_layers_reuse_buffer(self):
        """Simulate multiple MoE layers reusing the same managed buffer.

        Each layer saves activations during forward, then frees them
        during backward, mimicking the forward-then-backward pattern.
        """
        self._init_process()
        try:
            self._init_device()

            tokens_per_expert = 2
            experts_per_rank = 2
            total_local_tokens = tokens_per_expert * experts_per_rank
            dim = 64
            num_layers = 3

            # --- Set up buffer + allocator ---
            g = _get_granularity()
            row_bytes = dim * 2  # bf16
            max_rows = (total_local_tokens * 4 + 256) * num_layers
            total_bytes = max_rows * row_bytes
            gpu_bytes = ((total_bytes // g) + 1) * g

            buffer = create_unified_buffer(cpu_bytes=0, gpu_bytes=gpu_bytes)
            buf_2d = buffer.view(torch.bfloat16).view(-1, dim)
            total_rows = buf_2d.shape[0]
            allocator = CUDAAllocator(buffer, [total_rows])
            allocator.alloc(128)  # sentinel

            # --- Forward pass: save activations for each layer ---
            saved_offsets = []
            saved_inputs = []
            for layer_idx in range(num_layers):
                start_value = self.rank * total_local_tokens + layer_idx * 100 + 1
                input_tensor = torch.stack(
                    [
                        torch.full(
                            (dim,),
                            float(start_value + i),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        for i in range(total_local_tokens)
                    ]
                )

                # In production, the compute kernel writes at the GPU-side offset.
                # Here we use .item() for Python slicing — test-only sync.
                offset = allocator.alloc(total_local_tokens)
                off_val = offset.item()
                buf_2d[off_val : off_val + total_local_tokens].copy_(input_tensor)

                saved_offsets.append(offset)
                saved_inputs.append(input_tensor.clone())

            # --- Backward pass: verify and free in reverse order ---
            for layer_idx in reversed(range(num_layers)):
                offset = saved_offsets[layer_idx]
                off_val = offset.item()
                saved = buf_2d[off_val : off_val + total_local_tokens]

                assert torch.equal(saved, saved_inputs[layer_idx]), (
                    f"Layer {layer_idx}: saved input mismatch"
                )

                # allocator.free() accepts GPU tensor offsets (no sync needed)
                allocator.free(offset)

            stats = allocator.stats()
            assert stats.sum_allocated[0].item() == 128, (
                "Only sentinel should remain after backward"
            )

            # --- Cleanup ---
            allocator.free_all()
            free_unified_buffer(buffer.data_ptr())

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
