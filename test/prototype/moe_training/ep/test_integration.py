import sys
from pathlib import Path

import pytest

# Get the repo root (5 levels up from this file: ep -> moe_training -> prototype -> test -> ao).
# This avoids having to add PYTHONPATH manually when running the tests.
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

if not (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
):
    pytest.skip("Test requires CUDA 12.8+ with SM >= 100", allow_module_level=True)


import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import (
    all_to_all_single,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.ep import (
    a2a_dispatch_and_group_mxfp8_fwd_hp_bwd,
    a2a_dispatch_mxfp8_fwd_hp_bwd,
    unpermute_hp_fwd_mxfp8_bwd,
)
from torchao.prototype.moe_training.ep.permute import _permute_bf16
from torchao.prototype.moe_training.ep.unpermute import _unpermute_bf16
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.quantize_.common import KernelPreference


class TestIntegration(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 4

    @property
    def device(self):
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

    def test_full_pipeline(self):
        self._init_process()
        try:
            self._init_device()
            symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

            tokens = 64
            dim = 256
            hidden_dim = 512
            num_experts = 4

            # Create input activations and expert weights
            input_tensor = torch.randn(
                tokens,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)

            expert_weights = torch.randn(
                num_experts,
                hidden_dim,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            ref_expert_weights = expert_weights.detach().clone().requires_grad_(True)

            # Generate random tokens per expert that sum to total tokens
            # Shape should be [ep_degree * num_experts] where each rank has num_experts values
            ep_degree = self.world_size
            num_tokens_per_expert = generate_split_sizes(
                ep_degree * num_experts, tokens, self.device
            )

            # Compute tokens per expert group using tokens per expert using TorchTitan reference:
            # https://github.com/pytorch/torchtitan/blob/795a7a027eccf282c51a5ae1cc0c1c3459120c9b/torchtitan/distributed/expert_parallel.py#L111
            with torch.no_grad():
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=dist.group.WORLD,
                )
                # Need to wait explicitly because it is used by a triton kernel later
                # which doesn't realize that AsyncCollectiveTensor needs unwrapping
                num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                    num_tokens_per_expert_group
                )
                input_splits = (
                    num_tokens_per_expert.view(ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=True)
                )
                output_splits = (
                    num_tokens_per_expert_group.view(ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )

            group = dist.group.WORLD

            # Use deterministic permutation based on rank to avoid randomness issues
            torch.manual_seed(42)  # Same seed on all ranks

            rank_0_print(
                f"\n[Rank {self.rank}] ========== Side-by-Side BF16 vs MXFP8 Pipeline =========="
            )
            block_size = 32

            # ======================================
            # Stage 1: All-to-all dispatch
            # ======================================
            rank_0_print(f"[Rank {self.rank}] Stage 1: a2a_dispatch")

            # BF16: All-to-all dispatch
            bf16_dispatched = all_to_all_single(
                ref_input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=group,
            )
            bf16_dispatched = torch.ops._c10d_functional.wait_tensor(bf16_dispatched)

            # MXFP8: All-to-all dispatch with quantization
            mx_dispatched = a2a_dispatch_mxfp8_fwd_hp_bwd(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group_name=group.group_name,
            )
            assert isinstance(mx_dispatched, MXTensor)

            # Compare: Dequantize MXTensor and compute SQNR
            mx_dispatched_dq = mx_dispatched.dequantize()
            dispatched_sqnr = compute_error(
                bf16_dispatched.detach(), mx_dispatched_dq.detach()
            ).item()
            rank_0_print(
                f"[Rank {self.rank}]   Dispatched SQNR: {dispatched_sqnr:.2f} dB"
            )
            assert dispatched_sqnr >= 30.0, f"Dispatched SQNR {dispatched_sqnr} too low"

            # ======================================
            # Stage 2: Group by expert / padding
            # ======================================
            rank_0_print(f"[Rank {self.rank}] Stage 2: group + pad")

            # BF16: Permute
            (
                bf16_input_shape,
                bf16_permuted,
                bf16_permuted_indices,
                bf16_num_tokens_per_expert_padded,
                bf16_group_offsets,
            ) = _permute_bf16(
                bf16_dispatched,
                num_tokens_per_expert_group,
                ep_degree,
                num_experts,
                block_size,
            )
            active_rows = int(bf16_group_offsets[-1].item())
            bf16_permuted = bf16_permuted[:active_rows]
            bf16_permuted_indices = bf16_permuted_indices[:active_rows]

            # MXFP8: Dispatch directly into grouped+padded layout
            (
                padded_mx_shape,
                mx_permuted,
                mx_permuted_indices,
                num_tokens_per_expert_padded,
                mx_group_offsets,
            ) = a2a_dispatch_and_group_mxfp8_fwd_hp_bwd(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                num_tokens_per_expert,
                group_name=group.group_name,
            )
            assert isinstance(mx_permuted, MXTensor)

            # Compare: Dequantize and compare non-padded portion
            mx_permuted_dq = mx_permuted.dequantize()
            permuted_sqnr = compute_error(
                bf16_permuted.detach(), mx_permuted_dq.detach()
            ).item()
            rank_0_print(f"[Rank {self.rank}]   Permuted SQNR: {permuted_sqnr:.2f} dB")
            assert permuted_sqnr >= 30.0, f"Permuted SQNR {permuted_sqnr} too low"

            # ======================================
            # Stage 3: Grouped GEMM
            # ======================================
            rank_0_print(f"[Rank {self.rank}] Stage 3: grouped_mm")

            # BF16: Grouped MM
            bf16_gemm_output = torch._grouped_mm(
                bf16_permuted,
                ref_expert_weights.transpose(-2, -1),
                offs=bf16_group_offsets,
                out_dtype=torch.bfloat16,
            )

            # MXFP8: Grouped MM with quantization
            mx_gemm_output = _to_mxfp8_then_scaled_grouped_mm(
                mx_permuted,
                expert_weights.transpose(-2, -1),
                offs=mx_group_offsets,
                block_size=block_size,
                kernel_preference=KernelPreference.EMULATED,
                # wgrad_with_hp must be true if inputs are pre-quantized (MXTensor)
                wgrad_with_hp=True,
            )
            assert mx_gemm_output.dtype == torch.bfloat16

            # Compare: GEMM outputs (excluding padding)
            gemm_sqnr = compute_error(
                bf16_gemm_output.detach(), mx_gemm_output.detach()
            ).item()
            rank_0_print(f"[Rank {self.rank}]   GEMM output SQNR: {gemm_sqnr:.2f} dB")
            assert gemm_sqnr >= 27.5, f"GEMM SQNR {gemm_sqnr} too low"

            # ======================================
            # Stage 4: Unpermute
            # ======================================
            rank_0_print(f"[Rank {self.rank}] Stage 4: unpermute")

            # BF16: Unpermute
            bf16_output_shape = (bf16_input_shape[0], bf16_gemm_output.shape[-1])
            bf16_unpermuted = _unpermute_bf16(
                bf16_gemm_output,
                bf16_permuted_indices,
                bf16_output_shape,
            )

            # MXFP8: Unpermute
            # Update padded_shape to have output dimension instead of input dimension
            padded_output_shape = torch.Size(
                [padded_mx_shape[0], mx_gemm_output.shape[-1]]
            )
            mx_unpermuted = unpermute_hp_fwd_mxfp8_bwd(
                mx_gemm_output,
                mx_permuted_indices,
                padded_output_shape,
            )
            assert mx_unpermuted.dtype == torch.bfloat16

            # Compare: Unpermuted outputs
            unpermuted_sqnr = compute_error(
                bf16_unpermuted.detach(), mx_unpermuted.detach()
            ).item()
            rank_0_print(
                f"[Rank {self.rank}]   Unpermuted SQNR: {unpermuted_sqnr:.2f} dB"
            )
            assert unpermuted_sqnr >= 27.5, f"Unpermuted SQNR {unpermuted_sqnr} too low"

            rank_0_print(
                f"[Rank {self.rank}] ========== End Forward Validation ==========\n"
            )

        finally:
            dist.destroy_process_group()


def rank_0_print(msg: str):
    if dist.get_rank() == 0:
        print(msg)


if __name__ == "__main__":
    run_tests()
