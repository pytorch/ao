import sys
from pathlib import Path

# Get the repo root (5 levels up from this file: ep -> moe_training -> prototype -> test -> ao)
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    all_to_all_single,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.ep import (
    a2a_combine,
    a2a_dispatch,
    permute,
    unpermute,
)
from torchao.prototype.moe_training.ep.kernels import generate_permute_indices
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.mx_formats.kernels import triton_mxfp8_dequant_dim0
from torchao.prototype.mx_formats.mx_tensor import MXTensor


def _round_up(x, y):
    return ((x + y - 1) // y) * y


def _dequantize_mxtensor(mx_tensor: MXTensor, block_size: int = 32) -> torch.Tensor:
    """Dequantize an MXTensor to high precision for comparison."""
    return triton_mxfp8_dequant_dim0(
        mx_tensor.qdata,
        mx_tensor.scale.view(
            torch.uint8
        ),  # Triton can't handle e8m0 scale dtype directly
        out_dtype=mx_tensor._orig_dtype,
        scale_block_size=block_size,
    )


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts, alignment):
    x_padded_per_expert = x.shape[0] + num_local_experts * alignment
    padded_max_len = _round_up(x_padded_per_expert, alignment)

    with torch.no_grad():
        (
            permuted_indices,
            _sizes,
            offsets,
        ) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            alignment,
        )

    x = torch.vstack((x, x.new_zeros((1, x.shape[-1]))))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, offsets


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]  # Remove padding row
    return out


class TestIntegration(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

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

    def test_full_pipeline(self):
        self._init_process()
        try:
            tokens = 64
            dim = 256
            hidden_dim = 512
            num_experts = 4

            # Create input activations and expert weights
            # NOTE: Initially create with requires_grad=False, then enable gradients
            # right before forward pass to avoid tracking operations during setup
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

            # Compute tokens per expert group using tokens per expert
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
                # NOTE: this would incur a device-to-host sync
                output_splits = (
                    num_tokens_per_expert_group.view(ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )

            group = dist.group.WORLD

            # Use deterministic permutation based on rank to avoid randomness issues
            torch.manual_seed(42)  # Same seed on all ranks

            print(
                f"\n[Rank {self.rank}] ========== Side-by-Side BF16 vs MXFP8 Pipeline =========="
            )
            block_size = 32

            # ======================================
            # Stage 1: All-to-all dispatch
            # ======================================
            print(f"[Rank {self.rank}] Stage 1: a2a_dispatch")

            # BF16: All-to-all dispatch
            bf16_dispatched = all_to_all_single(
                ref_input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=group,
            )
            bf16_dispatched = torch.ops._c10d_functional.wait_tensor(bf16_dispatched)

            # MXFP8: All-to-all dispatch with quantization
            mx_dispatched = a2a_dispatch(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=group,
            )
            assert isinstance(mx_dispatched, MXTensor)

            # Compare: Dequantize MXTensor and compute SQNR
            mx_dispatched_dq = _dequantize_mxtensor(mx_dispatched, block_size)
            dispatched_sqnr = compute_error(bf16_dispatched, mx_dispatched_dq)
            print(f"[Rank {self.rank}]   Dispatched SQNR: {dispatched_sqnr:.2f} dB")
            assert dispatched_sqnr >= 30.0, f"Dispatched SQNR {dispatched_sqnr} too low"

            # ======================================
            # Stage 2: Permute
            # ======================================
            print(f"[Rank {self.rank}] Stage 2: permute")

            # BF16: Permute
            (
                bf16_input_shape,
                bf16_permuted,
                bf16_permuted_indices,
                bf16_group_offsets,
            ) = _permute(
                bf16_dispatched,
                num_tokens_per_expert_group,
                ep_degree,
                num_experts,
                block_size,
            )

            # MXFP8: Permute (with padding for MXTensor)
            (
                padded_mx_shape,
                mx_permuted,
                mx_permuted_indices,
                num_tokens_per_expert_padded,
                mx_group_offsets,
            ) = permute(
                mx_dispatched,
                num_tokens_per_expert_group,
                ep_degree,
                num_experts,
                block_size,
                use_mxfp8=True,
                use_triton_for_bwd=True,
            )
            assert isinstance(mx_permuted, MXTensor)

            # Compare: Dequantize and compare non-padded portion
            mx_permuted_dq = _dequantize_mxtensor(mx_permuted, block_size)
            permuted_sqnr = compute_error(bf16_permuted, mx_permuted_dq)
            print(f"[Rank {self.rank}]   Permuted SQNR: {permuted_sqnr:.2f} dB")
            assert permuted_sqnr >= 30.0, f"Permuted SQNR {permuted_sqnr} too low"

            # ======================================
            # Stage 3: Grouped GEMM
            # ======================================
            print(f"[Rank {self.rank}] Stage 3: grouped_mm")

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
                use_cuda_for_blocked_layout=False,
                wgrad_with_hp=True,
            )
            assert mx_gemm_output.dtype == torch.bfloat16

            # Compare: GEMM outputs (excluding padding)
            gemm_sqnr = compute_error(bf16_gemm_output, mx_gemm_output)
            print(f"[Rank {self.rank}]   GEMM output SQNR: {gemm_sqnr:.2f} dB")
            assert gemm_sqnr >= 28.0, f"GEMM SQNR {gemm_sqnr} too low"

            # ======================================
            # Stage 4: Unpermute
            # ======================================
            print(f"[Rank {self.rank}] Stage 4: unpermute")

            # BF16: Unpermute
            bf16_output_shape = (bf16_input_shape[0], bf16_gemm_output.shape[-1])
            bf16_unpermuted = _unpermute(
                bf16_gemm_output, bf16_output_shape, bf16_permuted_indices
            )

            # MXFP8: Unpermute
            # Update padded_shape to have output dimension instead of input dimension
            padded_output_shape = torch.Size(
                [padded_mx_shape[0], mx_gemm_output.shape[-1]]
            )
            mx_unpermuted = unpermute(
                mx_gemm_output, mx_permuted_indices, padded_output_shape, use_mxfp8=True
            )
            assert mx_unpermuted.dtype == torch.bfloat16

            # Compare: Unpermuted outputs
            unpermuted_sqnr = compute_error(bf16_unpermuted, mx_unpermuted)
            print(f"[Rank {self.rank}]   Unpermuted SQNR: {unpermuted_sqnr:.2f} dB")
            assert unpermuted_sqnr >= 28.0, f"Unpermuted SQNR {unpermuted_sqnr} too low"

            # ======================================
            # Stage 5: All-to-all combine
            # ======================================
            print(f"[Rank {self.rank}] Stage 5: a2a_combine")

            # BF16: All-to-all combine
            bf16_output = all_to_all_single(
                bf16_unpermuted,
                input_splits.tolist(),
                output_splits.tolist(),
                group=group,
            )
            bf16_output = torch.ops._c10d_functional.wait_tensor(bf16_output)

            # MXFP8: All-to-all combine
            mxfp8_output = a2a_combine(
                mx_unpermuted,
                output_splits=input_splits.tolist(),
                input_splits=output_splits.tolist(),
                group=group,
            )
            assert mxfp8_output.dtype == torch.bfloat16

            # Compare: Final outputs
            final_sqnr = compute_error(bf16_output, mxfp8_output)
            print(f"[Rank {self.rank}]   Final output SQNR: {final_sqnr:.2f} dB")
            assert final_sqnr >= 28.0, f"Final SQNR {final_sqnr} too low"

            print(f"[Rank {self.rank}] ========== End Forward Validation ==========\n")

            # ======================================
            # Backward passes
            # ======================================

            # BF16 backward
            bf16_labels = torch.ones_like(bf16_output)
            bf16_loss = F.mse_loss(bf16_output, bf16_labels)
            bf16_loss.backward()

            assert ref_input_tensor.grad is not None
            assert ref_input_tensor.grad.dtype == torch.bfloat16
            assert ref_expert_weights.grad is not None
            assert ref_expert_weights.grad.dtype == torch.bfloat16

            # MXFP8 backward
            mxfp8_labels = torch.ones_like(mxfp8_output)
            mxfp8_loss = F.mse_loss(mxfp8_output, mxfp8_labels)
            mxfp8_loss.backward()

            assert input_tensor.grad is not None
            assert input_tensor.grad.dtype == torch.bfloat16
            assert expert_weights.grad is not None
            assert expert_weights.grad.dtype == torch.bfloat16

            # Compare input gradients
            input_grad_sqnr = compute_error(ref_input_tensor.grad, input_tensor.grad)
            min_input_grad_sqnr = 26.0
            print(f"[Rank {self.rank}]   Input grad SQNR: {input_grad_sqnr:.2f} dB")
            assert input_grad_sqnr >= min_input_grad_sqnr, (
                f"Input grad SQNR {input_grad_sqnr} is too low, must be >= {min_input_grad_sqnr}"
            )

            # Verify weight gradients exist
            assert ref_expert_weights.grad is not None, (
                "ref_expert_weights.grad is None"
            )
            assert expert_weights.grad is not None, "expert_weights.grad is None"

            weight_grad_sqnr = compute_error(
                ref_expert_weights.grad, expert_weights.grad
            )
            min_weight_grad_sqnr = 25.0
            print(f"[Rank {self.rank}]   Weight grad SQNR: {weight_grad_sqnr:.2f} dB")
            assert weight_grad_sqnr >= min_weight_grad_sqnr, (
                f"Weight grad SQNR {weight_grad_sqnr} is too low, must be >= {min_weight_grad_sqnr}"
            )

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
