# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Single-GPU NVFP4 training example for one DeepSeek-V3 671B expert shard.

The input is pre-routed: its rows must be contiguous by expert in the same
order as ``num_tokens_per_expert``. This example intentionally has no router,
permutation, distributed initialization, or expert-parallel collectives.

Run with::

    python -m torchao.prototype.moe_training.nvfp4_training.nvfp4_single_gpu_example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

NUM_LOCAL_EXPERTS = 128
MODEL_DIM = 7168
EXPERT_HIDDEN_DIM = 2048
TOKENS_PER_EXPERT = 128
NUM_PACKED_ROWS = NUM_LOCAL_EXPERTS * TOKENS_PER_EXPERT
RHT_SIGN_VECTOR = tuple(1 if index % 2 == 0 else -1 for index in range(16))


class SimplifiedMoE(nn.Module):
    """MoE layer whose input has already been routed and packed by expert."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.experts = GroupedExperts(device)

    def forward(
        self, routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        return self.experts(routed_input, num_tokens_per_expert)


class GroupedExperts(nn.Module):
    """DeepSeek-V3 experts backed by differentiable NVFP4 grouped GEMMs."""

    def __init__(self, device: torch.device):
        super().__init__()
        parameter_kwargs = {"device": device, "dtype": torch.bfloat16}
        self.w1 = nn.Parameter(
            torch.empty(
                NUM_LOCAL_EXPERTS,
                EXPERT_HIDDEN_DIM,
                MODEL_DIM,
                **parameter_kwargs,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                NUM_LOCAL_EXPERTS,
                MODEL_DIM,
                EXPERT_HIDDEN_DIM,
                **parameter_kwargs,
            )
        )
        self.w3 = nn.Parameter(
            torch.empty(
                NUM_LOCAL_EXPERTS,
                EXPERT_HIDDEN_DIM,
                MODEL_DIM,
                **parameter_kwargs,
            )
        )
        self.register_buffer(
            "sr_seed", torch.tensor([1234], dtype=torch.int64, device=device)
        )

        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        nn.init.normal_(self.w3, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
            _to_nvfp4_then_scaled_grouped_mm,
        )

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate = _to_nvfp4_then_scaled_grouped_mm(
            x,
            self.w1,
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets,
            pad_token_groups_for_grouped_mm=False,
        )
        up = _to_nvfp4_then_scaled_grouped_mm(
            x,
            self.w3,
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets,
            pad_token_groups_for_grouped_mm=False,
        )
        hidden = F.silu(gate) * up
        return _to_nvfp4_then_scaled_grouped_mm(
            hidden,
            self.w2,
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets,
            pad_token_groups_for_grouped_mm=False,
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("Skipping NVFP4 example: CUDA is not available.")
        return
    if not has_triton():
        print("Skipping NVFP4 example: Triton is not available.")
        return
    if not torch_version_at_least("2.10.0"):
        print("Skipping NVFP4 example: PyTorch 2.10 or newer is required.")
        return

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability < (10, 0):
        print("Skipping NVFP4 example: an SM100 or newer GPU is required.")
        return

    torch.manual_seed(42)
    model = SimplifiedMoE(device)
    routed_input = torch.randn(
        NUM_PACKED_ROWS,
        MODEL_DIM,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    num_tokens_per_expert = torch.full(
        (NUM_LOCAL_EXPERTS,),
        TOKENS_PER_EXPERT,
        device=device,
        dtype=torch.int32,
    )

    output = model(routed_input, num_tokens_per_expert)
    loss = output.float().square().mean()
    loss.backward()

    expected_output_shape = (NUM_PACKED_ROWS, MODEL_DIM)
    assert output.shape == expected_output_shape
    assert routed_input.grad is not None and torch.isfinite(routed_input.grad).all()
    print(f"output: {tuple(output.shape)}")
    print(f"input grad: {tuple(routed_input.grad.shape)} (finite)")
    for name, parameter in model.experts.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        print(f"{name} grad: {tuple(parameter.grad.shape)} (finite)")
    print("NVFP4 single-GPU forward/backward completed successfully.")


if __name__ == "__main__":
    main()
