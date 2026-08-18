# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
A script demonstrating weight-only quantization of a Mixture-of-Experts (MoE)
model with `quantize_()`.

It builds a small token-choice top-2 MoE block (a router plus ``nn.Linear``
experts), quantizes only the expert weights — the router is left in high
precision so routing decisions are unchanged — and verifies the result:
expert weights become quantized tensor subclasses, the serialized model
shrinks roughly 4x (int8), and a forward pass stays numerically close to the
float32 baseline (SQNR).

Run on CPU (default, int8 weight-only):

    python examples/quantize_moe.py

Run int4 weight-only (requires CUDA and the `mslk` package):

    python examples/quantize_moe.py --dtype int4 --device cuda

Note: real MoE checkpoints (e.g. `meta-llama/Llama-4-Scout-17B-16E-Instruct`)
often store experts as fused 3D ``(num_experts, K, N)`` parameters instead of
``nn.Linear`` modules. For those, configure quantization per-parameter with
``FqnToConfig`` and ``PerRow(1)`` granularity — see
``examples/quantize_llama_4.py`` for a complete float8 version of that flow.
"""

import argparse
import copy
import io

import torch
import torch.nn.functional as F
from torch import nn

from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
from torchao.quantization.utils import compute_error


class Expert(nn.Module):
    """A standard FFN expert: up-projection, SiLU, down-projection."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class MoEBlock(nn.Module):
    """A token-choice top-k MoE block: softmax router + ``nn.Linear`` experts."""

    def __init__(self, dim=256, hidden_dim=512, num_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(dim, hidden_dim) for _ in range(num_experts)
        )
        self.top_k = top_k

    def forward(self, x):
        batch, seq, dim = x.shape
        flat = x.reshape(-1, dim)
        probs = F.softmax(self.router(flat), dim=-1)
        weights, expert_idx = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            routed = expert_idx == e
            token_idx = routed.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_idx.numel() == 0:
                continue
            gate = (weights * routed).sum(dim=-1)[token_idx].unsqueeze(-1)
            out[token_idx] += gate * expert(flat[token_idx])
        return out.reshape(batch, seq, dim)


def serialized_size_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def is_expert_linear(module, fqn):
    return isinstance(module, nn.Linear) and "experts" in fqn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weight-only quantization of an MoE model with quantize_()"
    )
    parser.add_argument(
        "--dtype",
        choices=["int8", "int4"],
        default="int8",
        help="weight dtype: int8 runs anywhere, int4 requires CUDA and mslk",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device to run on (default: cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dtype == "int4" and args.device == "cpu":
        print(
            "warning: the default int4 weight-only path targets CUDA "
            "(tinygemm) and requires the `mslk` package; expect a failure "
            "on CPU"
        )

    torch.manual_seed(0)
    model = MoEBlock().eval().to(args.device)
    x = torch.randn(4, 64, 256, device=args.device)
    with torch.no_grad():
        baseline_out = model(x)
    baseline_bytes = serialized_size_bytes(model)
    baseline_weight = model.experts[0].up.weight
    print(
        f"baseline | expert weight: {type(baseline_weight).__name__} "
        f"({baseline_weight.dtype}) | serialized: {baseline_bytes / 1e6:.2f} MB"
    )

    config = Int8WeightOnlyConfig() if args.dtype == "int8" else Int4WeightOnlyConfig()
    quantized_model = copy.deepcopy(model)
    # quantize only the expert weights; the router keeps full precision so
    # that token-to-expert assignments are identical to the baseline model
    quantize_(quantized_model, config, filter_fn=is_expert_linear)

    quantized_bytes = serialized_size_bytes(quantized_model)
    quantized_weight = quantized_model.experts[0].up.weight
    with torch.no_grad():
        quantized_out = quantized_model(x)
    sqnr = compute_error(baseline_out, quantized_out)
    print(
        f"{args.dtype} wo  | expert weight: {type(quantized_weight).__name__} | "
        f"router: {type(quantized_model.router.weight).__name__} "
        f"({quantized_model.router.weight.dtype}) | "
        f"serialized: {quantized_bytes / 1e6:.2f} MB "
        f"({baseline_bytes / quantized_bytes:.2f}x smaller) | "
        f"SQNR vs float32: {sqnr:.1f} dB"
    )

    assert type(quantized_weight) is not nn.Parameter, (
        "expert weights were not quantized"
    )
    assert type(quantized_model.router.weight) is nn.Parameter, (
        "router should not be quantized"
    )
    assert baseline_bytes / quantized_bytes > 1.5, "model did not shrink"
    assert sqnr > 25, f"SQNR too low: {sqnr:.1f} dB"
    print("quantization of the MoE model succeeded")


if __name__ == "__main__":
    main()
