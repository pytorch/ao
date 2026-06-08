"""Microbenchmark: permute_and_pad backward — Triton non-atomic scatter vs the
default PyTorch indexing_backward_kernel (atomic scatter).

The permute is applied to BF16 tokens upstream of FP8 quantization, so this
kernel is identical for both fp8_rowwise and fp8_tensorwise recipes.

Run (inside the training container, fresh torchao installed):
  python bench_permute_backward.py
"""

import torch

from torchao.prototype.moe_training.ep.kernels import generate_permute_indices
from torchao.prototype.moe_training.ep.permute import _PermuteBF16FwdBF16Bwd

DEVICE = "cuda"
ALIGNMENT = 16
DTYPE = torch.bfloat16


def _tokens_per_expert(M, n):
    w = torch.rand(n, device=DEVICE)
    c = (w / w.sum() * M).long()
    c[0] += M - c.sum()
    return c.clamp(min=0)


def bench(M, D, num_experts, ep_degree, warmup=20, iters=100):
    num_local = num_experts // ep_degree
    tpe = _tokens_per_expert(M, num_experts)
    padded_max = ((M + num_local * ALIGNMENT + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    with torch.no_grad():
        idx, _, _ = generate_permute_indices(
            tpe, num_local, ep_degree, padded_max, ALIGNMENT
        )
    base = torch.randn(M, D, device=DEVICE, dtype=DTYPE)

    def run_default():
        x = base.clone().requires_grad_(True)
        x_pad = torch.vstack((x, x.new_zeros((1, D))))
        y = x_pad[idx, :]
        x.grad = None
        y.sum().backward()

    def run_triton():
        x = base.clone().requires_grad_(True)
        y = _PermuteBF16FwdBF16Bwd.apply(x, idx)
        x.grad = None
        y.sum().backward()

    for fn in (run_default, run_triton):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    def timed(fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    return timed(run_default), timed(run_triton)


def main():
    print(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n")
    print("permute_and_pad backward: default indexing_backward vs Triton non-atomic scatter")
    hdr = ("M", "D", "experts", "ep", "default(ms)", "triton(ms)", "speedup")
    print(f"{hdr[0]:>7}{hdr[1]:>7}{hdr[2]:>9}{hdr[3]:>4}{hdr[4]:>13}{hdr[5]:>12}{hdr[6]:>9}")
    for M, D, ne, ep in [(4096, 7168, 256, 8), (8192, 7168, 256, 8), (16384, 7168, 256, 8)]:
        d, t = bench(M, D, ne, ep)
        print(f"{M:>7}{D:>7}{ne:>9}{ep:>4}{d:>13.4f}{t:>12.4f}{d / t:>8.2f}x")


if __name__ == "__main__":
    main()
