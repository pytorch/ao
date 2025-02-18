import argparse
import math
import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo import config as dynamo_config
from torch.nn.parallel import DistributedDataParallel as DDP

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4


class LoRALinear(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        lora_rank: int = None,
        lora_alpha: float = 16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        if lora_rank is None:
            lora_rank = hidden_dim // 2

        weight = torch.randn(hidden_dim, hidden_dim, dtype=dtype)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.register_parameter(
            "weight", nn.Parameter(to_nf4(weight), requires_grad=False)
        )
        self.lora_a = nn.Linear(
            in_features=hidden_dim, out_features=self.lora_rank, bias=False
        )
        self.lora_b = nn.Linear(
            in_features=self.lora_rank, out_features=hidden_dim, bias=False
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_b.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = linear_nf4(input=x, weight=self.weight)
        lora_out = self.lora_a(x)
        lora_out = (self.lora_alpha / self.lora_rank) * self.lora_b(lora_out)
        return out + lora_out


def _init_model(dim, num_linears, device, dtype) -> nn.Module:
    with torch.device(device):
        modules = []
        for i in range(num_linears):
            modules += [LoRALinear(hidden_dim=dim, dtype=dtype)]
        seq = nn.Sequential(*modules)

    return seq


def dist_print(*args, delay=0.5):
    rank = dist.get_rank()
    time.sleep(delay * rank)
    print(f"[rank{rank}]: ", *args, flush=True)


def make_batch(global_bs, dim, dtype, device):
    batch = torch.randn((global_bs, dim), dtype=dtype, device=device)
    if dist.get_world_size() > 1:
        batch = batch.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]
    return batch


def run_ddp(global_bs, dim, num_linears, device, dtype, num_steps, save_dir, compile):
    os.makedirs(save_dir, exist_ok=True)
    model = _init_model(dim, num_linears, device, dtype)
    model = DDP(model, device_ids=[device])

    if compile:
        model = torch.compile(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []

    for i in range(num_steps):
        inp = make_batch(global_bs, dim, dtype, device)
        loss = model(inp).sum()
        losses.append(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    dist.barrier()

    save_path = f"{save_dir}/ddp-{dist.get_rank()}.pt"
    torch.save(model.state_dict(), save_path)
    dist_print("Saved model to", save_path)


def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    dist_print("Dist initialized with world size", dist.get_world_size())


def cleanup_dist():
    dist.barrier()
    if dist.get_rank() == 0:
        print("Cleaning up dist")
    dist.destroy_process_group()


@contextmanager
def distributed_context():
    init_dist()
    yield
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--global_bs", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_linears", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--optimize_ddp", type=str, default="ddp_optimizer")
    args = parser.parse_args()

    args.dtype = getattr(torch, args.dtype)
    dynamo_config.optimize_ddp = args.optimize_ddp

    if args.optimize_ddp == "python_reducer":
        dynamo_config.compiled_autograd = True

    with distributed_context():
        torch.manual_seed(args.seed)
        run_ddp(
            global_bs=args.global_bs,
            dim=args.dim,
            num_linears=args.num_linears,
            device=args.device,
            dtype=args.dtype,
            num_steps=args.num_steps,
            save_dir=args.save_dir,
            compile=args.compile,
        )
