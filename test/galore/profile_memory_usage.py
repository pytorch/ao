# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib
import logging
import os

import model_configs
import profiling_utils
import torch
import torch.nn as nn
import torch.utils.data
from bitsandbytes.optim import AdamW8bit
from torch.profiler import record_function
from transformers import LlamaConfig, LlamaForCausalLM

from torchao.prototype.galore.optim.galore_torch import AdamW as GaLoreAdamW
from torchao.prototype.galore.optim.galore_torch import AdamW8bit as GaLoreAdamW8bit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_galore(model, lr, weight_decay, rank, galore_scale, update_proj_gap):
    galore_params = []
    target_modules_list = ["attn", "mlp"]
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if not any(target_key in module_name for target_key in target_modules_list):
            continue

        logger.debug("Enabling GaLore for weights in module: ", module_name)
        galore_params.append(module.weight)
    id_galore_params = [id(p) for p in galore_params]
    # make parameters without "rank" to another group
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    # then call galore_adamw

    total_galore_params = sum(p.numel() for p in galore_params)
    total_regular_params = sum(p.numel() for p in regular_params)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_galore_params + total_regular_params == total_params

    print(
        f"Total params: {total_params} = GaLore params: {total_galore_params} + Regular params: {total_regular_params}"
    )
    param_groups = [
        {"params": regular_params},
        {
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": galore_scale,
            "proj_type": "std",
        },
    ]
    if "adamw" in args.optimizer:
        if "8bit" in args.optimizer:
            optimizer = GaLoreAdamW8bit(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = GaLoreAdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return optimizer


def train_step(model, batch, labels, optimizer, profiler=None):
    with record_function("MODEL_FORWARD"):
        loss = model(**batch, labels=labels).loss

    with record_function("MODEL_BACKWARD"):
        loss.backward()

    with record_function("OPTIMIZER_STEP"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if profiler:
        profiler.step()


def run(args, file_prefix):
    torch.manual_seed(args.seed)

    # Initialize model from config dict
    model_config = LlamaConfig()
    try:
        model_config_dict = getattr(model_configs, args.model_config.upper())
    except:
        raise ValueError(f"Model config {args.model_config} not found")
    model_config.update(model_config_dict)
    model = LlamaForCausalLM(model_config).to("cuda")

    # Load sample batch
    input_ids = torch.randint(
        0,
        model_config.vocab_size,
        size=(args.batch_size, args.max_seq_len),
        dtype=torch.int64,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    batch = dict(input_ids=input_ids, attention_mask=attention_mask)
    labels = batch["input_ids"].clone()

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable params: {sum(p.numel() for p in trainable_params)} / {n_total_params}"
    )

    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    elif "galore" in args.optimizer.lower():
        optimizer = setup_galore(
            model,
            args.learning_rate,
            args.weight_decay,
            rank=args.rank,
            galore_scale=args.galore_scale,
            update_proj_gap=args.update_proj_gap,
        )
    elif args.optimizer.lower() == "adamw8bit":
        optimizer = AdamW8bit(
            trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise "Unsupported optimizer"

    if args.torch_profiler:
        prof_ctx = profiling_utils.get_torch_profiler(
            name=file_prefix,
            output_dir=args.output_dir,
            wait_steps=args.wait_steps,
            warmup_steps=args.warmup_steps,
            active_steps=args.profiler_steps,
        )
    elif args.nsys_profiler:
        prof_ctx = profiling_utils.nsys_profiler()
    else:
        prof_ctx = contextlib.nullcontext()

    total_steps = min(
        args.wait_steps + args.warmup_steps + args.profiler_steps, args.max_steps
    )
    print(
        f"Profiling {args.model_config} with {args.optimizer.upper()} for {total_steps} steps (wait_steps={args.wait_steps}, warmup_steps={args.warmup_steps}, profiler_steps={args.profiler_steps})"
    )
    with prof_ctx as prof:
        logger.debug(f"Profiler: {prof}")
        for _ in range(total_steps):
            with record_function("TRAIN_STEP"):
                train_step(
                    model,
                    batch,
                    labels,
                    optimizer,
                    profiler=prof if args.torch_profiler else None,
                )
    if args.torch_profiler:
        print(f"Finished profiling, outputs saved to {args.output_dir}/{file_prefix}*")
    else:
        print("Finished profiling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t", "--torch_profiler", action="store_true", help="Enable torch profiler"
    )
    parser.add_argument(
        "-m",
        "--torch_memory_snapshot",
        action="store_true",
        help="Enable torch memory snapshot",
    )

    parser.add_argument(
        "-ns",
        "--nsys_profiler",
        action="store_true",
        help="Enable nsys profiling context manager"
        "Surrounds training loop with cudaProfilerApi.{Start,Stop}",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "galore_adamw", "adamw8bit", "galore_adamw8bit"],
        help="Which optimizer to use",
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    # parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument(
        "--wait_steps",
        type=int,
        default=0,
        help="Number of steps to run before starting torch profiler",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for torch profiler",
    )

    parser.add_argument(
        "--profiler_steps",
        type=int,
        default=5,
        help="Number of active steps for torch profiler",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Max number of train steps to run."
        "Total train steps will be min of `max_steps` and the sum of torch profiler steps (`wait_steps` + `warmup_steps` + `profiler_steps`).",
    )
    parser.add_argument(
        "--model_config",
        default="llama100M",
        type=str,
        choices=["llama100M", "llama1B"],
        help="Model configuration (see model_configs.py)",
    )
    parser.add_argument(
        "--batch_size", default=5, type=int, help="Batch size to use for train step"
    )
    parser.add_argument(
        "--max_seq_len",
        default=256,
        type=int,
        help="Sequence length to use for train step, should be less than that in the specific model config",
    )
    parser.add_argument(
        "--output_dir",
        default="profiler_out",
        type=str,
        help="Directory for profiler outputs",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="Weight decay for AdamW",
    )

    parser.add_argument("--seed", default=0, type=int, help="Random seed for torch")
    args = parser.parse_args()
    output_dir = args.output_dir
    # output_prefix = args.output_prefix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if "galore" not in args.optimizer.lower():
        file_prefix = args.optimizer.lower()
    else:
        file_prefix = "-".join(
            [
                args.optimizer.lower(),
                str(args.rank),
                str(args.galore_scale),
                str(args.update_proj_gap),
            ]
        )
    mem_ctx = (
        profiling_utils.memory_recorder(
            file_name=os.path.join(output_dir, f"{file_prefix}-memory-snapshot")
        )
        if args.torch_memory_snapshot
        else contextlib.nullcontext()
    )
    profiling_utils.flush_cuda_mem()
    with mem_ctx:
        run(args, file_prefix)

    profiling_utils.get_cuda_memory_usage(units="MB", show=True)
