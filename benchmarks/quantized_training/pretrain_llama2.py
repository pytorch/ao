# pre-train a mini Llama2 on TinyStories with INT8 quantized training
# pip install huggingface_hub sentencepiece wandb
#
# BF16 baseline: python benchmarks/quantized_training/pretrain_llama2.py --seed 2024 --bf16_model --compile
# INT8 QT:       python benchmarks/quantized_training/pretrain_llama2.py --seed 2024 --bf16_model --compile --quantize int8_weight_only
# INT8 MP:       python benchmarks/quantized_training/pretrain_llama2.py --seed 2024 --bf16_model --compile --quantize int8_mixed_precision
# BitNet:        python benchmarks/quantized_training/pretrain_llama2.py --seed 2024 --bf16_model --compile --quantize bitnet --modify_rmsnorm_for_bitnet

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from torchao import quantize_
from torchao._models.llama.model import (
    ModelArgs,
    RMSNorm,
    Transformer,
    transformer_configs,
)
from torchao.prototype import low_bit_optim
from torchao.prototype.quantized_training import (
    bitnet_training,
    int8_mixed_precision_training,
    int8_weight_only_quantized_training,
)

# not official models
transformer_configs.update(
    (
        ("470M", dict(n_layer=24, n_head=16, dim=1024, intermediate_size=4096)),
        ("1B", dict(n_layer=24, n_head=24, dim=1536, intermediate_size=6144)),
    )
)


# hack from fairseq
# https://github.com/facebookresearch/fairseq/blob/920a548ca770fb1a951f7f4289b4d3a0c1bc226f/fairseq/modules/checkpoint_activations.py
def enable_activation_checkpointing(m: torch.nn.Module):
    assert not hasattr(m, "_forward")
    m._forward = m.forward
    m.forward = partial(checkpoint, m.forward, use_reentrant=False)


def get_loss(model: Transformer, batch: torch.Tensor):
    logits = model(batch)[:, :-1].float().flatten(0, 1)
    labels = batch[:, 1:].flatten()
    return torch.nn.functional.cross_entropy(logits, labels)


def get_tinystories():
    save_path = Path("tinystories.bin")

    if not save_path.exists():
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download

        tokenizer_path = hf_hub_download("meta-llama/Llama-2-7b", "tokenizer.model")
        tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        assert tokenizer.vocab_size() < (1 << 16)  # make sure we can use uint16

        # do everything in memory. we have enough RAM
        filepath = hf_hub_download(
            "roneneldan/TinyStories",
            "TinyStoriesV2-GPT4-train.txt",
            repo_type="dataset",
        )
        stories = open(filepath).read().split("\n<|endoftext|>\n")

        tokens_list = []
        chunk_size = 10_000
        for i in tqdm(
            range(0, len(stories), chunk_size), desc="Tokenizing TinyStories"
        ):
            chunk = stories[i : min(i + chunk_size, len(stories))]
            tokens_list.extend(
                tokenizer.Encode(chunk, add_bos=True, add_eos=True, num_threads=4)
            )

        total_size = sum(len(x) for x in tokens_list)
        mmap_tokens = np.memmap(save_path, dtype=np.uint16, mode="w+", shape=total_size)
        i = 0
        for tokens in tokens_list:
            mmap_tokens[i : i + len(tokens)] = tokens
            i += len(tokens)
        mmap_tokens.flush()

    tokens = np.memmap(save_path, dtype=np.uint16, mode="r")
    return torch.from_numpy(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="470M", choices=transformer_configs.keys())
    parser.add_argument("--bf16_model", action="store_true")
    parser.add_argument("--bf16_amp", action="store_true")
    parser.add_argument("--quantize")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--modify_rmsnorm_for_bitnet", action="store_true")

    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)

    parser.add_argument("--optim", default="AdamW")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())

    parser.add_argument("--project", default="quantized_training")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    config = ModelArgs.from_name(args.model)
    config.block_size = args.seq_len
    model = Transformer(config)
    if args.bf16_model:
        model.bfloat16()
    model.cuda()
    with torch.device("cuda"):
        model.setup_caches(args.batch_size, args.seq_len, training=True)
    if args.activation_checkpointing:
        for layer in model.layers:
            enable_activation_checkpointing(layer)

    # as recommended by https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
    # section 3
    if args.modify_rmsnorm_for_bitnet:
        # remove old RMSNorm
        for layer in model.layers:
            layer.attention_norm = torch.nn.Identity()
            layer.ffn_norm = torch.nn.Identity()

        # insert new RMSNorm
        def insert_rmsnorm(module: torch.nn.Module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    w = child.weight
                    norm = RMSNorm(child.in_features).to(device=w.device, dtype=w.dtype)
                    setattr(module, name, torch.nn.Sequential(norm, child))
                else:
                    insert_rmsnorm(child)

        insert_rmsnorm(model.layers)

    # don't apply int8_mixed_precision to LM head, since it can cause convergence issue.
    # TODO: might want to do the same for int8_weight_only to standardize.
    if args.quantize == "int8_weight_only":
        quantize_(
            model, int8_weight_only_quantized_training(), set_inductor_config=False
        )

    elif args.quantize == "int8_mixed_precision":
        quantize_(
            model.layers, int8_mixed_precision_training(), set_inductor_config=False
        )

    elif args.quantize == "int8_mixed_precision_module_swap":
        quantize_(
            model.layers,
            int8_mixed_precision_training(module_swap=True),
            set_inductor_config=False,
        )

    elif args.quantize == "bitnet":
        quantize_(model.layers, bitnet_training(), set_inductor_config=False)

    elif args.quantize is not None:
        raise ValueError(f"Unsupported quantize={args.quantize}")

    print(f"No. of params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")
    torch.cuda.reset_peak_memory_stats()  # don't count memory occupied by unquantized weights

    # only use optimizers from torchao.prototype.low_bit_optim to support quantized training
    if args.optim == "AdamW":
        args.optim = "_AdamW"
    optim = getattr(low_bit_optim, args.optim)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **args.optim_kwargs,
    )

    data = get_tinystories().cuda()
    args.torch_version = torch.__version__
    run = wandb.init(dir="/tmp", config=args, project=args.project, name=args.run_name)

    step = 0
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()
    _get_loss = torch.compile(get_loss) if args.compile else get_loss
    time0 = time.time()

    while step < args.n_steps:
        # randomly select a continuous chunk, then reshape it
        idx = torch.randint(
            0, data.shape[0] - args.batch_size * args.seq_len, (1,)
        ).item()
        batch = (
            data[idx : idx + args.batch_size * args.seq_len]
            .view(args.batch_size, args.seq_len)
            .long()
        )

        with torch.autocast("cuda", torch.bfloat16, enabled=args.bf16_amp):
            loss = _get_loss(model, batch)
        loss.backward()

        if step % args.log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                lr=optim.param_groups[0]["lr"],
                max_memory_allocated=torch.cuda.max_memory_allocated() / 1e9,
                max_memory_reserved=torch.cuda.max_memory_reserved() / 1e9,
            )
            run.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()

        if step % args.log_interval == 0:
            time1 = time.time()
            log_dict = dict(
                tokens_per_second=(args.log_interval * args.batch_size * args.seq_len)
                / (time1 - time0)
            )
            time0 = time1
            run.log(log_dict, step=step)

    run.finish()
