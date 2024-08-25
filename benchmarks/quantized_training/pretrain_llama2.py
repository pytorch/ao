# pre-train a mini Llama2 on TinyStories with INT8 quantized training
# pip install huggingface_hub sentencepiece wandb
#
# BF16 baseline: python benchmarks/quantized_training/pretrain_llama2.py --seed 2024 --n_steps 10_000 --compile
# INT8 QT:       python benchmarks/quantized_training/pretrain_llama2.py --seed 2024 --n_steps 10_000 --compile --quantize int8_weight_only

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from torchao._models.llama.model import ModelArgs, Transformer
from torchao.prototype import low_bit_optim
from torchao.prototype.quantized_training import int8_weight_only_quantized_training
from torchao.quantization.quant_api import quantize_


# hack from fairseq
# https://github.com/facebookresearch/fairseq/blob/920a548ca770fb1a951f7f4289b4d3a0c1bc226f/fairseq/modules/checkpoint_activations.py
def enable_activation_checkpointing(m: torch.nn.Module):
    assert not hasattr(m, "_forward")
    m._forward = m.forward
    m.forward = partial(checkpoint, m.forward)


def get_loss(model: Transformer, batch: torch.Tensor):
    logits = model(batch)[:, :-1].flatten(0, 1)
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
        for i in tqdm(range(0, len(stories), chunk_size), desc="Tokenizing TinyStories"):
            chunk = stories[i : min(i + chunk_size, len(stories))]
            tokens_list.extend(tokenizer.Encode(chunk, add_bos=True, add_eos=True, num_threads=4))

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
    # default config is 470M
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--ffn_size", type=int, default=4096)
    parser.add_argument("--head_dim", type=int, default=64)

    parser.add_argument("--quantize")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)

    parser.add_argument("--optim", default="AdamW")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--project", default="int8_quantized_training")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    config = ModelArgs(
        block_size=args.seq_len,
        n_layer=args.depth,
        n_head=args.d_model // args.head_dim,
        dim=args.d_model,
        intermediate_size=args.ffn_size,
    )
    model = Transformer(config).bfloat16().cuda()
    with torch.device("cuda"):
        model.setup_caches(args.batch_size, args.seq_len, training=True)
    if args.activation_checkpointing:
        for layer in model.layers:
            enable_activation_checkpointing(layer)
    if args.quantize == "int8_weight_only":
        quantize_(model, int8_weight_only_quantized_training(), set_inductor_config=False)
    elif args.quantize is not None:
        raise ValueError(f"Unsupported quantize={args.quantize}")
    print(f"No. of params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")

    # only use optimizers from torchao.prototype.low_bit_optim to support quantized training
    if args.optim == "AdamW":
        args.optim = "_AdamW"
    optim = getattr(low_bit_optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data = get_tinystories().cuda()
    run = wandb.init(dir="/tmp", config=args, project=args.project, name=args.run_name)

    step = 0
    log_interval = 50
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()
    _get_loss = torch.compile(get_loss) if args.compile else get_loss

    while step < args.n_steps:
        # randomly select a continuous chunk, then reshape it
        idx = torch.randint(0, data.shape[0] - args.batch_size * args.seq_len, (1,)).item()
        batch = data[idx : idx + args.batch_size * args.seq_len].view(args.batch_size, args.seq_len).long()

        loss = _get_loss(model, batch)
        loss.backward()

        if step % log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                lr=optim.param_groups[0]["lr"],
                max_memory_allocated=torch.cuda.max_memory_allocated() / 1e9,
                max_memory_active=torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1e9,
            )
            run.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()

    run.finish()
