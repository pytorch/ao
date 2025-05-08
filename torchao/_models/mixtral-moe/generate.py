# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

from torchao.utils import get_model_size_in_bytes

torch.manual_seed(0)


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
torch._dynamo.config.capture_scalar_outputs = True

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from sentencepiece import SentencePieceProcessor


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            next_token, next_prob = next_token.clone(), next_prob.clone()

            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    callback=lambda x: x,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    device, _ = prompt.device, prompt.dtype

    T = prompt.size(-1)
    max_seq_length = (
        min(T + max_new_tokens, model.config.block_size) if not interactive else 350
    )
    new_tokens = max_seq_length - T

    # duplicate prompt for batch_size
    prompt = prompt.repeat(batch_size, 1)

    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty(batch_size, max_seq_length, dtype=prompt.dtype, device=device)
    seq[:, :T] = prompt

    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(
        model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs
    )
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(batch_size, -1),
        input_pos,
        new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )
    seq = torch.cat((seq[:, : T + 1], *generated_tokens), dim=-1)

    return seq


def encode_tokens(tokenizer, string, bos=True, device="cuda"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision):
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    try:
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
    except:
        model = Transformer.from_name(checkpoint_path.parent.name)

    model = model.to(device=device, dtype=precision)
    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/mistralai/Mixtral-8x7B-v0.1/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    moe_quant: Optional[str] = None,
    profile: Optional[Path] = None,
    memory_profile: Optional[Path] = None,
    device="cuda",
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)
    print(f"Using device={device}")
    precision = torch.bfloat16
    is_chat = "chat" in str(checkpoint_path)

    if device == "cuda" and memory_profile is not None:
        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=500000, trace_alloc_record_context=True
        )

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, "cpu", precision)

    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    t0 = time.time()

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )

    from torchao.quantization.prototype.moe_quant.utils import (
        MoEQuantConfig,
        cond_ffn_filter,
        UseFakeExtraDimTensor
    )
    from torchao.quantization.quant_api import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8DynamicActivationIntxWeightConfig,
        Int8WeightOnlyConfig,
        PackedLinearInt8DynamicActivationIntxWeightLayout,
        PerRow,
        quantize_,
    )

    if moe_quant:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        config = None
        if "int8wo-base" in moe_quant:
            config = MoEQuantConfig(Int8WeightOnlyConfig())

        elif "int8wo" in moe_quant:
            config = MoEQuantConfig(Int8WeightOnlyConfig(), use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE)

        elif "int8dq-base" in moe_quant:
            config = MoEQuantConfig(Int8DynamicActivationInt8WeightConfig())

        elif "int8dq" in moe_quant:
            config = MoEQuantConfig(Int8DynamicActivationInt8WeightConfig(), use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE)

        elif "int4wo-base" in moe_quant:
            config = MoEQuantConfig(Int4WeightOnlyConfig())

        elif "int4wo" in moe_quant:
            config = MoEQuantConfig(Int4WeightOnlyConfig(), use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE)

        elif "fp8wo-base" in moe_quant:
            config = MoEQuantConfig(Float8WeightOnlyConfig())

        elif "fp8wo" in moe_quant:
            config = MoEQuantConfig(Float8WeightOnlyConfig(), use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE)

        elif "fp8dq-base" in moe_quant:
            config = MoEQuantConfig(Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

        elif "fp8dq" in moe_quant:
            config = MoEQuantConfig(
                Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
                use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE,
            )

        elif "intxdq" in moe_quant:
            config = MoEQuantConfig(
                Int8DynamicActivationIntxWeightConfig(
                    layout=PackedLinearInt8DynamicActivationIntxWeightLayout(),
                ),
                use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE,
            )
        else:
            assert config is not None, (
                f"expected moe_quant to match one of the options but got {moe_quant}"
            )

        if config is not None:
            quantize_(model, config, filter_fn=cond_ffn_filter, device=device)
            print(
                f"Time to apply quantization with config {config} to model: {time.time() - t0:.02f} seconds"
            )

    model.to(device=device)
    device_sync(device=device)

    if compile:
        # moe quant + compile causes repeated warnings
        import warnings

        warnings.simplefilter("ignore", lineno=84)
        warnings.simplefilter("ignore", lineno=105)

        torch._inductor.config.assert_indirect_indexing = False

        global decode_one_token, prefill

        if batch_size == 1 and (isinstance(moe_quant, str) and "base" in moe_quant):
            decode_one_token = torch.compile(
                decode_one_token, mode="reduce-overhead", fullgraph=True
            )
        else:
            decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead")

        if args.compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device)  # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        import contextlib

        if i != num_samples - 1 or not profile:
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0

        if not interactive:
            pass
            print(tokenizer.decode(y[0].tolist()))
        else:
            print()
        tokens_generated = y.size(-1) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        if i == 0 and device == "cuda" and memory_profile is not None:
            snapshot = torch.cuda.memory._snapshot()
            with open(f"{memory_profile}.pickle", "wb") as f:
                from pickle import dump

                dump(snapshot, f)
            print(
                f"\nmemory profile {memory_profile}.pickle saved, to convert that to a usable file, use",
                "python pytorch/torch/cuda/_memory_viz.py trace_plot <pickle file> -o <desired output name>.html",
            )
            break

    tokpersec = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    print(f"Average tokens/sec: {tokpersec:.2f}")
    if batch_size > 1:
        print(f"Average tokens/sec including batches {batch_size * tokpersec:.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    print(f"model size: {get_model_size_in_bytes(model) / 1e9:.02f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to benchmark with"
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    # parser.add_argument('-q', '--quantization', type=str, help='Which quantization techniques to apply: int8dq, int8wo, int4wo, fp8')
    parser.add_argument(
        "--moe_quant",
        type=str,
        help="Which quantization techniques to apply: int8dq, int8wo, int4wo, fp8wo, fp8dq",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--memory_profile", type=Path, default=None, help="filename for memory profile."
    )
    parser.add_argument("--device", type=str, default="cuda", help="device to use")

    args = parser.parse_args()
    print(args)
    main(
        args.prompt,
        args.interactive,
        args.num_samples,
        args.max_new_tokens,
        args.batch_size,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.compile_prefill,
        args.moe_quant,
        args.profile,
        args.memory_profile,
        args.device,
    )
