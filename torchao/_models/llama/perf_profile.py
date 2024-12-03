"""

## Performance Profiling Example

An minimal version of `gpt-fast generate.py` that demonstrates usage of `torchao.prototype.profiler.TransformerPerformanceCounter`.
- Outputs from gpt-fast are prefixed with GPT-Fast
- Outputs from `torchao.prototype.profiler.TransformerPerformanceCounter` are prefixed with `TransformerPerfCounter`.

## Usage
```python
python perf_profile.py --prompt "Hello my name is" --checkpoint_path path/to/model.pth --num_samples 1 --max_new_tokens 2 --save_path performance_stats.json
```
where `checkpoint_path` is the checkpoint path of the converted model weights per `gpt-fast` and `save_path` specifies where to save performance stats.


Running the above command for `llama2-7b` should print the following, with accumulated stats saved to `performance_stats.json`

```
Loading model ...
Time to load model: 20.14 seconds

==============================

Using DeviceSpec(device_type=cuda, name=NVIDIA GeForce RTX 3090, dtype=torch.bfloat16, bandwidth=936.1GB/s, flops=35.6TFLOPs, vram=25.4GB)
Model Config: ModelArgs(block_size=2048, vocab_size=32000, n_layer=32, n_head=32, dim=4096, intermediate_size=11008, n_local_heads=32, head_dim=128, rope_base=10000, norm_eps=1e-05)
Active params, Total Params: 6607343616, 6738415616

==============================

TransformerPerfCounter Metrics
PREFILL_SEQLEN-6:
  Latency = 1.26 s
  Tokens
    Total: 6 tokens
    Throughput: 5 tokens/s
  IO
    Total: 13.25 GB
    Throughput: 10.54 GB/s
    Theoretical Latency: 14.15 ms
  FLOPs
    Total: 79.31 GFLOPs
    Throughput: 63.06 GFLOPs/s
    Theoretical Latency: 2.23 ms
  Utilization
    Bandwidth: 0.0113 %
    FLOPs: 0.0018 %

==============================

TransformerPerfCounter Metrics
DECODE_CTX-6_NUM_TOKS-1:
  Latency = 0.16 s
  Tokens
    Total: 1 tokens
    Throughput: 6 tokens/s
  IO
    Total: 13.22 GB
    Throughput: 83.27 GB/s
    Theoretical Latency: 14.13 ms
  FLOPs
    Total: 13.22 GFLOPs
    Throughput: 83.24 GFLOPs/s
    Theoretical Latency: 0.37 ms
  Utilization
    Bandwidth: 0.0890 %
    FLOPs: 0.0023 %

==============================

Generated text for sample 0: Hello, my name is [Name

GPTFast Sample Metrics
  Time for inference 1: 6 prompt tokens 2 tokens generated, 1.57 sec total, 1.28 tokens/sec
  Bandwidth achieved: 17.22 GB/s

==============================

GPTFast Aggregate Stats
  Average tokens/sec: 1.28
  Memory used: 13.51 GB

==============================

TransformerPerfCounter
Performance Summary:
  Latency = 1.42 s
  Tokens
    Total: 7 tokens
    Throughput: 5 tokens/s
  IO
    Total: 26.47 GB
    Throughput: 18.69 GB/s
    Theoretical Latency: 28.28 ms
  FLOPs
    Total: 92.53 GFLOPs
    Throughput: 65.33 GFLOPs/s
    Theoretical Latency: 2.60 ms
  Utilization
    Bandwidth: 0.0200 %
    FLOPs: 0.0018 %

Saving performance results to performance_stats.json
```

**Notes**
- The discrepancy between `gpt-fast` token throughput and that of `TransformerPerformanceCounter` is due to the fact that gpt-fast` only counts generated tokens (no prefill)
-- so even though the `prefill` phase technically generates `len(prompt) + 1` tokens, it counts the number of tokens generated during this phase as `1`,
whereas `TransformerPerformanceCounter` includes all `prefill` tokens in the total token count.
"""

import textwrap
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.nn.attention import SDPBackend

from torchao._models.llama.model import Transformer
from torchao._models.llama.tokenizer import get_tokenizer
from torchao.prototype.profiler import (
    CUDADeviceSpec,
    TransformerPerformanceCounter,
    total_model_params,
)

DEVICE_SPEC: CUDADeviceSpec
PERF_COUNTER: TransformerPerformanceCounter
PERF_COUNTER_PREFIX = "TransformerPerfCounter"
GPT_FAST_PREFIX = "GPTFast"
DELIMITER = "\n" + "=" * 30 + "\n"


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet supported")


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
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    seqlen = input_pos.shape[-1]
    num_tokens = input_pos.numel()
    assert num_tokens == seqlen

    step_name = f"prefill_seqlen-{seqlen}".upper()
    with PERF_COUNTER.count(step_name, num_tokens=num_tokens):
        logits = model(x, input_pos)
        next_token = sample(logits, **sampling_kwargs)[0]
    print(DELIMITER)
    stats_str = PERF_COUNTER.print_summary(labels=[step_name], show=False)
    print(f"{PERF_COUNTER_PREFIX} Metrics\n{stats_str}")

    return next_token


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    context_len = input_pos[-1].item()
    num_tokens = input_pos.numel()
    assert input_pos.shape[-1] == 1
    assert num_tokens == 1

    step_name = f"decode_ctx-{context_len}_num_toks-{num_tokens}".upper()
    with PERF_COUNTER.count(step_name, num_tokens=num_tokens):
        logits = model(x, input_pos)
        next_token = sample(logits, **sampling_kwargs)
    print(DELIMITER)
    stats_str = PERF_COUNTER.print_summary(labels=[step_name], show=False)
    print(f"{PERF_COUNTER_PREFIX} Metrics\n{stats_str}")

    return next_token


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
        with torch.nn.attention.sdpa_kernel(
            backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    callback=lambda x: x,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(
        model, prompt.view(1, -1), input_pos, **sampling_kwargs
    ).clone()
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        max_new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )
    seq[T + 1 :] = torch.cat(generated_tokens)

    return seq


def encode_tokens(tokenizer, string, bos=True, device="cuda"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision):
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def main(
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    checkpoint_path: Union[Path, str],
    save_path: Union[Path, str],
    device: str = "cuda",
    precision: torch.dtype = torch.bfloat16,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    print(f"{GPT_FAST_PREFIX}")
    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    global DEVICE_SPEC
    global PERF_COUNTER

    DEVICE_SPEC = CUDADeviceSpec(dtype=precision)
    PERF_COUNTER = TransformerPerformanceCounter(depth=3, device_spec=DEVICE_SPEC)
    print(DELIMITER)
    print(f"{PERF_COUNTER_PREFIX}")
    print(f"Using {DEVICE_SPEC}")
    print(f"Model Config: {model.config}")

    num_active_params = total_model_params(model, exclude_embeddings=True)
    num_params = total_model_params(model, exclude_embeddings=False)
    model_size = num_params * precision.itemsize
    print(f"Active params, Total Params: {num_active_params}, {num_params}")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)

    aggregate_metrics = {
        "tokens_per_sec": [],
    }

    start = 0

    for i in range(start, num_samples):
        t0 = time.perf_counter()

        y = generate(
            model,
            encoded,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        t = time.perf_counter() - t0
        txt = tokenizer.decode(y.tolist())
        print(DELIMITER)
        print(f"{GPT_FAST_PREFIX}")
        print(f"Generated text for sample {i}: {txt}\n")

        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        sample_metrics = textwrap.dedent(f"""\
            {GPT_FAST_PREFIX} Sample Metrics
            Time for inference {i+1}: {prompt_length} prompt tokens {tokens_generated} tokens generated, {t:.02f} sec total, {tokens_sec:.02f} tokens/sec
            Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s""")
        print(
            textwrap.indent(
                sample_metrics,
                prefix="  ",
                predicate=lambda line: not line.startswith(GPT_FAST_PREFIX),
            )
        )
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)

    # First print aggregate stats from original gpt-fast script
    print(DELIMITER)
    gpt_stats = textwrap.dedent(f"""\
        {GPT_FAST_PREFIX} Aggregate Stats
        Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}
        Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB""")

    print(
        textwrap.indent(
            gpt_stats,
            prefix="  ",
            predicate=lambda line: not line.startswith(GPT_FAST_PREFIX),
        )
    )

    # Print performance summary from TransformerPerformanceCounter
    print(DELIMITER)
    total_stats_str = PERF_COUNTER.print_summary(show=False)
    print(f"{PERF_COUNTER_PREFIX}\n{total_stats_str}")
    print(f"\nSaving performance results to {save_path}")
    PERF_COUNTER.to_json(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TransformerPerformanceCounter Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=2, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("./checkpoints/7B/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("performance_stats.json"),
        help="Path to save performance stats.",
    )
    args = parser.parse_args()
    main(**vars(args))
