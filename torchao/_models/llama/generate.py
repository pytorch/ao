# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import torch
import torchao
import torch._dynamo.config
import torch._inductor.config
from torchao.utils import get_model_size_in_bytes
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from torchao._models.llama.model import Transformer, prepare_inputs_for_model
from torchao._models.llama.tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
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

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            next_token, next_prob = next_token.clone(), next_prob.clone()
            input_pos += 1
            new_tokens.append(next_token)
            callback(new_tokens[-1])
            new_probs.append(next_prob)
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
    interactive: bool,
    callback = lambda x: x,
    kv_cache_quantization: bool = False,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool=False,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    device = prompt.device
    T = prompt.numel()

    # calculate how many tokens to generate based on max_new_tokens and model's upper bound (block_size)
    max_seq_length = min(T + max_new_tokens, model.config.block_size) if not interactive else 350
    new_tokens = max_seq_length - T

    # full prompt+output will be stored in seq
    seq = torch.empty(max_seq_length, dtype=prompt.dtype, device=device)
    seq[:T] = prompt.view(-1)

    # setup model caches
    with torch.device(device):
        if cache_size is None:
            cache_size = max_seq_length
        assert cache_size >= max_seq_length, "need cache_size to be greater than max_new_tokens + size-of-prompt"
        model.setup_caches(max_batch_size=1, max_seq_length=cache_size, kv_cache_quantization=kv_cache_quantization, linear_causal_mask=linear_causal_mask, prompt_length=T)

    # format model input
    x, input_pos = prepare_inputs_for_model(prompt, max_new_tokens)

    # execute prefill
    next_token = prefill(model, x, input_pos, **sampling_kwargs).clone()
    seq[T] = next_token
    # execute token generation
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, new_tokens-1, callback=callback, **sampling_kwargs)

    seq = torch.cat((seq[:T+1], *generated_tokens))

    return seq

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision):
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]

    model = Transformer.from_name(checkpoint_path.parent.name)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=precision)

    return model.eval()

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    quantization: Optional[str] = None,
    calibration_limit: int = 10,
    calibration_seq_length: int = 256,
    kv_cache_quantization: bool = False,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool=False,
    save: bool = False,
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    memory_profile: Optional[Path] = None,
    device=default_device,
    precision=torch.bfloat16,
    write_result: Optional[Path] = None,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """

    torchao.quantization.utils.recommended_inductor_config_setter()

    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    print(f"Using device={device}")
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision)


    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)


    if quantization:
        from torchao.quantization.quant_api import (
            quantize_,
            int8_weight_only,
            int8_dynamic_activation_int8_weight,
            int4_weight_only,
            fpx_weight_only,
            uintx_weight_only,
            autoquant,
            unwrap_tensor_subclass,
            float8_weight_only,
            float8_dynamic_activation_float8_weight,
        )
        from torchao.quantization.granularity import PerTensor, PerRow
        if "spinquant" in quantization:
            from torchao.prototype.spinquant import apply_spinquant
            apply_spinquant(model)
        if "int8wo" in quantization:
            quantize_(model, int8_weight_only())
        if "int8dq" in quantization:
            quantize_(model, int8_dynamic_activation_int8_weight())
        if "int4wo" in quantization:
            if "hqq" in quantization:
                use_hqq=True
            else:
                use_hqq=False
            groupsize=int(quantization.split("-")[1])
            assert groupsize in [32,64,128,256], f"int4wo groupsize needs to be one of [32,64,128,256] but got {groupsize}"
            quantize_(model, int4_weight_only(group_size=groupsize))
        if "marlin" in quantization:
            from torchao.dtypes import MarlinSparseLayout
            quantize_(model, int4_weight_only(layout=MarlinSparseLayout()))
        if "fp6" in quantization:
            quantize_(model, fpx_weight_only(3, 2))
        if quantization.startswith("awq"):
            from torchao._models._eval import TransformerEvalWrapper
            from torchao.utils import TORCH_VERSION_AT_LEAST_2_3
            from torchao.prototype.awq.example import get_calib_dataset
            if not TORCH_VERSION_AT_LEAST_2_3:
                print("Awq requires torch2.3+")
                exit()
            from torchao.prototype.awq import insert_awq_observer_, awq_uintx, AWQObservedLinear
            quant_dtype = quantization.split("-")[1]
            group_size = int(quantization.split("-")[2])
            quant_dtype = getattr(torch, quant_dtype, torch.uint8)
            model=model.to(device)
            # get calibration data
            insert_awq_observer_(model, calibration_limit, calibration_seq_length, quant_dtype=quant_dtype, group_size=group_size)
            TransformerEvalWrapper(
                model=model.to(device),
                tokenizer=tokenizer,
                max_seq_length=calibration_seq_length,
                input_prep_func=prepare_inputs_for_model,
                device=device,
            ).run_eval(
                tasks=['wikitext'], 
                limit=calibration_limit,
            )
            is_observed_linear = lambda m, fqn: isinstance(m, AWQObservedLinear)
            use_hqq = "hqq" in quantization
            quantize_(model, awq_uintx(quant_dtype=quant_dtype, group_size = group_size, use_hqq=use_hqq), is_observed_linear)
        if "uintx" in quantization:
            # uintx-nbits-groupsize, e.g. "uintx-2-64"
            if "hqq" in quantization:
                # uintx-nbits-groupsize-hqq
                use_hqq = True
            else:
                use_hqq = False
            _quant_args = quantization.split("-")
            nbits = int(_quant_args[1])
            assert nbits >= 1 and nbits <= 8, "nbits must be 1 to 8"
            _NBITS_TO_DTYPE = {1: torch.uint1, 2: torch.uint2, 3: torch.uint3, 4: torch.uint4, 5: torch.uint5, 6: torch.uint6, 7: torch.uint7, 8: torch.uint8}
            dtype = _NBITS_TO_DTYPE[nbits]
            group_size = int(_quant_args[2])
            quantize_(model, uintx_weight_only(dtype, group_size, use_hqq=use_hqq))
        if "float8wo" in quantization:
            quantize_(model, float8_weight_only())
        if "float8dq" in quantization:
            granularity = str(quantization.split("-")[-1])
            if granularity=="tensor":
                granularity = PerTensor()
            elif granularity=="row":
                granularity = PerRow()
            else:
                granularity = PerTensor()
            quantize_(model, float8_dynamic_activation_float8_weight(granularity=granularity))
        if "autoquant" in quantization:
            if "autoquant-int4" == quantization:
                model = autoquant(model, manual=True, qtensor_class_list = torchao.quantization.DEFAULT_INT4_AUTOQUANT_CLASS_LIST)
            elif "autoquant-float8" == quantization:
                model = autoquant(model, manual=True, qtensor_class_list = torchao.quantization.OTHER_AUTOQUANT_CLASS_LIST)
            else:
                model = autoquant(model, manual=True)

            generate(
                model,
                encode_tokens(tokenizer, prompt, bos=True, device=device),
                max_new_tokens,
                interactive=False,
                temperature=temperature,
                top_k=top_k,
            )

            # do autoquantization
            model.finalize_autoquant()
        else:
            if not TORCH_VERSION_AT_LEAST_2_5:
                unwrap_tensor_subclass(model)

    model_size = get_model_size_in_bytes(model, ignore_embeddings=True) / 1e9

    if save:
        output_dir = str(checkpoint_path.cwd())
        filename = str(checkpoint_path.name).split(".")[0]
        torch.save(model.state_dict(), os.path.join(output_dir, filename + f"-{quantization}.pt"))

    if compile:
        print("Compiling Model")
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    if memory_profile:
        torch.cuda.memory._record_memory_history(True,trace_alloc_max_entries=250000, trace_alloc_record_context=True)
    aggregate_metrics = {
        'tokens_per_sec': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        if i==0:
            torch.cuda.reset_peak_memory_stats()
        device_sync(device=device) # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                kv_cache_quantization=kv_cache_quantization,
                cache_size=cache_size,
                linear_causal_mask=linear_causal_mask,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        if not interactive:
                tok_list = y.tolist()
                # truncate text after end of string token
                tokens = tok_list if not tokenizer.eos_id() in y else tok_list[:tok_list.index(tokenizer.eos_id())]
                print(tokenizer.decode(tokens))
        else:
            print()
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec:.02f} GB/s")

        if memory_profile and i==0:
            snapshot = torch.cuda.memory._snapshot()
            with open(f"{memory_profile}.pickle", 'wb') as f:
                from pickle import dump
                dump(snapshot, f)
            print(
                f"\nmemory profile {memory_profile}.pickle saved, to convert that to a usable file, use",
                "python pytorch/torch/cuda/_memory_viz.py trace_plot <pickle file> -o <desired output name>.html"
            )
            break

    print("==========")

    tokpersec = torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item()
    bandwidth = model_size * tokpersec
    mem = torch.cuda.max_memory_reserved() /1e9
    print(f"Average tokens/sec: {tokpersec:.2f}")
    print(f"Average Bandwidth: {bandwidth:.02f} GB/s")
    print(f"Peak Memory Usage: {mem:.02f} GB")
    print(f"Model Size: {model_size:.02f} GB")
    if write_result:
        result_txt = f"\n{datetime.today().strftime('%Y%m%d%H%M%S')}, tok/s={tokpersec:6.2f}, mem/s={bandwidth:7.2f} GB/s, peak_mem={mem:5.2f} GB, model_size={model_size:5.2f} GB "
        result_txt += f"quant: {quantization}, mod: {checkpoint_path.parent.name}, kv_quant: {kv_cache_quantization}, compile: {compile}, compile_prefill: {compile_prefill}, dtype: {precision}, device: {device} "
        result_txt += f"repro: python generate.py "
        result_txt += f"--quantization {quantization} " if quantization else ""
        result_txt += f"--checkpoint_path {checkpoint_path} "
        result_txt += f"--device {device} "
        result_txt += f"--precision {precision} "
        result_txt += f"--compile " if compile else ""
        result_txt += f"--compile_prefill " if compile_prefill else ""
        result_txt += f"--profile {profile} " if profile else ""
        result_txt += f"--profile {memory_profile} " if memory_profile else ""
        result_txt += f"--interactive " if interactive else ""
        result_txt += f"--num_samples {num_samples} "
        result_txt += f"--max_new_tokens {max_new_tokens} "
        result_txt += f"--top_k {top_k} "
        result_txt += f"--temperature {temperature} "
        result_txt += f"--cache_size {cache_size}" if cache_size else ""
        result_txt += f"--kv_cache_quantization " if kv_cache_quantization else ""
        result_txt += f"--linear_causal_mask " if linear_causal_mask else ""

        f=open(write_result, "a")
        f.write(result_txt)
        f.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("../../../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('-q', '--quantization', type=str, 
        help=(
            'Which quantization techniques to apply: int8dq, int8wo, fp6, int4wo-<groupsize>, int4wo-<groupsize>-hqq, autoquant, '
            +'autoquant-int4, autoquant-float8, uintx-<nbits>-<groupsize>, uintx-<nbits>-<groupsize>-hqq, sparse-marlin, spinquant'
        )
    )
    parser.add_argument("--calibration_limit", type=int, default=10, help="Number of calibration examples")
    parser.add_argument("--calibration_seq_length", type=int, default=256, help="Sequence length for calibration")
    parser.add_argument('--kv_cache_quantization', action='store_true', help='Whether to quantize the KV cache')
    parser.add_argument('--cache_size', type=int, default=None, help='Force size of cache to be a certain number of tokens, if not set, will use max_new_tokens+prompt_size')
    parser.add_argument('--linear_causal_mask', action='store_true', help='Whether to use the memory efficient, but slightly less fast, linear causal mask (important for long context lengths)')
    parser.add_argument('--save', action='store_true', help='Whether to save the quantized model.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--memory_profile', type=Path, default=None, help='filename for memory profile.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--precision', type=lambda x: getattr(torch, x.split(".")[-1]), default=torch.bfloat16, help='dtype precision to use')
    parser.add_argument('--write_result', type=Path, default=None, help='Path where to write the result')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k,
        args.temperature, args.checkpoint_path, args.quantization, args.calibration_limit, args.calibration_seq_length, args.kv_cache_quantization, args.cache_size, args.linear_causal_mask, args.save, args.compile, args.compile_prefill, args.profile, args.memory_profile, args.device, args.precision, args.write_result
    )
