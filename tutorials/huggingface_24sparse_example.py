# This script shows how to accelerate an off-the-shelf 2:4 sparse checkpoint
# using pytorch's `to_sparse_semi_structured`

# It takes advantage of the model checkpoints offered by neuralmagic:
# https://huggingface.co/nm-testing/SparseLlama-3-8B-pruned_50.2of4-FP8

import os
import torch
from torchao.sparsity import sparsify_, semi_sparse_weight

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false" # silence warnings when compiling

torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
torch.set_float32_matmul_precision('high')

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def benchmark(fn, WARMUP=5, N=25):
    time_per_batch = []
    with torch.no_grad():
        # warmup steps
        for _ in range(WARMUP):
            timed(fn)
    
        # benchmark
        for _ in tqdm(range(N)):
            with torch.no_grad():
                _ , time_sec =  timed(fn)
                time_per_batch.append(time_sec)
            
    # each time we generate 128 tokens - 7 for the prompt = 121 tokens at a time.
    total_time = sum(time_per_batch)
    tokens_per_second = 121 * N / total_time
    print(f"Total time: {total_time:.3f}s | Tokens/second: {tokens_per_second:.3f}")

# define model and tokenizer
model = AutoModelForCausalLM.from_pretrained("nm-testing/SparseLlama-3-8B-pruned_50.2of4", torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("nm-testing/SparseLlama-3-8B-pruned_50.2of4")

# Even though we need to pad the matmul shapes from (1, hidden) @ (hidden, output)
# to (8, hidden) @ (hidden, output) we are still able to achieve speedups on 
# the mlp.up and mlp.gate linear layers of the FFN.
def is_mlp_up_or_mlp_gate(mod, name):
    return isinstance(mod, torch.nn.Linear) and ('mlp.gate' in name or 'mlp.up' in name)

# apply sparsity
sparsify_(model, semi_sparse_weight(), filter_fn=is_mlp_up_or_mlp_gate)

# Specify the max length (including both the prompt and the response)
# When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
# with sequence length = `max_length`. The longer the more you will re-use it
model.generation_config.max_length = 128
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.generation_config.cache_implementation = "static"

prompt = "Why dogs are so cute?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# without `torch.compile`: each call takes ~ 5.0 seconds (on A100 80G + torch 2.3)
# Total time: 168.715s | Tokens/second: 17.930
outputs = model.generate(**inputs)
response = tokenizer.batch_decode(outputs)[0]
print(response)

# `torch.compile(model, ...)` is not recommended as you compile callbacks
# and full generate. We recommend compiling only the forward for now. 
# "reduce-overhead" will use cudagraphs.
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

benchmark(lambda: model.generate(**inputs))

# sanity check we get same output as non-compiled model
outputs = model.generate(**inputs)
response = tokenizer.batch_decode(outputs)[0]
print(response)

## Run torch.compile baseline

del model
model = AutoModelForCausalLM.from_pretrained("nm-testing/SparseLlama-3-8B-pruned_50.2of4", torch_dtype=torch.float16).cuda()

model.generation_config.max_length = 128
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.generation_config.cache_implementation = "static"

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
benchmark(lambda: model.generate(**inputs))

outputs = model.generate(**inputs)
response = tokenizer.batch_decode(outputs)[0]
print(response)
