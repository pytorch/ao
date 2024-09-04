# This script shows how to accelerate an off-the-shelf 2:4 sparse checkpoint
# using pytorch's `to_sparse_semi_structured`

# Also shows how to use marlin

# It takes advantage of the model checkpoints offered by neuralmagic:
# https://huggingface.co/nm-testing/SparseLlama-3-8B-pruned_50.2of4-FP8

import os
import torch
from torchao.sparsity import sparsify_, semi_sparse_weight

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.utils import benchmark_model, profiler_runner
from torchao.quantization import int4_weight_only, quantize_
from torchao.dtypes import MarlinSparseLayoutType

os.environ["TOKENIZERS_PARALLELISM"] = "false" # silence warnings when compiling
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
warmup = 5
num_runs = 25

torch.set_float32_matmul_precision('high')


# Even though we need to pad the matmul shapes from (1, hidden) @ (hidden, output)
# to (8, hidden) @ (hidden, output) we are still able to achieve speedups on 
# the mlp.up and mlp.gate linear layers of the FFN.
def is_mlp_up_or_mlp_gate(mod, name):
    return isinstance(mod, torch.nn.Linear) and ('mlp.gate' in name or 'mlp.up' in name)

def run_benchmark(compression_config="baseline", dtype=torch.float16):
    print (f"\n Running: {compression_config} benchmark with dtype={dtype}\n")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Why dogs are so cute?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Specify the max length (including both the prompt and the response)
    # When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
    # with sequence length = `max_length`. The longer the more you will re-use it
    model.generation_config.max_length = 128
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.cache_implementation = "static"

    if compression_config == "24_sparse":
        sparsify_(model, semi_sparse_weight(), filter_fn=is_mlp_up_or_mlp_gate)
    elif compression_config == "int4_wo":
        assert dtype == torch.bfloat16, "int4 quantization only works with bf16"
        quantize_(model, int4_weight_only())
    elif compression_config == "sparse_marlin":
        assert dtype == torch.float16, "sparse_marlin only works with fp16"
        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
    elif compression_config == "baseline":
        pass
    else:
        raise ValueError(f"Unknown compression config: {compression_config}")

    # `torch.compile(model, ...)` is not recommended as you compile callbacks
    # and full generate. We recommend compiling only the forward for now. 
    # "reduce-overhead" will use cudagraphs.
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    # WARMUP
    benchmark_model(lambda: model.generate(**inputs), warmup, device_type="cuda")
    # res is in ms so multiply by 1000 to get tok/s
    res = benchmark_model(lambda: model.generate(**inputs), num_runs, device_type="cuda")
    tokens_per_second = 1000 * (121 / res)
    print(f"Average time: {res:.3f}ms | Tokens/second: {tokens_per_second:.3f}")

    # sanity check we get same output as non-compiled model
    outputs = model.generate(**inputs)
    response = tokenizer.batch_decode(outputs)[0]
    print(response)

    del model

## baseline
# run_benchmark(compression_config="baseline", dtype=torch.bfloat16)

# # ## int4_wo
# run_benchmark(compression_config="int4_wo", dtype=torch.bfloat16)

# ## sparse marlin
run_benchmark(compression_config="sparse_marlin", dtype=torch.float16)

## sparse 
# run_benchmark(compression_config="24_sparse", dtype=torch.bfloat16)
